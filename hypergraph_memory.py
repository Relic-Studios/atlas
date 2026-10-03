"""Per-agent semantic hypergraph memory with Hebbian learning and decay (BGE-M3).

Structure (one store per agent, never shared):
  node      = a concept key: a person ("@maya") or a content word ("guitar")
  hyperedge = one memory (a thing someone said / an approved fact). It joins ALL the
              nodes it mentions at once, and carries a BGE-M3 dense embedding.
  hebb      = learned edge<->edge association weights ("fired together").

Retrieval (no LLM, ever -- the mem0 lesson):
  1. cosine(query, every edge)            -> semantic seeds
  2. spread one hop from the seeds through shared nodes and Hebbian links
  3. score = similarity + spread + small strength bonus, relevance-gated, top-k

Learning:
  - Hebbian: memories retrieved AND actually used in the reply are strengthened, and
    every pair of them gets a stronger association link. New memories link to what was
    active when they were formed (temporal association).
  - Decay: every weight decays exponentially with time since it was last touched
    (lazy: stored w + t, decayed on read). Weak, old, unpinned memories are pruned.
    Owner-approved memories are pinned (never pruned, decay floored).

Only USER lines are stored (never the agent's own replies: self-copying caused every
loop we fought). S-tags, PII and slurs are scrubbed before storage.

Storage: agent_state/hypergraph/<agent>/{graph.json,emb.npy}. agent_state/ is
gitignored and excluded from the public export -- dev agents' memories never ship.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent / "agent_state" / "hypergraph"
ENABLED = os.environ.get("ATLAS_HG", "1") != "0"
MODEL = os.environ.get("ATLAS_HG_MODEL", "BAAI/bge-m3")
DEVICE = os.environ.get("ATLAS_HG_DEVICE", "cpu")   # cpu by default: never share the live CUDA ctx

HALF_LIFE_S = float(os.environ.get("ATLAS_HG_HALF_LIFE_H", "72")) * 3600.0
HEBB_HALF_LIFE_S = float(os.environ.get("ATLAS_HG_HEBB_HALF_LIFE_H", "168")) * 3600.0
ETA = 0.25            # edge strengthening rate on use
ETA_PAIR = 0.30       # pair association rate on co-use
ETA_TEMPORAL = 0.10   # link new memory -> currently active memories
INIT_W = 0.35         # strength of a fresh memory
PIN_FLOOR = 0.6       # approved memories never decay below this
PRUNE_W = 0.04        # unpinned memories weaker than this (and old) are dropped
PRUNE_MIN_AGE_S = 6 * 3600.0
MAX_EDGES = int(os.environ.get("ATLAS_HG_MAX", "4000"))
MIN_SIM = float(os.environ.get("ATLAS_HG_MIN_SIM", "0.53"))   # relevance gate (calibrated: sim_hypergraph)
NAME_SIM = 0.42      # gate when the query names the memory's person
SPREAD_SIM = 0.33     # a spread-reached memory still needs this much direct similarity
SEEDS = 12
NODE_SPREAD = 0.15
PERSON_SPREAD = 0.45
HEBB_RECALL = 0.05
PAIR_SIM = 0.48
PAIR_DISTINCT = 0.85  # two "same person" memories must differ this much to count as joint evidence
HUB_PENALTY = float(os.environ.get("ATLAS_HG_HUB", "1.0"))
RARE_DEG = 3            # a shared content word this rare counts like a name       # see pair_ok in retrieve()
SPREAD_SCORE = 0.3    # spread lets a memory IN; its own similarity still ranks it     # a strongly co-used memory comes along even if not similar   # same-person bridge (multi-hop: "the guy who moved to Denver, ...")
HEBB_SPREAD = 0.20
STRENGTH_BONUS = 0.05

_STOP = set("""a an the and or but so to of in on at for with is are was were be been being am i you he she
it we they me him her us them my your his its our their this that these those there here what which who whom
whose when where why how do does did done doing have has had having not no yes yeah yep nah ok okay just like
really very much many more most some any all can could would should will shall may might must if then than
too also only even still about into over under out up down off again once oh um uh hmm lol bro dude man guys
get got gonna wanna kinda sorta thing things stuff something anything nothing everything one two lot
know think thought say said tell told go going went come came see saw look make made want need let right
well now time today yesterday tomorrow im ive dont cant wont didnt isnt its thats whats youre theyre were
wait anyway anyways honestly literally basically actually maybe sure fine cool nice good bad huh yo hey""".split())
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_TAG = re.compile(r"\[[^\]]*\]|\bS\d{1,4}\b")


UNCOMMON_ZIPF = 4.2     # "play"/"favourite" (~5) are everyday words, "ska"/"chess" (<4) carry meaning
try:
    from wordfreq import zipf_frequency as _zipf
except Exception:  # pragma: no cover
    _zipf = None


@functools.lru_cache(maxsize=8192)
def _uncommon(word: str) -> bool:
    """A shared word only counts as evidence if it's uncommon in English too, not just
    rare in a small store (a 40-memory graph makes 'play' look rare)."""
    if _zipf is None:
        return True
    return _zipf(word.lstrip("@"), "en") < UNCOMMON_ZIPF


def _norm_word(w: str) -> str:
    w = w.strip("'-")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content_nodes(text: str) -> list:
    out = []
    for w in _WORD.findall(_TAG.sub(" ", text or "").lower().replace("’", "'")):
        w = _norm_word(w.replace("'", ""))
        if len(w) >= 3 and w not in _STOP and w not in out:
            out.append(w)
    return out[:24]


def _clean(text: str) -> str:
    """Scrub before storage: speaker tags, slurs (whole memory dropped), emails, phone numbers."""
    t = _TAG.sub(" ", text or "")
    try:
        import speech_safety
        if speech_safety.has_slur(t):
            return ""
    except Exception:  # noqa: BLE001
        pass
    t = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "[email]", t)
    t = re.sub(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)", "[number]", t)
    return " ".join(t.split())[:300]


def embed_text(text: str, who: str = "") -> str:
    """What gets embedded: the speaker's name is part of the memory's meaning."""
    return f"{who}: {text}" if who else text


# ---------------------------------------------------------------- embedder
class Embedder:
    """Lazy BGE-M3 dense embedder. Thread-safe; tiny LRU for repeated queries."""

    def __init__(self, model: str = MODEL, device: str = DEVICE):
        self.model_name, self.device = model, device
        self._m = None
        self._lock = threading.Lock()
        self._cache: dict = {}
        self.dim = 1024

    def _load(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(self.model_name, device=self.device)
            self._m.max_seq_length = 256
            self.dim = self._m.get_sentence_embedding_dimension()
        return self._m

    def encode(self, texts: list) -> np.ndarray:
        texts = [t or "" for t in texts]
        with self._lock:
            m = self._load()
            miss = [t for t in texts if t not in self._cache]
            if miss:
                v = m.encode(miss, batch_size=16, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
                for t, e in zip(miss, v):
                    self._cache[t] = e
                if len(self._cache) > 512:
                    for k in list(self._cache)[:256]:
                        self._cache.pop(k, None)
            return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


_SHARED_EMBEDDER = None
_EMB_LOCK = threading.Lock()


def shared_embedder() -> Embedder:
    global _SHARED_EMBEDDER
    with _EMB_LOCK:
        if _SHARED_EMBEDDER is None:
            _SHARED_EMBEDDER = Embedder()
        return _SHARED_EMBEDDER


# ---------------------------------------------------------------- the graph
class HyperMemory:
    def __init__(self, agent: str, root: Path = None, embedder=None, clock=time.time,
                 autosave: bool = True):
        self.agent = (agent or "unknown").lower()
        self.dir = Path(root or ROOT) / self.agent
        self.emb_model = embedder
        self.clock = clock
        self.autosave = autosave
        self._lock = threading.RLock()
        self.edges: list = []           # dicts: id,text,nodes,who,kind,w,t,hits,pinned,created
        self.vecs = np.zeros((0, 0), np.float32)
        self.hebb: dict = {}            # "idA|idB" (sorted) -> [w, t]
        self.node_index: dict = {}      # node -> set(edge idx)
        self.next_id = 1
        self.active: list = []          # ids active in the current turn (temporal Hebbian)
        self._dirty = False
        self._load()

    # ---- persistence ------------------------------------------------------
    def _load(self):
        g, e = self.dir / "graph.json", self.dir / "emb.npy"
        if not g.exists():
            return
        try:
            d = json.loads(g.read_text(encoding="utf-8"))
            edges = d.get("edges", [])
            vecs = np.load(e) if e.exists() else np.zeros((0, 0), np.float32)
            if len(vecs) != len(edges):
                raise ValueError(f"emb/edges mismatch {len(vecs)} vs {len(edges)}")
            self.edges, self.vecs = edges, vecs.astype(np.float32)
            self.hebb = d.get("hebb", {})
            self.next_id = int(d.get("next_id", len(edges) + 1))
            self._reindex()
        except Exception as ex:  # noqa: BLE001  corrupt store -> keep a copy, start fresh
            logger.warning("hypergraph[%s]: unreadable store (%s); starting empty", self.agent, ex)
            bad = self.dir / f"corrupt_{int(self.clock())}"
            try:
                bad.mkdir(parents=True, exist_ok=True)
                for f in (g, e):
                    if f.exists():
                        os.replace(f, bad / f.name)
            except OSError:
                pass
            self.edges, self.vecs, self.hebb = [], np.zeros((0, 0), np.float32), {}

    def save(self):
        with self._lock:
            if not self._dirty and (self.dir / "graph.json").exists():
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            d = {"agent": self.agent, "next_id": self.next_id, "edges": self.edges, "hebb": self.hebb,
                 "model": getattr(self.emb_model, "model_name", "")}
            gt, et = self.dir / "graph.tmp", self.dir / "emb.tmp.npy"
            gt.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            np.save(et, self.vecs if self.vecs.size else np.zeros((0, 0), np.float32))
            os.replace(et, self.dir / "emb.npy")
            os.replace(gt, self.dir / "graph.json")
            self._dirty = False

    def _reindex(self):
        self._hub = None
        self.node_index = {}
        for i, e in enumerate(self.edges):
            for n in e["nodes"]:
                self.node_index.setdefault(n, set()).add(i)

    # ---- decay ------------------------------------------------------------
    def _w(self, e: dict, now: float) -> float:
        w = e["w"] * 0.5 ** (max(0.0, now - e["t"]) / HALF_LIFE_S)
        return max(w, PIN_FLOOR) if e.get("pinned") else w

    def _h(self, key: str, now: float) -> float:
        v = self.hebb.get(key)
        return v[0] * 0.5 ** (max(0.0, now - v[1]) / HEBB_HALF_LIFE_S) if v else 0.0

    @staticmethod
    def _pk(a: str, b: str) -> str:
        return f"{a}|{b}" if a < b else f"{b}|{a}"

    # ---- writing ----------------------------------------------------------
    def add(self, text: str, who: str = "", kind: str = "episodic", pinned: bool = False,
            vec: np.ndarray = None) -> str | None:
        text = _clean(text)
        nodes = content_nodes(text)
        if len(nodes) < 2:
            return None
        who = (who or "").strip()
        if who and not re.match(r"^S\d+$", who):
            nodes = [f"@{who.lower()}"] + nodes
        else:
            who = ""
        if vec is None:
            vec = self.emb_model.encode([embed_text(text, who)])[0]
        now = self.clock()
        with self._lock:
            # near-duplicate -> reinforce the existing memory instead of storing it twice
            if len(self.edges) and self.vecs.shape[1] == vec.shape[0]:
                sims = self.vecs @ vec
                j = int(np.argmax(sims))
                if sims[j] >= 0.95:
                    e = self.edges[j]
                    e["w"] = min(1.0, self._w(e, now) + ETA * (1 - self._w(e, now)))
                    e["t"], e["hits"] = now, e.get("hits", 0) + 1
                    e["pinned"] = e.get("pinned") or pinned
                    self._dirty = True
                    return e["id"]
            eid = f"{self.agent}-e{self.next_id}"
            self.next_id += 1
            e = {"id": eid, "text": text, "nodes": nodes, "who": who, "kind": kind,
                 "w": max(INIT_W, PIN_FLOOR if pinned else 0.0), "t": now, "created": now,
                 "hits": 0, "pinned": bool(pinned)}
            self.edges.append(e)
            self.vecs = vec[None, :].astype(np.float32) if self.vecs.size == 0 else \
                np.vstack([self.vecs, vec[None, :].astype(np.float32)])
            i = len(self.edges) - 1
            self._hub = None
            for n in nodes:
                self.node_index.setdefault(n, set()).add(i)
            for a in self.active[-6:]:             # temporal Hebbian: formed while these were active
                k = self._pk(eid, a)
                h = self._h(k, now)
                self.hebb[k] = [h + ETA_TEMPORAL * (1 - h), now]
            self._dirty = True
            if len(self.edges) > MAX_EDGES:
                self.maintain(force=True)
            return eid

    def reinforce(self, ids: list, used_ids: list = None):
        """Hebbian update after a turn. `used_ids` = retrieved memories the reply drew on."""
        now = self.clock()
        used = list(dict.fromkeys(used_ids if used_ids is not None else ids))
        with self._lock:
            by_id = {e["id"]: e for e in self.edges}
            for i in used:
                e = by_id.get(i)
                if e:
                    w = self._w(e, now)
                    e["w"], e["t"], e["hits"] = w + ETA * (1 - w), now, e.get("hits", 0) + 1
            for x in range(len(used)):
                for y in range(x + 1, len(used)):
                    if used[x] in by_id and used[y] in by_id:
                        k = self._pk(used[x], used[y])
                        h = self._h(k, now)
                        self.hebb[k] = [h + ETA_PAIR * (1 - h), now]
            self._dirty = bool(used) or self._dirty

    def maintain(self, force: bool = False):
        """Prune weak/old unpinned memories and dead Hebbian links; cap size."""
        now = self.clock()
        with self._lock:
            keep = []
            for i, e in enumerate(self.edges):
                w = self._w(e, now)
                if e.get("pinned") or w >= PRUNE_W or now - e["created"] < PRUNE_MIN_AGE_S:
                    keep.append((w, i))
            if len(keep) > MAX_EDGES:
                pinned = [k for k in keep if self.edges[k[1]].get("pinned")]
                rest = sorted([k for k in keep if not self.edges[k[1]].get("pinned")], reverse=True)
                keep = pinned + rest[:max(0, MAX_EDGES - len(pinned))]
            idx = sorted(i for _, i in keep)
            if len(idx) != len(self.edges):
                self.edges = [self.edges[i] for i in idx]
                self.vecs = self.vecs[idx] if len(idx) else np.zeros((0, 0), np.float32)
                self._reindex()
                self._dirty = True
            alive = {e["id"] for e in self.edges}
            for k in list(self.hebb):
                a, b = k.split("|", 1)
                if a not in alive or b not in alive or self._h(k, now) < 0.01:
                    del self.hebb[k]
                    self._dirty = True

    # ---- reading ----------------------------------------------------------
    def retrieve(self, query: str, k: int = 4, qvec: np.ndarray = None, mark_active: bool = True) -> list:
        """Top-k relevant memories: [{id,text,who,score,sim,age_s}]. Empty if nothing relevant."""
        query = _clean(query)
        if not query.strip():
            return []
        with self._lock:
            n = len(self.edges)
            if n == 0:
                return []
        if qvec is None:
            qvec = self.emb_model.encode([query])[0]
        now = self.clock()
        qnodes = set(content_nodes(query))
        if not qnodes:          # "lol", "anyway", "wait what": nothing to remember about
            return []
        with self._lock:
            if self.vecs.shape[1] != qvec.shape[0]:
                return []
            raw = self.vecs @ qvec
            # hubness: generic memories ("I watched some videos") sit close to every
            # query; subtract how much more central than average each one is
            hub = self._hub_scores()
            sims = raw - HUB_PENALTY * hub
            seeds = np.argsort(-sims)[:SEEDS]
            act = {int(i): float(sims[i]) for i in seeds}
            spread: dict = {}
            hspread: dict = {}
            id_to_i = {e["id"]: i for i, e in enumerate(self.edges)}
            for i in seeds:
                i = int(i)
                # only confident seeds spread: weak ones dragged unrelated memories in
                a = float(sims[i]) if sims[i] >= MIN_SIM else 0.0
                if a <= 0:
                    continue
                e = self.edges[i]
                nodes = set(e["nodes"])
                # through shared nodes (hyperedge -> node -> hyperedge)
                for nd in nodes:
                    w_nd = PERSON_SPREAD * a if nd.startswith("@") else NODE_SPREAD * a / max(1, len(nodes))
                    for j in self.node_index.get(nd, ()):
                        if j != i:
                            spread[j] = spread.get(j, 0.0) + w_nd
                # through learned Hebbian links
                for k2, v in list(self.hebb.items()):
                    if e["id"] in k2.split("|"):
                        other = k2.replace(e["id"], "").strip("|")
                        j = id_to_i.get(other)
                        if j is not None:
                            v2 = HEBB_SPREAD * a * self._h(k2, now)
                            spread[j] = spread.get(j, 0.0) + v2
                            hspread[j] = hspread.get(j, 0.0) + v2
            qpeople = {f"@{w}" for w in re.findall(r"[a-z]+", query.lower())} & set(self.node_index)
            # joint evidence: two memories of the SAME person that are both fairly close
            # ("whoever broke their wrist, what's their favourite film?")
            def _specific(i):   # shares a word that's rare here AND uncommon in English
                return any(len(self.node_index.get(nd, ())) <= RARE_DEG and _uncommon(nd)
                           for nd in (qnodes & set(self.edges[i]["nodes"])))
            pair_ok = set()
            by_person: dict = {}
            for i in act:
                if act[i] >= PAIR_SIM:
                    for nd in self.edges[i]["nodes"]:
                        if nd.startswith("@"):
                            by_person.setdefault(nd, []).append(i)
            for lst in by_person.values():
                # joint evidence needs two DIFFERENT memories: near-duplicates of one
                # line ("I watched some videos lol" / "... today") are one fact, not two
                # and one of them must be a confident match on its own (the multi-hop
                # anchor); two loose matches of a chatty person aren't evidence
                if not any(act[x] >= MIN_SIM or (act[x] >= NAME_SIM and _specific(x)) for x in lst):
                    continue
                for x in lst:
                    if any(y != x and float(self.vecs[x] @ self.vecs[y]) < PAIR_DISTINCT for y in lst):
                        pair_ok.add(x)
            cand = set(act) | set(spread)
            out = []
            for i in cand:
                e = self.edges[i]
                s = float(sims[i])
                node_hit = len(qnodes & set(e["nodes"])) / max(1, len(qnodes)) if qnodes else 0.0
                named = bool(qpeople & set(e["nodes"]))
                rare = _specific(i)
                gate = (s >= MIN_SIM or ((named or rare) and s >= NAME_SIM) or i in pair_ok
                        or (s >= SPREAD_SIM and spread.get(i, 0.0) >= 0.03)
                        or hspread.get(i, 0.0) >= HEBB_RECALL)   # learned association alone
                if not gate:
                    continue
                score = s + SPREAD_SCORE * spread.get(i, 0.0) + 0.05 * node_hit + STRENGTH_BONUS * self._w(e, now)
                out.append({"id": e["id"], "text": e["text"], "who": e.get("who", ""),
                            "score": round(score, 4), "sim": round(s, 4),
                            "age_s": now - e["created"], "pinned": e.get("pinned", False)})
            out.sort(key=lambda r: -r["score"])
            out = out[:k]
            if mark_active:
                self.active = (self.active + [r["id"] for r in out])[-12:]
            return out

    def _hub_scores(self) -> np.ndarray:
        """Per-memory centrality minus the store average (cached; recomputed on change)."""
        n = len(self.edges)
        if getattr(self, "_hub_n", -1) != n or getattr(self, "_hub", None) is None:
            if n < 8:
                self._hub = np.zeros(n, np.float32)
            else:
                m = self.vecs @ self.vecs.T
                np.fill_diagonal(m, np.nan)
                c = np.nanmean(m, axis=1)
                self._hub = np.clip(c - float(np.mean(c)), 0.0, None).astype(np.float32)
            self._hub_n = n
        return self._hub

    def stats(self) -> dict:
        now = self.clock()
        with self._lock:
            ws = [self._w(e, now) for e in self.edges]
            return {"agent": self.agent, "edges": len(self.edges), "nodes": len(self.node_index),
                    "links": len(self.hebb), "pinned": sum(1 for e in self.edges if e.get("pinned")),
                    "mean_w": round(float(np.mean(ws)), 3) if ws else 0.0}


# ---------------------------------------------------------------- prompt glue
def _age(s: float) -> str:
    if s < 3600:
        return "earlier this call" if s < 3 * 3600 else "today"
    if s < 86400:
        return "today"
    d = int(s // 86400)
    return "yesterday" if d == 1 else f"{d} days ago"


def note(hits: list) -> str:
    if not hits:
        return ""
    lines = []
    for h in hits:
        who = h["who"] or "someone"
        lines.append(f"- {who} ({_age(h['age_s'])}): {h['text']}")
    return ("THINGS YOU REMEMBER THAT MIGHT MATTER NOW (only use one if it genuinely fits; "
            "never recite the list; don't invent details beyond it):\n" + "\n".join(lines))


def used_by(reply: str, hits: list) -> list:
    """Which retrieved memories the reply actually drew on (content-word overlap)."""
    rw = set(content_nodes(reply))
    out = []
    for h in hits:
        hw = set(content_nodes(h["text"]))
        if hw and len(rw & hw) >= min(2, len(hw)):
            out.append(h["id"])
    return out


class MemoryService:
    """Owns one HyperMemory per agent + a background writer. The reply path only reads."""

    def __init__(self, root: Path = None, embedder=None, clock=time.time):
        self.root, self.clock = root, clock
        self.embedder = embedder
        self.graphs: dict = {}
        self._lock = threading.Lock()
        self._q: "queue.Queue" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._last_hits: dict = {}
        self._worker = threading.Thread(target=self._run, name="HypergraphWriter", daemon=True)
        self._worker.start()

    def graph(self, agent: str) -> HyperMemory:
        agent = (agent or "unknown").lower()
        with self._lock:
            g = self.graphs.get(agent)
            if g is None:
                g = HyperMemory(agent, root=self.root, embedder=self.embedder or shared_embedder(),
                                clock=self.clock)
                self.graphs[agent] = g
            return g

    # called from the live turn handler; never blocks
    def observe_user(self, agent: str, who: str, text: str):
        if not ENABLED or len(content_nodes(text)) < 3:
            return
        try:
            self._q.put_nowait(("add", agent, who, text))
        except queue.Full:
            pass

    def observe_reply(self, agent: str, reply: str):
        hits = self._last_hits.pop((agent or "").lower(), None)
        if hits:
            try:
                self._q.put_nowait(("reinforce", agent, hits, reply))
            except queue.Full:
                pass

    def recall_note(self, agent: str, query: str, k: int = 4, budget_s: float = 0.35) -> str:
        """Retrieve within a time budget; on timeout the turn simply goes without memory."""
        if not ENABLED or not query or len(content_nodes(query)) < 1:
            return ""
        box = {}

        def work():
            try:
                box["hits"] = self.graph(agent).retrieve(query, k=k)
            except Exception as ex:  # noqa: BLE001
                logger.warning("hypergraph recall failed: %s", ex)
        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(budget_s)
        hits = box.get("hits") or []
        self._last_hits[(agent or "").lower()] = hits
        return note(hits)

    def sync_approved(self, agent: str, texts: list):
        for t in texts:
            try:
                self._q.put_nowait(("pin", agent, "", t))
            except queue.Full:
                pass

    def flush(self, timeout: float = 30.0):
        end = time.time() + timeout
        while not self._q.empty() and time.time() < end:
            time.sleep(0.02)
        self._q.join()
        for g in list(self.graphs.values()):
            g.save()

    def close(self):
        self.flush()
        self._stop.set()

    def _run(self):
        last_save = time.time()
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=1.0)
            except queue.Empty:
                job = None
            try:
                if job:
                    op, agent = job[0], job[1]
                    g = self.graph(agent)
                    if op == "add":
                        g.add(job[3], who=job[2])
                    elif op == "pin":
                        g.add(job[3], kind="approved", pinned=True)
                    elif op == "reinforce":
                        hits, reply = job[2], job[3]
                        g.reinforce([h["id"] for h in hits], used_ids=used_by(reply, hits))
            except Exception as ex:  # noqa: BLE001
                logger.warning("hypergraph writer: %s", ex)
            finally:
                if job:
                    self._q.task_done()
            if time.time() - last_save > 20:
                for g in list(self.graphs.values()):
                    try:
                        g.maintain()
                        g.save()
                    except Exception as ex:  # noqa: BLE001
                        logger.warning("hypergraph save: %s", ex)
                last_save = time.time()
