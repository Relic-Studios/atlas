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

import agent_memory as AM
ROOT = AM.ROOT          # agent_state/memory/<agent>/graph.json + emb.npy (one folder per agent)
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
MIN_SIM = float(os.environ.get("ATLAS_HG_MIN_SIM", "0.53"))
# Live call 10-06 ("he's remembering too much"): recall kept handing back lines from the
# last few minutes, which are already in the conversation log. Seeing them twice made the
# agent fixate (the "Canadian" loop) and even recalled the very line being answered.
# Live recall skips anything younger than this; the conversation log already covers it.
RECALL_MIN_AGE_S = float(os.environ.get("ATLAS_HG_RECALL_MIN_AGE", "900"))
RECALL_K = int(os.environ.get("ATLAS_HG_RECALL_K", "3"))


def strip_agent_name(agent: str, text: str) -> str:
    """'Max, ...' would match every stored line that ever named him."""
    if not agent or not text:
        return text or ""
    return re.sub(r"(?i)\b" + re.escape(agent) + r"\b[,!?.]*", " ", text).strip()


def _only_questions(text: str) -> bool:
    """A line that is nothing but questions ('Max, do you have hair?') says what someone
    asked, not anything worth remembering; live recall of these confused who said what."""
    parts = [x.strip() for x in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if x.strip()]
    return bool(parts) and all(x.endswith("?") for x in parts)   # relevance gate (calibrated: sim_hypergraph)
NAME_SIM = 0.42      # gate when the query names the memory's person
SPREAD_SIM = 0.33     # a spread-reached memory still needs this much direct similarity
SEEDS = 12
NODE_SPREAD = 0.15
PERSON_SPREAD = 0.45
HEBB_RECALL = 0.05
PAIR_SIM = 0.48
PAIR_DISTINCT = 0.85  # two "same person" memories must differ this much to count as joint evidence
HUB_PENALTY = float(os.environ.get("ATLAS_HG_HUB", "1.0"))
PF_GAP_S = float(os.environ.get("ATLAS_HG_PF_GAP", "0.08"))   # pause between prefetches (bounds CPU)
PF_TTL_S = 20.0                                                # a prefetched recall is reusable this long
RARE_DEG = 3            # a shared content word this rare counts like a name       # see pair_ok in retrieve()
SPREAD_SCORE = 0.3    # spread lets a memory IN; its own similarity still ranks it     # a strongly co-used memory comes along even if not similar   # same-person bridge (multi-hop: "the guy who moved to Denver, ...")
HEBB_SPREAD = 0.20
STRENGTH_BONUS = 0.05
# Named-chatter gate (owner 10-03, tests/sim_named_chatter.py): a person's name alone
# inflated similarity, so "Sam you're muted" pulled up everything about Sam. When a
# line names someone, a memory must also match the line WITHOUT the name, minus how
# much that remainder looks like generic call chatter.
NAMED_TOPIC = 0.17
PHATIC_W = 0.5
_PHATIC = ["is my mic working", "are you there", "hello can anyone hear me", "you cut out",
           "you're breaking up", "hold on one sec", "brb", "ok cool", "yeah sure", "sounds good",
           "never mind", "what did you say", "say that again", "lol", "wait what", "thank you",
           "good night", "see you later", "what's up", "how are you", "you there?",
           "turn your volume down", "you're too loud", "I can't hear you", "what are we doing now",
           "anyway", "fine", "let's go", "who's talking", "be quiet for a second"]

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


def qkey(text: str) -> str:
    """Prefetch match key: words only (tags, case and punctuation don't change recall)."""
    return " ".join(re.findall(r"[\w']+", _TAG.sub(" ", text or "").lower()))


def same_query(a: str, b: str) -> bool:
    """Final transcript vs last partial: identical, or one word swapped/added/dropped in a
    line long enough that recall can't change meaningfully (finals often re-punctuate or
    fix one word). Short lines must match exactly: one word IS the meaning there."""
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    if min(len(wa), len(wb)) < 6 or abs(len(wa) - len(wb)) > 1:
        return False
    sa, sb = set(wa), set(wb)
    return len(sa ^ sb) <= 2 and len(sa & sb) / max(1, len(sa | sb)) >= 0.8


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


# Never stored at all (whole line dropped), checked before anything else:
#  - secrets: passwords, PINs, keys, card/bank/social-security numbers
#  - where someone lives: "I live at 42 ...", street addresses
#  - the speaker asked: "don't remember this", "off the record", "between us"
_PRIVATE = re.compile(
    r"\b(pass ?words?|passcodes?|pin (?:code|number)|security code|api ?keys?|secret keys?|"
    r"seed phrase|social security|ssn|credit ?card|debit ?card|card number|bank account|"
    r"routing number|login (?:is|was)|my (?:home )?address|i live at|we live at|"
    r"don'?t (?:remember|save|store|record|repeat|tell anyone)|do not (?:remember|save|store|record|repeat)|"
    r"off the record|between (?:you and me|us)|keep (?:this|it|that) (?:a )?secret|"
    r"(?:it'?s|this is) (?:private|confidential))\b", re.I)
# a street address = number + Capitalised name + street word ("42 Wallaby Way");
# case-sensitive so "drove 20 minutes down the road" is not an address
_STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|"
                     r"Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl|Terrace|Crescent)\b")
_FORGET = re.compile(r"\b(?:forget (?:that|this|what i (?:just )?said|it|everything i said)|"
                     r"scratch that,? (?:don'?t|do not) remember|delete that)\b", re.I)


_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def is_private(text: str) -> bool:
    """A line with a secret, an address, an email or a phone number is not stored at all
    (a redacted 'my number is [number]' is useless as a memory and still says too much)."""
    t = text or ""
    return bool(_PRIVATE.search(t) or _STREET.search(t) or _EMAIL.search(t) or _PHONE.search(t))


def is_forget_request(text: str) -> bool:
    return bool(_FORGET.search(text or ""))


def _clean(text: str) -> str:
    """Scrub before storage: speaker tags, slurs (whole memory dropped), emails, phone numbers."""
    t = _TAG.sub(" ", text or "")
    if is_private(t):
        return ""
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
            if not texts:
                return np.zeros((0, self.dim), np.float32)
            got = {t: self._cache[t] for t in texts if t in self._cache}
            miss = list(dict.fromkeys(t for t in texts if t not in got))
            if miss:
                v = m.encode(miss, batch_size=16, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
                got.update(zip(miss, v))
            # Build the result BEFORE trimming the cache: a batch with >256 new
            # texts used to evict its own vectors -> KeyError (sim_memory_crowding).
            out = np.stack([got[t] for t in texts])
            for t in miss[-256:]:
                self._cache[t] = got[t]
            if len(self._cache) > 512:
                for k in list(self._cache)[:len(self._cache) - 256]:
                    self._cache.pop(k, None)
            return out


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
        self.dir = AM.folder(self.agent, root or ROOT)
        AM.migrate(root or ROOT)
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
        self.root = root or ROOT
        self.gen = AM.ident(self.agent, self.root)   # folder identity at load time
        self.dead = False
        self._load()

    def stale(self) -> bool:
        """The folder was archived (agent deleted) or re-created since this graph loaded."""
        return self.dead or AM.ident(self.agent, self.root, create=False) != self.gen

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
            if self.stale():
                # never resurrect a deleted agent's memories into its folder or its successor
                if not self.dead:
                    logger.info("hypergraph[%s]: folder replaced/archived; dropping stale save", self.agent)
                self.dead = True
                return
            if not self._dirty and (self.dir / "graph.json").exists():
                return
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

    # ---- owner controls ---------------------------------------------------
    def _drop(self, idxs: set):
        keep = [i for i in range(len(self.edges)) if i not in idxs]
        gone = {self.edges[i]["id"] for i in idxs}
        self.edges = [self.edges[i] for i in keep]
        self.vecs = self.vecs[keep] if keep else np.zeros((0, 0), np.float32)
        for k in list(self.hebb):
            a, b = k.split("|", 1)
            if a in gone or b in gone:
                del self.hebb[k]
        self.active = [a for a in self.active if a not in gone]
        self._reindex()
        self._dirty = True

    def forget(self, ids) -> int:
        """Remove memories completely: text, embedding and every Hebbian link."""
        ids = set(ids)
        with self._lock:
            idx = {i for i, e in enumerate(self.edges) if e["id"] in ids}
            if idx:
                self._drop(idx)
            return len(idx)

    def forget_since(self, cutoff: float) -> int:
        """Owner wipe: drop every memory FORMED at/after `cutoff` (epoch seconds), whatever
        its kind, pinned or not. cutoff <= 0 wipes the whole graph, links included."""
        with self._lock:
            if cutoff <= 0:
                n = len(self.edges)
                self.edges, self.vecs, self.hebb, self.active = [], np.zeros((0, 0), np.float32), {}, []
                self._reindex()
                self._dirty = True
                return n
            idx = {i for i, e in enumerate(self.edges) if float(e.get("created", e.get("t", 0))) >= cutoff}
            if idx:
                self._drop(idx)
            self.active = []
            return len(idx)

    def forget_last(self, who: str = "", within_s: float = 600.0) -> str | None:
        """'forget that': drop the newest memory from that speaker (recent ones only)."""
        now = self.clock()
        who = (who or "").strip().lower()
        with self._lock:
            for i in range(len(self.edges) - 1, -1, -1):
                e = self.edges[i]
                if now - e["created"] > within_s:
                    continue
                if e.get("kind") == "episodic" and (not who or e.get("who", "").lower() == who):
                    self._drop({i})
                    return e["id"]
        return None

    def set_pinned(self, eid: str, pinned: bool = True) -> bool:
        with self._lock:
            for e in self.edges:
                if e["id"] == eid:
                    e["pinned"] = bool(pinned)
                    if pinned:
                        e["kind"] = "approved"
                    self._dirty = True
                    return True
        return False

    def sync_approved(self, texts: list) -> int:
        """Approved memories are pinned; anything pinned as approved that the owner has
        since rejected or forgotten is REMOVED (it must never be recalled again)."""
        want = {_clean(t).lower() for t in texts}
        with self._lock:
            idx = {i for i, e in enumerate(self.edges)
                   if e.get("kind") == "approved" and e["text"].lower() not in want}
            if idx:
                self._drop(idx)
            have = {e["text"].lower() for e in self.edges if e.get("kind") == "approved"}
        added = 0
        for t in texts:
            if _clean(t).lower() not in have:
                if self.add(t, kind="approved", pinned=True):
                    added += 1
        return added

    def listing(self, limit: int = 200) -> list:
        """For the owner's memory panel: pinned first, then strongest."""
        now = self.clock()
        with self._lock:
            rows = [{"id": e["id"], "text": e["text"], "who": e.get("who", ""), "kind": e.get("kind", ""),
                     "pinned": bool(e.get("pinned")), "hits": e.get("hits", 0),
                     "strength": round(self._w(e, now), 3), "created": e["created"]}
                    for e in self.edges]
        rows.sort(key=lambda r: (-r["pinned"], -r["strength"]))
        return rows[:limit]

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
                    if pinned:
                        e["kind"] = kind
                    self._dirty = True
                    return e["id"]
            eid = f"{self.agent}-e{self.next_id}"
            self.next_id += 1
            e = {"id": eid, "text": text, "nodes": nodes, "who": who, "kind": kind,
                 "w": max(INIT_W, PIN_FLOOR if pinned else 0.0), "t": now, "created": now,
                 "hits": 0, "pinned": bool(pinned)}
            self.edges.append(e)
            old_vecs = self.vecs
            self.vecs = vec[None, :].astype(np.float32) if self.vecs.size == 0 else \
                np.vstack([self.vecs, vec[None, :].astype(np.float32)])
            i = len(self.edges) - 1
            self._hub = None
            # incremental hub sums: O(n) per add instead of an O(n^2) rebuild on the
            # next recall (136 ms at 4000 memories, and every live line is an add)
            hs = getattr(self, "_hub_sum", None)
            if (hs is not None and getattr(self, "_hub_src", None) is old_vecs
                    and len(hs) == i and i > 0):
                s = (old_vecs @ self.vecs[i]).astype(np.float64)
                self._hub_sum = np.concatenate([hs + s, [float(s.sum())]])
                self._hub_src = self.vecs
            else:
                self._hub_sum = None
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
    def retrieve(self, query: str, k: int = 4, qvec: np.ndarray = None, mark_active: bool = True,
                 min_age_s: float = 0.0) -> list:
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
            names = [w for w in dict.fromkeys(re.findall(r"[a-z]+", query.lower()))
                     if f"@{w}" in self.node_index]
        # "what's Nina's job": a possessive means the line is ABOUT that person, so the
        # name is the topic, not an address ("Sam, what time is it"). Don't gate those.
        if any(re.search(r"\b" + w + r"'s\b", query, re.I) for w in names):
            names = []
        topical = None
        if names:
            rest = query
            for w in names:
                rest = re.sub(rf"\b{w}\b('s)?,?", " ", rest, flags=re.I)
            rest = " ".join(rest.split()) or "."
            svec = self.emb_model.encode([rest])[0]
            if svec.shape[0] == qvec.shape[0]:
                ph = float(np.max(self._phatic_vecs() @ svec))
                topical = (svec, ph)
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
            tscore = None
            if topical is not None:
                tscore = self.vecs @ topical[0] - PHATIC_W * topical[1]
            out = []
            for i in cand:
                if tscore is not None and float(tscore[i]) < NAMED_TOPIC:
                    continue
                e = self.edges[i]
                if min_age_s and now - e["created"] < min_age_s:
                    continue      # still in the conversation log: recalling it is an echo
                if min_age_s and _only_questions(e["text"]):
                    continue      # live recall: a bare question isn't a memory
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
                            "age_s": now - e["created"], "created": e["created"],
                            "pinned": e.get("pinned", False), "kind": e.get("kind", "")})
            out.sort(key=lambda r: -r["score"])
            out = out[:k]
            if mark_active:
                self.active = (self.active + [r["id"] for r in out])[-12:]
            return out

    def _phatic_vecs(self) -> np.ndarray:
        v = getattr(self, "_phatic", None)
        if v is None:
            v = self._phatic = np.asarray(self.emb_model.encode(_PHATIC), np.float32)
        return v

    def _hub_scores(self) -> np.ndarray:
        """Per-memory centrality minus the store average (cached; recomputed on change)."""
        n = len(self.edges)
        if getattr(self, "_hub_n", -1) != n or getattr(self, "_hub", None) is None:
            hs = getattr(self, "_hub_sum", None)
            if hs is None or getattr(self, "_hub_src", None) is not self.vecs or len(hs) != n:
                # full rebuild: row sums of the similarity matrix, diagonal excluded
                if n:
                    m = (self.vecs @ self.vecs.T).astype(np.float64)
                    hs = m.sum(axis=1) - np.diagonal(m)
                else:
                    hs = np.zeros(0, np.float64)
                self._hub_sum, self._hub_src = hs, self.vecs
            if n < 8:
                self._hub = np.zeros(n, np.float32)
            else:
                c = hs / (n - 1)
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
def _age(s: float, this_call: bool = False) -> str:
    if this_call:
        return "earlier this call"
    if s < 86400:
        return "today"
    d = int(s // 86400)
    return "yesterday" if d == 1 else f"{d} days ago"


def note(hits: list, session_start: float = None) -> str:
    if not hits:
        return ""
    lines = []
    for h in hits:
        who = h["who"] or "someone"
        this_call = session_start is not None and h.get("created", 0) >= session_start
        lines.append(f"- {who} ({_age(h['age_s'], this_call)}): {h['text']}")
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
        self._epoch: dict = {}           # agent -> bumped on delete/wipe; older queued jobs are dropped
        self._job_lock = threading.Lock()  # a wipe never interleaves with a half-done write
        self.session_start = clock()
        # recall prefetch from partial transcripts: latest-wins, one job at a time
        self._pf_cv = threading.Condition()
        self._pf_pending = None          # (agent, key, text, epoch, k)
        self._pf_running = None          # (agent, key)
        self._pf: dict = {}              # agent -> {"key","hits","ep","k","t"}
        self._pf_thread = None
        self.pf_stats = {"hit": 0, "waited": 0, "miss": 0}
        self._worker = threading.Thread(target=self._run, name="HypergraphWriter", daemon=True)
        self._worker.start()

    def graph(self, agent: str) -> HyperMemory:
        agent = (agent or "unknown").lower()
        with self._lock:
            g = self.graphs.get(agent)
            if g is not None and g.stale():
                g = None                     # deleted/re-created on disk: load the new folder
            if g is None:
                g = HyperMemory(agent, root=self.root, embedder=self.embedder or shared_embedder(),
                                clock=self.clock)
                self.graphs[agent] = g
            return g

    # called from the live turn handler; never blocks
    def observe_user(self, agent: str, who: str, text: str):
        if not ENABLED:
            return
        if is_forget_request(text):
            try:
                self._q.put_nowait(("forget_last", agent, who, text, self._ep(agent)))
            except queue.Full:
                pass
            return
        if len(content_nodes(text)) < 3 or is_private(text):
            return
        try:
            self._q.put_nowait(("add", agent, who, text, self._ep(agent)))
        except queue.Full:
            pass

    def observe_reply(self, agent: str, reply: str):
        hits = self._last_hits.pop((agent or "").lower(), None)
        if hits:
            try:
                self._q.put_nowait(("reinforce", agent, hits, reply, self._ep(agent)))
            except queue.Full:
                pass

    # ---- recall prefetch ---------------------------------------------------
    def prefetch(self, agent: str, text: str, k: int = None):
        """Start recall for a PARTIAL transcript in the background (never blocks).
        When the turn's final text matches, recall_note() reuses the result, so memory
        costs ~0 ms at turn end instead of ~140 ms (embedding is ~85% of it)."""
        k = RECALL_K if k is None else k
        text = strip_agent_name(agent, text)
        if not ENABLED or not text or len(content_nodes(text)) < 1:
            return
        a, key = (agent or "").lower(), qkey(text)
        if not key:
            return
        with self._pf_cv:
            c = self._pf.get(a)
            if c is not None and c["key"] == key and c["ep"] == self._ep(a):
                return
            if self._pf_running == (a, key):
                return
            self._pf_pending = (a, key, text, self._ep(a), k)
            if self._pf_thread is None or not self._pf_thread.is_alive():
                self._pf_thread = threading.Thread(target=self._pf_run, name="HypergraphPrefetch",
                                                   daemon=True)
                self._pf_thread.start()
            self._pf_cv.notify_all()

    def _pf_run(self):
        while not self._stop.is_set():
            with self._pf_cv:
                while self._pf_pending is None and not self._stop.is_set():
                    self._pf_cv.wait(1.0)
                if self._pf_pending is None:
                    continue
                a, key, text, ep, k = self._pf_pending
                self._pf_pending = None
                self._pf_running = (a, key)
            hits = None
            try:
                hits = self.graph(a).retrieve(text, k=k, mark_active=False, min_age_s=RECALL_MIN_AGE_S)
            except Exception as ex:  # noqa: BLE001
                logger.warning("hypergraph prefetch failed: %s", ex)
            with self._pf_cv:
                self._pf_running = None
                if hits is not None and ep == self._ep(a):
                    self._pf[a] = {"key": key, "hits": hits, "ep": ep, "k": k, "t": time.monotonic()}
                self._pf_cv.notify_all()
            time.sleep(PF_GAP_S)        # bound CPU while someone is talking

    def _pf_take(self, agent: str, query: str, k: int, budget_s: float):
        """Prefetched hits for exactly this text (waiting for an in-flight one), else None."""
        a, key = (agent or "").lower(), qkey(query)
        with self._pf_cv:
            waited = False
            # a prefetch only QUEUED for this text would run after the current one: that's
            # two retrieves of wait. Cancel it and let the caller compute directly.
            if self._pf_pending is not None and self._pf_pending[0] == a:
                self._pf_pending = None
            run = self._pf_running
            if run is not None and run[0] == a and same_query(run[1], key):
                waited = True     # already computing (nearly) this text: finishing it is cheaper
                self._pf_cv.wait_for(lambda: self._pf_running != run, timeout=budget_s)
            c = self._pf.get(a)
            if (c is not None and same_query(c["key"], key) and c["ep"] == self._ep(a)
                    and c["k"] >= k and time.monotonic() - c["t"] < PF_TTL_S):
                self.pf_stats["waited" if waited else "hit"] += 1
                return [dict(h) for h in c["hits"][:k]]
        self.pf_stats["miss"] += 1
        return None

    def _pf_clear(self, agent: str):
        with self._pf_cv:
            self._pf.pop((agent or "").lower(), None)

    def recall_note(self, agent: str, query: str, k: int = None, budget_s: float = 0.35,
                    exclude: set = None) -> str:
        """Retrieve within a time budget; on timeout the turn simply goes without memory."""
        if not ENABLED or not query or len(content_nodes(query)) < 1:
            return ""
        k = RECALL_K if k is None else k
        query = strip_agent_name(agent, query)
        if len(content_nodes(query)) < 1:
            return ""
        t0 = time.monotonic()
        hits = self._pf_take(agent, query, k, budget_s)
        if hits is not None:
            try:   # the prefetch didn't mark these active (a partial isn't a turn yet)
                g = self.graph(agent)
                g.active = (g.active + [h["id"] for h in hits])[-12:]
            except Exception:  # noqa: BLE001
                pass
        else:
            box = {}

            def work():
                try:
                    box["hits"] = self.graph(agent).retrieve(query, k=k, min_age_s=RECALL_MIN_AGE_S)
                except Exception as ex:  # noqa: BLE001
                    logger.warning("hypergraph recall failed: %s", ex)
            t = threading.Thread(target=work, daemon=True)
            t.start()
            t.join(max(0.0, budget_s - (time.monotonic() - t0)))
            hits = box.get("hits") or []
        if exclude:   # already in the prompt as an approved memory: don't list it twice
            ex = {x.strip().lower() for x in exclude}
            hits = [h for h in hits if h["text"].strip().lower() not in ex]
        self._last_hits[(agent or "").lower()] = hits
        return note(hits, self.session_start)

    def sync_approved(self, agent: str, texts: list):
        """Replace the agent's approved set (rejected/forgotten ones are removed)."""
        try:
            self._q.put_nowait(("sync", agent, "", list(texts), self._ep(agent)))
        except queue.Full:
            pass

    # owner controls: synchronous (rare, and the owner is waiting for the answer)
    def forget(self, agent: str, ids) -> int:
        self._pf_clear(agent)
        g = self.graph(agent)
        n = g.forget(ids)
        g.save()
        return n

    def set_pinned(self, agent: str, eid: str, pinned: bool = True) -> bool:
        self._pf_clear(agent)
        g = self.graph(agent)
        ok = g.set_pinned(eid, pinned)
        g.save()
        return ok

    def listing(self, agent: str, limit: int = 200) -> list:
        return self.graph(agent).listing(limit)

    def _ep(self, agent: str) -> int:
        return self._epoch.get((agent or "").lower(), 0)

    def drop_agent(self, agent: str):
        """Agent deleted: drop the in-memory graph (its folder is archived separately) and
        invalidate every write still queued for it."""
        with self._lock:
            a = (agent or "").lower()
            self._epoch[a] = self._epoch.get(a, 0) + 1
            g = self.graphs.pop(a, None)
            if g is not None:
                g.dead = True
        self._last_hits.pop((agent or "").lower(), None)
        self._pf_clear(agent)

    def wipe(self, agent: str, cutoff: float) -> int:
        """Owner wipe for ONE agent: invalidate its queued writes (a line heard before the wipe
        can't land after it), then drop memories formed since `cutoff` (<= 0: all of them)."""
        a = (agent or "").lower()
        with self._job_lock:
            with self._lock:
                self._epoch[a] = self._epoch.get(a, 0) + 1
            self._last_hits.pop(a, None)
            self._pf_clear(a)
            g = self.graph(a)
            n = g.forget_since(cutoff)
            g.save()
            return n

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
        with self._pf_cv:
            self._pf_cv.notify_all()

    def _run(self):
        last_save = time.time()
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=1.0)
            except queue.Empty:
                job = None
            try:
                if job:
                    self._job_lock.acquire()
                stale = bool(job) and len(job) > 4 and job[4] != self._ep(job[1])
                if job and not stale:            # stale = queued before the agent was deleted/wiped
                    op, agent = job[0], job[1]
                    g = self.graph(agent)
                    if op == "add":
                        g.add(job[3], who=job[2])
                    elif op == "pin":
                        g.add(job[3], kind="approved", pinned=True)
                    elif op == "sync":
                        g.sync_approved(job[3])
                    elif op == "forget_last":
                        g.forget_last(job[2])
                    elif op == "reinforce":
                        hits, reply = job[2], job[3]
                        g.reinforce([h["id"] for h in hits], used_ids=used_by(reply, hits))
            except Exception as ex:  # noqa: BLE001
                logger.warning("hypergraph writer: %s", ex)
            finally:
                if job:
                    self._job_lock.release()
                    self._q.task_done()
            if time.time() - last_save > 20:
                for g in list(self.graphs.values()):
                    try:
                        g.maintain()
                        g.save()
                    except Exception as ex:  # noqa: BLE001
                        logger.warning("hypergraph save: %s", ex)
                last_save = time.time()
