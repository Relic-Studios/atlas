"""Experience memory stores (docs/designs/experience_memory.md, D1, S1, S2).

  agent_state/memory/<agent>/episodes.json   what one agent remembers of the calls it was in
  agent_state/memory/_owner_facts.json        short facts about the owner, shared by every agent
  agent_state/memory/_settings.json           "Use memories", pause, the cloud switch
  agent_state/memory/_wipes.json              wiped ranges and deleted ids, so a waiting pass cannot undo them
  agent_state/memory/_store.lock              the one cross-process lock (filelock)

Every write takes the store lock, reads the file again inside it, and replaces it atomically (unique temp file,
flushed to disk, os.replace retried on a Windows PermissionError). The server and the post-call pass process both
write this way. Reply-path writes use `store_lock(timeout=0)` and skip on contention.

Recall reads the stores through `Recall`, which keeps a word index per agent (memory_redact.Matcher), so a lookup only
reads items that share a word with the turn.
"""
from __future__ import annotations

import contextlib
import json
import logging
import re
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

SETTINGS_DEFAULT = {"use_memories": False, "paused": False, "cloud_allowed": False}
_tlock = threading.RLock()
_flocks: dict = {}


class StoreBusy(Exception):
    """The store lock could not be taken in time."""


def root() -> Path:
    import agent_memory
    return Path(agent_memory.ROOT)


def _p(name: str) -> Path:
    return root() / name


@contextlib.contextmanager
def store_lock(timeout: float = 2.0):
    """In-process RLock, then the cross-process file lock. timeout=0 tries once (reply path); StoreBusy on failure."""
    from filelock import FileLock, Timeout
    if not _tlock.acquire(timeout=-1 if timeout is None else max(timeout, 0.0)):
        raise StoreBusy("store busy")
    try:
        r = root()
        r.mkdir(parents=True, exist_ok=True)
        key = str(r / "_store.lock")
        fl = _flocks.get(key)
        if fl is None:
            fl = _flocks[key] = FileLock(key, thread_local=False)
        try:
            fl.acquire(timeout=timeout if timeout is not None else -1)
        except Timeout as e:
            raise StoreBusy("store busy") from e
        try:
            yield
        finally:
            fl.release()
    finally:
        _tlock.release()


class StoreLock:
    """`with StoreLock():` = `with store_lock():`, for modules that used a plain `with _lock:`."""

    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self._cm = threading.local()

    def __enter__(self):
        cm = store_lock(self.timeout)
        stack = getattr(self._cm, "stack", None)
        if stack is None:
            stack = self._cm.stack = []
        cm.__enter__()
        stack.append(cm)
        return self

    def __exit__(self, *exc):
        return self._cm.stack.pop().__exit__(*exc)


def read_json(path: Path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data, retry_s: float = 2.0) -> None:
    """Atomic write: unique temp file, flushed and synced, then os.replace (retried for `retry_s` on Windows'
    PermissionError when another process has the file open). retry_s=0 never retries (reply path)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(3)}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    deadline = time.monotonic() + retry_s
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(0.05)


# ------------------------------------------------------------------ settings
def settings() -> dict:
    d = read_json(_p("_settings.json"), {})
    out = dict(SETTINGS_DEFAULT)
    out.update({k: bool(v) for k, v in d.items() if k in SETTINGS_DEFAULT})
    return out


def set_setting(key: str, value: bool) -> dict:
    if key not in SETTINGS_DEFAULT:
        raise KeyError(key)
    with store_lock():
        d = read_json(_p("_settings.json"), {})
        d[key] = bool(value)
        write_json(_p("_settings.json"), d)
    _changed("settings")
    return settings()


# ------------------------------------------------------------------ wipes and deleted ids
def wipes() -> dict:
    d = read_json(_p("_wipes.json"), {})
    return {"wipes": list(d.get("wipes", [])), "deleted": list(d.get("deleted", []))}


def _record(wipe: Optional[dict] = None, deleted: Iterable[str] = ()) -> None:
    """Inside the store lock."""
    d = wipes()
    if wipe:
        d["wipes"].append(wipe)
    for i in deleted:
        if i not in d["deleted"]:
            d["deleted"].append(i)
    d["deleted"] = d["deleted"][-5000:]
    write_json(_p("_wipes.json"), d)


def wiped(agent: str, first: float, last: float) -> bool:
    """True when a wipe of this agent (or "_owner") overlaps a call that ran from `first` to `last`."""
    for w in wipes()["wipes"]:
        if w.get("agent") == agent and float(w.get("from", 0)) <= last and first <= float(w.get("to", 0)):
            return True
    return False


def deleted_ids() -> set:
    return set(wipes()["deleted"])


# ------------------------------------------------------------------ owner facts
def _facts_path() -> Path:
    return _p("_owner_facts.json")


def owner_facts() -> list:
    """Every stored owner fact, newest first."""
    d = read_json(_facts_path(), {})
    facts = [f for f in d.get("facts", []) if isinstance(f, dict)]
    return sorted(facts, key=lambda f: -float(f.get("t", 0)))


def new_fact_id() -> str:
    return "f-" + secrets.token_hex(4)


def add_owner_fact(fact: str, quote: str, source: str, confirmed: bool, call: Optional[str] = None,
                   src_t: Optional[float] = None, tagger_checked: bool = False, fid: Optional[str] = None,
                   now: Optional[float] = None, timeout: float = 2.0) -> Optional[dict]:
    """Store one owner fact (S2). Returns it, or None when its id was deleted by the owner."""
    now = time.time() if now is None else now
    rec = {"id": fid or new_fact_id(), "call": call, "date": time.strftime("%Y-%m-%d", time.localtime(now)),
           "t": now, "src_t": src_t if src_t is not None else now, "source": source, "confirmed": bool(confirmed),
           "tagger_checked": bool(tagger_checked), "fact": fact, "quote": quote}
    with store_lock(timeout):
        gone = deleted_ids()
        # deleted by the owner: never stored again by id, and the pass never extracts the same words again (an
        # owner saying "remember that" again is a new request and is kept)
        if rec["id"] in gone or (source == "extracted" and _fact_hash(fact) in gone):
            return None
        d = read_json(_facts_path(), {})
        facts = [f for f in d.get("facts", []) if f.get("id") != rec["id"]]
        same = next((f for f in facts if _fact_key(f.get("fact", "")) == _fact_key(fact)), None)
        if same is not None:          # the same fact again (a "remember that" line the pass also read): kept once
            if confirmed and not same.get("confirmed"):
                same["confirmed"] = True
                write_json(_facts_path(), {"facts": facts})
                _changed("owner_facts")
            return same
        facts.append(rec)
        write_json(_facts_path(), {"facts": facts})
    _changed("owner_facts")
    return rec


def _fact_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", (text or "").lower()).split())


def _fact_hash(text: str) -> str:
    """What a deletion remembers of a fact's text: a hash, never the words (the owner wanted them gone)."""
    import hashlib
    return "key:" + hashlib.sha256(_fact_key(text).encode("utf-8")).hexdigest()[:16]


def update_facts(fids: Iterable[str], **fields) -> int:
    ids = set(fids)
    with store_lock():
        d = read_json(_facts_path(), {})
        n = 0
        for f in d.get("facts", []):
            if f.get("id") in ids:
                f.update(fields)
                n += 1
        if n:
            write_json(_facts_path(), d)
    if n:
        _changed("owner_facts")
    return n


def confirm_fact(fid: str) -> bool:
    return update_facts([fid], confirmed=True) > 0


def to_confirm(fids: Iterable[str]) -> int:
    """Turn facts into "to confirm" (unconfirmed, unused) - after a later "Not me", or a failed tagger check."""
    return update_facts(fids, confirmed=False)


def delete_fact(fid: str) -> bool:
    with store_lock():
        d = read_json(_facts_path(), {})
        facts = d.get("facts", [])
        keep = [f for f in facts if f.get("id") != fid]
        if len(keep) == len(facts):
            return False
        write_json(_facts_path(), {"facts": keep})
        gone = next(f for f in facts if f.get("id") == fid)
        _record(deleted=[fid, _fact_hash(gone.get("fact", ""))])
    _changed("owner_facts")
    return True


def clear_owner_facts(now: Optional[float] = None) -> int:
    now = time.time() if now is None else now
    with store_lock():
        n = len(read_json(_facts_path(), {}).get("facts", []))
        write_json(_facts_path(), {"facts": []})
        _record(wipe={"agent": "_owner", "from": 0.0, "to": now})
    _changed("owner_facts")
    return n


def usable_owner_facts(use_memories: Optional[bool] = None) -> list:
    """Facts agents may use (D10): confirmed `said` facts always; confirmed `extracted` facts with "Use memories"."""
    if use_memories is None:
        use_memories = settings()["use_memories"]
    return [f for f in owner_facts() if f.get("confirmed") and (f.get("source") == "said" or use_memories)]


def review_count() -> int:
    """Items waiting for the owner: unconfirmed facts and failed calls (journey.py's hint)."""
    n = sum(1 for f in owner_facts() if not f.get("confirmed"))
    pend = root() / "_pending"
    if pend.exists():
        for f in pend.glob("*.fails"):
            try:
                if int(f.read_text(encoding="utf-8").strip() or 0) >= 3:
                    n += 1
            except (OSError, ValueError):
                pass
    return n


# ------------------------------------------------------------------ episodes
def _ep_path(agent: str) -> Path:
    import agent_memory
    return agent_memory.folder(agent) / "episodes.json"


def episodes(agent: str) -> list:
    """The agent's episodes, newest first."""
    try:
        d = read_json(_ep_path(agent), {})
    except ValueError:
        return []
    eps = [e for e in d.get("episodes", []) if isinstance(e, dict)]
    return sorted(eps, key=lambda e: -float(e.get("end") or e.get("start") or 0))


def put_episode(agent: str, ep: dict, ident: Optional[str] = None) -> bool:
    """Add or replace (same id) an episode. Writes nothing when the id was deleted, a wipe overlaps the call, or the
    agent's folder is no longer the one the call ran with (`ident`)."""
    import agent_memory
    with store_lock():
        if ident is not None and agent_memory.ident(agent, create=False) != ident:
            return False
        if ep.get("id") in deleted_ids() or wiped(agent, float(ep.get("start") or 0), float(ep.get("end") or 0)):
            return False
        path = _ep_path(agent)
        d = read_json(path, {})
        eps = [e for e in d.get("episodes", []) if e.get("id") != ep.get("id")]
        eps.append(ep)
        write_json(path, {"episodes": eps})
    _changed("episodes", agent)
    return True


def delete_episode(agent: str, eid: str) -> bool:
    with store_lock():
        path = _ep_path(agent)
        d = read_json(path, {})
        eps = d.get("episodes", [])
        keep = [e for e in eps if e.get("id") != eid]
        if len(keep) == len(eps):
            return False
        write_json(path, {"episodes": keep})
        _record(deleted=[eid])
    _changed("episodes", agent)
    return True


def wipe_episodes(agent: str, cutoff: float, now: Optional[float] = None) -> int:
    """The per-agent wipe (windowed): episodes whose end is at or after `cutoff` go, and the range is recorded."""
    now = time.time() if now is None else now
    with store_lock():
        path = _ep_path(agent)
        d = read_json(path, {})
        eps = d.get("episodes", [])
        keep = [e for e in eps if float(e.get("end") or 0) < cutoff] if cutoff else []
        n = len(eps) - len(keep)
        if n:
            write_json(path, {"episodes": keep})
        _record(wipe={"agent": agent, "from": float(cutoff or 0), "to": now})
    if n:
        _changed("episodes", agent)
    return n


EPISODE_FIELDS = ("summary", "topics", "games", "decisions", "jokes", "moments")


def episode_items(ep: dict) -> list:
    """[(item id, kind, text)] for one episode (S2 ids: <episode id>/<field>/<index>)."""
    out = []
    for field in EPISODE_FIELDS:
        vals = ep.get(field) or []
        if isinstance(vals, str):
            vals = [vals]
        for i, v in enumerate(vals):
            if isinstance(v, dict):
                v = v.get("text", "")
            if v:
                out.append((f"{ep.get('id')}/{field}/{i}", field, str(v)))
    return out


# ------------------------------------------------------------------ change notifications
_listeners: list = []


def on_change(cb) -> None:
    """cb(kind, agent_or_None) after a store write in this process (the pipeline composes its prompt again)."""
    _listeners.append(cb)


def _changed(kind: str, agent: Optional[str] = None) -> None:
    for cb in list(_listeners):
        try:
            cb(kind, agent)
        except Exception as e:  # noqa: BLE001
            logger.debug("memory change listener failed: %s", e)


# ------------------------------------------------------------------ recall index
class Recall:
    """Word index over one agent's episode items and the owner facts (D1 Size, D4). Rebuilt when a store changes
    (`invalidate`), lazily on the next lookup."""

    def __init__(self, freq_limit: float = 5.5):
        self.freq_limit = freq_limit
        self._lock = threading.Lock()
        self._build = threading.Lock()  # one build at a time: a lookup during a warm-up waits for it, never builds twice
        self._episodes: dict = {}       # agent -> (Matcher, {item id: (kind, text)})
        self._facts = None              # (Matcher, {fact id: fact})

    def warm(self, agent: Optional[str] = None) -> None:
        """Build the indexes now (D1 Size: after each write, off the reply path), so the next lookup finds them."""
        try:
            self._fact_index()
            if agent:
                self._episode_index(agent)
        except Exception as e:  # noqa: BLE001
            logger.debug("recall warm-up failed: %s", e)

    def warm_later(self, agent: Optional[str] = None) -> None:
        threading.Thread(target=self.warm, args=(agent,), name="RecallWarm", daemon=True).start()

    def invalidate(self, kind: str = "", agent: Optional[str] = None) -> None:
        with self._lock:
            self._gen = getattr(self, "_gen", 0) + 1     # a build that started before this must not be cached
            if kind in ("", "owner_facts", "settings"):
                self._facts = None
            if kind in ("", "episodes", "settings"):
                if agent is None:
                    self._episodes.clear()
                else:
                    self._episodes.pop(agent, None)

    def _episode_index(self, agent: str):
        from memory_redact import Matcher
        with self._lock:
            got = self._episodes.get(agent)
            if got is not None:
                return got
        with self._build:
            with self._lock:
                got = self._episodes.get(agent)
                if got is not None:
                    return got
            for _ in range(5):          # a store change during the read means read again (it may have deleted)
                with self._lock:
                    gen = getattr(self, "_gen", 0)
                m, texts = Matcher(self.freq_limit), {}
                for ep in episodes(agent):
                    rec = float(ep.get("end") or 0)
                    for iid, kind, text in episode_items(ep):
                        m.add(iid, text, recency=rec)
                        texts[iid] = (kind, text)
                with self._lock:
                    if getattr(self, "_gen", 0) == gen:
                        self._episodes[agent] = (m, texts)
                        break
            return m, texts

    def _fact_index(self):
        from memory_redact import Matcher
        with self._lock:
            if self._facts is not None:
                return self._facts
        with self._build:
            with self._lock:
                if self._facts is not None:
                    return self._facts
            for _ in range(5):
                with self._lock:
                    gen = getattr(self, "_gen", 0)
                m, by_id = Matcher(self.freq_limit), {}
                for f in owner_facts():
                    m.add(f["id"], f.get("fact", ""), recency=float(f.get("t", 0)))
                    by_id[f["id"]] = f
                with self._lock:
                    if getattr(self, "_gen", 0) == gen:
                        self._facts = (m, by_id)
                        break
            return m, by_id

    def episode_hits(self, agent: str, text: str, limit: int = 3, exclude=lambda _i: False) -> list:
        """[(item id, score, text)] from the agent's episodes."""
        m, texts = self._episode_index(agent)
        return [(i, s, texts[i][1]) for i, s in m.lookup(text, limit=limit, exclude=exclude)]

    def recency(self, agent: str, item_id: str) -> float:
        """How new an episode item or owner fact is (its episode's end, the fact's store time): D4 breaks score ties
        in favour of the newer item."""
        with self._lock:
            got, facts = self._episodes.get(agent), self._facts
        for idx in (got, facts):
            if idx is not None and item_id in idx[1]:
                return idx[0].recency(item_id)
        return 0.0

    def fact_hits(self, text: str, use_memories: bool, limit: int = 3, exclude=lambda _i: False) -> list:
        """[(fact id, score, text)] among the facts agents may use."""
        m, by_id = self._fact_index()

        def ok(i):
            f = by_id.get(i)
            return (f is None or not f.get("confirmed") or (f.get("source") != "said" and not use_memories)
                    or exclude(i))
        return [(i, s, by_id[i]["fact"]) for i, s in m.lookup(text, limit=limit, exclude=ok)]


class CallLines:
    """The current call's human lines, indexed by word as they are noted (D5), so finding earlier lines that share
    words with a turn costs the same at 4,000 lines as at 40. Cleared at call end."""

    def __init__(self, freq_limit: float = 5.5):
        from memory_redact import Matcher
        self._lock = threading.Lock()
        self._m = Matcher(freq_limit)
        self._lines: dict = {}        # id -> (time, who, text)
        self.freq_limit = freq_limit

    def add(self, t: float, who: str, text: str, agent: bool = False) -> None:
        if agent or not (text or "").strip():
            return
        iid = f"line/{int(t * 1000)}"
        with self._lock:
            while iid in self._lines:
                iid += "+"
            self._lines[iid] = (t, who or "", text)
            self._m.add(iid, text, recency=t)

    def clear(self) -> None:
        from memory_redact import Matcher
        with self._lock:
            self._lines.clear()
            self._m = Matcher(self.freq_limit)

    def __len__(self) -> int:
        return len(self._lines)

    def recency(self, item_id: str) -> float:
        with self._lock:
            return float(self._lines.get(item_id, (0.0,))[0])

    def hits(self, text: str, before: float, limit: int = 3, exclude=lambda _i: False) -> list:
        """[(line id, score, who, text)] for lines noted before `before`."""
        with self._lock:
            lines = self._lines
            got = self._m.lookup(text, limit=limit,
                                 exclude=lambda i: lines.get(i, (before,))[0] >= before or exclude(i))
            return [(i, s, lines[i][1], lines[i][2]) for i, s in got]


def forget_window(t_from: float, t_to: float) -> dict:
    """Off the record, backward (D7): delete `said` owner facts, lore entries, lessons and predictions whose source
    line was committed from `t_from` to `t_to`."""
    out = {"facts": 0, "lore": 0, "lessons": 0, "predictions": 0}
    for f in owner_facts():
        if f.get("source") == "said" and t_from <= float(f.get("src_t") or 0) <= t_to:
            if delete_fact(f["id"]):
                out["facts"] += 1
    for mod, key in (("lore", "lore"), ("teach", "lessons"), ("self_tools", "predictions")):
        try:
            out[key] = int(getattr(__import__(mod), "drop_window")(t_from, t_to))
        except Exception as e:  # noqa: BLE001
            logger.warning("%s not cleared for an off-the-record window: %s", mod, e)
    return out


def code_hash() -> str:
    """Hash of the memory code a D9 report was measured with (D10): a report only enables "Use memories" for the
    code it measured."""
    import hashlib
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("memory_redact.py", "memory_writer.py", "experience_memory.py"):
        try:
            h.update((here / f).read_bytes().replace(b"\r\n", b"\n"))
        except OSError:
            h.update(f.encode())
    return h.hexdigest()[:16]
