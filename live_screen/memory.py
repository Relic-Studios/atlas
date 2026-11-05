"""The screen log (docs/designs/live_screen.md, section 6): what the watcher saw, as short self-contained lines with
times, for one Live screen session (one period of being switched on), in RAM only. Shared by all agents; never
written to long-term memory; cleared when the session ends; "forget what you saw" drops the last two minutes."""
from __future__ import annotations

import re
import threading
import time

from live_screen import config as CFG
from live_screen import filters

_lock = threading.Lock()
_entries: list = []                 # {"t", "kind", "text", "norm", "title", "label"}
_current: dict = {}                 # the newest look: {"t", "scene", "scene_kind", "title", "label"}
_seen_entities: dict = {}           # entity key -> last "seen" text
_STOP = set("""the a an and or but of to in on at for with from by is are was were be been it its this that these
those i you he she they we me my your his her their our what where when who how did do does done have has had
just still now then there here not no yes so up out about into over can could would should will well right okay
ok put left leave get got thing things""".split())


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']{3,}", (text or "").lower()) if w not in _STOP}


def clear() -> None:
    with _lock:
        _entries.clear()
        _current.clear()
        _seen_entities.clear()


def _add(kind: str, text: str, t: float, title: str, label: str) -> bool:
    text = filters.clean(text, limit=200)
    if not text:
        return False
    n = _norm(text)
    for e in reversed(_entries):
        if t - e["t"] > CFG.MERGE_WINDOW_S:
            break
        if e["norm"] == n:
            e["t"] = t                     # the same thing again: keep one line, newest time
            return False
    _entries.append({"t": t, "kind": kind, "text": text, "norm": n, "title": filters.clean(title, 60),
                     "label": label})
    del _entries[:-CFG.LOG_CAP]
    return True


def add(look: dict, t: float, title: str, label: str) -> int:
    """One watcher look -> new log lines (events, entities whose state changed, text worth keeping)."""
    if not isinstance(look, dict):
        return 0
    added = 0
    with _lock:
        if not look.get("unclear"):
            _current.update(t=t, scene=filters.clean(look.get("scene") or "", 200),
                            scene_kind=look.get("scene_kind") or "other", title=filters.clean(title, 60), label=label)
        for ev in (look.get("events") or [])[:3]:
            added += _add("event", str(ev), t, title, label)
        for ent in (look.get("entities") or [])[:5]:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            key = f"{ent.get('kind', '')}:{_norm(ent['name'])}"
            seen = str(ent.get("seen") or "")
            if _seen_entities.get(key) == _norm(seen):
                continue
            _seen_entities[key] = _norm(seen)
            added += _add(ent.get("kind") or "item", f"{ent['name']}: {seen}" if seen else str(ent["name"]),
                          t, title, label)
        for tx in (look.get("text") or [])[:2]:
            added += _add("text", f"on screen: \"{tx}\"", t, title, label)
    return added


def note_app(title: str, process: str, t: float, label: str) -> None:
    """A skipped look (mail, password manager): only that the program was open."""
    with _lock:
        _add("app", f"{process} open", t, title, label)


def line(e: dict, now: float | None = None) -> str:
    stamp = time.strftime("%H:%M", time.localtime(e["t"]))
    ago = (now or time.time()) - e["t"]
    rel = "just now" if ago < 60 else (f"about {round(ago / 60)} min ago" if ago < 5400 else
                                       f"about {round(ago / 3600, 1):g} h ago")
    where = f"{e['label']}" + (f", {e['title']}" if e.get("title") else "")
    return f"{stamp} ({rel}), {where}: {e['text']}"


def search(question: str, n: int = CFG.SEARCH_RESULTS) -> list:
    """Log entries sharing the most keywords with the question, newest first on ties."""
    q = _words(question)
    if not q:
        return []
    with _lock:
        scored = [(len(q & _words(e["text"] + " " + e.get("title", ""))), e["t"], e) for e in _entries]
    hits = [x for x in scored if x[0] > 0]
    hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [e for _, _, e in hits[:n]]


def last_seen(name: str) -> dict | None:
    k = _words(name)
    with _lock:
        for e in reversed(_entries):
            if k and k <= _words(e["text"]):
                return e
    return None


def current(max_age: float = CFG.DESCRIPTION_FRESH_S) -> dict | None:
    with _lock:
        if _current and time.time() - _current.get("t", 0) <= max_age:
            return dict(_current)
    return None


def forget(seconds: float = CFG.FORGET_S) -> int:
    cut = time.time() - seconds
    with _lock:
        before = len(_entries)
        _entries[:] = [e for e in _entries if e["t"] < cut]
        if _current.get("t", 0) >= cut:
            _current.clear()
        return before - len(_entries)


def snapshot(limit: int = 500) -> list:
    with _lock:
        return [{"t": e["t"], "kind": e["kind"], "text": e["text"], "title": e.get("title", ""),
                 "label": e.get("label", "")} for e in _entries[-limit:]]


def size() -> int:
    with _lock:
        return len(_entries)
