"""Friend-group lore keeper (owner 10-05 plugin list): the room's shared canon - running jokes,
campaign events, "the time someone fell off the roof" - kept and brought back at the right
moment weeks later. Entries hold no names: they are scrubbed and checked when written, and
checked again by the post-call pass (docs/designs/experience_memory.md, D8).

Storage is per agent (agent_state/memory/<agent>/lore.json), so lore one agent keeps never
reaches another. By default every new entry waits for the owner's approval in the plugin
window before the agent ever uses it ("owner" mode); "auto" mode trusts the room.

A lore entry comes back into the prompt only when the current line touches its words, at most
one per turn, and not the same entry again within COOLDOWN_S - so it lands as a callback, not
a catchphrase.

Private details (phone numbers, emails, addresses, keys) are refused, same filter as memory.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

MAX_LORE = 80
MAX_TITLE = 60
MAX_STORY = 500
COOLDOWN_S = 20 * 60

from experience_memory import StoreLock as _StoreLock  # noqa: E402

_lock = _StoreLock()     # the experience memory's cross-process store lock (design D1)
_last_used: dict = {}          # (agent, id) -> time it was last put in a prompt (this session)
_STOP = {"the", "a", "an", "our", "my", "your", "that", "this", "time", "when", "what", "was", "were",
         "is", "are", "of", "for", "to", "in", "on", "and", "we", "do", "did", "it", "about", "with",
         "remember", "lore", "joke", "story", "guys", "just", "like", "so", "then", "there", "they",
         "you", "he", "she", "him", "her", "his", "them", "from", "have", "had", "has", "all", "one"}


def _path(agent: str, root: Optional[Path] = None) -> Path:
    import agent_memory
    return agent_memory.folder(agent or "default", root) / "lore.json"


def _load(agent: str, root=None) -> list:
    try:
        d = json.loads(_path(agent, root).read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001
        return []


def _save(agent: str, items: list, root=None, retry_s: float = 2.0) -> None:
    import experience_memory
    experience_memory.write_json(_path(agent, root), items, retry_s=retry_s)


def _private(text: str) -> bool:
    try:
        from memory_redact import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001 - fail closed: treat the text as private
        return True


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if len(w) > 2 and w not in _STOP}


def _mode() -> str:
    try:
        import plugins
        return (plugins.settings_of("lore") or {}).get("approval", "owner")
    except Exception:  # noqa: BLE001
        return "owner"


def add(agent: str, title: str, story: str, people: Iterable[str] = (), added_by: str = "",
        root=None, mode: Optional[str] = None, clock: Callable[[], float] = time.time) -> str:
    """Keep a piece of the group's lore without names (docs/designs/experience_memory.md, D8): the call's names (and
    any names passed in `people`/`added_by` by older callers) become "someone" before the entry is checked and
    stored. Nothing about who added it is kept."""
    import memory_redact as MR
    if _off_record():
        return "Not kept: you're off the record right now. Say so briefly."
    legacy = [p for p in list(people or ()) + [added_by] if p]
    story = MR.store_scrub(re.sub(r"\s+", " ", (story or "").strip())[:MAX_STORY], legacy)
    title = MR.store_scrub(re.sub(r"\s+", " ", (title or "").strip(" .:,'\""))[:MAX_TITLE], legacy)
    if not story:
        return "Nothing to keep: say what the lore is."
    if not title:
        title = " ".join(story.split()[:6])
    if _private(title + " " + story):
        return "Not kept: that has a private detail (phone, email, address or key). Say you don't keep those."
    reason = MR.store_check(title + ". " + story, legacy)
    if reason:
        return MR.refusal(reason)
    mode = mode or _mode()
    status = "approved" if mode == "auto" else "pending"
    now = clock()
    with _lock:
        items = _load(agent, root)
        key = title.lower()
        for it in items:
            if it.get("title", "").lower() == key and it.get("status") != "rejected":
                it.update(story=story, at=now, src_t=_src_t(now), tagger_checked=False)
                _save(agent, items, root)
                return f"Updated the lore '{title}'."
        items.append({"id": max([i.get("id", 0) for i in items] + [0]) + 1, "title": title, "story": story,
                      "at": now, "src_t": _src_t(now), "tagger_checked": False, "status": status, "uses": 0})
        # keep rejected out of the cap's way first, then oldest
        if len(items) > MAX_LORE:
            items = [i for i in items if i.get("status") != "rejected"][-MAX_LORE:]
        _save(agent, items, root)
    if status == "pending":
        return (f"Kept '{title}' for the room's lore. The owner approves new lore before you use it, so "
                "say you've noted it, without promising you'll bring it up.")
    return f"Added '{title}' to the room's lore. Acknowledge it briefly."


def _match(items: list, text: str, present: Iterable[str] = ()) -> List[dict]:
    """Approved entries this text touches: title words count three times, story words once (the shared matcher's
    core, memory_redact.overlap_score). `present` is ignored: lore no longer knows who is in it."""
    from memory_redact import overlap_score
    q = _words(text)
    scored = []
    for it in items:
        if it.get("status") != "approved":
            continue
        tw, sw = _words(it.get("title", "")), _words(it.get("story", ""))
        s = overlap_score(q, [(tw, 3), (sw, 1)])
        if (tw and q & tw) or s >= 3:
            scored.append((s, it))
    scored.sort(key=lambda x: (-x[0], x[1].get("uses", 0)))
    return [it for _, it in scored]


def recall(agent: str, topic: str = "", root=None) -> str:
    with _lock:
        items = [i for i in _load(agent, root) if i.get("status") == "approved"]
    if not items:
        return "The room has no lore yet. Say so, and that you'd love to keep some."
    if topic:
        hits = _match(items, topic)
        if not hits:
            return f"No lore about '{topic}'. Lore you do have: " + ", ".join(i["title"] for i in items[-8:]) + "."
        return " ".join(f"'{h['title']}': {h['story']}" for h in hits[:3])
    return "The room's lore: " + "; ".join(i["title"] for i in items[-15:]) + "."


def forget(agent: str, title: str = "", lid=None, root=None) -> str:
    with _lock:
        items = _load(agent, root)
        keep, gone = [], []
        for it in items:
            hit = (lid is not None and str(it.get("id")) == str(lid)) or \
                  (title and title.lower().strip() in it.get("title", "").lower())
            (gone if hit else keep).append(it)
        if not gone:
            return f"No lore matching '{title or lid}'."
        _save(agent, keep, root)
    return "Dropped from the lore: " + ", ".join(g["title"] for g in gone) + "."


def forget_since(agent: str, cutoff: float, root=None) -> int:
    with _lock:
        items = _load(agent, root)
        keep = [i for i in items if float(i.get("at", 0)) < cutoff] if cutoff else []
        n = len(items) - len(keep)
        if n:
            _save(agent, keep, root)
    return n


def note(agent: str, text: str, present: Iterable[str] = (), root=None,
         clock: Callable[[], float] = time.time) -> str:
    """Per-turn note: at most one approved lore entry this line touches, with a cooldown."""
    try:
        items = _load(agent, root)          # reply path: read without the lock (files are replaced atomically)
        now = clock()
        hits = [h for h in _match(items, text, present)
                if now - _last_used.get((agent, h["id"]), 0) > COOLDOWN_S]
    except Exception:  # noqa: BLE001
        return ""
    if not hits:
        return ""
    h = hits[0]
    _last_used[(agent, h["id"])] = now
    try:   # the use count is best effort: on the reply path the lock is tried once and never waited for
        import experience_memory
        with experience_memory.store_lock(timeout=0):
            items = _load(agent, root)
            for it in items:
                if it.get("id") == h["id"]:
                    it["uses"] = int(it.get("uses", 0)) + 1
            _save(agent, items, root, retry_s=0)
    except Exception:  # noqa: BLE001
        pass
    return ("ROOM LORE (background only - it doesn't change whether you answer. This group's own shared "
            "history: mention it only if it fits what's being said right now, and most of the time "
            f"don't; never retell the whole story): '{h['title']}': {h['story']}")


# ------------------------------------------------------------------ owner window
def _mem_root(root=None) -> Path:
    import agent_memory
    return Path(root) if root else agent_memory.ROOT


def feed(root=None) -> List[dict]:
    base = _mem_root(root)
    out = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for x in reversed(_load(d.name, root)):
            st = x.get("status", "pending")
            if st == "rejected":
                continue
            acts = ([{"act": "approve", "label": "Approve"}, {"act": "reject", "label": "Reject"}]
                    if st in ("pending", "review") else [{"act": "delete", "label": "Remove"}])
            out.append({"title": f"{x['title']}: {x['story']}",
                        "meta": f"{d.name.title()} · {'needs review' if st == 'review' else st}",
                        "id": str(x["id"]), "agent": d.name, "actions": acts})
    return out[:60]


def action(agent: str, lid: str, act: str, root=None) -> bool:
    if act not in ("approve", "reject", "delete"):
        raise ValueError("unknown action")
    if not re.fullmatch(r"[a-z0-9_\-]{1,40}", agent or ""):
        raise ValueError("bad agent")
    with _lock:
        items = _load(agent, root)
        for i, it in enumerate(items):
            if str(it.get("id")) == str(lid):
                if act == "delete":
                    items.pop(i)
                else:
                    it["status"] = "approved" if act == "approve" else "rejected"
                _save(agent, items, root)
                return True
    return False


# ------------------------------------------------------------------ intent
# Explicit "this is lore: <story>" requests only; "add that to the lore" needs the model to
# say what "that" was, so it goes through the tool.
_ADD_RE = re.compile(
    r"(?i)\b(?:add\s+(?:this|that)\s+to\s+(?:the|our)\s+lore|(?:this|that)(?:'s|\s+is)\s+(?:now\s+)?(?:official\s+)?canon|"
    r"make\s+(?:this|that)\s+(?:lore|canon|a\s+running\s+joke)|remember\s+this\s+as\s+lore)\b"
    r"\s*[:\-]\s*(?P<story>.{8,})")
_RECALL_RE = re.compile(
    r"(?i)\b(?:what(?:'s|\s+is)\s+(?:our|the)\s+lore|tell\s+(?:us|me)\s+(?:our|the)\s+lore|"
    r"what\s+lore\s+do\s+you\s+(?:have|know))\b(?:\s+(?:about|on)\s+(?P<topic>[^?.!]{2,50}))?")


def intent(text: str) -> Optional[tuple]:
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "")
    m = _ADD_RE.search(t)
    if m:
        return ("add_lore", {"title": "", "story": m["story"].strip()})
    m = _RECALL_RE.search(t)
    if m:
        return ("recall_lore", {"topic": (m["topic"] or "").strip()})
    return None


def strip_legacy_names(root=None) -> int:
    """One-time (idempotent) migration for entries made before names were removed (D8): the entry's own `people` and
    `added_by` are used as its name list, the text is scrubbed and checked again, and both fields are dropped; an
    entry that now fails a check gets status "review" (unused until the owner looks at it). Returns entries changed."""
    import memory_redact as MR
    base = _mem_root(root)
    changed = 0
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        with _lock:
            items = _load(d.name, root)
            dirty = False
            for it in items:          # every entry from before (legacy fields, or never checked: D8)
                if "people" not in it and "added_by" not in it and "tagger_checked" in it:
                    continue
                names = list(it.pop("people", None) or []) + [it.pop("added_by", None) or ""]
                names = [n for n in names if n]
                it["title"] = MR.scrub(it.get("title", ""), names)
                it["story"] = MR.scrub(it.get("story", ""), names)
                it["tagger_checked"] = False
                if MR.write_time_check(it["title"] + ". " + it["story"], names) and it.get("status") != "rejected":
                    it["status"] = "review"
                dirty = True
                changed += 1
            if dirty:
                _save(d.name, items, root)
    return changed


def drop_window(t_from: float, t_to: float, root=None) -> int:
    """Off the record, backward (docs/designs/experience_memory.md, D7): drop entries from lines committed in the
    window, for every agent. Returns how many."""
    import agent_memory
    base = Path(root) if root else agent_memory.ROOT
    n = 0
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        with _lock:
            items = _load(d.name, root)
            keep = [i for i in items if not (t_from <= float(i.get("src_t") or -1) <= t_to)]
            if len(keep) != len(items):
                n += len(items) - len(keep)
                _save(d.name, keep, root)
    return n


def _off_record() -> bool:
    """An open span, or a write from a line a backward trigger marked (D7)."""
    try:
        import call_boundary
        return call_boundary.off_record_at()
    except Exception:  # noqa: BLE001
        return False


def _src_t(now: float) -> float:
    try:
        import call_boundary
        return call_boundary.source_time(now)
    except Exception:  # noqa: BLE001
        return now
