"""Friend-group lore keeper (owner 10-05 plugin list): the room's shared canon - running jokes,
nicknames, campaign events, "the time Sam fell off the roof" - kept and brought back at the
right moment weeks later.

Storage is per agent (agent_state/memory/<agent>/lore.json), so lore one agent keeps never
reaches another. By default every new entry waits for the owner's approval in the plugin
window before the agent ever uses it ("owner" mode); "auto" mode trusts the room.

A lore entry comes back into the prompt only when the current line touches its words or one
of its people is talking, at most one per turn, and not the same entry again within
COOLDOWN_S - so it lands as a callback, not a catchphrase.

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

_lock = threading.Lock()
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


def _save(agent: str, items: list, root=None) -> None:
    p = _path(agent, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def _private(text: str) -> bool:
    try:
        from hypergraph_memory import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001
        return False


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
    story = re.sub(r"\s+", " ", (story or "").strip())[:MAX_STORY]
    title = re.sub(r"\s+", " ", (title or "").strip(" .:,'\""))[:MAX_TITLE]
    if not story:
        return "Nothing to keep: say what the lore is."
    if not title:
        title = " ".join(story.split()[:6])
    if _private(title + " " + story):
        return "Not kept: that has a private detail (phone, email, address or key). Say you don't keep those."
    mode = mode or _mode()
    status = "approved" if mode == "auto" else "pending"
    people = sorted({p.strip() for p in people if p and p.strip() and not re.fullmatch(r"S\d+", p.strip())})
    with _lock:
        items = _load(agent, root)
        key = title.lower()
        for it in items:
            if it.get("title", "").lower() == key and it.get("status") != "rejected":
                it.update(story=story, people=sorted(set(it.get("people", [])) | set(people)), at=clock())
                _save(agent, items, root)
                return f"Updated the lore '{title}'."
        items.append({"id": max([i.get("id", 0) for i in items] + [0]) + 1, "title": title, "story": story,
                      "people": people, "added_by": added_by, "at": clock(), "status": status, "uses": 0})
        # keep rejected out of the cap's way first, then oldest
        if len(items) > MAX_LORE:
            items = [i for i in items if i.get("status") != "rejected"][-MAX_LORE:]
        _save(agent, items, root)
    if status == "pending":
        return (f"Kept '{title}' for the room's lore. The owner approves new lore before you use it, so "
                "say you've noted it, without promising you'll bring it up.")
    return f"Added '{title}' to the room's lore. Acknowledge it briefly."


def _match(items: list, text: str, present: Iterable[str] = ()) -> List[dict]:
    q = _words(text)
    pres = {p.lower() for p in present if p}
    scored = []
    for it in items:
        if it.get("status") != "approved":
            continue
        tw, sw = _words(it.get("title", "")), _words(it.get("story", ""))
        s = 3 * len(q & tw) + len(q & sw)
        named = {p.lower() for p in it.get("people", [])}
        if named & q:                 # someone in the story is being talked about
            s += 2
        if named & pres:              # or is in the room talking
            s += 1
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
        with _lock:
            items = _load(agent, root)
        now = clock()
        hits = [h for h in _match(items, text, present)
                if now - _last_used.get((agent, h["id"]), 0) > COOLDOWN_S]
    except Exception:  # noqa: BLE001
        return ""
    if not hits:
        return ""
    h = hits[0]
    _last_used[(agent, h["id"])] = now
    try:
        with _lock:
            items = _load(agent, root)
            for it in items:
                if it.get("id") == h["id"]:
                    it["uses"] = int(it.get("uses", 0)) + 1
            _save(agent, items, root)
    except Exception:  # noqa: BLE001
        pass
    return ("ROOM LORE (background only - it doesn't change whether you answer. This group's own shared "
            "history: if it fits what's being said, a quick callback is welcome; don't retell the whole "
            f"story): '{h['title']}': {h['story']}")


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
                    if st == "pending" else [{"act": "delete", "label": "Remove"}])
            who = ", ".join(x.get("people", []))
            out.append({"title": f"{x['title']}: {x['story']}",
                        "meta": f"{d.name.title()} · {st}" + (f" · {who}" if who else "")
                                + (f" · added by {x['added_by']}" if x.get("added_by") else ""),
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
