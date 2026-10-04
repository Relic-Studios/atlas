"""Owner-approved memory across calls, per persona.

Store: agent_state/memory/<persona>/items.json  ->  {"items": [{id, text, status, src, t}]}
(one folder per agent, see agent_memory.py)
  status: "candidate" (proposed off-call by tools/memory_review.py) | "approved" | "rejected"

Only APPROVED items ever reach the model, as a short list. Nothing is extracted
during a call and no model is called on the reply path (the mem0 lesson).
agent_state/ is gitignored and excluded from the public export.
"""
import json
import os
import threading
import time
from pathlib import Path

import agent_memory as AM

ROOT = AM.ROOT
MAX_IN_PROMPT = int(os.environ.get("ATLAS_MEMORY_MAX", "8"))
ENABLED = os.environ.get("ATLAS_MEMORY", "1") != "0"

_lock = threading.Lock()
_cache: dict = {}   # persona -> (mtime, items)


def _path(persona: str, root: Path = None) -> Path:
    AM.migrate(root or ROOT)
    return AM.folder(persona, root or ROOT) / AM.ITEMS


def load(persona: str, root: Path = None) -> list:
    p = _path(persona, root)
    try:
        mt = p.stat().st_mtime
    except OSError:
        return []
    key = str(p)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mt:
            return hit[1]
    try:
        items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
    except (OSError, ValueError):
        items = []
    with _lock:
        _cache[key] = (mt, items)
    return items


def save(persona: str, items: list, root: Path = None) -> None:
    p = _path(persona, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def add_candidates(persona: str, texts: list, src: str = "", root: Path = None) -> int:
    items = load(persona, root)
    have = {i["text"].strip().lower() for i in items}
    n = max([int(i["id"].rsplit("m", 1)[-1]) for i in items if i["id"].rsplit("m", 1)[-1].isdigit()] or [0])
    added = 0
    for t in texts:
        t = " ".join(str(t).split())[:220]
        if not t or t.lower() in have:
            continue
        n += 1
        items.append({"id": f"{persona}-m{n:04d}", "text": t, "status": "candidate",
                      "src": src, "t": round(time.time())})
        have.add(t.lower())
        added += 1
    save(persona, items, root)
    return added


def set_status(persona: str, ids, status: str, root: Path = None) -> int:
    ids = set(ids)
    items = load(persona, root)
    n = 0
    for i in items:
        if i["id"] in ids:
            i["status"] = status
            n += 1
    save(persona, items, root)
    return n


def forget(persona: str, ids, root: Path = None) -> int:
    """Remove items entirely (not just 'rejected'): the owner said forget it."""
    ids = set(ids)
    items = load(persona, root)
    keep = [i for i in items if i["id"] not in ids]
    if len(keep) != len(items):
        save(persona, keep, root)
    return len(items) - len(keep)


def forget_since(persona: str, cutoff: float, root: Path = None) -> int:
    """Owner wipe: drop approved/candidate memories added at/after `cutoff` (cutoff <= 0:
    all of them). Rejected items stay: they are the owner's "no" list, not memories, and
    they stop the reviewer from proposing the same thing again."""
    items = load(persona, root)
    keep = [i for i in items if i.get("status") == "rejected"
            or (cutoff > 0 and float(i.get("t", 0) or 0) < cutoff)]
    if len(keep) != len(items):
        save(persona, keep, root)
    return len(items) - len(keep)


def add_approved(persona: str, text: str, src: str = "owner", root: Path = None) -> str | None:
    """Owner kept a learned memory in the app: store it as approved directly."""
    t = " ".join(str(text).split())[:220]
    if not t:
        return None
    items = load(persona, root)
    for i in items:
        if i["text"].strip().lower() == t.lower():
            i["status"] = "approved"
            save(persona, items, root)
            return i["id"]
    n = max([int(i["id"].rsplit("m", 1)[-1]) for i in items if i["id"].rsplit("m", 1)[-1].isdigit()] or [0]) + 1
    iid = f"{AM.slug(persona)}-m{n:04d}"
    items.append({"id": iid, "text": t, "status": "approved", "src": src, "t": round(time.time())})
    save(persona, items, root)
    return iid


def approved(persona: str, root: Path = None) -> list:
    return [i for i in load(persona, root) if i.get("status") == "approved"]


def memory_note(persona: str, root: Path = None) -> str:
    """Short block for the system context. Empty when nothing is approved."""
    if not ENABLED:
        return ""
    items = sorted(approved(persona, root), key=lambda i: i.get("t", 0), reverse=True)[:MAX_IN_PROMPT]
    if not items:
        return ""
    lines = "\n".join(f"- {i['text']}" for i in items)
    return ("FROM EARLIER CALLS (things you actually remember; use them only when they fit, "
            "never recite the list, and don't invent anything beyond it):\n" + lines)
