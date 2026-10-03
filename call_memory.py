"""Owner-approved memory across calls, per persona.

Store: agent_state/memory/<persona>.json  ->  {"items": [{id, text, status, src, t}]}
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

ROOT = Path(__file__).resolve().parent / "agent_state" / "memory"
MAX_IN_PROMPT = int(os.environ.get("ATLAS_MEMORY_MAX", "8"))
ENABLED = os.environ.get("ATLAS_MEMORY", "1") != "0"

_lock = threading.Lock()
_cache: dict = {}   # persona -> (mtime, items)


def _path(persona: str, root: Path = None) -> Path:
    return Path(root or ROOT) / f"{(persona or 'unknown').lower()}.json"


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
    n = len(items)
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
