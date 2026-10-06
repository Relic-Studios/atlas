"""Teach-me mode (owner 10-05 plugin list): the room teaches the agent something out loud
("here's how our league scoring works: ...") and it keeps that as a lesson it uses next time.

Lessons are per agent (agent_state/memory/<agent>/lessons.json), so one agent never sees
another's. They are taught facts about the group's own stuff (rules, house customs, how a
game works), not a general memory: the hypergraph already remembers what was said. A lesson
comes back into the prompt only when the current line is about its topic.

Private details (phone numbers, emails, addresses, keys) are refused, same filter as memory.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

MAX_LESSONS = 60
MAX_TOPIC = 60
MAX_CONTENT = 600
NOTE_MAX = 2          # lessons injected per turn

_lock = threading.Lock()
_STOP = {"the", "a", "an", "our", "my", "your", "how", "what", "works", "work", "is", "are", "of",
         "for", "to", "in", "on", "and", "we", "do", "does", "it", "that", "this", "about", "rules",
         "rule", "thing", "things", "game"}


def _path(agent: str, root: Optional[Path] = None) -> Path:
    import agent_memory
    return agent_memory.folder(agent or "default", root) / "lessons.json"


def _load(agent: str, root=None) -> list:
    p = _path(agent, root)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
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


def learn(agent: str, topic: str, content: str, taught_by: str = "", root=None,
          clock: Callable[[], float] = time.time) -> str:
    topic = re.sub(r"\s+", " ", (topic or "").strip(" .:,"))[:MAX_TOPIC]
    content = re.sub(r"\s+", " ", (content or "").strip())[:MAX_CONTENT]
    if not content:
        return "Nothing to learn: give the lesson's content."
    if not topic:
        topic = " ".join(content.split()[:5])
    if _private(topic + " " + content):
        return ("Not saved: that lesson contains a private detail (phone, email, address or key). "
                "Tell them you don't keep those.")
    with _lock:
        items = _load(agent, root)
        key = topic.lower()
        for it in items:
            if it.get("topic", "").lower() == key:
                it.update(content=content, taught_by=taught_by or it.get("taught_by", ""), at=clock())
                _save(agent, items, root)
                return f"Updated the lesson '{topic}'. Say it back in your own words to confirm you got it."
        items.append({"id": max([i.get("id", 0) for i in items] + [0]) + 1, "topic": topic,
                      "content": content, "taught_by": taught_by, "at": clock()})
        items = items[-MAX_LESSONS:]
        _save(agent, items, root)
    by = f" (taught by {taught_by})" if taught_by else ""
    return f"Learned '{topic}'{by}. Say it back briefly in your own words so they can correct you."


def _match(items: list, query: str) -> List[dict]:
    q = _words(query)
    if not q:
        return []
    scored = []
    for it in items:
        tw = _words(it.get("topic", ""))
        cw = _words(it.get("content", ""))
        s = 3 * len(q & tw) + len(q & cw)
        if tw and q & tw:
            scored.append((s, it))
        elif s >= 3:
            scored.append((s, it))
    scored.sort(key=lambda x: (-x[0], -x[1].get("at", 0)))
    return [it for _, it in scored]


def recall(agent: str, topic: str = "", root=None) -> str:
    with _lock:
        items = _load(agent, root)
    if not items:
        return "You haven't been taught anything yet. Say so plainly, and invite them to teach you."
    if topic:
        hits = _match(items, topic)
        if not hits:
            names = ", ".join(i["topic"] for i in items[-8:])
            return f"No lesson about '{topic}'. Lessons you do have: {names}."
        return " ".join(f"Lesson '{h['topic']}': {h['content']}" for h in hits[:3])
    return "Lessons you've been taught: " + "; ".join(i["topic"] for i in items[-15:]) + "."


def forget(agent: str, topic: str = "", lid=None, root=None) -> str:
    with _lock:
        items = _load(agent, root)
        keep, gone = [], []
        for it in items:
            hit = (lid is not None and str(it.get("id")) == str(lid)) or \
                  (topic and topic.lower().strip() in it.get("topic", "").lower())
            (gone if hit else keep).append(it)
        if not gone:
            return f"No lesson matching '{topic or lid}'."
        _save(agent, keep, root)
    return "Forgot: " + ", ".join(g["topic"] for g in gone) + "."


def forget_since(agent: str, cutoff: float, root=None) -> int:
    """Owner wipe: drop lessons taught at/after cutoff (0 = all). Returns how many."""
    with _lock:
        items = _load(agent, root)
        keep = [i for i in items if float(i.get("at", 0)) < cutoff] if cutoff else []
        n = len(items) - len(keep)
        if n:
            _save(agent, keep, root)
    return n


def note(agent: str, text: str, root=None) -> str:
    """Per-turn note: lessons whose topic the current line touches. Empty when none."""
    try:
        with _lock:
            items = _load(agent, root)
        hits = _match(items, text)[:NOTE_MAX] if items else []
    except Exception:  # noqa: BLE001
        return ""
    if not hits:
        return ""
    body = " ".join(f"'{h['topic']}': {h['content']}" for h in hits)
    return ("WHAT THIS ROOM TAUGHT YOU (use it if relevant; it's their own rule, trust it over "
            "general knowledge): " + body)


# ------------------------------------------------------------------ intent
# Clear, deterministic teach requests only; the model handles the rest via the tool.
_TEACH_RE = re.compile(
    r"(?i)\b(?:let\s+me\s+teach\s+you|i(?:'ll|\s+will|\s+wanna|\s+want\s+to)\s+teach\s+you|"
    r"learn\s+this|remember\s+this\s+rule|here'?s\s+how)\b\s*(?P<topic>[^:]{0,60}?)\s*:\s*(?P<body>.{8,})")
_RECALL_RE = re.compile(
    r"(?i)\bwhat\s+(?:did\s+(?:we|i)\s+teach\s+you|have\s+(?:we|you)\s+(?:been\s+)?taught\s+you|"
    r"have\s+you\s+learned|did\s+you\s+learn)\b(?:\s+about\s+(?P<topic>[^?.!]{2,50}))?")


def intent(text: str) -> Optional[tuple]:
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "")
    m = _TEACH_RE.search(t)
    if m:
        topic = re.sub(r"(?i)^(?:about|on|for|how)\s+(?:our\s+|the\s+)?", "", (m["topic"] or "").strip(" ,"))
        topic = re.sub(r"(?i)\s+works?$", "", topic)
        return ("learn_lesson", {"topic": topic, "content": m["body"].strip()})
    m = _RECALL_RE.search(t)
    if m:
        return ("recall_lessons", {"topic": (m["topic"] or "").strip()})
    return None
