"""Voice-addressed mail (Voice mail plugin).

"Tell Riley the raid moved to 9 when she joins." The message is addressed to a
*person the room knows by voice*, not an account. When that person next speaks and
the name book recognises them (same call or days later, via voiceprints), the agent
passes it on at the first gap in the talk.

State: memory_db/voice_mail.json (never exported). Messages expire after a few days.
Private details (phone numbers, emails, addresses, keys) are refused, same filter
as memory.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "memory_db" / "voice_mail.json"
_lock = threading.RLock()

MAX_PENDING = 30
MAX_TEXT = 300
DEFAULT_DAYS = 7
HEARD_WINDOW_S = 120.0    # recipient must have spoken this recently to get it now

MAIL_CUE_RE = re.compile(r"\(MAIL CUE #(\d+)\)")

# recipient -> (speaker label, time last heard). In memory only.
_heard: dict = {}


def _load() -> dict:
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _human(sec: float) -> str:
    sec = max(0.0, sec)
    if sec < 90:
        return "just now"
    if sec < 3600:
        return f"{round(sec / 60)} minutes ago"
    if sec < 2 * 86400:
        h = round(sec / 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    return f"{round(sec / 86400)} days ago"


def _prune(d: dict, now: float, days: float) -> list:
    msgs = [m for m in d.get("messages", []) if now - m.get("created", now) < days * 86400]
    d["messages"] = msgs
    return msgs


def _private(text: str) -> bool:
    try:
        from hypergraph_memory import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001
        return False


def leave(to: str, text: str, sender: str = "", agent_names=(), days: float = DEFAULT_DAYS,
          clock: Callable[[], float] = time.time) -> str:
    to = re.sub(r"[^A-Za-z' -]", "", (to or "")).strip(" '-")
    text = re.sub(r"\s+", " ", (text or "")).strip(" ,.;:")
    if not to or len(to) > 30:
        return "Couldn't leave that: who is it for? (a first name)"
    if _key(to) in {_key(n) for n in agent_names if n}:
        return "That's me - just tell me now."
    if sender and _key(to) == _key(sender):
        return f"That message is for {sender} themselves - nothing to pass on."
    if not text:
        return "Couldn't leave that: what's the message?"
    if _private(text):
        return ("Not saved: it looks like it has a phone number, email, address or key in it. "
                "I don't keep those.")
    text = text[:MAX_TEXT]
    now = clock()
    with _lock:
        d = _load()
        msgs = _prune(d, now, days)
        if len(msgs) >= MAX_PENDING:
            return f"Mailbox is full ({MAX_PENDING} waiting). Cancel one first."
        mid = int(d.get("next_id", 1))
        d["next_id"] = mid + 1
        msgs.append({"id": mid, "to": to, "from": sender or "", "text": text, "created": now})
        _save(d)
    who = f" from {sender}" if sender else ""
    return (f"Saved message #{mid} for {to}{who}: \"{text}\". I'll pass it on the next time I hear "
            f"{to}'s voice (kept {days:g} days).")


def list_messages(days: float = DEFAULT_DAYS, clock: Callable[[], float] = time.time) -> str:
    now = clock()
    with _lock:
        d = _load()
        msgs = _prune(d, now, days)
        _save(d)
    if not msgs:
        return "No messages waiting."
    return "Waiting: " + "; ".join(
        f"#{m['id']} for {m['to']}" + (f" from {m['from']}" if m.get("from") else "")
        + f" ({_human(now - m['created'])}): \"{m['text']}\"" for m in msgs)


def cancel(mid=None, words: str = "") -> str:
    with _lock:
        d = _load()
        msgs = d.get("messages", [])
        hit = None
        try:
            if mid not in (None, ""):
                hit = next((m for m in msgs if m["id"] == int(mid)), None)
        except (TypeError, ValueError):
            hit = None
        if hit is None and words.strip():
            w = words.lower()
            hit = next((m for m in msgs if _key(m["to"]) == _key(words) or w in m["text"].lower()), None)
        if hit is None:
            return "No such message."
        d["messages"] = [m for m in msgs if m is not hit]
        _save(d)
    return f"Cancelled message #{hit['id']} for {hit['to']}."


def heard(name: Optional[str], label: Optional[str], clock: Callable[[], float] = time.time) -> None:
    """The name book recognised `name` speaking (as `label`) on this line."""
    if name and label and label != "self":
        _heard[_key(name)] = (label, clock())


def forget_heard() -> None:
    _heard.clear()


def due(days: float = DEFAULT_DAYS, clock: Callable[[], float] = time.time) -> Optional[dict]:
    """Oldest message whose recipient spoke in the last couple of minutes."""
    now = clock()
    with _lock:
        msgs = _prune(_load(), now, days)
    for m in sorted(msgs, key=lambda x: x["created"]):
        h = _heard.get(_key(m["to"]))
        if h and now - h[1] < HEARD_WINDOW_S:
            return dict(m, label=h[0])
    return None


def claim(mid: int, clock: Callable[[], float] = time.time) -> str:
    """Remove a message as delivered and return the agent's cue line."""
    with _lock:
        d = _load()
        msgs = d.get("messages", [])
        m = next((x for x in msgs if x["id"] == mid), None)
        if m is None:
            return ""
        d["messages"] = [x for x in msgs if x is not m]
        _save(d)
    h = _heard.get(_key(m["to"]))
    lead = f"[{h[0]}] " if h else ""
    frm = m.get("from") or "someone"
    return (f"{lead}(MAIL CUE #{m['id']}) This is NOT a new line from anyone: it's your own cue. "
            f"{m['to']} is here (you just heard their voice). {frm} left a message for {m['to']} "
            f"{_human(clock() - m['created'])}: \"{m['text']}\". Pass it on to {m['to']} now, by "
            "name, in one short line, in character. Don't [HOLD].")


def cue_note(text: str) -> str:
    if MAIL_CUE_RE.search(text or ""):
        return ("MESSAGE: this turn is your own cue to pass on a message someone left. Say it to "
                "the person by name now, in one short line, in character. Do not [HOLD].")
    return ""


# ------------------------------------------------------------------ intent
_PRON = r"(?i:he|she|they|him|her|them)"
_NAME = r"[A-Z][a-z]+"
_ARRIVE = (r"(?i:joins?|gets?\s+(?:here|on|back|in)|comes?\s+(?:on|back|in|by)|shows?\s+up|is\s+back|"
           r"logs?\s+on|hops?\s+on|is\s+here|talks?|speaks?|(?:you\s+)?(?:hear|see)\s+(?:him|her|them))")
_WHEN = (r"(?:(?i:when(?:ever)?|once|if|next\s+time)\s+(?:(?:" + _PRON + r"|(?P<n2>" + _NAME + r"))\s+"
         + _ARRIVE + r"|(?i:you\s+(?:hear|see))\s+(?:" + _PRON + r"|(?P<n3>" + _NAME + r"))))")
_TELL_RE = re.compile(
    r"\b(?i:tell|remind|let)\s+(?P<to>" + _NAME + r")\s+(?i:know\s+)?(?i:that\s+)?(?P<msg>.{3,300}?)\s*,?\s*"
    + _WHEN + r"\b", re.S)
_TELL_PRE_RE = re.compile(
    r"\b" + _WHEN + r"\s*,?\s*(?i:can\s+you\s+|could\s+you\s+|please\s+)?(?i:tell|let)\s+"
    r"(?:" + _PRON + r"|(?P<to>" + _NAME + r"))\s+(?i:know\s+)?(?i:that\s+)?(?P<msg>.{3,300}?)\s*[.!?]*$", re.S)
_TELL_MID_RE = re.compile(
    r"\b(?i:tell|let)\s+(?P<to>" + _NAME + r")\s+(?i:know\s+)?" + _WHEN + r"\s*,?\s*(?i:that\s+)?"
    r"(?P<msg>.{3,300}?)\s*[.!?]*$", re.S)
_LEAVE_RE = re.compile(
    r"\b(?i:leave)\s+(?i:a\s+)?(?i:message|note)\s+(?i:for)\s+(?P<to>" + _NAME + r")\s*(?:[:,-]|(?i:saying|that))\s*"
    r"(?P<msg>.{3,300}?)\s*[.!?]*$", re.S)
_NOT_NAMES = {"Me", "Us", "Everyone", "Him", "Her", "Them", "You", "The", "Him."}


def intent(text: str) -> Optional[tuple]:
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "").strip()
    t = re.sub(r"^[A-Z][a-z]+,\s*", "", t)    # leading vocative ("Fae, ...")
    for rx in (_LEAVE_RE, _TELL_MID_RE, _TELL_RE, _TELL_PRE_RE):
        m = rx.search(t)
        if not m:
            continue
        g = m.groupdict()
        to = g.get("to") or g.get("n2") or g.get("n3")
        if not to or to in _NOT_NAMES:
            continue
        msg = re.sub(r"^(?:that\s+)?", "", m["msg"].strip(" ,.;:"), flags=re.I)
        if msg:
            return ("leave_message", {"to": to, "message": msg})
    return None
