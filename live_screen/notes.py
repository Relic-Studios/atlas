"""The status line at the end of each normal reply prompt while Live screen is on (docs/designs/live_screen.md,
section 4): what the agent can see and whether it is visible right now. Short and unchanged turn to turn unless
the status changes, so it costs almost nothing."""
from __future__ import annotations

_STATE = {
    "capturing": "", "minimized": "minimized right now", "app_not_open": "the program isn't open right now",
    "black": "showing nothing right now", "hdr_unsupported": "unreadable (HDR)", "off": "not visible right now",
}


import re

# "Where did I put the key / what did he say / when did we" (docs/designs/live_screen.md, section 6).
EARLIER_RE = re.compile(
    r"\b(?:where\s+(?:did|do|is|was|are|were)\s+(?:i|we|my|the)|what\s+did\s+(?:he|she|they|it|the\s+\w+)\s+say|"
    r"when\s+did\s+(?:i|we|you)|did\s+(?:i|we)\s+(?:get|pick\s+up|grab|find|already)|(?:do|did)\s+you\s+remember\s+"
    r"(?:when|what|where)|what\s+was\s+(?:the|that)\s+\w+\s+(?:called|again))\b", re.IGNORECASE)


def turn_note(line: str, frames: bool = False) -> str:
    """The screen log on the turns that need it: questions about earlier (matching log lines with times, or that
    nothing matches), and screen questions whose look returned no frames (the watcher's latest description, if
    fresh). Empty otherwise. Screen text is wrapped as data."""
    try:
        import live_screen
        from live_screen import filters, memory
        if not live_screen.enabled():
            return ""
        body = re.sub(r"^\s*\[S\d+\]\s*", "", line or "")
        if EARLIER_RE.search(body):
            hits = sorted(memory.search(body), key=lambda e: e["t"])
            if not hits:
                return ("Your screen log has nothing about this: if it's about something on screen earlier, say you "
                        "didn't see it rather than guessing.")
            log = "\n".join(memory.line(e) for e in hits)
            return ("From your screen log (what you saw earlier, with times; use it only if it answers them, and say "
                    "when you saw it):\n" + filters.wrap(log))
        if not frames:
            cur = memory.current()
            if cur and cur.get("scene"):
                age = max(0, round(__import__("time").time() - cur["t"]))
                return (f"What the screen showed {age} s ago, from your screen log:\n" +
                        filters.wrap(f"{cur['label']}, {cur.get('title') or ''}: {cur['scene']}"))
    except Exception:  # noqa: BLE001
        return ""
    return ""


def status_line(can_look: bool = True) -> str:
    """can_look=False: a look of its own would be refused this turn (screen.can_glance). The line still invited one,
    and each refused call cost a whole extra model round before the reply (heavy-call test 11-04: 1-2 s, even for
    "what's the time where you are?")."""
    try:
        import live_screen
        if not live_screen.enabled():
            return ""
        st = live_screen.status()
    except Exception:  # noqa: BLE001
        return ""
    from live_screen.plan import _clean_title
    title = _clean_title(st.get("title") or "")
    state = st.get("state") or "off"
    if state == "capturing":
        front = f" (in front: \"{title}\")" if title and st.get("mode") == "screen" else ""
        if not can_look:
            return (f"Live screen: you're watching {st['watching']}{front}, but you looked recently, so don't look "
                    "again now; you only know what's on it when you look.")
        return (f"Live screen: you can look at {st['watching']}{front} whenever it's useful; "
                "you only know what's on it when you look.")
    words = _STATE.get(state, "not visible right now")
    return f"Live screen: {st['watching']} is {words}; you can't see it at the moment."
