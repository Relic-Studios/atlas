"""Call summary plugin: "what did we decide?" / "recap the last 10 minutes".

Keeps an in-memory transcript of the current session (never written to disk by
this module) and hands the model the relevant slice so it can summarise. The
model writes the summary; this module only supplies real lines, so nothing in a
recap can be invented from thin air.

Speaker labels are stored raw (S1, S2) and resolved to names at recap time, so a
name learned later in the call still shows up.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, Optional

MAX_LINES = 4000
MAX_AGE_S = 4 * 3600
MAX_CHARS = 3000          # what the model gets back; fits small-model context
DEFAULT_MIN = 10
_DECIDE = re.compile(r"\b(decid|agree|plan|let'?s|we('| wi)ll|gonna|going to|tomorrow|tonight|at \d|"
                     r"o'?clock|deal|settled|vote|won|pick|choose|meet|schedule|remind|todo|need to)\w*",
                     re.I)

_lock = threading.Lock()
_buf: deque = deque(maxlen=MAX_LINES)


def note(who: str, text: str, agent: bool = False, clock: Callable[[], float] = time.time) -> None:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return
    with _lock:
        _buf.append((clock(), who or "", text, bool(agent)))


def clear() -> None:
    with _lock:
        _buf.clear()


def lines(minutes: float = DEFAULT_MIN, clock: Callable[[], float] = time.time):
    now = clock()
    cut = now - max(0.5, min(float(minutes or DEFAULT_MIN), MAX_AGE_S / 60)) * 60
    with _lock:
        return [x for x in _buf if x[0] >= cut and now - x[0] <= MAX_AGE_S]


def recap(minutes=DEFAULT_MIN, focus: str = "", name_of: Optional[Callable] = None,
          clock: Callable[[], float] = time.time) -> str:
    try:
        minutes = float(minutes or DEFAULT_MIN)
    except (TypeError, ValueError):
        minutes = DEFAULT_MIN
    rows = lines(minutes, clock)
    if not rows:
        return (f"No conversation recorded in the last {minutes:g} minutes of this session "
                "(I only hold what was said since ATLAS started). Say so plainly.")

    def label(who: str, agent: bool) -> str:
        if agent:
            return "You" + (f" ({who})" if who else "")
        n = None
        if name_of:
            try:
                n = name_of(who)
            except Exception:  # noqa: BLE001
                n = None
        return n or ("someone" if not who or re.fullmatch(r"S\d+|user", who) else who)

    out = [f"[{label(w, a)}] {t}" for _, w, t, a in rows]
    total = len(out)
    text = "\n".join(out)
    if len(text) > MAX_CHARS:
        # Too long: keep lines that look like decisions/plans or match the focus,
        # plus the most recent lines, in original order.
        fw = [w for w in re.findall(r"[a-z']{4,}", (focus or "").lower())]
        keep = set(range(max(0, total - 12), total))
        for i, (_, _, t, _) in enumerate(rows):
            lt = t.lower()
            if _DECIDE.search(t) or any(w in lt for w in fw):
                keep.add(i)
        picked = [out[i] for i in sorted(keep)]
        while len("\n".join(picked)) > MAX_CHARS and len(picked) > 12:
            picked.pop(0)
        text = "\n".join(picked)
        head = (f"Transcript of the last {minutes:g} min ({total} lines; showing {len(picked)} "
                "key lines: decisions, plans and the most recent talk):")
    else:
        head = f"Transcript of the last {minutes:g} min ({total} lines):"
    tail = ("\nSummarise this for the room in your own words: what was decided or planned, "
            "who is doing what, anything still open. Only use what is in these lines; if "
            "nothing was decided, say so. Keep it short (spoken).")
    if focus:
        tail += f" Focus on: {focus[:120]}."
    return head + "\n" + text + tail


_RECAP_RE = re.compile(
    r"\b(?:what\s+did\s+(?:we|y'?all|you\s+guys|everyone)\s+(?:decide|agree|settle|say|plan)|"
    r"what(?:'s|\s+is|\s+was)\s+the\s+plan|what\s+did\s+i\s+miss|catch\s+me\s+up|"
    r"(?:recap|summari[sz]e|sum\s+up|summary\s+of)\s*(?:it\s+all|everything|(?:the|this)\s+(?:call|conversation|chat|"
    r"discussion|last|past|whole|entire)|what\s+(?:we|you\s+guys|y'?all|everyone)|(?:the\s+)?(?:last|past)\s|so\s+far)|"
    r"(?:give\s+(?:us|me)\s+a\s+)?(?:quick\s+)?recap\s*[.!?]*$|\bsum\s+(?:it|that)\s+up\b)", re.I)
_MIN_RE = re.compile(r"\b(?:last|past)\s+(\d+|few|couple(?:\s+of)?|ten|five|fifteen|twenty|thirty)?\s*"
                     r"(minutes?|mins?|hours?|hrs?)\b", re.I)
_W = {"few": 5, "couple": 2, "couple of": 2, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}


def intent(text: str):
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "")
    if not _RECAP_RE.search(t):
        return None
    mins = DEFAULT_MIN
    m = _MIN_RE.search(t)
    if m:
        n = (m[1] or "").lower()
        v = float(n) if n.isdigit() else float(_W.get(n, 1 if not n else 5))
        mins = v * (60 if m[2].lower().startswith("h") else 1)
    elif re.search(r"\b(whole|entire|full)\s+(call|thing|night)\b", t, re.I):
        mins = MAX_AGE_S / 60
    args = {"minutes": mins}
    f = re.search(r"\babout\s+(.{3,60}?)\s*[.!?]*$", t, re.I)
    if f:
        args["focus"] = f[1]
    return ("recap_call", args)
