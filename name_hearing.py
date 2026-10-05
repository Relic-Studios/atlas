"""Recover the active agent's name when speech-to-text mishears it while someone
is addressing the agent ("Hey Ran, you there?" -> "Hey Wren, you there?").

Demo 10-04: Whisper heard "Wren" as Ren / Ran / Rand in 4 of 7 address lines, so
the agent could not tell it was being called. Only rewrites a word that

  * sits in a direct-address slot: after hey/hi/yo/okay/alright/thanks/so/right,
    or at the start of a clause and followed by , ? ! or .
  * is capitalised (Whisper capitalises names) and is one sound-edit from the name
    after dropping silent leading letters (wr->r, kn->n, wh->w, h).

Pure text, no model calls; safe on partials and finals.
"""
from __future__ import annotations

import re
import threading

_LOCK = threading.Lock()
_NAMES: list[str] = []
_PROTECT = None  # callable -> iterable of known human names (never rewritten)

_LEAD = "hey|hi|hello|yo|okay|ok|alright|all right|right|thanks|thank you|so|well|oh|um|uh|no|yes|yeah"
_WORD = r"[A-Z][a-z']{1,14}"
# (prefix)(word)(suffix): vocative after a greeting/discourse word ...
_AFTER_LEAD = re.compile(rf"(\b(?:{_LEAD})[,.]?\s+)({_WORD})(?=\s*[,.?!]|\s*$)", re.I)
# ... or a clause-initial word followed by punctuation ("Ran, settle it.")
_CLAUSE = re.compile(rf"((?:^|[.?!]\s+))({_WORD})(?=\s*[,?!])")


def set_protect(fn) -> None:
    global _PROTECT
    _PROTECT = fn


def set_names(names) -> None:
    with _LOCK:
        _NAMES[:] = [n for n in (names or []) if n and len(n) >= 3]


def _skeleton(w: str) -> str:
    w = w.lower().replace("'", "")
    for a, b in (("wr", "r"), ("kn", "n"), ("wh", "w"), ("ph", "f"), ("ck", "k")):
        if w.startswith(a):
            w = b + w[len(a):]
    if len(w) > 2 and w[0] == "h":
        w = w[1:]
    return w


def _dist(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return 2
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _match(word: str, names: list[str]) -> str | None:
    if not word[:1].isupper():
        return None
    sw = _skeleton(word)
    for n in names:
        if word.lower() == n.lower():
            return None  # already right
        sn = _skeleton(n)
        if len(sn) < 3:
            continue
        # Same first sound plus at most one edit; or identical skeleton.
        if sw == sn or (sw[:1] == sn[:1] and _dist(sw, sn) <= 1):
            return n
    return None


def fix(text: str | None, names=None) -> str | None:
    if not text:
        return text
    with _LOCK:
        ns = list(names) if names is not None else list(_NAMES)
    if not ns:
        return text
    try:
        humans = {h.lower() for h in (_PROTECT() if _PROTECT else ()) if h}
    except Exception:  # noqa: BLE001
        humans = set()

    def sub(m):
        if m.group(2).lower() in humans:
            return m.group(0)
        hit = _match(m.group(2), ns)
        return m.group(1) + (hit or m.group(2))

    out = _AFTER_LEAD.sub(sub, text)
    out = _CLAUSE.sub(sub, out)
    return out
