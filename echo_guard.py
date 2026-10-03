"""Text-level echo backstop.

The voiceprint catches most of the agent's own voice leaking back through the
call (AEC is imperfect on Discord), but short or noisy TTS lines can miss it and
come back as an anonymous speaker. If the words STT just heard are mostly the
words the agent itself spoke a moment ago, it's our echo, whoever the diarizer
thinks it was. Pure, no model, microseconds.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from difflib import SequenceMatcher

WINDOW_S = 8.0        # how long after speaking a line can still come back as echo
MIN_WORDS = 3         # "yeah" / "okay" are too common to call echo
MATCH_FRAC = 0.7      # share of heard words that must line up with our words

_WORD = re.compile(r"[a-z0-9']+")
_TAG = re.compile(r"^\s*\[[^\]]*\]\s*")


def _words(text: str) -> list[str]:
    return _WORD.findall(_TAG.sub("", text or "").lower())


def overlap(heard: str, spoken: str) -> float:
    """Fraction of heard words that appear, in order, inside the spoken text."""
    h, s = _words(heard), _words(spoken)
    if not h or not s:
        return 0.0
    m = SequenceMatcher(None, h, s, autojunk=False)
    return sum(b.size for b in m.get_matching_blocks()) / len(h)


class EchoGuard:
    def __init__(self, window_s: float = WINDOW_S):
        self.window_s = window_s
        self._spoken: deque[tuple[float, str]] = deque(maxlen=12)
        self._lock = threading.Lock()

    def spoke(self, text: str, now: float | None = None) -> None:
        if text and text.strip():
            with self._lock:
                self._spoken.append((time.time() if now is None else now, text))

    def is_echo(self, heard: str, in_flight: str = "", now: float | None = None) -> bool:
        """True if `heard` is mostly the agent's own recent (or currently playing) words."""
        if len(_words(heard)) < MIN_WORDS:
            return False
        now = time.time() if now is None else now
        with self._lock:
            recent = [t for ts, t in self._spoken if now - ts <= self.window_s]
        if in_flight:
            recent.append(in_flight)
        return any(overlap(heard, t) >= MATCH_FRAC for t in recent)

    def reset(self) -> None:
        with self._lock:
            self._spoken.clear()
