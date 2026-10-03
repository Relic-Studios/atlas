"""Loop-bait guard: notice people trying to trap the agent in a loop.

Pure + deterministic (no model call). Three patterns seen live in Discord:
  repeat - someone repeats a line the agent ALREADY answered (broken record)
  parrot - someone echoes the agent's own last line back at it
  spam   - one sentence repeated 3+ times inside a single utterance

A repeat the agent never answered ("Max are you there?" x3 while he was
silent) is a real request, NOT bait.

Escalation per bait line:
  1st detection -> 'roast': steering note -> one quick roast, then pivot
  continued     -> 'hold' : deterministic silence on that line (no LLM call),
                            so the roast can't become its own loop.
"""
from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass

_LABEL_RE = re.compile(r'^\s*\[(S\d+|to=[^\]]*|SPEAK[^\]]*)\]\s*')
ROAST_COOLDOWN_S = 120.0
MIN_WORDS = 2
_SPEECH_CMD = re.compile(r"^(?:say|repeat|tell|sing|call|read|spell)\b")
# A repeated CLAIM ("she's a woman" x4, live 13:57) is bait; a chant has no claim.
_CLAIM = re.compile(r"\b(?:you're|youre|you are|he's|she's|he is|she is|i'm|im|i am|it's|its|is|are|am)\b")


def _body(text: str) -> str:
    return _LABEL_RE.sub('', text or '', count=1).strip()


def _speaker(text: str) -> str | None:
    m = re.match(r'^\s*\[(S\d+)\]', text or '')
    return m[1] if m else None


def key(text: str) -> str:
    return ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', _body(text).lower()).split())


def _same(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _meaningful(k: str) -> bool:
    if len(k.split()) < MIN_WORDS:
        return False
    try:
        from floor import is_stock_filler
        return not is_stock_filler(k)
    except Exception:  # noqa: BLE001
        return True


@dataclass
class Verdict:
    kind: str = ''        # 'repeat' | 'parrot' | 'spam' | ''
    action: str = ''      # 'roast' | 'hold' | ''
    line: str = ''
    count: int = 0
    speaker: str | None = None
    note: str = ''


def _note(kind: str, who: str, line: str, count: int) -> str:
    q = line if len(line) <= 80 else line[:77] + '...'
    return (f'{who} keeps repeating "{q}". You already heard it. Don\'t answer it again; '
            "ignore it or say one short, good-natured thing about it, then move on.")


def detect(txt: str, history: list) -> Verdict:
    """Classify the current utterance against the raw history (no escalation)."""
    cur = key(txt)
    spk = _speaker(txt)
    who = spk or 'This person'

    # spam: one sentence 3+ times inside the utterance
    sents = [key(s) for s in re.split(r'(?<=[.!?])\s+|,\s+', _body(txt)) if key(s)]
    for s in set(sents):
        n = sum(_same(s, t) for t in sents)
        # Chanting ("let's go, let's go", "GG GG GG", "no no no") is hype, not bait.
        # In-utterance spam is bait only when it's a command to make the agent
        # say/repeat something, or an absurd run (6+).
        if n >= 3 and _meaningful(s) and (_SPEECH_CMD.match(s) or _CLAIM.search(s) or n >= 6):
            return Verdict('spam', 'roast', s, n, spk, _note('spam', who, s, n))

    if not _meaningful(cur):
        return Verdict()

    hist = list(history or [])
    # the in-flight turn may already be committed at the tail; don't count it twice
    if hist and hist[-1].get('role') == 'user' and key(hist[-1].get('content', '')) == cur:
        hist = hist[:-1]

    # parrot: echo of the agent's last reply
    last_agent = next((m for m in reversed(hist) if m.get('role') == 'assistant'), None)
    if last_agent is not None:
        ak = key(last_agent.get('content', ''))
        if len(ak.split()) >= 3 and _same(cur, ak):
            return Verdict('parrot', 'roast', cur, 1, spk, _note('parrot', who, cur, 1))

    # repeat: prior occurrences in the last ~24 messages, at least one already answered
    window = hist[-24:]
    prior = answered = 0
    for i, m in enumerate(window):
        if m.get('role') != 'user' or not _same(cur, key(m.get('content', ''))):
            continue
        prior += 1
        if any(n.get('role') == 'assistant' for n in window[i + 1:]):
            answered += 1
    if prior >= 2 and answered >= 1:
        return Verdict('repeat', 'roast', cur, prior + 1, spk, _note('repeat', who, cur, prior + 1))
    return Verdict()


class LoopBaitGuard:
    """Adds escalation: once a bait line has been roasted AND the agent has spoken
    since, further bait on that line -> hold. Re-preparing the same turn (partial
    transcript refinements) keeps returning 'roast', never an accidental hold."""

    def __init__(self, cooldown_s: float = ROAST_COOLDOWN_S, clock=time.monotonic):
        self.cooldown_s = cooldown_s
        self.clock = clock
        self._roasted: dict[str, tuple[float, int]] = {}

    @staticmethod
    def _spoken_count(history: list) -> int:
        return sum(1 for m in history or [] if m.get('role') == 'assistant')

    def check(self, txt: str, history: list) -> Verdict:
        v = detect(txt, history)
        if not v.kind:
            return v
        now, spoken = self.clock(), self._spoken_count(history)
        self._roasted = {k: r for k, r in self._roasted.items() if now - r[0] < self.cooldown_s}
        for k, (_, spoken_at) in self._roasted.items():
            if _same(k, v.line):
                if spoken > spoken_at:          # he already replied since the roast
                    v.action, v.note = 'hold', ''
                return v                        # same turn re-prepared -> still roast
        self._roasted[v.line] = (now, spoken)
        return v

    def reset(self) -> None:
        self._roasted.clear()
