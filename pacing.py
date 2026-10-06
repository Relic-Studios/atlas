"""Conversational pacing: how much of the talking each person is doing.

Owner 10-06: "our agent needs to talk less ... analyze the length everyone is
speaking and match it with healthy pacing". Measured on the 10-06 call: the agent
said 31% of all words in a 7-person room (fair share ~14%), and only ~7% of its
turns followed a line that used its name.

This module is pure (no I/O, injectable clock) so the live gate, the sims and the
offline replay (tools/pacing_replay.py) all share one implementation.

Shares are counted in WORDS from the transcript, because that is the one unit we
have identically for humans and the agent (live and in recordings). Measured
audio (voiced seconds, mean level, overlap) is logged alongside for analysis and
training data, but never changes a decision on its own.
"""
from __future__ import annotations

import math
import re
import statistics
import time
from collections import deque
from typing import Callable, Optional

WINDOW_S = 300.0          # rolling window for shares (5 minutes)
MIN_WORDS = 60            # below this the room hasn't said enough to judge pacing
MIN_HUMANS = 2            # 1-on-1 conversations are never paced
FRAGMENT_SHARE = 0.05     # a voice label with <5% of human words is a diarizer fragment, not a person
SOFT_PRESSURE = 1.15      # over budget: only plausible requests get through
HARD_PRESSURE = 1.6       # well over budget: only direct address gets through
AGENT_WPS = 2.7           # TTS speaking rate, used to estimate agent seconds
HUMAN_WPS = 2.5

_LABEL = re.compile(r"^\s*\[(?:S\d+|user|self)\]\s*")


def count_words(text: str) -> int:
    body = _LABEL.sub("", text or "")
    # CJK has no spaces: count characters / 2 as a rough word count
    cjk = sum(1 for ch in body if "぀" <= ch <= "鿿" or "가" <= ch <= "힯")
    latin = len(re.findall(r"[^\W_]+(?:['’][^\W_]+)*", re.sub(r"[぀-鿿가-힯]", " ", body)))
    return latin + (cjk + 1) // 2


def audio_features(audio, sample_rate: int = 16000) -> dict:
    """Cheap per-turn audio stats from the final-turn float32 mono buffer.

    voiced_s: 20 ms frames within 30 dB of the turn's loudest frame (and above
    -55 dBFS); mean_dbfs: mean level of those frames; dur_s: buffer length.
    """
    if audio is None:
        return {}
    try:
        import numpy as np
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return {}
        if float(np.max(np.abs(x))) > 4.0:   # int16 samples slipped through
            x = x / 32768.0
        hop = max(1, int(sample_rate * 0.02))
        n = x.size // hop
        if n == 0:
            return {"dur_s": round(x.size / sample_rate, 2)}
        fr = x[: n * hop].reshape(n, hop)
        rms = np.sqrt(np.mean(fr * fr, axis=1) + 1e-12)
        db = 20.0 * np.log10(rms + 1e-12)
        peak = float(np.max(db))
        voiced = (db > peak - 30.0) & (db > -55.0)
        vs = float(np.sum(voiced)) * hop / sample_rate
        out = {"dur_s": round(x.size / sample_rate, 2), "voiced_s": round(vs, 2),
               "peak_dbfs": round(peak, 1)}
        if np.any(voiced):
            out["mean_dbfs"] = round(float(np.mean(db[voiced])), 1)
        return out
    except Exception:  # noqa: BLE001 - analysis must never break a turn
        return {}


class Pacing:
    """Rolling per-speaker talk accounting for one conversation."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 window_s: float = WINDOW_S):
        self.clock = clock
        self.window_s = window_s
        self.turns: deque = deque()          # (t, who, words, is_agent)
        self.last_end = 0.0                  # estimated end of the previous turn

    # ---------------------------------------------------------------- input
    def note_human(self, speaker: Optional[str], text: str,
                   feats: Optional[dict] = None) -> dict:
        """Record a finished human turn; returns the per-turn pacing record."""
        now = self.clock()
        who = speaker or "user"
        w = count_words(text)
        feats = dict(feats or {})
        dur = feats.get("dur_s") or (w / HUMAN_WPS)
        start = now - float(dur)
        rec = {"spk": who, "words": w, **feats}
        if self.last_end:
            rec["gap_s"] = round(start - self.last_end, 2)   # negative = overlap
        self.last_end = max(self.last_end, now)
        if w:
            self.turns.append((now, who, w, False))
        self._trim(now)
        return rec

    def note_agent(self, text: str) -> None:
        now = self.clock()
        w = count_words(text)
        if w:
            self.turns.append((now, "agent", w, True))
        self.last_end = max(self.last_end, now + w / AGENT_WPS)
        self._trim(now)

    def reset(self) -> None:
        self.turns.clear()
        self.last_end = 0.0

    def _trim(self, now: float) -> None:
        while self.turns and now - self.turns[0][0] > self.window_s:
            self.turns.popleft()

    # ---------------------------------------------------------------- state
    def snapshot(self, talkativeness: float = 0.35) -> dict:
        now = self.clock()
        self._trim(now)
        words: dict = {}
        human_turn_words = []
        for _, who, w, is_agent in self.turns:
            words[who] = words.get(who, 0) + w
            if not is_agent:
                human_turn_words.append(w)
        total = sum(words.values())
        human_total = sum(v for k, v in words.items() if k != "agent")
        humans = [k for k in words if k != "agent"
                  and words[k] >= max(1.0, FRAGMENT_SHARE * human_total)]
        agent = words.get("agent", 0)
        fair = 1.0 / (len(humans) + 1) if humans else 1.0
        # talkativeness 0 -> 60% of a fair share, 0.35 -> 88%, 1 -> 140%
        target = fair * (0.6 + 0.8 * max(0.0, min(1.0, float(talkativeness))))
        share = agent / total if total else 0.0
        enough = total >= MIN_WORDS and len(humans) >= MIN_HUMANS
        pressure = (share / target) if (enough and target > 0) else 0.0
        med = statistics.median(human_turn_words) if human_turn_words else 0
        return {
            "total_words": total,
            "humans": len(humans),
            "shares": {k: round(v / total, 3) for k, v in sorted(
                words.items(), key=lambda kv: -kv[1])} if total else {},
            "agent_share": round(share, 3),
            "fair_share": round(fair, 3),
            "target_share": round(target, 3),
            "pressure": round(pressure, 2),
            "median_turn_words": med,
        }

    def tier(self, talkativeness: float = 0.35) -> str:
        p = self.snapshot(talkativeness)["pressure"]
        if p >= HARD_PRESSURE:
            return "hard"
        if p >= SOFT_PRESSURE:
            return "soft"
        return "ok"

    def target_words(self, talkativeness: float = 0.35) -> int:
        """Reply length that matches how this room talks (6..25 words)."""
        snap = self.snapshot(talkativeness)
        med = snap["median_turn_words"] or 14
        cap = med * (0.7 if snap["pressure"] >= SOFT_PRESSURE else 1.0)
        return int(max(6, min(25, math.ceil(cap))))

    # ---------------------------------------------------------------- prompt
    def note(self, name_of: Callable[[str], Optional[str]] = lambda s: None,
             talkativeness: float = 0.35) -> str:
        """One pacing line for the agent's per-turn context ('' when not useful).

        Only ever about HOW MUCH to say -- never whether to answer: a note that
        reads like 'stay quiet' made the model hold on direct questions (lore
        keeper, 10-05)."""
        snap = self.snapshot(talkativeness)
        if snap["humans"] < MIN_HUMANS or snap["total_words"] < MIN_WORDS:
            return ""
        parts = []
        for who, sh in list(snap["shares"].items())[:6]:
            if who == "agent":
                label = "you"
            elif who in ("user",):
                label = "the owner"
            else:
                label = name_of(who) or "someone"
            parts.append(f"{label} {round(sh * 100)}%")
        tw = self.target_words(talkativeness)
        line = (f"Room pacing (last {int(self.window_s // 60)} min, share of the talking): "
                f"{', '.join(parts)}. A fair share for you here is about "
                f"{round(snap['target_share'] * 100)}%; people here say about "
                f"{snap['median_turn_words'] or tw} words a turn.")
        if snap["pressure"] >= SOFT_PRESSURE:
            line += (f" You've been talking more than your share: when you answer, "
                     f"use one short sentence (about {tw} words), no follow-up question.")
        else:
            line += f" When you answer, keep it to about {tw} words."
        return line
