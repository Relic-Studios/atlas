"""Playout shaping for agent speech: jitter buffer, soft edges, stutter meter.

Pure numpy, no audio device. CallAudio owns one Playout and calls:
  push(audio)        -- enqueue TTS samples (already at the device rate)
  drain(frames)      -- device callback pulls exactly `frames` samples
  stop(fade_ms)      -- barge-in: fade out over fade_ms, drop the rest
  end_reply()        -- optional: reply is complete (starts a short reply early)

Behaviour
- Reply start: hold audio until `prebuffer_s` is buffered, `prebuffer_s` of wall
  time has passed, or end_reply() was called. Bounded extra latency, and a
  cushion that absorbs synthesis hiccups.
- Never hard silence mid-reply: when the buffer is about to run dry the last
  FADE_MS of real audio is faded out; when audio resumes it re-primes a smaller
  cushion and fades in. A gap becomes a soft pause instead of a click.
- Joints: a sample-value jump between consecutive chunks is ramped over ~3 ms.
- Loudness: slow per-reply gain toward a target RMS (bounded, speech blocks only)
  plus a soft limiter so no chunk clips or jumps in volume.
- Adaptive: a reply that underran raises the next reply's prebuffer; a run of
  clean replies lowers it again.
- Stats: underruns (dry -> audio resumed within the same reply) and silence ms.
  A reply that simply ENDS is not an underrun.
"""
from __future__ import annotations

import collections
import os
import threading
import time

import numpy as np

FADE_MS = 20.0
JOINT_MS = 3.0
NEW_REPLY_IDLE_S = 0.8          # queue empty this long => next audio is a new reply
TARGET_RMS = 0.08
GAIN_MIN, GAIN_MAX = 0.6, 1.8
GAIN_UP_DB = 1.0                # max gain RISE per pushed chunk (slow release)
GAIN_DOWN_DB = 6.0              # max gain DROP per pushed chunk (fast attack)
SPEECH_RMS = 0.025              # only voiced chunks steer the gain (breaths don't)
PEAK_CEIL = 0.80                # gain never pushes a chunk's peak past the limiter knee


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Playout:
    def __init__(self, sample_rate: int = 48000, *, prebuffer_s: float | None = None,
                 min_prebuffer_s: float | None = None, max_prebuffer_s: float | None = None,
                 clock=time.monotonic):
        self.sr = int(sample_rate)
        self.clock = clock
        self.min_pre = _env_f("ATLAS_PREBUFFER_MIN", 0.25) if min_prebuffer_s is None else min_prebuffer_s
        self.max_pre = _env_f("ATLAS_PREBUFFER_MAX", 0.80) if max_prebuffer_s is None else max_prebuffer_s
        self.prebuffer_s = _env_f("ATLAS_PREBUFFER", 0.40) if prebuffer_s is None else prebuffer_s
        self.prebuffer_s = min(max(self.prebuffer_s, self.min_pre), self.max_pre)
        self.fade_n = max(1, int(self.sr * FADE_MS / 1000))
        # Reply start: TTS already begins near-silent; a 20 ms fade there swallowed the
        # first consonant ('Rokay' -> 'okay', live 10-01). 3 ms only de-clicks.
        self.start_fade_n = max(1, int(self.sr * 0.003))
        self._fade_in_n = self.start_fade_n
        self.joint_n = max(1, int(self.sr * JOINT_MS / 1000))
        self._lock = threading.Lock()
        self._q: collections.deque[np.ndarray] = collections.deque()
        self._n = 0                      # queued samples
        self._primed = False             # currently releasing audio
        self._wait_since = None          # when priming started
        self._need_fade_in = True
        self._ended = False
        self._last_out_at = 0.0          # last time real audio was emitted
        self._last_tail = 0.0            # last pushed sample (joint smoothing)
        self._dry_at = None              # time we ran dry mid-reply
        self._reply_underruns = 0
        self._clean_streak = 0
        self._gain = 1.0
        self._in_reply = False
        # meters
        self._reply_played = 0           # samples emitted in the current reply
        self._reply_pushed = 0           # samples queued in the current reply
        self.underruns = 0
        self.silence_ms = 0.0
        self.replies = 0
        self.last_reply_underruns = 0
        self.stops = 0

    # ------------------------------------------------------------------ input
    def push(self, audio: np.ndarray) -> None:
        a = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not a.size:
            return
        a = a.copy()
        with self._lock:
            now = self.clock()
            if not self._in_reply or (self._n == 0 and not self._primed
                                      and now - self._last_out_at > NEW_REPLY_IDLE_S
                                      and self._dry_at is None):
                self._begin_reply_locked(now)
            elif self._dry_at is not None:
                # audio resumed after running dry inside a reply => a real underrun
                self.underruns += 1
                self._reply_underruns += 1
                self.silence_ms += (now - self._dry_at) * 1000.0
                self._dry_at = None
                self._need_fade_in = True
                self._fade_in_n = self.fade_n
                self._wait_since = now
            # joint smoothing against the previous chunk's last sample
            if self._n or self._primed:
                jump = a[0] - self._last_tail
                if abs(jump) > 0.02:
                    k = min(self.joint_n, a.size)
                    a[:k] -= jump * np.linspace(1.0, 0.0, k, dtype=np.float32)
            self._last_tail = float(a[-1])
            a = self._level_locked(a)
            self._q.append(a)
            self._n += a.size
            self._reply_pushed += a.size

    def end_reply(self) -> None:
        with self._lock:
            self._ended = True

    def _begin_reply_locked(self, now: float) -> None:
        self._finish_reply_locked()
        self._in_reply = True
        self.replies += 1
        self._reply_underruns = 0
        self._reply_played = 0
        self._reply_pushed = 0
        self._primed = False
        self._wait_since = now
        self._need_fade_in = True
        self._fade_in_n = self.start_fade_n
        self._ended = False
        self._dry_at = None

    def _finish_reply_locked(self) -> None:
        """Close out stats for the previous reply and adapt the prebuffer."""
        if not self._in_reply:
            return
        self.last_reply_underruns = self._reply_underruns
        if self._reply_underruns:
            self.prebuffer_s = min(self.max_pre, self.prebuffer_s + 0.10)
            self._clean_streak = 0
        else:
            self._clean_streak += 1
            if self._clean_streak >= 5:
                self.prebuffer_s = max(self.min_pre, self.prebuffer_s - 0.05)
                self._clean_streak = 0
        self._in_reply = False

    # ------------------------------------------------------------------ loudness
    def _level_locked(self, a: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(a * a)))
        g0 = self._gain
        peak = float(np.abs(a).max()) if a.size else 0.0
        if rms > SPEECH_RMS:
            want = min(GAIN_MAX, max(GAIN_MIN, TARGET_RMS / rms))
            # Quiet onsets/breaths used to pump the gain up, and the next loud
            # vowel then slammed the limiter ("clips in", live 10-02 Fae).
            up, down = 10 ** (GAIN_UP_DB / 20.0), 10 ** (GAIN_DOWN_DB / 20.0)
            self._gain = min(max(want, self._gain / down), self._gain * up)
        if peak > 0 and self._gain * peak > PEAK_CEIL:
            self._gain = max(GAIN_MIN, PEAK_CEIL / peak)
        # ramp gain across the chunk so a gain change never steps at a joint
        out = a * np.linspace(g0, self._gain, a.size, dtype=np.float32)
        # soft limiter: linear below 0.8, smooth knee to +-1.0
        over = np.abs(out) > 0.8
        if over.any():
            x = out[over]
            s = np.sign(x)
            out[over] = s * (0.8 + 0.2 * np.tanh((np.abs(x) - 0.8) / 0.2))
        return out

    # ------------------------------------------------------------------ output
    def drain(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        with self._lock:
            now = self.clock()
            if not self._primed:
                if not self._n:
                    if self._in_reply and self._ended and self._dry_at is None:
                        self._finish_reply_locked()
                    return out
                cushion = self.prebuffer_s if self._dry_at is None and self._reply_underruns == 0 \
                    else max(self.min_pre, self.prebuffer_s * 0.6)
                waited = now - (self._wait_since or now)
                if self._n < cushion * self.sr and waited < cushion and not self._ended:
                    return out
                self._primed = True
            take = min(frames, self._n)
            filled = 0
            while filled < take:
                head = self._q[0]
                k = min(head.size, take - filled)
                out[filled:filled + k] = head[:k]
                filled += k
                if k == head.size:
                    self._q.popleft()
                else:
                    self._q[0] = head[k:]
            self._n -= filled
            self._reply_played += filled
            if filled:
                self._last_out_at = now
                if self._need_fade_in:
                    k = min(self._fade_in_n, filled)
                    out[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
                    self._need_fade_in = False
            if self._n == 0:
                # about to be dry: fade the tail of what we emitted
                k = min(self.fade_n, filled)
                if k:
                    out[filled - k:filled] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
                self._primed = False
                if self._ended:
                    self._finish_reply_locked()
                else:
                    self._dry_at = now
                    self._wait_since = now
        return out

    def tick(self) -> None:
        """Close a reply that ran dry and never resumed (no end_reply signal)."""
        with self._lock:
            if self._dry_at is not None and self.clock() - self._dry_at > NEW_REPLY_IDLE_S:
                self._dry_at = None
                self._finish_reply_locked()

    # ------------------------------------------------------------------ barge-in
    def stop(self, fade_ms: float = 80.0) -> float:
        """Soft stop: keep `fade_ms` of queued audio with a fade-out, drop the rest.
        Returns seconds dropped."""
        with self._lock:
            keep = int(self.sr * fade_ms / 1000) if self._primed else 0
            data = np.concatenate(list(self._q)) if self._q else np.zeros(0, np.float32)
            keep = min(keep, data.size)
            dropped = (data.size - keep) / self.sr
            self._q.clear()
            if keep:
                tail = data[:keep] * np.linspace(1.0, 0.0, keep, dtype=np.float32)
                self._q.append(tail)
            self._n = keep
            self._ended = True
            self._dry_at = None
            if not keep:
                self._primed = False
                self._finish_reply_locked()
            self.stops += 1
            return dropped

    # ------------------------------------------------------------------ queries
    def pending_seconds(self) -> float:
        with self._lock:
            return self._n / self.sr

    def reply_progress(self) -> tuple[float, float]:
        """(seconds played, seconds queued) for the current/last reply."""
        with self._lock:
            return self._reply_played / self.sr, self._reply_pushed / self.sr

    def last_out_at(self) -> float:
        with self._lock:
            return self._last_out_at

    def stats(self) -> dict:
        with self._lock:
            return {"underruns": self.underruns, "silence_ms": round(self.silence_ms, 1),
                    "replies": self.replies, "last_reply_underruns": self.last_reply_underruns,
                    "prebuffer_ms": round(self.prebuffer_s * 1000), "stops": self.stops,
                    "gain": round(self._gain, 2)}
