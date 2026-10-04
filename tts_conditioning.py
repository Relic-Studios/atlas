"""Clean TTS audio on its way to the call (owner 10-04: Fae "clips and crackles sometimes").

Two measured defects, both upstream of playout's soft limiter:

1. Upsampling 24k -> 48k used linear interpolation per chunk. On Fae's reference
   that left spectral images at -60 dB in 16-24 kHz (ideal polyphase: -95 dB),
   only ~18 dB under the voice band: audible fizz that Discord's Opus then mangles.
   `Upsampler2x` is a stateful polyphase FIR: continuous across chunk joints,
   images pushed down to the ideal level, ~0.5 ms delay.

2. RealtimeTTS' Qwen engine converts float -> int16 with a HARD clip at +-1.0.
   A bright voice that overshoots gets square-topped before we ever see it
   (crackle on loud syllables). `soft_pcm16` is identity below 0.9 and a smooth
   knee to full scale above, and counts how often it engaged so live logs show
   whether overshoot actually happens.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi

log = logging.getLogger(__name__)

KNEE = 0.9
_TAPS = firwin(63, 10500, fs=48000, window=("kaiser", 8.0)).astype(np.float64) * 2.0

stats = {"chunks": 0, "overshoot_chunks": 0, "overshoot_samples": 0, "max_peak": 0.0}


class Upsampler2x:
    """24 kHz -> 48 kHz, stateful across calls (no clicks at chunk joints). Thread-safe:
    TTS and backchannels feed the same output from different threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._zi = lfilter_zi(_TAPS, 1.0) * 0.0

    def reset(self) -> None:
        with self._lock:
            self._zi = self._zi * 0.0

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n == 0:
            return np.asarray(audio, dtype=np.float32)
        up = np.zeros(2 * n, dtype=np.float64)
        up[0::2] = audio
        with self._lock:
            out, self._zi = lfilter(_TAPS, 1.0, up, zi=self._zi)
        return out.astype(np.float32)


def soft_limit(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)
    over = np.abs(x) > KNEE
    if over.any():
        v = x[over]
        x = x.copy()
        x[over] = np.sign(v) * (KNEE + (1.0 - KNEE) * np.tanh((np.abs(v) - KNEE) / (1.0 - KNEE)))
    return x


CEIL = 0.97
_WIN = 72          # 3 ms at 24 kHz: gain starts dipping this far before a peak
_lim = {"g": 1.0}  # gain at the end of the previous chunk (joint continuity)


def lookahead_limit(a: np.ndarray) -> np.ndarray:
    """Gain-envelope limiter: per-sample required gain, widened by a min filter and
    smoothed by a box average (so it stays under the requirement AND has no corners),
    ramped in from the previous chunk's gain. Identity when nothing overshoots."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    a = np.nan_to_num(np.asarray(a, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0, neginf=-1.0)
    g_prev = _lim["g"]
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak <= CEIL and g_prev >= 0.999:
        _lim["g"] = 1.0
        return a
    need = np.minimum(1.0, CEIL / np.maximum(np.abs(a), 1e-9))
    env = minimum_filter1d(need, size=2 * _WIN + 1, mode="nearest")
    env = uniform_filter1d(env, size=_WIN + 1, mode="nearest")
    env = np.minimum(env, need)  # box edges can't lift it above the requirement
    # continuity: release from the previous gain over ~20 ms, never above env
    rel = g_prev + (1.0 - g_prev) * np.minimum(1.0, np.arange(a.size) / 480.0)
    g = np.minimum(env, rel)
    _lim["g"] = float(g[-1])
    return (a * g).astype(np.float32)


def soft_pcm16(samples: np.ndarray) -> bytes:
    """Drop-in for qwen_engine._float_to_pcm16: look-ahead limiter, soft knee as
    backstop, instead of the engine's hard clip at +-1.0."""
    a = np.asarray(samples, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return b""
    peak = float(np.nanmax(np.abs(a)))
    stats["chunks"] += 1
    stats["max_peak"] = max(stats["max_peak"], round(peak, 3))
    n_over = int((np.abs(a) > 1.0).sum())
    if n_over:
        stats["overshoot_chunks"] += 1
        stats["overshoot_samples"] += n_over
        log.info("[tts] overshoot: %d samples > 1.0 (peak %.2f) limited instead of hard-clipped", n_over, peak)
    return np.rint(soft_limit(lookahead_limit(a)) * 32767.0).astype("<i2", copy=False).tobytes()


def install_qwen_soft_clip() -> bool:
    """Swap the engine's hard-clip converter for soft_pcm16 (module global, looked up per call)."""
    try:
        from RealtimeTTS.engines import qwen_engine
    except Exception:  # noqa: BLE001
        return False
    if getattr(qwen_engine, "_float_to_pcm16", None) is soft_pcm16:
        return True
    if not hasattr(qwen_engine, "_float_to_pcm16"):
        log.warning("[tts] qwen_engine has no _float_to_pcm16; soft clip not installed")
        return False
    qwen_engine._float_to_pcm16 = soft_pcm16
    return True
