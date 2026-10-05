"""Stop the voice model's stutter loops (owner 10-05: "it goes glitchy and cascades
the same sound like r-r-r-r-r a-a-a-a-a f-f-f-f-f").

Cause: Qwen3-TTS generates audio as 80 ms codec frames (12.5 Hz, 1920 samples at
24 kHz). Occasionally the talker gets stuck re-emitting the same frame (or a 2-3
frame cycle). The runaway frame cap (audio_module.tts_frame_cap) allows ~2x a
normal sentence length, so a loop could stutter for seconds before it was cut.

Guard: watch each sentence's audio as it streams. A loop repeats with a period of
exactly k codec frames, so compare the log-band spectrogram of the last `REPS`
periods with the same span shifted by one period (per-band mean removed, so a
steady held vowel scores low and only a repeating *pattern* scores high). When it
clears THRESH, stop that sentence (closing the native stream cancels generation on
the GPU) and let the next sentence play.

Calibration (10-05, CPU): on 130 rendered backchannel clips, 12 voice references
and ~10 min of live demo speech, the highest score real speech reached was 0.85
(a laugh); THRESH is 0.93. Suspicious sentences (strong score, or far longer than
the text) are saved to logs/tts_suspect/ so the threshold can be re-checked on
real loops.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SR = 24000
FRAME = 1920          # one codec frame at 12.5 Hz
HOP = 240             # 10 ms spectrogram hop -> 8 hops per codec frame
NFFT = 480
REPS = 4              # periods compared (a cut needs REPS+1 periods of loop)
THRESH = float(os.environ.get("ATLAS_TTS_LOOP_THRESH", "0.93"))
CHECK_EVERY = 960     # re-score every 40 ms of new audio
MIN_RMS = 0.02        # near-silence never counts as a loop
MIN_PATTERN = 0.5     # std of the repeating pattern (log units, ~2 dB): a steady tone has none
KEEP_S = (REPS + 2) * 3 * FRAME / SR  # enough history for k=3
ENABLED = os.environ.get("ATLAS_TTS_LOOP_GUARD", "1") != "0"
SUSPECT_DIR = Path(__file__).resolve().parent / "logs" / "tts_suspect"
SUSPECT_MAX = 300

_win = np.hanning(NFFT).astype(np.float32)
_edges = np.unique(np.geomspace(2, NFFT // 2 + 1, 25).astype(int))

stats = {"sentences": 0, "cut": 0, "saved": 0}


def _logspec(x: np.ndarray):
    n = (len(x) - NFFT) // HOP + 1
    if n <= 0:
        return np.zeros((0, len(_edges) - 1), np.float32), np.zeros(0, np.float32)
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    rms = np.sqrt((fr * fr).mean(1))
    S = np.abs(np.fft.rfft(fr * _win[None, :], axis=1)) ** 2
    B = np.stack([S[:, a:b].sum(1) for a, b in zip(_edges[:-1], _edges[1:])], 1)
    floor = max(float(B.max()) * 1e-3, 1e-8)   # bands >30 dB under the peak are noise, not pattern
    return np.log(B + floor), rms


def loop_score(x: np.ndarray) -> tuple[float, int]:
    """Best (score, k) over periods of k = 1..3 codec frames at the END of x."""
    S, R = _logspec(np.asarray(x, np.float32))
    best, best_k = 0.0, 0
    for k in (1, 2, 3):
        L = 8 * k
        N = REPS * L
        if len(S) < N + L or R[-N:].mean() < MIN_RMS:
            continue
        a = S[-N:] - S[-N:].mean(0)
        b = S[-N - L:-L] - S[-N - L:-L].mean(0)
        if float(a.std()) < MIN_PATTERN:
            continue
        den = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12
        s = float((a * b).sum() / den)
        if s > best:
            best, best_k = s, k
    return best, best_k


class LoopDetector:
    """Feed float32 24 kHz chunks; `feed` returns True once a loop is detected."""

    def __init__(self):
        self.buf = np.zeros(0, np.float32)
        self.since = 0
        self.total = 0
        self.max_score = 0.0
        self.k = 0
        self.tripped = False

    def feed(self, a: np.ndarray) -> bool:
        a = np.asarray(a, np.float32).reshape(-1)
        self.total += a.size
        self.buf = np.concatenate((self.buf, a))[-int(KEEP_S * SR):]
        self.since += a.size
        if self.since < CHECK_EVERY:
            return False
        self.since = 0
        s, k = loop_score(self.buf)
        if s > self.max_score:
            self.max_score, self.k = s, k
        if s >= THRESH:
            self.tripped = True
        return self.tripped


def _chars(text: str) -> int:
    t = (text or "").strip()
    return max(1, len(t) + 2 * sum(c.isdigit() for c in t))


def _suspicious(text: str, frames: float, det: LoopDetector) -> bool:
    if det.tripped or det.max_score >= 0.85:
        return True
    # far beyond the per-length p99 measured on 29k logged syntheses
    return frames > 2.0 * _chars(text) + 10


def _save(text: str, audio: list[np.ndarray], det: LoopDetector, cut: bool) -> None:
    try:
        SUSPECT_DIR.mkdir(parents=True, exist_ok=True)
        if len(list(SUSPECT_DIR.glob("*.wav"))) >= SUSPECT_MAX:
            return
        import soundfile as sf
        stem = SUSPECT_DIR / time.strftime("%Y%m%d_%H%M%S")
        stem = Path(f"{stem}_{int(time.time() * 1000) % 1000:03d}")
        sf.write(str(stem) + ".wav", np.concatenate(audio) if audio else np.zeros(1, np.float32), SR)
        meta = {"text": text, "cut": cut, "score": round(det.max_score, 3), "k": det.k,
                "frames": round(det.total / FRAME, 1), "chars": _chars(text)}
        Path(str(stem) + ".json").write_text(json.dumps(meta), encoding="utf-8")
        stats["saved"] += 1
    except Exception as e:  # noqa: BLE001 - diagnostics must never break speech
        log.debug("tts_suspect save failed: %s", e)


def guard_stream(inner, text: str):
    """Wrap a native (chunk, sample_rate) iterator; stop it when it loops."""
    det = LoopDetector()
    audio: list[np.ndarray] = []
    cut = False
    stats["sentences"] += 1
    try:
        for chunk, sr in inner:
            a = np.asarray(chunk, np.float32).reshape(-1)
            if int(sr) != SR:          # unknown rate: pass through untouched
                yield chunk, sr
                continue
            audio.append(a)
            if ENABLED and det.feed(a):
                cut = True
                stats["cut"] += 1
                log.warning("[tts] stutter loop cut after %.1fs (score %.2f, period %d frame%s): %r",
                            det.total / SR, det.max_score, det.k, "" if det.k == 1 else "s",
                            (text or "")[:60])
                return
            yield chunk, sr
    finally:
        close = getattr(inner, "close", None)
        if close:
            try:
                close()                # sets the native cancel flag -> GPU stops
            except Exception:  # noqa: BLE001
                pass
        if _suspicious(text, det.total / FRAME, det):
            _save(text, audio, det, cut)


def install(engine) -> bool:
    """Wrap QwenEngine._stream_native on this engine instance."""
    orig = getattr(engine, "_stream_native", None)
    if orig is None:
        log.warning("[tts] QwenEngine has no _stream_native; stutter guard not installed")
        return False

    def guarded(stream_kwargs, *, stream_started_ns=None):
        inner = orig(stream_kwargs, stream_started_ns=stream_started_ns)
        max_new = int(stream_kwargs.get("max_new_tokens") or 0)
        if max_new and max_new <= 16:  # warmup pass
            return inner
        return guard_stream(inner, stream_kwargs.get("text") or "")

    engine._stream_native = guarded
    return True
