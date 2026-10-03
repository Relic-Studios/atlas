"""Import a user's reference audio as a cloneable voice (public build has none built in).

Accepts any format soundfile reads (wav/flac/ogg) or, via ffmpeg if present, mp3/m4a.
Normalises to what the TTS cloner expects (mono, 24 kHz, -1 dBFS peak), trims silence,
keeps 6-20 s of the clip and stores it under voices/user/<key>.wav with an optional
transcript, then registers it in voices/user/voices.json. Never touches dev voices.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
USER_DIR = ROOT / "voices" / "user"
INDEX = USER_DIR / "voices.json"
SR = 24000
MIN_S, MAX_S = 4.0, 20.0
MAX_UPLOAD = 40 * 1024 * 1024
_lock = threading.Lock()


class VoiceImportError(ValueError):
    pass


def slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (label or "").lower()).strip("-")[:24]
    return s


def _decode(data: bytes, filename: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    except Exception:  # noqa: BLE001 - fall through to ffmpeg
        pass
    ff = shutil.which("ffmpeg")
    if not ff:
        raise VoiceImportError("Couldn't read that file. Use WAV/FLAC/OGG, or install ffmpeg for MP3/M4A.")
    suffix = Path(filename or "x.bin").suffix or ".bin"
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / ("in" + suffix), Path(td) / "out.wav"
        src.write_bytes(data)
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not dst.exists():
            raise VoiceImportError("Couldn't decode that audio file.")
        x, sr = sf.read(str(dst), dtype="float32", always_2d=True)
        return x.mean(axis=1), sr


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, SR)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def _trim(x: np.ndarray) -> np.ndarray:
    """Drop leading/trailing silence (20 ms frames below -45 dBFS relative to peak)."""
    f = int(0.02 * SR)
    n = len(x) // f
    if n == 0:
        return x
    rms = np.sqrt(np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12)
    loud = np.where(rms > rms.max() * 10 ** (-45 / 20))[0]
    if len(loud) == 0:
        return x[:0]
    return x[max(0, loud[0] - 5) * f: min(n, loud[-1] + 6) * f]


def prepare(data: bytes, filename: str = "") -> np.ndarray:
    if not data:
        raise VoiceImportError("Empty upload.")
    if len(data) > MAX_UPLOAD:
        raise VoiceImportError("File too big (40 MB max).")
    x, sr = _decode(data, filename)
    x = _trim(_resample(np.nan_to_num(x), sr))
    if len(x) < MIN_S * SR:
        raise VoiceImportError(f"Need at least {MIN_S:.0f} seconds of clear speech (got {len(x) / SR:.1f}s).")
    x = x[: int(MAX_S * SR)]
    peak = float(np.max(np.abs(x))) or 1.0
    return (x * (10 ** (-1 / 20) / peak)).astype(np.float32)


def list_user() -> dict:
    try:
        d = json.loads(INDEX.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {"voices": {}}
    except FileNotFoundError:
        return {"voices": {}}


def import_voice(data: bytes, filename: str, label: str, transcript: str = "",
                 taken: set | None = None) -> dict:
    """-> {"key", "seconds", "label"}. Raises VoiceImportError with a user-facing message."""
    import soundfile as sf
    key = slug(label)
    if len(key) < 2:
        raise VoiceImportError("Give the voice a name (2+ letters).")
    with _lock:
        idx = list_user()
        existing = set(idx.get("voices", {})) | set(taken or ())
        if key in existing:
            raise VoiceImportError(f"A voice called '{key}' already exists.")
        x = prepare(data, filename)
        USER_DIR.mkdir(parents=True, exist_ok=True)
        wav, txt = USER_DIR / f"{key}.wav", USER_DIR / f"{key}.txt"
        sf.write(str(wav), x, SR, subtype="PCM_16")
        entry = {"wav": f"voices/user/{key}.wav", "label": (label or key).strip()[:40]}
        transcript = " ".join((transcript or "").split())[:1000]
        if transcript:
            txt.write_text(transcript, encoding="utf-8")
            entry["text"] = f"voices/user/{key}.txt"
        idx.setdefault("voices", {})[key] = entry
        tmp = INDEX.with_suffix(".tmp")
        tmp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
        tmp.replace(INDEX)
    import agent_registry
    agent_registry.reload()
    return {"key": key, "seconds": round(len(x) / SR, 1), "label": entry["label"]}
