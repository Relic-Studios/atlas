"""Structured call recording for offline replay (owner-only, local disk).

Every live call writes one JSONL file under recordings/<date>/<HHMMSS>.jsonl:
  {"t": epoch, "k": "user",     "spk": "S2", "text": "..."}
  {"t": epoch, "k": "agent",    "persona": "ivy", "text": "..."}
  {"t": epoch, "k": "decision", "action": "SPEAK", "target": "S2", "ms": 412.0}
  {"t": epoch, "k": "persona",  "persona": "max"}
  {"t": epoch, "k": "drop",     "why": "parrot", "text": "..."}
tests/sim_long_call.py --log <file.jsonl> replays it against the real model,
so a live bug becomes a permanent regression case.

Optional raw call audio (ATLAS_RECORD_AUDIO=1): the cleaned 16 kHz input that
STT hears, as <HHMMSS>.wav next to the JSONL. Off by default (size + privacy).

Never raises into the realtime path: every write is best-effort.
Disable entirely with ATLAS_RECORD=0. recordings/ is gitignored and excluded
from the public export.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent / "recordings"
MAX_BYTES = 50 * 1024 * 1024  # roll to a new file past this


class CallRecorder:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        # Only the live server records by default; sims/tests importing the
        # pipeline used to drop junk files into recordings/. ATLAS_RECORD=1 forces on.
        flag = os.environ.get("ATLAS_RECORD", "")
        import sys as _sys
        self.enabled = flag == "1" or (flag != "0" and Path(_sys.argv[0] or "").name == "server.py")
        self.audio_enabled = os.environ.get("ATLAS_RECORD_AUDIO", "0") == "1"
        self._lock = threading.Lock()
        self._fh = None
        self._path: Path | None = None
        self._wav = None

    # -- files -----------------------------------------------------------
    def _open(self):
        stamp = time.strftime("%H%M%S")
        d = self.root / time.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        self._path = d / f"{stamp}.jsonl"
        self._fh = open(self._path, "a", encoding="utf-8")
        if self.audio_enabled:
            try:
                import soundfile as sf
                self._wav = sf.SoundFile(str(d / f"{stamp}.wav"), "w", 16000, 1, "PCM_16")
            except Exception as e:  # noqa: BLE001
                logger.warning("recorder audio disabled: %s", e)
                self._wav = None

    @property
    def path(self):
        return self._path

    # -- api -------------------------------------------------------------
    def event(self, kind: str, **fields):
        if not self.enabled:
            return
        try:
            line = json.dumps({"t": round(time.time(), 3), "k": kind, **fields}, ensure_ascii=False)
            with self._lock:
                if self._fh is None or (self._path and self._path.stat().st_size > MAX_BYTES):
                    self.close_locked()
                    self._open()
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception as e:  # noqa: BLE001
            logger.debug("recorder event failed: %s", e)

    def audio(self, pcm_float32):
        if not (self.enabled and self.audio_enabled):
            return
        try:
            with self._lock:
                if self._fh is None:
                    self._open()
                if self._wav is not None:
                    self._wav.write(pcm_float32)
        except Exception as e:  # noqa: BLE001
            logger.debug("recorder audio failed: %s", e)

    def close_locked(self):
        for f in (self._fh, self._wav):
            try:
                if f is not None:
                    f.close()
            except Exception:  # noqa: BLE001
                pass
        self._fh = self._wav = None

    def close(self):
        with self._lock:
            self.close_locked()


REC = CallRecorder()


def load(path) -> list[tuple]:
    """Recording -> sim_long_call events: ('user', spk, text) / ('agent', persona, text)."""
    ev = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            try:
                d = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if d.get("k") == "user" and (d.get("text") or "").strip():
                ev.append(("user", d.get("spk") or "S1", d["text"].strip()))
            elif d.get("k") == "agent" and (d.get("text") or "").strip():
                ev.append(("agent", d.get("persona", ""), d["text"].strip()))
    return ev
