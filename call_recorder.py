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
# Off the record (docs/designs/experience_memory.md, D7): the field holding text, per event kind
MASK_FIELDS = {"user": "text", "agent": "text", "drop": "text", "retry": "dropped", "name": "name"}
MASKED = "[off the record]"
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
        self._paths: list = []          # this session's files (a call can span a roll)
        self._wav = None

    # -- files -----------------------------------------------------------
    def _open(self):
        stamp = time.strftime("%H%M%S")
        d = self.root / time.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        self._path = d / f"{stamp}.jsonl"
        self._paths = (self._paths + [self._path])[-4:]
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
            field = MASK_FIELDS.get(kind)
            if field and field in fields and (_off_record_open() or (kind == "agent" and _reply_off_record())):
                fields = {**fields, field: MASKED}
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
            if _off_record_open():                       # D7: an off-the-record span is recorded as silence
                import numpy as _np
                pcm_float32 = _np.zeros_like(pcm_float32)
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

    def mask_since(self, t_from: float, t_to: float) -> int:
        """Off the record, backward (D7): mask the text of events from `t_from` to `t_to` in this session's files.
        Events are appended in time order, so only each file's tail from the first event at or after `t_from` is read
        and rewritten in place: a few kilobytes for the last minute, however large the file, so the lock (which the
        audio recorder shares) is held for milliseconds. Returns how many events were masked."""
        if not self.enabled:
            return 0
        n, failed = 0, None
        with self._lock:
            reopen = self._fh is not None
            if self._fh is not None:                     # only the text file; the audio keeps recording
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
            for path in list(self._paths)[-2:]:          # the window is a minute; older files end before it
                try:
                    n += _mask_tail(path, t_from, t_to)
                except OSError as e:
                    failed = e
            if reopen and self._path is not None:
                self._fh = open(self._path, "a", encoding="utf-8")
        if failed is not None:   # the caller (the writer thread) retries; masking an already masked line is a no-op
            raise failed
        return n


def _mask_tail(path: Path, t_from: float, t_to: float) -> int:
    """Mask the window's events in one file, rewriting only its tail from the first line at or after `t_from`."""
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        chunk = 65536
        while True:
            start = max(0, size - chunk)
            f.seek(start)
            data = f.read(size - start)
            if start > 0:
                nl = data.find(b"\n")
                if nl < 0:
                    chunk *= 2
                    continue
                data, start = data[nl + 1:], start + nl + 1
            try:
                t0 = float(json.loads(data.split(b"\n", 1)[0]).get("t", 0))
            except (ValueError, AttributeError):
                t0 = None
            if start == 0 or (t0 is not None and t0 < t_from):
                break
            chunk *= 2
        out, n = [], 0
        for raw in data.split(b"\n"):
            try:
                d = json.loads(raw) if raw.strip() else None
            except ValueError:
                d = None
            field = MASK_FIELDS.get(d.get("k")) if isinstance(d, dict) else None
            if field and field in d and t_from <= float(d.get("t", 0)) <= t_to and d[field] != MASKED:
                d[field] = MASKED
                n += 1
                raw = json.dumps(d, ensure_ascii=False).encode("utf-8")
            out.append(raw)
        if n:
            f.seek(start)
            f.write(b"\n".join(out))
            f.truncate()
    return n


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


def _reply_off_record() -> bool:
    try:
        import call_boundary
        return call_boundary.reply_off_record()
    except Exception:  # noqa: BLE001
        return False


def _off_record_open() -> bool:
    try:
        import call_boundary
        return call_boundary.OFF.is_open()
    except Exception:  # noqa: BLE001
        return False
