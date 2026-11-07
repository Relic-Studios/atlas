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
import copy
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent / "recordings"
# Off the record (docs/designs/experience_memory.md, D7): the field holding text, per event kind
MASK_FIELDS = {"user": "text", "agent": "text", "drop": "text", "retry": "dropped", "name": "name"}
MASKED = "[off the record]"
MAX_BYTES = 50 * 1024 * 1024  # roll to a new file past this
MAX_PENDING_DATA = 256
MAX_PENDING_BYTES = 4 * 1024 * 1024


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
        self._enqueue_lock = threading.Lock()  # bookkeeping only; never held during disk I/O
        self._queue = queue.Queue()
        self._thread = None
        self._pending_data = self._pending_bytes = 0
        self.dropped_events = self.dropped_audio = self.write_errors = 0
        self._mask_windows = []
        self._event_captures = {}

    def _prune_masks_locked(self, now):
        # Queued earlier events are covered by the FIFO mask barrier; later ones
        # already carry their submission-time redaction. Only a producer still
        # capturing an older event can require an expired interval.
        captures = self._event_captures.values()
        self._mask_windows = [(a, b) for a, b in self._mask_windows
                              if b >= now - .001 or any(a <= t <= b for t in captures)]

    def _enqueue_locked(self, kind, payload=None, size=None):
        if size is not None:
            if self._pending_data >= MAX_PENDING_DATA or self._pending_bytes + size > MAX_PENDING_BYTES:
                if kind == "audio":
                    self.dropped_audio += 1
                else:
                    self.dropped_events += 1
                return None
            self._pending_data += 1
            self._pending_bytes += size
        future = Future() if size is None else None
        self._queue.put((kind, payload, size, future))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="CallRecorderWriter", daemon=True)
            self._thread.start()
        return future

    def _control(self, kind, payload=None, timeout=10.0):
        with self._enqueue_lock:
            future = self._enqueue_locked(kind, payload)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            raise TimeoutError(f"recorder {kind} did not finish within {timeout} seconds") from exc

    def _run(self):
        reported_drops = 0
        while True:
            try:
                kind, payload, size, future = self._queue.get(timeout=1.0)
            except queue.Empty:
                with self._enqueue_lock:
                    if self._queue.empty():
                        self._thread = None
                        return
                continue
            try:
                if kind == "event":
                    result = self._append_event(payload)
                elif kind == "audio":
                    result = self._append_audio(payload)
                elif kind == "mask":
                    result = self._mask_since(*payload)
                elif kind == "close":
                    with self._lock:
                        result = self.close_locked()
                else:  # flush barrier
                    with self._lock:
                        for fh in (self._fh, self._wav):
                            if fh is not None:
                                fh.flush()
                    result = True
                if future is not None:
                    future.set_result(result)
            except Exception as exc:  # recording data remains best effort; controls propagate failure
                if future is not None:
                    future.set_exception(exc)
                else:
                    self.write_errors += 1
                    logger.debug("recorder %s failed: %s", kind, exc)
            finally:
                if size is not None:
                    with self._enqueue_lock:
                        self._pending_data -= 1
                        self._pending_bytes -= size
                        drops = self.dropped_events + self.dropped_audio
                    if drops != reported_drops:
                        logger.warning("recorder overloaded: dropped %d events and %d audio blocks",
                                       self.dropped_events, self.dropped_audio)
                        reported_drops = drops
                self._queue.task_done()

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
        capture = object()
        try:
            with self._enqueue_lock:
                stamp = round(time.time(), 3)
                self._event_captures[capture] = stamp
                self._prune_masks_locked(stamp)
            field = MASK_FIELDS.get(kind)
            private = bool(field and field in fields and
                           (_off_record_open() or (kind == "agent" and _reply_off_record())))
            fields = copy.deepcopy(fields)
            if private:
                fields = {**fields, field: MASKED}
            with self._enqueue_lock:
                if field and field in fields and any(a <= stamp <= b for a, b in self._mask_windows):
                    fields[field] = MASKED
                line = json.dumps({"t": stamp, "k": kind, **fields}, ensure_ascii=False)
                self._enqueue_locked("event", line, len(line.encode("utf-8")))
        except Exception as e:  # noqa: BLE001
            logger.debug("recorder event failed: %s", e)
        finally:
            with self._enqueue_lock:
                self._event_captures.pop(capture, None)
                self._prune_masks_locked(round(time.time(), 3))

    def _append_event(self, line):
        with self._lock:
            if self._fh is None or (self._path and self._path.stat().st_size > MAX_BYTES):
                self.close_locked()
                self._open()
            self._fh.write(line + "\n")
            self._fh.flush()

    def audio(self, pcm_float32):
        if not (self.enabled and self.audio_enabled):
            return
        try:
            private = _off_record_open()
            import numpy as _np
            pcm_float32 = _np.asarray(pcm_float32, dtype=_np.float32).copy()
            if private:                                 # D7: an off-the-record span is recorded as silence
                pcm_float32 = _np.zeros_like(pcm_float32)
            with self._enqueue_lock:
                self._enqueue_locked("audio", pcm_float32, pcm_float32.nbytes)
        except Exception as e:  # noqa: BLE001
            logger.debug("recorder audio failed: %s", e)

    def _append_audio(self, pcm_float32):
        with self._lock:
            if self._fh is None:
                self._open()
            if self._wav is not None:
                self._wav.write(pcm_float32)

    def close_locked(self):
        first_error = None
        for f in (self._fh, self._wav):
            try:
                if f is not None:
                    f.close()
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = exc
        self._fh = self._wav = None
        if first_error is not None:
            raise first_error

    def flush(self, timeout=10.0):
        """Wait for previously accepted data; never call this barrier on the event loop."""
        try:
            return bool(self._control("flush", timeout=timeout))
        except TimeoutError:
            return False

    def close(self, timeout=10.0):
        """Ordered file close; reusable for a new recording, raises on barrier timeout."""
        return self._control("close", timeout=timeout)

    def mask_since(self, t_from: float, t_to: float) -> int:
        """FIFO privacy barrier. PendingWriter retries failures; never call on the event loop."""
        if not self.enabled:
            return 0
        with self._enqueue_lock:
            self._prune_masks_locked(round(time.time(), 3))
            windows = sorted(self._mask_windows + [(t_from, t_to)])
            merged = []
            for a, b in windows:
                if merged and a <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
                else:
                    merged.append((a, b))
            self._mask_windows = merged
            future = self._enqueue_locked("mask", (t_from, t_to))
        try:
            return future.result(timeout=10.0)
        except FutureTimeout as exc:
            raise TimeoutError("recorder mask did not finish within 10 seconds") from exc

    def _mask_since(self, t_from: float, t_to: float) -> int:
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
