"""Atlas's side of the capture process (docs/designs/live_screen.md, item 9): launch screen_sidecar.py with pipes,
tie it to Atlas with a Windows job object, send requests, and hand answers back by request id.

Every request waits at most REQUEST_TIMEOUT_S; a late answer is dropped. The sidecar's stderr is copied into
Atlas's log. Same waiting-by-id idea as nemotron_client.SlotClient, over pipes instead of a socket.
"""
from __future__ import annotations

import base64
import itertools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from live_screen import config as CFG
from live_screen import wire as W

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
SIDECAR = ROOT / "screen_sidecar.py"


def _job_object_for(proc) -> object | None:
    """A job object that kills the sidecar when Atlas exits, however it exits (Windows only)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                      "OtherOperationCount", "ReadTransferCount",
                                                      "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    h = k32.OpenProcess(0x0101, False, proc.pid)          # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if h:
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(h))
        k32.CloseHandle(wintypes.HANDLE(h))
    return job                                            # keep the handle: closing it kills the sidecar


class SidecarClient:
    def __init__(self, on_event=None, backend: str = "real"):
        self.on_event = on_event
        self.backend = backend
        self.proc = None
        self._job = None
        self._lock = threading.Lock()                 # one writer
        self._ids = itertools.count(1)
        self._waiting: dict = {}                      # id -> [event, answer]
        self._waiting_lock = threading.Lock()
        self.ready = threading.Event()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 15.0) -> bool:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(SIDECAR), "--parent-pid", str(os.getpid()), "--backend", self.backend],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT), env=env,
            creationflags=flags)
        try:
            self._job = _job_object_for(self.proc)
        except Exception as e:  # noqa: BLE001
            logger.warning("👁️ live screen: no job object (%s); the sidecar stops when its pipe closes", e)
        threading.Thread(target=self._read, name="live-screen-read", daemon=True).start()
        threading.Thread(target=self._stderr, name="live-screen-log", daemon=True).start()
        if not self.ready.wait(timeout):
            logger.warning("👁️ live screen: capture process did not start")
            self.stop()
            return False
        return True

    def _read(self):
        try:
            while True:
                kind, payload = W.read_msg(self.proc.stdout)
                if kind == W.ANSWER:
                    header, jpegs = W.unpack_answer(payload)
                    with self._waiting_lock:
                        slot = self._waiting.pop(header.get("id"), None)
                    if slot is not None:                     # else: late, dropped
                        slot[1] = (header, jpegs)
                        slot[0].set()
                elif kind == W.EVENT:
                    ev = json.loads(payload.decode("utf-8"))
                    if ev.get("type") == "ready":
                        self.ready.set()
                    if self.on_event is not None:
                        try:
                            self.on_event(ev)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("live screen event handler: %s", e)
        except (EOFError, OSError, ValueError):
            pass
        finally:
            with self._waiting_lock:
                for slot in self._waiting.values():
                    slot[0].set()
                self._waiting.clear()

    def _stderr(self):
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    logger.info("👁️ %s", line[:300])
        except (OSError, ValueError):
            pass

    def request(self, op: str, timeout: float = CFG.REQUEST_TIMEOUT_S, **args):
        """(header, [jpeg bytes]) or None on timeout / no sidecar."""
        if not self.running:
            return None
        rid = next(self._ids)
        slot = [threading.Event(), None]
        with self._waiting_lock:
            self._waiting[rid] = slot
        try:
            W.write_msg(self.proc.stdin, self._lock, W.REQUEST, W.pack_json({"id": rid, "op": op, **args}))
        except (OSError, ValueError):
            with self._waiting_lock:
                self._waiting.pop(rid, None)
            return None
        if not slot[0].wait(timeout):
            with self._waiting_lock:
                self._waiting.pop(rid, None)
            return None
        return slot[1]

    def look(self, kind: str, timeout: float = CFG.REQUEST_TIMEOUT_S):
        """{"ok", "error", "title", "frames": [{"age_s", "role"}], "images": [base64 jpeg]} or None."""
        got = self.request("look", timeout=timeout, kind=kind)
        if got is None:
            return None
        header, jpegs = got
        return {"ok": bool(header.get("ok")), "error": header.get("error"), "title": header.get("title") or "",
                "process": header.get("process") or "", "frames": header.get("images") or [],
                "images": [base64.b64encode(j).decode("ascii") for j in jpegs]}

    def stop(self, wait: float = 1.0):
        p = self.proc
        if p is None:
            return
        try:
            if p.poll() is None:
                try:
                    W.write_msg(p.stdin, self._lock, W.REQUEST, W.pack_json({"id": 0, "op": "stop"}))
                    p.stdin.close()
                except (OSError, ValueError):
                    pass
                deadline = time.time() + wait
                while p.poll() is None and time.time() < deadline:
                    time.sleep(0.05)
                if p.poll() is None:
                    p.kill()
        finally:
            self.proc = None
            if self._job is not None and sys.platform == "win32":
                import ctypes
                ctypes.windll.kernel32.CloseHandle(self._job)
            self._job = None
