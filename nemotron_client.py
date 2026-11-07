"""Atlas side of the Nemotron side process (nemotron_sidecar.py), for the hybrid diarizer.

Starts the side process in its own Python (it needs a newer transformers than Atlas), streams
it the same 16 kHz call audio the STT gets, and asks it which slot was active during a finished
turn. Every failure degrades to "no slot", so the voiceprints alone decide: a missing, slow or
crashed side process can never cost a turn or delay a reply beyond QUERY_TIMEOUT_S.

Python: ATLAS_NEMOTRON_PYTHON, private/nemotron_python.txt, or
%LOCALAPPDATA%/atlas_sidecar/venv/Scripts/python.exe. Model cache: ATLAS_NEMOTRON_HF_HOME or
~/.cache/atlas_models/hf.
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
SR = 16000
QUERY_TIMEOUT_S = 0.15
READY_TIMEOUT_S = 240.0
RESTART_BACKOFF_S = 30.0


def python_path() -> str:
    try:
        import plugins
        configured = str(plugins.settings_of("nemotron_diarization").get("python") or "").strip()
        if configured:
            return configured
    except Exception:
        pass
    if os.environ.get("ATLAS_NEMOTRON_PYTHON"):
        return os.environ["ATLAS_NEMOTRON_PYTHON"]
    try:
        p = (ROOT / "private" / "nemotron_python.txt").read_text(encoding="utf-8").strip()
        if p:
            return p
    except OSError:
        pass
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    exe = "python.exe" if sys.platform == "win32" else "python"
    sub = "Scripts" if sys.platform == "win32" else "bin"
    return str(base / "atlas_sidecar" / "venv" / sub / exe)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SlotClient:
    def __init__(self, mode: str = "low_latency", device: str = "cuda"):
        self.mode, self.device = mode, device
        self.ready = False
        self.proc = None
        self.sock = None
        self._send_lock = threading.Lock()
        self._out: queue.Queue = queue.Queue(maxsize=2000)
        self._waiting: dict = {}
        self._ids = itertools.count(1)
        self._fed = 0                       # samples queued for the side process
        self._stop = threading.Event()
        self._last_fail = 0.0
        self._starting = False
        self.stats = {"queries": 0, "answered": 0, "timeouts": 0, "restarts": 0}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Launch in the background; ready flips True once the model is loaded and connected."""
        if self._starting or self.ready or time.time() - self._last_fail < RESTART_BACKOFF_S:
            return
        self._starting = True
        threading.Thread(target=self._launch, name="NemotronLaunch", daemon=True).start()

    def _launch(self) -> None:
        try:
            py = python_path()
            if not Path(py).exists():
                raise FileNotFoundError(f"side-process Python not found: {py}")
            port = _free_port()
            env = dict(os.environ)
            env.setdefault("HF_HOME", os.environ.get("ATLAS_NEMOTRON_HF_HOME",
                                                     str(Path.home() / ".cache" / "atlas_models" / "hf")))
            env["PYTHONIOENCODING"] = "utf-8"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self.proc = subprocess.Popen(
                [py, str(ROOT / "nemotron_sidecar.py"), "--port", str(port), "--mode", self.mode,
                 "--device", self.device],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT),
                creationflags=flags, text=True, encoding="utf-8", errors="replace")
            t0 = time.time()
            while time.time() - t0 < READY_TIMEOUT_S:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"side process exited (code {self.proc.poll()})")
                if line.startswith("READY"):
                    break
            else:
                raise TimeoutError("side process did not get ready")
            threading.Thread(target=self._drain_stdout, daemon=True, name="NemotronLog").start()
            self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
            self.sock.settimeout(None)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._fed = 0
            self.ready = True          # before the threads: both loop `while self.ready`
            threading.Thread(target=self._reader, daemon=True, name="NemotronRead").start()
            threading.Thread(target=self._sender, daemon=True, name="NemotronSend").start()
            logger.info("👥 Nemotron side process ready in %.0fs (%s, %s)", time.time() - t0, self.mode, self.device)
        except Exception as e:  # noqa: BLE001
            logger.warning("👥 Nemotron side process unavailable (%s); voiceprints alone decide", e)
            self._fail()
        finally:
            self._starting = False

    def _drain_stdout(self):
        try:
            for line in self.proc.stdout:
                if "Error" in line or "Traceback" in line:
                    logger.warning("👥 nemotron: %s", line.rstrip()[:200])
        except Exception:  # noqa: BLE001
            pass

    def _fail(self):
        self.ready = False
        self._last_fail = time.time()
        for ev in list(self._waiting.values()):
            ev[0].set()
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        self.sock = None
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        self._fail()
        self._last_fail = 0.0
        self._stop = threading.Event()

    # ------------------------------------------------------------ io
    def _send(self, kind: bytes, payload: bytes) -> None:
        with self._send_lock:
            self.sock.sendall(kind + struct.pack("<I", len(payload)) + payload)

    def _sender(self):
        try:
            while self.ready:
                try:
                    x = self._out.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._send(b"A", x.tobytes())
        except (OSError, AttributeError) as e:
            if self.ready:
                logger.warning("👥 Nemotron send failed: %s", e)
                self.stats["restarts"] += 1
                self._fail()

    def _reader(self):
        try:
            while self.ready:
                head = self._recv(5)
                (n,) = struct.unpack("<I", head[1:])
                msg = json.loads(self._recv(n))
                w = self._waiting.get(msg.get("id"))
                if w is not None:
                    w[1] = msg
                    w[0].set()
        except (OSError, ConnectionError, ValueError, AttributeError) as e:
            if self.ready:
                logger.warning("👥 Nemotron side process lost: %s", e)
                self.stats["restarts"] += 1
                self._fail()

    def _recv(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("closed")
            buf += chunk
        return bytes(buf)

    # ------------------------------------------------------------ api
    @property
    def fed_s(self) -> float:
        return self._fed / SR

    def feed(self, audio16: np.ndarray) -> None:
        if not self.ready:
            return
        x = np.ascontiguousarray(audio16, dtype=np.float32)
        try:
            self._out.put_nowait(x)
            self._fed += len(x)
        except queue.Full:
            # the side process can't keep up: restart it rather than drift out of time
            logger.warning("👥 Nemotron fell behind; restarting it")
            self.stats["restarts"] += 1
            self._fail()

    def window(self, duration_s: float, timeout: float = QUERY_TIMEOUT_S):
        """Slot activities (frames x 8) for the last duration_s of the stream, or None."""
        if not self.ready:
            return None
        t1 = self.fed_s
        qid = next(self._ids)
        slot = [threading.Event(), None]
        self._waiting[qid] = slot
        self.stats["queries"] += 1
        try:
            self._send(b"Q", json.dumps({"id": qid, "t0": max(0.0, t1 - duration_s), "t1": t1}).encode())
            if not slot[0].wait(timeout) or slot[1] is None:
                self.stats["timeouts"] += 1
                return None
            self.stats["answered"] += 1
            return np.asarray(slot[1].get("frames") or [], dtype=np.float32).reshape(-1, 8)
        except (OSError, AttributeError):
            return None
        finally:
            self._waiting.pop(qid, None)
