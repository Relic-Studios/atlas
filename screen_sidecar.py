"""Live screen capture process (docs/designs/live_screen.md, section 1 and item 8).

Started by Atlas (live_screen/sidecar_client.py) with stdin/stdout pipes and tied to it by a job object; never
imported by the Atlas server process. It watches one target the owner picked in the plugin settings:

  screen  the whole Eyes monitor (DXGI desktop duplication), Windows notification pop-ups blacked out
  app     one program's window (Windows Graphics Capture; PrintWindow if that fails), found by its process name

and keeps the newest frame at native resolution plus a 10 s ring of small JPEGs, all in RAM. Looks:

  now     the newest frame at the asked-look budget
  recent  that plus three ring frames from about 7.5, 5 and 2.5 s earlier (small budget), oldest first

Logs go to stderr only; stdout carries only messages (live_screen/wire.py).

  python screen_sidecar.py [--parent-pid N] [--backend real|fake]      (the target comes from set_target)
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from live_screen import capture as C  # noqa: E402
from live_screen import config as CFG  # noqa: E402
from live_screen import wire as W  # noqa: E402


def log(*a):
    print("[screen_sidecar]", *a, file=sys.stderr, flush=True)


class FakeScreen:
    """Tests: a moving box on a dark background; no real screen is touched."""
    name = "fake"

    def __init__(self, *_a):
        self.rect, self.k = (0, 0, 1280, 720), 0

    def grab(self):
        self.k += 1
        f = np.full((720, 1280, 3), 25, np.uint8)
        x = (self.k * 40) % 1100
        f[300:420, x:x + 160] = (220, 60, 60)
        return f

    def release(self):
        pass


class Sidecar:
    def __init__(self, backend="real", parent_pid=0):
        self.fake = backend == "fake"
        self.exclude = {os.getpid(), parent_pid}
        self.out = sys.stdout.buffer
        self.out_lock = threading.Lock()
        self.target = {"mode": "off"}
        self.target_lock = threading.Lock()
        self.backend = None
        self.backend_key = None          # what the backend was built for: ("screen", monitor) / ("app", hwnd)
        self.retry_at = 0.0
        self.lock = threading.Lock()
        self.newest = None
        self.t = 0.0
        self.hash = None                 # difference hash of the newest frame (change events)
        self.exact = None                # checksum of the newest frame's pixels (identical-frame skip)
        self.prev_hash = None
        self.ref_hash = None             # the picture at the last change event
        self.ring = deque()              # (t, small jpeg)
        self.state = "off"
        self.black_since = None
        self.title = ""
        self.process = ""
        self.running = True
        self.last_status = None

    # ----------------------------------------------------------- capture
    def _release(self):
        if self.backend is not None:
            self.backend.release()
        self.backend, self.backend_key = None, None

    def _clear(self):
        with self.lock:
            self.newest, self.t, self.hash = None, 0.0, None
            self.prev_hash = self.ref_hash = None
            self.ring.clear()

    def _ensure_backend(self, target, now):
        """The backend for the current target, rebuilt when the target (or the app's window) changes."""
        mode = target.get("mode")
        if mode == "screen":
            key = ("screen", int(target.get("monitor") or 1))
        elif mode == "app":
            proc = (target.get("process") or "").lower()
            wins = [w for w in C.list_windows(proc) if w.title and w.process == proc] if proc and not self.fake else []
            if self.fake:
                key = ("app", 4242)
                self.title, self.process = "Fake Game", target.get("process") or "fake.exe"
            elif not wins:
                self._release()
                self.state, self.title = "app_not_open", ""
                return None
            else:
                w = max(wins, key=lambda x: (not x.minimized, x.area))
                self.title, self.process = w.title, w.process
                if w.minimized:
                    self.state = "minimized"
                    return None
                key = ("app", w.hwnd)
        else:
            self._release()
            self.state = "off"
            return None
        if self.backend_key != key:
            self._release()
            self._clear()
            self.retry_at = 0.0
        if self.backend is None:
            if now < self.retry_at:
                return None
            try:
                if self.fake:
                    self.backend = FakeScreen()
                elif key[0] == "screen":
                    self.backend = C.DxgiMonitor(key[1])
                    if C.hdr_enabled(self.backend.rect):
                        self._release()
                        self.state = "hdr_unsupported"
                        self.retry_at = now + 30.0
                        return None
                else:
                    try:
                        self.backend = C.WgcWindow(key[1])
                    except Exception as e:  # noqa: BLE001 - some windows refuse WGC
                        log("graphics capture failed, using PrintWindow:", e)
                        self.backend = C.WindowGrab(key[1])
                self.backend_key = key
            except Exception as e:  # noqa: BLE001
                log("capture unavailable:", e)
                self.state = "error"
                self.retry_at = now + 5.0
                return None
        return self.backend

    def _tick(self, now):
        with self.target_lock:
            target = dict(self.target)
        b = self._ensure_backend(target, now)
        if b is None:
            return
        f = b.grab()
        if f is None:
            if now - self.t > 3.0:
                self.state = "not_visible"
            return
        if target.get("mode") == "screen" and not self.fake:
            wins = C.list_windows()
            C.apply_masks(f, C.mask_rects(b.rect, wins))
            game = C.largest_window(b.rect, wins, exclude_pids=self.exclude)
            self.title, self.process = (game.title, game.process) if game else ("", "")
        elif self.fake:
            self.title = self.title or "Fake Game"
        black = C.is_black(f, CFG.BLACK_LEVEL)
        if black:
            self.black_since = self.black_since or now
            if now - self.black_since >= CFG.BLACK_TIME_S:
                self.state = "black"
                return
        else:
            self.black_since = None
        jpg, _, _ = C.fast_jpeg(f, CFG.SMALL_AREA, CFG.RING_JPEG_QUALITY)
        h = C.dhash(f)
        with self.lock:
            self.newest, self.t, self.hash = f, now, h
            self.exact = hashlib.md5(jpg).hexdigest()[:16]    # same small JPEG = same picture (skip a look)
            self.ring.append((now, jpg))
            while self.ring and now - self.ring[0][0] > CFG.RING_S + 0.5:
                self.ring.popleft()
        self.state = "capturing"
        self._change_gate(h, now)

    def _change_gate(self, h, now):
        """A change event when the picture moved away from the last reported one and then held still for a
        moment (the watcher's cue to look; docs/designs/live_screen.md section 6)."""
        prev, self.prev_hash = self.prev_hash, h
        if self.ref_hash is None or (C.hamming(h, self.ref_hash) >= CFG.CHANGE_BITS and prev is not None
                                     and C.hamming(h, prev) < CFG.CHANGE_BITS):
            self.ref_hash = h
            W.write_msg(self.out, self.out_lock, W.EVENT, W.pack_json({"type": "change", "t": round(now, 2)}))

    def capture_loop(self):
        period = 1.0 / CFG.CAPTURE_FPS
        next_t = time.time()
        while self.running:
            now = time.time()
            try:
                self._tick(now)
                self._maybe_event()
            except Exception as e:  # noqa: BLE001 - never die in the loop
                log("capture tick failed:", repr(e))
            next_t += period
            delay = next_t - time.time()
            if delay < -period:                    # fell behind (UAC, mode switch): don't burst
                next_t, delay = time.time(), 0.0
            time.sleep(max(0.0, delay))
        self._release()

    # ----------------------------------------------------------- answers
    def look(self, req):
        kind, now = req.get("kind", "now"), time.time()
        with self.lock:
            frame, t, ring = self.newest, self.t, list(self.ring)
        if frame is None or self.state != "capturing" or now - t > CFG.NOW_MAX_AGE_S:
            err = self.state if self.state != "capturing" else "stale"
            return {"ok": False, "error": err, "images": [], "title": self.title}, []
        images, jpegs = [], []
        if kind == "recent":
            for off in CFG.RECENT_OFFSETS:
                target = t - off
                best = min(ring, key=lambda it: abs(it[0] - target), default=None)
                if best is not None and abs(best[0] - target) <= 1.5 and best[0] < t - 0.25:
                    images.append({"age_s": round(now - best[0], 1), "role": "earlier"})
                    jpegs.append(best[1])
        jpg, w, h = C.fast_jpeg(frame, CFG.ASKED_AREA)
        images.append({"age_s": round(now - t, 2), "w": w, "h": h, "role": "frame"})
        jpegs.append(jpg)
        return {"ok": True, "images": images, "title": self.title, "process": self.process}, jpegs

    def watch(self):
        """The newest frame at the watcher's budget, for the screen log (Phase 2)."""
        now = time.time()
        with self.lock:
            frame, t, exact = self.newest, self.t, self.exact
        if frame is None or self.state != "capturing" or now - t > CFG.NOW_MAX_AGE_S:
            return {"ok": False, "error": self.state, "images": []}, []
        jpg, w, h_px = C.fast_jpeg(frame, CFG.WATCHER_AREA)
        return {"ok": True, "images": [{"age_s": round(now - t, 2), "w": w, "h": h_px, "role": "watch"}],
                "title": self.title, "process": self.process, "hash": exact, "t": round(t, 2)}, [jpg]

    def status(self):
        with self.target_lock:
            target = dict(self.target)
        return {"state": self.state, "mode": target.get("mode", "off"), "title": self.title,
                "process": self.process, "t": round(self.t, 2)}

    def _maybe_event(self):
        st = self.status()
        key = json.dumps({k: v for k, v in st.items() if k != "t"}, sort_keys=True)
        if key != self.last_status:
            self.last_status = key
            W.write_msg(self.out, self.out_lock, W.EVENT, W.pack_json({"type": "status", "status": st}))

    def handle(self, req):
        op = req.get("op")
        if op == "look":
            header, jpegs = self.look(req)
        elif op == "watch":
            header, jpegs = self.watch()
        elif op == "status":
            header, jpegs = {"ok": True, "status": self.status(), "images": []}, []
        elif op == "set_target":
            with self.target_lock:
                self.target = {k: req.get(k) for k in ("mode", "monitor", "process")}
            header, jpegs = {"ok": True, "images": []}, []
        elif op == "stop":
            self.running = False
            header, jpegs = {"ok": True, "images": []}, []
        else:
            header, jpegs = {"ok": False, "error": f"unknown op {op!r}", "images": []}, []
        header["id"] = req.get("id")
        W.write_msg(self.out, self.out_lock, W.ANSWER, W.pack_answer(header, jpegs))

    def serve(self):
        threading.Thread(target=self.capture_loop, name="capture", daemon=True).start()
        W.write_msg(self.out, self.out_lock, W.EVENT, W.pack_json({"type": "ready", "pid": os.getpid()}))
        inp = sys.stdin.buffer
        while self.running:
            try:
                kind, payload = W.read_msg(inp)
            except (EOFError, OSError, ValueError):
                break                               # Atlas went away: stop
            if kind != W.REQUEST:
                continue
            try:
                self.handle(json.loads(payload.decode("utf-8")))
            except Exception as e:  # noqa: BLE001
                log("request failed:", repr(e))
        self.running = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--backend", choices=("real", "fake"), default="real")
    a = ap.parse_args()
    if a.backend == "real":
        C.set_dpi_aware()
    Sidecar(a.backend, a.parent_pid).serve()


if __name__ == "__main__":
    main()
