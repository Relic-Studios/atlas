"""Nemotron-3-Diarization side process for Atlas's hybrid diarizer (diarize_hybrid.py).

Runs in its own Python (transformers >= 5.18; Atlas's own stays on its version), started and
fed by nemotron_client.py. It streams the call audio through the model and keeps a rolling
timeline of 8 slot activities, 10 ms per frame.

Protocol: one localhost TCP connection, frames of 1-byte type + 4-byte little-endian length + payload.
  A  float32 PCM, 16 kHz mono, appended to the stream
  Q  JSON {"id", "t0", "t1"}  -> R JSON {"id", "frames": [[8 floats], ...], "done_s"}
  P  JSON {"id"}              -> R JSON {"id", "done_s", "fed_s"}
Seconds are measured on the stream itself (samples received / 16000).

  <python with transformers>=5.18> nemotron_sidecar.py --port 47110 [--mode low_latency] [--device cuda]
"""
import argparse
import json
import socket
import struct
import sys
import threading
import time

import numpy as np

SR = 16000
FRAME_S = 0.01


def send_frame(sock, kind: bytes, payload: bytes, lock):
    with lock:
        sock.sendall(kind + struct.pack("<I", len(payload)) + payload)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf += chunk
    return bytes(buf)


class Stream:
    """Audio in, slot activities out, both indexed from the start of the stream."""

    def __init__(self, model, processor, keep_s=900.0):
        self.model, self.processor = model, processor
        self.lock = threading.Lock()
        self.audio = np.zeros(0, dtype=np.float32)
        self.base = 0                 # absolute sample index of self.audio[0]
        self.fed = 0                  # samples received
        self.started = False
        self.mel = 0
        self.start = 0                # absolute sample index of the next chunk
        self.cache = None
        self.frames = np.zeros((0, 8), dtype=np.float32)
        self.frame_base = 0           # absolute frame index of self.frames[0]
        self.keep_frames = int(keep_s / FRAME_S)
        self.wake = threading.Event()

    def feed(self, x):
        with self.lock:
            self.audio = np.concatenate([self.audio, x])
            self.fed += len(x)
        self.wake.set()

    def _run(self, chunk, first):
        import torch
        inp = self.processor(chunk, sampling_rate=SR, is_streaming=True, is_first_audio_chunk=first)
        inp = inp.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            out = self.model(**inp, speaker_cache=self.cache)
        self.cache = out.speaker_cache
        p = torch.sigmoid(out.logits)[0].float().cpu().numpy()
        with self.lock:
            self.frames = np.concatenate([self.frames, p])
            extra = len(self.frames) - self.keep_frames
            if extra > 0:
                self.frames = self.frames[extra:]
                self.frame_base += extra

    def step(self) -> bool:
        """Process one chunk if enough audio has arrived. True when it did."""
        pr = self.processor
        with self.lock:
            have = self.base + len(self.audio)
            if not self.started:
                if have < pr.num_samples_first_audio_chunk:
                    return False
                chunk = self.audio[: pr.num_samples_first_audio_chunk].copy()
            else:
                end = self.start + pr.num_samples_per_audio_chunk
                if end > have:
                    return False
                chunk = self.audio[self.start - self.base: end - self.base].copy()
        self._run(chunk, first=not self.started)
        with self.lock:
            if not self.started:
                self.started = True
                self.mel = pr.num_mel_frames_per_step
            else:
                self.mel += pr.num_mel_frames_per_step
            self.start = pr.audio_chunk_start(self.mel)
            drop = max(0, self.start - self.base - SR)        # keep a second before the next chunk
            if drop:
                self.audio = self.audio[drop:]
                self.base += drop
        return True

    def window(self, t0, t1):
        with self.lock:
            done_s = (self.frame_base + len(self.frames)) * FRAME_S
            f0 = max(int(t0 / FRAME_S) - self.frame_base, 0)
            f1 = min(int(t1 / FRAME_S) - self.frame_base, len(self.frames))
            rows = self.frames[f0:f1] if f1 > f0 else np.zeros((0, 8), dtype=np.float32)
            return rows.round(3).tolist(), done_s

    def loop(self, stop):
        while not stop.is_set():
            if not self.step():
                self.wake.wait(0.05)
                self.wake.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", default="low_latency")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    processor = AutoProcessor.from_pretrained("nvidia/Nemotron-3-Diarization")
    if a.device == "cpu":
        torch.set_num_threads(8)
        model = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization")
    else:
        model = AutoModelForAudioFrameClassification.from_pretrained("nvidia/Nemotron-3-Diarization",
                                                                     device_map=a.device)
    model.eval()
    processor.set_streaming_mode(a.mode)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port))
    srv.listen(1)
    print(f"READY {a.port} {a.mode} {a.device}", flush=True)
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    stream = Stream(model, processor)
    stop = threading.Event()
    threading.Thread(target=stream.loop, args=(stop,), daemon=True, name="nemotron").start()
    send_lock = threading.Lock()
    try:
        while True:
            kind = recv_exact(conn, 1)
            (n,) = struct.unpack("<I", recv_exact(conn, 4))
            body = recv_exact(conn, n)
            if kind == b"A":
                stream.feed(np.frombuffer(body, dtype=np.float32))
            elif kind == b"Q":
                q = json.loads(body)
                rows, done_s = stream.window(float(q["t0"]), float(q["t1"]))
                send_frame(conn, b"R", json.dumps({"id": q["id"], "frames": rows, "done_s": done_s}).encode(),
                           send_lock)
            elif kind == b"P":
                q = json.loads(body)
                with stream.lock:
                    done_s = (stream.frame_base + len(stream.frames)) * FRAME_S
                    fed_s = stream.fed / SR
                send_frame(conn, b"R", json.dumps({"id": q["id"], "done_s": done_s, "fed_s": fed_s}).encode(),
                           send_lock)
    except (ConnectionError, OSError):
        pass
    finally:
        stop.set()
        time.sleep(0.1)
        sys.exit(0)


if __name__ == "__main__":
    main()
