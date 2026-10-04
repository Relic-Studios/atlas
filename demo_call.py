"""Demo mode: a virtual call device that plays scripted human voices INTO the real
ATLAS pipeline and records everything a Discord listener would hear.

Nothing is faked downstream: the human audio goes through Whisper STT, the
diarizer, the floor/turn logic, the real LLM and the real TTS, exactly as on a
call. Only the sound card is replaced.

Enable with   ATLAS_DEMO_SCENE=<scene.json>   (optional ATLAS_DEMO_OUT=<dir>).

scene.json:
  {"lead_in_s": 3.0,
   "lines": [{"who": "Sam", "wav": "sam_01.wav", "text": "...",
              "wait_reply": true,          # previous line addressed the agent: wait for its answer
              "gap_s": 0.6}, ...],
   "tail_s": 4.0}

Pacing is reactive, like real people: a line starts only once the agent is
quiet. If the line before it was addressed to the agent ("wait_reply"), the
director waits for the agent to start AND finish (up to reply_timeout_s) and
then leaves gap_s. Otherwise it leaves gap_s after the last sound.

The room stays silent until <out>/GO exists (the orchestrator touches it once
the agent is selected and the screen recorder runs).

Outputs (ATLAS_DEMO_OUT, default logs/demo/<stamp>/):
  master.wav   human + agent mix, 48 kHz mono (what the call sounds like)
  humans.wav / agent.wav   the two stems
  events.jsonl  wall-clock events: line start/end, agent audio start/end
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

from audio_io import BLOCKSIZE, SAMPLE_RATE, CallAudio

log = logging.getLogger(__name__)

AGENT_ON_RMS = 0.004       # agent block counts as audible above this
REPLY_TIMEOUT_S = 12.0     # max wait for an answer before the humans move on


def enabled() -> bool:
    return bool(os.environ.get("ATLAS_DEMO_SCENE"))


def call_audio_class():
    return VirtualCallAudio if enabled() else CallAudio


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g).astype(np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0:
        x = x * (0.5 / peak)   # humans at a sane, equal level
    return x.astype(np.float32)


class Director:
    """Decides, block by block, what the 'room' says next."""

    def __init__(self, scene: dict, base: Path, events):
        self.lines = scene["lines"]
        self.lead_in = float(scene.get("lead_in_s", 3.0))
        self.tail = float(scene.get("tail_s", 4.0))
        self.reply_timeout = float(scene.get("reply_timeout_s", REPLY_TIMEOUT_S))
        self.audio = [_load_wav(base / ln["wav"]) for ln in self.lines]
        self.ev = events
        self.i = 0                # next line index
        self.cur = None           # samples of the line being played
        self.pos = 0
        self.t = 0.0              # virtual clock (s)
        self.last_sound = 0.0     # end of the last human OR agent sound
        self.agent_on = False
        self.agent_started_after_line = False
        self.agent_finished_after_line = False
        self.line_end_t = None
        self.done_at = None

    def agent_block(self, audible: bool):
        if audible and not self.agent_on:
            self.ev("agent_start", t=self.t)
            if self.line_end_t is not None:
                self.agent_started_after_line = True
        if not audible and self.agent_on:
            self.ev("agent_end", t=self.t)
            if self.agent_started_after_line:
                self.agent_finished_after_line = True
        self.agent_on = audible
        if audible:
            self.last_sound = self.t

    def _ready(self) -> bool:
        if self.t < self.lead_in or self.agent_on:
            return False
        ln = self.lines[self.i]
        gap = float(ln.get("gap_s", 0.7))
        if ln.get("wait_reply") and self.line_end_t is not None:
            waited = self.t - self.line_end_t
            timeout = float(ln.get("reply_timeout_s", self.reply_timeout))
            if not self.agent_finished_after_line and waited < timeout:
                return False
            if not self.agent_finished_after_line:
                self.ev("reply_timeout", line=self.i, t=self.t)
        return self.t - self.last_sound >= gap

    def next_block(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if self.cur is None and self.i < len(self.lines) and self._ready():
            self.cur, self.pos = self.audio[self.i], 0
            ln = self.lines[self.i]
            self.ev("line_start", line=self.i, who=ln.get("who"), text=ln.get("text"), t=self.t)
        if self.cur is not None:
            k = min(n, len(self.cur) - self.pos)
            out[:k] = self.cur[self.pos:self.pos + k]
            self.pos += k
            self.last_sound = self.t
            if self.pos >= len(self.cur):
                self.ev("line_end", line=self.i, t=self.t)
                self.cur = None
                self.line_end_t = self.t
                self.agent_started_after_line = self.agent_finished_after_line = False
                self.i += 1
        if self.i >= len(self.lines) and self.cur is None and not self.agent_on and self.done_at is None \
                and self.t - self.last_sound > 1.0:
            self.done_at = self.t
        self.t += n / SAMPLE_RATE
        return out

    @property
    def finished(self) -> bool:
        return self.done_at is not None and self.t - self.done_at >= self.tail


class VirtualCallAudio(CallAudio):
    """CallAudio with the sound card replaced by a real-time virtual clock."""

    def __init__(self, *a, **kw):
        kw["enable_aec"] = False   # Discord never returns our own voice; nothing to cancel
        super().__init__(*a, **kw)
        scene_path = Path(os.environ["ATLAS_DEMO_SCENE"]).resolve()
        self.scene = json.loads(scene_path.read_text(encoding="utf-8"))
        out = os.environ.get("ATLAS_DEMO_OUT") or str(
            Path(__file__).resolve().parent / "logs" / "demo" / time.strftime("%Y%m%d_%H%M%S"))
        self.out_dir = Path(out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._evf = open(self.out_dir / "events.jsonl", "w", encoding="utf-8")
        self._evlock = threading.Lock()
        self.director = Director(self.scene, scene_path.parent, self._event)
        self._hum, self._agt = [], []
        self._thread = None
        self.wall_t0 = None

    def _event(self, kind, **kw):
        rec = dict(kind=kind, wall=time.time(), **kw)
        with self._evlock:
            self._evf.write(json.dumps(rec) + "\n")
            self._evf.flush()
        log.info("[demo] %s %s", kind, {k: v for k, v in kw.items() if k != "t"})

    def open(self):
        log.info("[audio] DEMO virtual call: %s -> %s", os.environ["ATLAS_DEMO_SCENE"], self.out_dir)
        self._running = True
        self._thread = threading.Thread(target=self._clock, name="demo-clock", daemon=True)
        self._thread.start()
        return -1, -1

    def _clock(self):
        # Hold the room silent until the orchestrator says GO (agent picked, recorder
        # running). Silence still flows so STT/VAD see a normal idle line.
        go = self.out_dir / "GO"
        while self._running and not go.exists():
            if self.on_clean_input is not None:
                try:
                    self.on_clean_input(np.zeros(BLOCKSIZE, dtype=np.float32))
                except Exception:  # noqa: BLE001
                    pass
            self.drain(BLOCKSIZE)
            time.sleep(BLOCKSIZE / SAMPLE_RATE)
        self.wall_t0 = time.time()
        self._event("start", t=0.0)
        block_s = BLOCKSIZE / SAMPLE_RATE
        nxt = time.perf_counter()
        while self._running:
            mic = self.director.next_block(BLOCKSIZE)
            self.level_in = self.level_clean = float(np.sqrt(np.mean(mic * mic)))
            self.last_clean_block = mic
            if self.on_clean_input is not None:
                try:
                    self.on_clean_input(mic)
                except Exception as e:  # noqa: BLE001
                    log.warning("[demo] clean-input callback error: %s", e)
            agent = self.drain(BLOCKSIZE)
            self.level_out = float(np.sqrt(np.mean(agent * agent))) if len(agent) else 0.0
            self.director.agent_block(self.level_out > AGENT_ON_RMS or self.pending_seconds() > 0.05)
            self._hum.append(mic.copy())
            self._agt.append(np.asarray(agent, dtype=np.float32).copy())
            if self.director.finished:
                self._event("finished", t=self.director.t)
                self._write()
                self._running = False
                break
            nxt += block_s
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            elif dt < -0.5:   # fell far behind (debugger, GC): resync, don't spin
                nxt = time.perf_counter()

    def _write(self):
        import soundfile as sf
        h = np.concatenate(self._hum) if self._hum else np.zeros(0, np.float32)
        a = np.concatenate(self._agt) if self._agt else np.zeros(0, np.float32)
        m = np.clip(h + a, -1.0, 1.0)
        sf.write(str(self.out_dir / "humans.wav"), h, SAMPLE_RATE)
        sf.write(str(self.out_dir / "agent.wav"), a, SAMPLE_RATE)
        sf.write(str(self.out_dir / "master.wav"), m, SAMPLE_RATE)
        (self.out_dir / "DONE").write_text(json.dumps(dict(seconds=len(m) / SAMPLE_RATE,
                                                           wall_t0=self.wall_t0)), encoding="utf-8")
        log.info("[demo] wrote %s (%.1fs)", self.out_dir, len(m) / SAMPLE_RATE)

    def close(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join(timeout=2)
            self._write()

    @property
    def running(self) -> bool:
        return self._running
