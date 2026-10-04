"""Audio routing for a Discord-call-resident agent, with echo cancellation.

Three-layer self-hearing defence:

1. **Routing** — input is the call-audio device (what the agent hears), output
   is a separate device that feeds the call (where the agent speaks). Separate
   devices, so the agent's own voice never physically loops back into its mic.
   Device names come from user settings or the agent pack's "audio" traits.
2. **Echoff AEC** — a WebRTC acoustic echo canceller. The agent's own output is
   fed as the far-end *reference*; the input capture is the near-end *microphone*.
   Any residual bleed of the agent's voice into the input is cancelled before STT.
3. **Diarization** — (separate module) tags the agent's cloned voice as "self"
   and separates the real participants.

Input and output are both 48 kHz mono, matching Echoff's WebRTC APM contract.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

def _bus(kind: str):
    try:
        import agent_registry as _reg
        return (_reg.traits().get("audio") or {}).get(kind)
    except Exception:  # noqa: BLE001
        return None


# Owner's call-bus routing comes from the dev pack; public builds have none.
INPUT_BUS = _bus("input")
OUTPUT_BUS = _bus("output")
SAMPLE_RATE = 48000
BLOCK_MS = 20                      # one WebRTC APM block = 2 x 10 ms frames
BLOCKSIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)


def list_audio_devices() -> dict:
    """Enumerate available input and output devices for the UI selector."""
    inputs, outputs = [], []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            inputs.append({"index": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            outputs.append({"index": i, "name": d["name"]})
    return {"inputs": inputs, "outputs": outputs}


def resolve_device(kind: str, spec):
    """Resolve a device spec to (index, name): None -> default bus, int -> index, str -> name."""
    if spec is None:
        # 1) the device the user picked in setup, 2) the Voicemeeter call buses,
        # 3) the system default device (public builds on a plain PC).
        import user_settings
        try:
            picked = user_settings.audio_devices().get(kind)
        except Exception:  # noqa: BLE001
            picked = None
        if picked not in (None, ""):
            try:
                return resolve_device(kind, picked)
            except (LookupError, ValueError, sd.PortAudioError):
                log.warning("[audio] saved %s device %r not found, falling back", kind, picked)
        try:
            # Voicemeeter call-bus routing is the owner's dev setup only; a
            # public user gets the system default until they pick in setup.
            bus = INPUT_BUS if kind == "input" else OUTPUT_BUS
            if user_settings.is_public() or not bus:
                raise LookupError("no pack routing: use the system default device")
            return find_device(kind, INPUT_BUS if kind == "input" else OUTPUT_BUS)
        except LookupError:
            idx = sd.default.device[0 if kind == "input" else 1]
            if idx is None or idx < 0:
                raise
            return int(idx), sd.query_devices(int(idx))["name"]
    if isinstance(spec, int):
        return spec, sd.query_devices(spec)["name"]
    return find_device(kind, str(spec))


def find_device(kind: str, name_substr: str):
    """Return the first (primary) device whose name contains name_substr.

    Voicemeeter exposes each bus multiple times (WDM/KS/Mapper aliases). The
    primary bus sorts first, so taking the lowest matching index is correct.
    """
    for i, d in enumerate(sd.query_devices()):
        max_ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if max_ch <= 0:
            continue
        if name_substr.lower() in d["name"].lower():
            return i, d["name"]
    raise LookupError(f"no {kind} device matching {name_substr!r}")


class CallAudio:
    """B2 in + VAIO3 out, with Echoff AEC inline on the input path."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, enable_aec: bool = True,
                 input_device=None, output_device=None):
        self.sample_rate = sample_rate
        self.blocksize = BLOCKSIZE
        self.enable_aec = enable_aec
        self.input_device = input_device
        self.output_device = output_device
        self._in = None
        self._out = None
        self._aec = None
        self._running = False

        # Reference (agent's own output) ring buffer, paired with B2 blocks.
        self._ref_buf = deque()
        self._ref_lock = threading.Lock()

        # Playback buffer drained by the OUTPUT CALLBACK. play() only appends, so
        # it never blocks the caller (it used to be a blocking stream.write()
        # called from the asyncio loop, freezing the server while Max spoke).
        self._play = deque()
        self._play_lock = threading.Lock()
        self._play_samples = 0
        self._last_audio_at = 0.0
        # Jitter buffer + soft edges + stutter meter (playout.py). All agent
        # speech goes through it; the deque above is kept only for old callers.
        from playout import Playout
        self._playout = Playout(sample_rate)
        from tts_conditioning import Upsampler2x
        self._up2x = Upsampler2x()   # stateful polyphase (linear interp left -60 dB images)

        # Live meters for the telemetry panel (RMS of the last 20 ms block).
        self.level_in = 0.0
        self.level_clean = 0.0
        self.level_out = 0.0
        self.last_clean_block = np.zeros(0, dtype=np.float32)

        # Hooks set by the caller.
        self.on_clean_input = None   # callable(float32 mono np.ndarray, 48 kHz)

    # ------------------------------------------------------------------ lifecycle
    def open(self):
        if self.enable_aec:
            from echoff import AecConfig, BufferedWebRtcAecProcessor
            self._aec = BufferedWebRtcAecProcessor(AecConfig())

        in_idx, in_name = resolve_device("input", self.input_device)
        out_idx, out_name = resolve_device("output", self.output_device)
        log.info("[audio] input  = %d %s", in_idx, in_name)
        log.info("[audio] output = %d %s", out_idx, out_name)
        log.info("[audio] AEC     = %s", "ON" if self._aec else "OFF")

        self._in = sd.InputStream(
            device=in_idx, channels=1, dtype="float32",
            samplerate=self.sample_rate, blocksize=self.blocksize,
            callback=self._in_callback,
        )
        self._out = sd.OutputStream(
            device=out_idx, channels=1, dtype="float32",
            samplerate=self.sample_rate, blocksize=self.blocksize,
            callback=self._out_callback,
        )
        self._in.start()
        self._out.start()
        self._running = True
        return in_idx, out_idx

    # ------------------------------------------------------------------ input path
    def _in_callback(self, indata, frames, time_info, status):
        mic = indata.reshape(-1).astype(np.float32)
        # Pop the agent's own output (reference) for this block, pad with silence.
        with self._ref_lock:
            ref = np.zeros(self.blocksize, dtype=np.float32)
            if self._ref_buf:
                chunk = self._ref_buf.popleft()
                n = min(len(chunk), self.blocksize)
                ref[:n] = chunk[:n]
        if self._aec is not None:
            try:
                clean = np.asarray(self._aec.process_pair(ref.tolist(), mic.tolist()), dtype=np.float32)
            except Exception as e:  # noqa: BLE001 - AEC must never take down capture
                log.warning("[audio] AEC error, passing mic through: %s", e)
                clean = mic
        else:
            clean = mic
        self.level_in = float(np.sqrt(np.mean(mic * mic))) if mic.size else 0.0
        self.level_clean = float(np.sqrt(np.mean(clean * clean))) if clean.size else 0.0
        self.last_clean_block = clean
        if self.on_clean_input is not None:
            try:
                self.on_clean_input(clean)
            except Exception as e:  # noqa: BLE001
                log.warning("[audio] clean-input callback error: %s", e)

    # ------------------------------------------------------------------ output path
    @staticmethod
    def _resample_2x(audio: np.ndarray) -> np.ndarray:
        """24k -> 48k upsample by linear interpolation. Stateless and continuous
        across chunk boundaries, unlike per-chunk resample_poly (which clicks)."""
        n = len(audio)
        if n == 0:
            return audio
        out = np.empty(2 * n, dtype=np.float32)
        out[0::2] = audio
        if n >= 2:
            out[1:-1:2] = 0.5 * (audio[:-1] + audio[1:])
        out[-1] = audio[-1]
        return out

    MAX_PLAY_SECONDS = 90.0

    def _out_callback(self, outdata, frames, time_info, status):
        """Drain the playback buffer into VAIO3 and mirror EXACTLY what was
        played, block-aligned, as the AEC reference (zeros when silent so the
        far-end stream stays continuous and paired 1:1 with input blocks)."""
        out = self.drain(frames)
        outdata[:, 0] = out
        self.level_out = float(np.sqrt(np.mean(out * out))) if len(out) else 0.0
        if self._aec is not None:
            with self._ref_lock:
                self._ref_buf.append(out)
                while len(self._ref_buf) > 50:   # ~1 s of 20 ms blocks
                    self._ref_buf.popleft()

    def drain(self, frames: int) -> np.ndarray:
        out = self._playout.drain(frames)
        self._playout.tick()
        return out

    def _drain_legacy(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with self._play_lock:
            while filled < frames and self._play:
                head = self._play[0]
                n = min(len(head), frames - filled)
                out[filled:filled + n] = head[:n]
                filled += n
                if n == len(head):
                    self._play.popleft()
                else:
                    self._play[0] = head[n:]
            self._play_samples -= filled
            if filled:
                self._last_audio_at = time.monotonic()
        return out

    def play(self, float32_audio: np.ndarray, sample_rate: int | None = None):
        """Queue agent TTS for VAIO3. Non-blocking."""
        audio = np.asarray(float32_audio, dtype=np.float32).reshape(-1)
        if sample_rate and sample_rate != self.sample_rate:
            from math import gcd
            g = gcd(sample_rate, self.sample_rate)
            up, down = self.sample_rate // g, sample_rate // g
            if (up, down) == (2, 1):
                audio = self._up2x(audio)
            else:
                from scipy.signal import resample_poly
                audio = resample_poly(audio, up, down).astype(np.float32)
        if not len(audio):
            return
        if self._playout.pending_seconds() + len(audio) / self.sample_rate > self.MAX_PLAY_SECONDS:
            log.warning("[audio] playback buffer full; dropping %.2fs", len(audio) / self.sample_rate)
            return
        self._playout.push(audio)

    def end_reply(self) -> None:
        """The current reply is fully synthesized (lets a short reply start at once)."""
        self._playout.end_reply()

    def flush(self, fade_ms: float | None = None) -> float:
        """Barge-in: soft stop (fade out over ~80 ms), drop the rest. Returns seconds dropped."""
        if fade_ms is None:
            try:
                fade_ms = float(os.environ.get("ATLAS_STOP_FADE_MS", "80"))
            except ValueError:
                fade_ms = 80.0
        return self._playout.stop(fade_ms)

    def pending_seconds(self) -> float:
        return self._playout.pending_seconds()

    def playout_stats(self) -> dict:
        return self._playout.stats()

    def reply_progress(self) -> tuple[float, float]:
        return self._playout.reply_progress()

    def _last_out(self) -> float:
        return self._playout.last_out_at()

    def audible_within(self, window_s: float) -> bool:
        """True if agent audio was queued or audible at any point in the last window_s."""
        if self._playout.pending_seconds() > 0:
            return True
        last = self._last_out()
        return last > 0 and time.monotonic() - last < window_s

    def is_playing(self, hangover_s: float = 0.3) -> bool:
        """True while audio is queued or was audible within hangover_s."""
        if self._playout.pending_seconds() > 0:
            return True
        last = self._last_out()
        return last > 0 and time.monotonic() - last < hangover_s

    # ------------------------------------------------------------------ teardown
    def close(self):
        self._running = False
        for s in (self._in, self._out):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
        self._in = self._out = None
        self._aec = None

    @property
    def running(self) -> bool:
        return self._running
