"""Bridge the Discord-call audio path into the RealtimeVoiceChat agent loop.

The input device carries the call audio; the output device is where the agent
speaks into the call. The bridge:

* captures the input, runs Echoff AEC (reference = the agent's own output) so
  the agent never transcribes its own voice, resamples 48k->16k, and feeds the
  RealtimeSTT recorder directly (no browser involved),
* routes synthesized TTS audio to the output device and mirrors it as the AEC reference,
* tags every final utterance with a diarized speaker so the agent can tell the
  participants apart and ignore its own voice.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

INPUT_RATE = 48000
STT_RATE = 16000


class CallBridge:
    def __init__(self, *, enable_aec: bool = True, self_reference_audio: str | None = None,
                 input_device=None, output_device=None):
        from demo_call import call_audio_class   # virtual device in demo/sim calls
        CallAudio = call_audio_class()  # noqa: N806
        self.enable_aec = enable_aec
        self.input_device = input_device
        self.output_device = output_device
        self.call_audio = CallAudio(enable_aec=enable_aec, input_device=input_device, output_device=output_device)

        self.diarizer = None
        # Diarize even with no voice installed yet (fresh public build): speaker
        # labels still work; the self-profile is set once a voice exists.
        from diarize import Diarizer
        self.diarizer = Diarizer(self_reference_audio=self_reference_audio or None, device="cuda")

        self._transcriber = None
        self._stop = threading.Event()

    # ---- lifecycle ------------------------------------------------------
    def start(self, transcriber):
        """Open devices and begin feeding AEC-clean B2 audio to the transcriber."""
        self._transcriber = transcriber
        self.call_audio.open()
        self.call_audio.on_clean_input = self._on_clean_input
        self._stop.clear()
        log.info("[bridge] B2 -> AEC -> STT feeding started")

    # Live-partial cadence. Every partial is a full Whisper pass (~90 ms on the
    # 4090); at a 30 ms pause Whisper holds the GPU ~75% of the time, which starves
    # TTS synthesis while the agent talks (-> late chunks -> stutter). While agent
    # audio is audible we slow partials down; barge-in still works, ~120 ms later.
    RT_PAUSE_IDLE = float(os.environ.get("ATLAS_RT_PAUSE", "0.03"))
    RT_PAUSE_SPEAKING = float(os.environ.get("ATLAS_RT_PAUSE_SPEAKING", "0.15"))

    def realtime_pause_for(self, agent_playing: bool) -> float:
        return self.RT_PAUSE_SPEAKING if agent_playing else self.RT_PAUSE_IDLE

    def _update_rt_pause(self) -> None:
        t = self._transcriber
        if t is None:
            return
        try:
            want = self.realtime_pause_for(self.call_audio.is_playing(hangover_s=0.3))
        except Exception:  # noqa: BLE001
            return
        if want != getattr(self, "_rt_pause", None):
            setter = getattr(t, "_set_recorder_param", None)
            if setter is not None:
                try:
                    setter("realtime_processing_pause", want)
                    self._rt_pause = want
                except Exception:  # noqa: BLE001
                    pass

    def _on_clean_input(self, clean_float32: np.ndarray):
        """clean_float32 is 48 kHz mono. Resample to 16 kHz int16 for RealtimeSTT."""
        self._update_rt_pause()
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(INPUT_RATE, STT_RATE)
        audio16 = resample_poly(clean_float32, STT_RATE // g, INPUT_RATE // g)
        pcm = (np.clip(audio16, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        try:
            from call_recorder import REC
            REC.audio(audio16.astype(np.float32))
        except Exception:  # noqa: BLE001
            pass
        if self._transcriber is not None:
            try:
                self._transcriber.feed_audio(pcm)
            except Exception as e:  # noqa: BLE001
                log.warning("[bridge] feed_audio error: %s", e)

    # ---- TTS out --------------------------------------------------------
    def play_tts(self, int16_24k_bytes: bytes):
        """Route a synthesized TTS chunk (24 kHz int16, QwenEngine output) to VAIO3."""
        audio = np.frombuffer(int16_24k_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self.call_audio.play(audio, sample_rate=24000)   # non-blocking enqueue

    def is_speaking(self) -> bool:
        return self.call_audio.is_playing()

    def pending_seconds(self) -> float:
        fn = getattr(self.call_audio, "pending_seconds", None)
        return float(fn()) if fn is not None else 0.0

    def reply_progress(self) -> tuple[float, float]:
        fn = getattr(self.call_audio, "reply_progress", None)
        return fn() if fn is not None else (0.0, 0.0)

    def agent_audible_during(self, utterance_s: float) -> bool | None:
        """Could the agent's own voice be inside an utterance that just ended?
        Window = utterance + end-of-turn silence + echo tail. None if unknown."""
        fn = getattr(self.call_audio, "audible_within", None)
        if fn is None:
            return None
        try:
            return bool(fn(float(utterance_s) + 1.5))
        except Exception:  # noqa: BLE001
            return None

    def flush_tts(self) -> float:
        """Stop the agent mid-sentence (barge-in, soft fade). Returns seconds cut."""
        return self.call_audio.flush()

    def end_reply(self) -> None:
        fn = getattr(self.call_audio, "end_reply", None)
        if fn is not None:
            fn()

    def playout_stats(self) -> dict:
        fn = getattr(self.call_audio, "playout_stats", None)
        return fn() if fn is not None else {}

    def identify_speaker(self, audio_float: np.ndarray, sample_rate: int = 16000):
        """Non-mutating speaker guess for the utterance in progress."""
        if self.diarizer is None:
            return None
        try:
            aud = self.agent_audible_during(len(audio_float) / float(sample_rate))
            return self.diarizer.identify(audio_float, sample_rate, agent_audible=aud)
        except Exception as e:  # noqa: BLE001
            log.warning("[bridge] identify error: %s", e)
            return None

    # ---- diarization ----------------------------------------------------
    def tag_speaker(self, audio_float: np.ndarray, sample_rate: int = 16000):
        if self.diarizer is None:
            return None
        try:
            aud = self.agent_audible_during(len(audio_float) / float(sample_rate))
            return self.diarizer.process(audio_float, sample_rate, agent_audible=aud).speaker
        except Exception as e:  # noqa: BLE001
            log.warning("[bridge] diarize error: %s", e)
            return None

    def call_summary(self) -> str:
        return self.diarizer.summary() if self.diarizer else ""

    def reconfigure(self, *, input_device=None, output_device=None) -> None:
        """Close and reopen the audio path with new in/out devices (hot-swap)."""
        from audio_io import CallAudio
        import demo_call
        if demo_call.enabled():
            log.info("[bridge] demo call active: device change ignored")
            return
        self.input_device = input_device
        self.output_device = output_device
        self.call_audio.close()
        self.call_audio = CallAudio(enable_aec=self.enable_aec,
                                    input_device=input_device, output_device=output_device)
        self.call_audio.on_clean_input = self._on_clean_input
        self.call_audio.open()
        log.info("[bridge] reconfigured devices: in=%r out=%r", input_device, output_device)

    def update_self_reference(self, audio_path: str) -> None:
        """Point the diarizer's self-profile at the active voice so the agent
        recognizes its own TTS output as 'self' (echo filter)."""
        if self.diarizer is None:
            from diarize import Diarizer
            self.diarizer = Diarizer(self_reference_audio=audio_path, device="cuda")
        else:
            self.diarizer.set_self_reference(audio_path)

    # ---- teardown -------------------------------------------------------
    def stop(self):
        self._stop.set()
        self.call_audio.close()
        log.info("[bridge] stopped")
