"""Optional final-pass transcription with NVIDIA Parakeet TDT 0.6B v3 (ONNX, onnx-asr).

Whisper large-v3-turbo keeps producing the live partials; only the authoritative final
transcript of each turn comes from Parakeet. Off unless ATLAS_FINAL_ASR=parakeet or
private/final_asr.txt says "parakeet". Any Parakeet failure falls back to Whisper for
that turn, so a broken model never costs a turn.

Model files: ATLAS_PARAKEET_DIR, private/parakeet_dir.txt, or the Hugging Face cache
(istupakov/parakeet-tdt-0.6b-v3-onnx).
"""
from __future__ import annotations

import copy
import base64
import glob
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
_REPO = "models--istupakov--parakeet-tdt-0.6b-v3-onnx"


def enabled() -> bool:
    if os.environ.get("ATLAS_FINAL_ASR"):
        return os.environ["ATLAS_FINAL_ASR"].strip().lower() == "parakeet"
    try:
        return (ROOT / "private" / "final_asr.txt").read_text(encoding="utf-8").strip().lower() == "parakeet"
    except OSError:
        return False


def model_dir() -> str | None:
    cands = [os.environ.get("ATLAS_PARAKEET_DIR", "")]
    try:
        cands.append((ROOT / "private" / "parakeet_dir.txt").read_text(encoding="utf-8").strip())
    except OSError:
        pass
    for hub in (Path.home() / ".cache" / "huggingface" / "hub",
                Path(os.environ.get("LOCALAPPDATA", "")) / "atlas_asr_bench" / "models" / "hub"):
        cands += sorted(glob.glob(str(hub / _REPO / "snapshots" / "*")), reverse=True)
    for c in cands:
        if c and (Path(c) / "encoder-model.onnx").exists():
            return c
    return None


class ParakeetFinal:
    """Loads once; recognize() is called from the recorder's transcription thread."""

    def __init__(self, path: str):
        import torch  # noqa: F401 - torch's CUDA/cuDNN DLLs must be loaded before onnxruntime's CUDA EP
        import onnxruntime
        import onnx_asr
        onnxruntime.set_default_logger_severity(3)   # memcpy-node notices are noise, errors still log
        t0 = time.time()
        self._lock = threading.Lock()
        self.model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", path,
                                         providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.recognize(np.zeros(16000, dtype=np.float32))     # warm the CUDA graphs
        logger.info("👂🦜 Parakeet TDT 0.6B v3 final pass ready in %.1fs (%s)", time.time() - t0, path)

    def recognize(self, audio: np.ndarray) -> str:
        with self._lock:
            return (self.model.recognize(np.asarray(audio, dtype=np.float32), sample_rate=16000) or "").strip()


def wrap(recorder, background: bool = True) -> bool:
    """Route recorder.perform_final_transcription through Parakeet. Returns True when installed
    (with background=True the model may still be loading; Whisper covers finals until then)."""
    if recorder is None or getattr(recorder, "_atlas_parakeet_wrapped", False) or not enabled():
        return False
    path = model_dir()
    if not path:
        logger.warning("👂🦜 final ASR is set to Parakeet but no model files were found; staying on Whisper")
        return False
    # A cold load reads 2.4 GB (~45 s): load in the background, Whisper does finals until ready.
    holder: dict = {}

    def _load():
        try:
            holder["engine"] = ParakeetFinal(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("👂🦜 Parakeet failed to load (%s); staying on Whisper", e)
    loader = threading.Thread(target=_load, name="ParakeetLoad", daemon=True)
    loader.start()
    if not background:
        loader.join()
    whisper_final = recorder.perform_final_transcription

    def final(audio_bytes=None, use_prompt=True):
        engine = holder.get("engine")
        if engine is None:
            return whisper_final(audio_bytes, use_prompt)
        audio = audio_bytes if audio_bytes is not None else copy.deepcopy(getattr(recorder, "audio", None))
        if audio is None or len(audio) == 0:
            return ""
        try:
            with recorder.transcription_lock:
                t0 = time.time()
                text = engine.recognize(audio)
                recorder.allowed_to_early_transcribe = True
                recorder._set_state("inactive")
                recorder.detected_language, recorder.detected_language_probability = None, 0.0
                recorder.last_transcription_bytes = copy.deepcopy(audio)
                recorder.last_transcription_bytes_b64 = base64.b64encode(
                    np.asarray(audio, dtype=np.float32).tobytes()).decode("utf-8")
                out = recorder._preprocess_output(text) if text else ""
                logger.info("👂🦜 Parakeet final in %d ms: %r", int((time.time() - t0) * 1000), out[:80])
                return "" if recorder.interrupt_stop_event.is_set() else out
        except Exception as e:  # noqa: BLE001
            logger.warning("👂🦜 Parakeet final failed (%s); using Whisper for this turn", e)
            return whisper_final(audio_bytes, use_prompt)

    recorder.perform_final_transcription = final
    recorder._atlas_parakeet_wrapped = True
    return True
