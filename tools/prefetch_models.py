"""Download every speech and voice model ATLAS needs, once, with visible progress.

Run by install.ps1 so the first launch downloads nothing. Each download is independent: a failure is reported and the
app retries that download on first launch. No accounts or tokens are needed.

    python tools/prefetch_models.py              plain progress lines
    python tools/prefetch_models.py --progress   also "MODELPROGRESS <done> <expected> <label>" lines for the setup program

With --progress a background thread measures how many bytes the Hugging Face hub cache has gained: the size of every
regular file (links are skipped) in each model folder, both blobs and snapshots. Where Windows allows symbolic links the
cache keeps each file in blobs and links to it from snapshots; where it doesn't, the finished file is moved into
snapshots. Either way each file is counted once. The shown value never drops below the expected sizes of the models
already done.
"""
from __future__ import annotations

import os
import stat
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tagger():
    """The model the post-call memory pass uses to keep people's names out of memories (memory_redact.TAGGER_FILES:
    the GLiNER weights and its backbone's tokenizer, nothing more)."""
    from huggingface_hub import snapshot_download
    sys.path.insert(0, str(ROOT))
    import memory_redact as MR
    for repo, files in MR.TAGGER_FILES:
        snapshot_download(repo, allow_patterns=list(files))


def _voice():
    """The voice engine's own model files. The model, its quantization and the file names all come from the engine's
    packages, so an engine update can't leave Setup downloading files the engine no longer loads. The quantization
    must be passed: resolve_gguf_paths defaults to BF16, which the engine doesn't use."""
    from RealtimeTTS.engines.qwen_engine import DEFAULT_MODEL, DEFAULT_QUANT
    from qwentts_cpp.models import resolve_gguf_paths
    resolve_gguf_paths(DEFAULT_MODEL, quant=DEFAULT_QUANT)


def _silero():
    """Voice activity detection: RealtimeSTT loads Silero VAD with torch.hub from GitHub when ATLAS starts
    (silero_use_onnx in transcribe.py). Fetch it into torch's hub cache now, trusted, so that start downloads nothing."""
    import torch
    # torch.hub downloads with Python's own SSL setup, which on some PCs can't verify GitHub (seen on the build PC:
    # "unable to get issuer certificate", which torch.hub reports as "no internet connection"). pip and Hugging Face
    # bring certifi's roots; give this one download the same.
    saved = os.environ.get("SSL_CERT_FILE")
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    try:
        torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", verbose=False, onnx=True, trust_repo=True)
    finally:
        if saved is None:
            os.environ.pop("SSL_CERT_FILE", None)
        else:
            os.environ["SSL_CERT_FILE"] = saved


def _punkt():
    """Sentence splitting for the voice: RealtimeTTS's default tokenizer ("nltk+rule-based", via stream2sentence)
    downloads NLTK's punkt_tab data the first time it runs."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError("NLTK punkt_tab download failed")


# (label, what it is, expected bytes, download). The sizes were measured in a full cache on 2026-11-06; they only
# scale the progress bar, so a small change upstream does no harm. The last two land outside the Hugging Face cache
# (torch's hub cache, nltk_data), so the bar counts them when they finish.
STEPS = [
    ("Speech recognition", "Whisper large-v3-turbo, ~1.6 GB", 1_621_667_008,
     lambda: __import__("faster_whisper").download_model("large-v3-turbo")),
    ("Turn detection", "~270 MB", 268_780_727,
     lambda: __import__("huggingface_hub").snapshot_download("KoljaB/SentenceFinishedClassification")),
    ("Speaker recognition", "ECAPA, ~90 MB", 89_135_280,
     lambda: __import__("huggingface_hub").snapshot_download("speechbrain/spkrec-ecapa-voxceleb")),
    ("Name tagger for memories", "GLiNER small, ~610 MB", 613_124_312, _tagger),
    ("Voice", "Qwen3-TTS, ~1.3 GB", 1_283_766_112, _voice),
    ("Voice activity detection", "Silero VAD, ~30 MB", 31_000_000, _silero),
    ("Sentence splitting", "NLTK punkt_tab, ~4 MB", 4_400_000, _punkt),
]


def hub_cache() -> Path:
    from huggingface_hub import constants
    return Path(constants.HF_HUB_CACHE)


def cache_bytes(cache: Path) -> int:
    """Bytes of every regular file (not links) under the cache's model folders."""
    total = 0
    try:
        tops = [e.path for e in os.scandir(cache) if e.name.startswith("models--") and e.is_dir(follow_symlinks=False)]
    except OSError:
        return 0
    for top in tops:
        for root, _dirs, files in os.walk(top):          # os.walk does not follow folder links
            for name in files:
                try:
                    st = os.lstat(os.path.join(root, name))
                except OSError:
                    continue                             # a file moved or removed while counting
                if stat.S_ISREG(st.st_mode):
                    total += st.st_size
    return total


class Meter:
    """Prints MODELPROGRESS lines while the downloads run: max(cache growth, finished models' sizes), capped."""

    def __init__(self, total: int, cache: Path, out=None, interval: float = 0.5):
        self.total, self.cache, self.out, self.interval = total, cache, out or sys.stdout, interval
        self.base = cache_bytes(cache)
        self.finished = 0
        self.label = ""
        self.last = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def value(self) -> int:
        grown = cache_bytes(self.cache) - self.base
        return min(self.total, max(self.finished, grown))

    def emit(self) -> None:
        with self.lock:
            line = (self.value(), self.label)
            if line != self.last:
                self.last = line
                print(f"MODELPROGRESS {line[0]} {self.total} {line[1]}", file=self.out, flush=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self.emit()

    def start(self) -> "Meter":
        self.emit()
        self.thread.start()
        return self

    def begin(self, label: str) -> None:
        self.label = label
        self.emit()

    def done(self, size: int) -> None:
        self.finished += size
        self.emit()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        self.emit()


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    meter = Meter(sum(s[2] for s in STEPS), hub_cache()).start() if "--progress" in args else None
    failed = 0
    for i, (label, about, size, fn) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label} ({about}) ...", flush=True)
        if meter:
            meter.begin(label)
        t0 = time.time()
        try:
            fn()
            print(f"      done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"      skipped ({type(e).__name__}: {e}); ATLAS will retry on first launch", flush=True)
        if meter:
            meter.done(size)                             # finished either way: the bar moves on
    if meter:
        meter.close()
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
