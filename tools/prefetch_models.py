"""Download every speech model ATLAS needs, once, with visible progress.

Run by install.ps1 so the first launch doesn't sit on a splash screen for ten
minutes. Each step is independent: a failure is reported and the app will
retry that download on first launch. No accounts or tokens are needed.
"""
import sys
import time

STEPS = [
    ("Speech recognition (Whisper large-v3-turbo, ~1.6 GB)",
     lambda: __import__("faster_whisper").download_model("large-v3-turbo")),
    ("Turn detection (~260 MB)",
     lambda: __import__("huggingface_hub").snapshot_download("KoljaB/SentenceFinishedClassification")),
    ("Speaker recognition (ECAPA, ~90 MB)",
     lambda: __import__("huggingface_hub").snapshot_download("speechbrain/spkrec-ecapa-voxceleb")),
    ("Memory embeddings (BGE-M3, ~2.3 GB)",
     lambda: __import__("huggingface_hub").snapshot_download(
         "BAAI/bge-m3", allow_patterns=["*.json", "*.txt", "sentencepiece*", "tokenizer*",
                                        "pytorch_model.bin", "1_Pooling/*", "modules.json"])),
]


def main() -> int:
    failed = 0
    for i, (label, fn) in enumerate(STEPS, 1):
        print(f"[{i}/{len(STEPS)}] {label} ...", flush=True)
        t0 = time.time()
        try:
            fn()
            print(f"      done in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"      skipped ({type(e).__name__}: {e}); ATLAS will retry on first launch", flush=True)
    print("Voice synthesis (Qwen3-TTS) downloads on first launch.", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
