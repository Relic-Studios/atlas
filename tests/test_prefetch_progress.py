"""tools/prefetch_models.py: the cache watcher behind the setup program's model progress, and the voice-model download.

The engine packages are replaced by stubs, so this runs where they are not installed (the public CI)."""
import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prefetch_models as PM  # noqa: E402


def write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


class CacheBytes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_moved_into_snapshots_counts_once(self):
        # Windows without symlink rights: Hugging Face moves the finished blob into snapshots
        write(self.cache / "models--a--b" / "snapshots" / "abc" / "model.bin", 1000)
        write(self.cache / "models--a--b" / "refs" / "main", 40)
        self.assertEqual(PM.cache_bytes(self.cache), 1040)

    def test_blob_with_snapshot_link_counts_once(self):
        blob = write(self.cache / "models--a--b" / "blobs" / "deadbeef", 2000)
        link = self.cache / "models--a--b" / "snapshots" / "abc" / "model.bin"
        link.parent.mkdir(parents=True)
        try:
            os.symlink(blob, link)
        except (OSError, NotImplementedError):
            self.skipTest("this PC can't create symbolic links")
        self.assertEqual(PM.cache_bytes(self.cache), 2000)

    def test_download_in_progress_counts(self):
        write(self.cache / "models--a--b" / "blobs" / "deadbeef.0123.incomplete", 500)
        self.assertEqual(PM.cache_bytes(self.cache), 500)

    def test_other_folders_and_missing_cache(self):
        write(self.cache / "xet" / "chunk-cache" / "x", 999)          # not a model folder
        self.assertEqual(PM.cache_bytes(self.cache), 0)
        self.assertEqual(PM.cache_bytes(self.cache / "nope"), 0)


class MeterLines(unittest.TestCase):
    def test_never_below_finished_models_and_capped(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            write(cache / "models--old--one" / "blobs" / "x", 300)       # already there: not growth
            out = io.StringIO()
            m = PM.Meter(1000, cache, out=out, interval=60)
            m.emit()
            m.begin("First")
            write(cache / "models--new--two" / "blobs" / "y.incomplete", 250)
            m.emit()
            m.done(400)                                                  # finished, though only 250 bytes grew
            m.begin("Second")
            write(cache / "models--new--three" / "blobs" / "z", 900)    # growth 1150 > total
            m.emit()
            values = [int(l.split()[1]) for l in out.getvalue().splitlines()]
            self.assertEqual(values[0], 0)
            self.assertIn(250, values)
            self.assertIn(400, values)
            self.assertEqual(values[-1], 1000)
            self.assertEqual(values, sorted(values))
            self.assertTrue(all(l.split()[2] == "1000" for l in out.getvalue().splitlines()))
            self.assertTrue(out.getvalue().splitlines()[-1].endswith(" Second"))


class VoiceModel(unittest.TestCase):
    def test_uses_the_engine_names_including_quant(self):
        calls = []
        qe = types.ModuleType("RealtimeTTS.engines.qwen_engine")
        qe.DEFAULT_MODEL, qe.DEFAULT_QUANT = "Org/Some-Model", "Q8_0"
        models = types.ModuleType("qwentts_cpp.models")
        models.resolve_gguf_paths = lambda *a, **k: calls.append((a, k))
        stubs = {"RealtimeTTS": types.ModuleType("RealtimeTTS"), "RealtimeTTS.engines": types.ModuleType("RealtimeTTS.engines"),
                 "RealtimeTTS.engines.qwen_engine": qe, "qwentts_cpp": types.ModuleType("qwentts_cpp"),
                 "qwentts_cpp.models": models}
        saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        try:
            PM._voice()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        self.assertEqual(calls, [(("Org/Some-Model",), {"quant": "Q8_0"})])

    def with_stubs(self, stubs, fn):
        saved = {k: sys.modules.get(k) for k in stubs}
        sys.modules.update(stubs)
        try:
            fn()
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_silero_vad_is_fetched_the_way_the_recorder_loads_it(self):
        calls = []
        torch = types.ModuleType("torch")
        torch.hub = types.SimpleNamespace(load=lambda **k: calls.append(k))
        self.with_stubs({"torch": torch}, PM._silero)
        self.assertEqual(calls, [{"repo_or_dir": "snakers4/silero-vad", "model": "silero_vad", "verbose": False,
                                  "onnx": True, "trust_repo": True}])

    def test_silero_uses_certificate_roots_and_restores_environment_on_failure(self):
        torch = types.ModuleType("torch")
        certifi = types.ModuleType("certifi")
        certifi.where = lambda: "test-roots.pem"
        seen = []

        def load(**kwargs):
            seen.append(os.environ.get("SSL_CERT_FILE"))
            raise RuntimeError("download failed")

        torch.hub = types.SimpleNamespace(load=load)
        with patch.dict(os.environ):
            for original in (None, "owner-roots.pem"):
                if original is None:
                    os.environ.pop("SSL_CERT_FILE", None)
                else:
                    os.environ["SSL_CERT_FILE"] = original
                with self.assertRaisesRegex(RuntimeError, "download failed"):
                    self.with_stubs({"torch": torch, "certifi": certifi}, PM._silero)
                self.assertEqual(os.environ.get("SSL_CERT_FILE"), original)
        self.assertEqual(seen, ["test-roots.pem", "owner-roots.pem"])

    def test_punkt_tab_is_downloaded_only_when_missing(self):
        downloads = []

        def make(found):
            nltk = types.ModuleType("nltk")

            def find(name):
                if not found:
                    raise LookupError(name)
            nltk.data = types.SimpleNamespace(find=find)
            nltk.download = lambda name, quiet=False: downloads.append(name) or True
            return nltk
        self.with_stubs({"nltk": make(True)}, PM._punkt)
        self.assertEqual(downloads, [])
        self.with_stubs({"nltk": make(False)}, PM._punkt)
        self.assertEqual(downloads, ["punkt_tab"])

    def test_no_model_file_names_of_its_own(self):
        src = (ROOT / "tools" / "prefetch_models.py").read_text(encoding="utf-8")
        self.assertNotIn(".gguf", src)
        self.assertNotIn("Qwen3-TTS-GGUF", src)


class MainWithProgress(unittest.TestCase):
    def test_lines_rise_to_the_total(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d)
            steps = [("One", "test", 100, lambda: write(cache / "models--a--one" / "blobs" / "a", 100)),
                     ("Two", "test", 300, lambda: (_ for _ in ()).throw(OSError("offline"))),
                     ("Three", "test", 600, lambda: write(cache / "models--a--three" / "blobs" / "c", 600))]
            saved = PM.STEPS, PM.hub_cache, sys.stdout
            PM.STEPS, PM.hub_cache = steps, (lambda: cache)
            sys.stdout = out = io.StringIO()
            try:
                rc = PM.main(["--progress"])
            finally:
                PM.STEPS, PM.hub_cache, sys.stdout = saved
            lines = [l for l in out.getvalue().splitlines() if l.startswith("MODELPROGRESS ")]
            values = [int(l.split()[1]) for l in lines]
            self.assertEqual(rc, 2)                                       # one model failed: Setup warns
            self.assertEqual(values, sorted(values))
            self.assertEqual(values[-1], 1000)
            self.assertIn("skipped (OSError: offline)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
