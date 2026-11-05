"""The pass runner (design docs/designs/experience_memory.md, D3; acceptance A3 runner cases and the real-clock kill
time)."""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory  # noqa: E402
import call_boundary as CB  # noqa: E402
import experience_memory as EM  # noqa: E402
import memory_writer as MW  # noqa: E402
import importlib.util  # noqa: E402
SPEECH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "huggingface_hub"))
NEEDS_SPEECH = unittest.skipUnless(SPEECH, "imports the server, which needs the speech stack "
                                   "(not in the light CI deps)")


def sleeper(args, **kw):
    """Stands in for the pass: a real process that would run for 30 s."""
    kw.pop("creationflags", None)
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], **kw)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        self.clock_t = 10_000.0
        p = CB.pending_dir() / "20261104-100000-abcd.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"k": "start", "t": time.time()}) + "\n" + json.dumps({"k": "end", "t": time.time()})
                     + "\n", encoding="utf-8")
        self.tagger = mock.patch("memory_redact.tagger_files_present", return_value=True)
        self.tagger.start()
        self.done = []
        self.r = MW.Runner(args_fn=lambda: ["--mode", "remote"], on_done=self.done.append, popen=sleeper,
                           clock=lambda: self.clock_t)

    def tearDown(self):
        if self.r.proc is not None and self.r.proc.poll() is None:
            self.r.proc.kill()
            self.r.proc.wait(5)
        self.tagger.stop()
        agent_memory.ROOT = self._root
        self.tmp.cleanup()


class StartConditions(Base):
    def test_waits_for_quiet_and_no_call(self):
        self.assertIsNone(self.r.tick(in_progress=True, last_activity=0.0))
        self.assertIsNone(self.r.tick(in_progress=False, last_activity=self.clock_t - 60))
        self.assertEqual(self.r.tick(in_progress=False, last_activity=self.clock_t - 301), "started")
        self.assertEqual(self.r.tick(in_progress=False, last_activity=0.0), "running")

    def test_paused_cloud_tagger_and_nothing_to_do(self):
        EM.set_setting("paused", True)
        self.assertIsNone(self.r.tick(False, 0.0))
        self.assertEqual(self.r.last_reason, "paused")
        EM.set_setting("paused", False)
        self.r.cloud_fn = lambda: True
        self.r.tick(False, 0.0)
        self.assertEqual(self.r.last_reason, "cloud_not_allowed")
        self.r.cloud_fn = lambda: False
        with mock.patch("memory_redact.tagger_files_present", return_value=False):
            self.r.tick(False, 0.0)
        self.assertEqual(self.r.last_reason, "tagger_missing")
        for p in CB.pending_dir().glob("*.jsonl"):
            p.with_suffix(".fails").write_text("3", encoding="utf-8")         # failed calls don't count
        self.r.tick(False, 0.0)
        self.assertEqual(self.r.last_reason, "nothing")

    def test_exit_reloads(self):
        self.r._popen = lambda args, **kw: subprocess.Popen([sys.executable, "-c", "pass"],
                                                            **{k: v for k, v in kw.items() if k != "creationflags"})
        self.r.tick(False, 0.0)
        self.r.proc.wait(10)
        self.assertEqual(self.r.tick(False, 0.0), "exited 0")
        self.assertEqual(self.done, [0])


class BackOff(Base):
    def exit_with(self, code):
        self.r._popen = lambda args, **kw: subprocess.Popen(
            [sys.executable, "-c", f"import sys; sys.exit({code})"],
            **{k: v for k, v in kw.items() if k != "creationflags"})

    def run_once(self):
        self.assertEqual(self.r.tick(False, 0.0), "started")
        self.r.proc.wait(10)
        self.r.tick(False, 0.0)                                  # reap

    def test_failed_runs_wait_longer_each_time(self):
        self.exit_with(2)
        self.run_once()
        self.assertIsNone(self.r.tick(False, 0.0))
        self.assertEqual(self.r.last_reason, "backoff")
        self.clock_t += MW.BACKOFF_S[0] + 1
        self.run_once()
        self.clock_t += MW.BACKOFF_S[0] + 1                      # not long enough after a second failure
        self.assertIsNone(self.r.tick(False, 0.0))
        self.clock_t += MW.BACKOFF_S[1]
        self.exit_with(0)
        self.run_once()
        self.assertEqual((self.r.fails, self.r.not_before), (0, 0.0))

    def test_a_missing_tagger_waits_an_hour_and_process_now_skips_the_wait(self):
        self.exit_with(3)
        self.run_once()
        self.clock_t += 600
        self.assertIsNone(self.r.tick(False, 0.0))
        self.assertEqual(self.r.start_now(), "started")

    def test_every_exit_reloads(self):
        self.exit_with(2)
        self.run_once()
        self.assertEqual(self.done, [2])


class Script(unittest.TestCase):
    def test_the_pass_runs_as_a_script(self):
        """The runner starts `python memory_writer.py` (not an import): with no tagger files it must log and exit 3,
        not crash (a NameError here once meant the pass never ran)."""
        with tempfile.TemporaryDirectory() as d:
            env = dict(__import__("os").environ, ATLAS_MEMORY_DIR=str(Path(d) / "memory"), HF_HOME=str(Path(d) / "hf"))
            root = Path(__file__).resolve().parents[1]
            r = subprocess.run([sys.executable, str(root / "memory_writer.py"), "--mode", "remote", "--call", "none"],
                               cwd=str(root), env=env, capture_output=True, timeout=180)
            self.assertEqual(r.returncode, 3, r.stderr.decode("utf-8", "replace")[-800:])
            self.assertIn("tagger_missing", (Path(d) / "memory" / "_pass_log.txt").read_text(encoding="utf-8"))


class Kill(Base):
    def test_speech_activity_kills_within_200ms(self):
        worst = 0.0
        for i in range(20):
            self.assertEqual(self.r.start_now(), "started")
            p = self.r.proc
            time.sleep(0.05)
            t0 = time.perf_counter()
            self.r.on_activity("partial")
            p.wait(5)
            worst = max(worst, time.perf_counter() - t0)
            self.r.tick(False, 0.0)                         # reap
        self.assertLess(worst, 0.2, f"slowest kill {worst * 1000:.0f} ms")
        self.assertEqual(self.r.kills, 20)
        self.assertNotIn(0, self.done)                       # a kill is not a finished run


@NEEDS_SPEECH
class QuitDecision(unittest.TestCase):
    def test_never_at_quit_with_a_local_model(self):
        import server
        app = SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=SimpleNamespace(
            llm=SimpleNamespace(remote=None, model="m"))))
        with mock.patch("memory_writer.may_run", return_value=None):
            self.assertFalse(server.start_pass_at_quit(app))
        app.state.SpeechPipelineManager.llm.remote = SimpleNamespace(cfg={"cloud": False})
        with mock.patch("memory_writer.may_run", return_value=None):
            self.assertTrue(server.start_pass_at_quit(app))
        with mock.patch("memory_writer.may_run", return_value="paused"):
            self.assertFalse(server.start_pass_at_quit(app))

    def test_the_pass_talks_to_the_live_model(self):
        """D3: the server names the live model (URL and name, never the key); a run whose configuration no longer
        matches refuses instead of sending calls somewhere else."""
        import server
        app = SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=SimpleNamespace(llm=SimpleNamespace(
            remote=SimpleNamespace(cfg={"url": "http://box:8000", "model": "m", "key": "k"})))))
        args = server._llm_mode(app)["args"]
        self.assertEqual(args, ["--mode", "remote", "--url", "http://box:8000", "--model", "m"])
        self.assertNotIn("k", args)
        with mock.patch("remote_llm.load_config", return_value={"url": "http://box:8000/", "model": "m", "key": "k"}):
            self.assertEqual(MW.Model("remote", url="http://box:8000", model="m").cfg["key"], "k")
        with mock.patch("remote_llm.load_config", return_value={"url": "http://other:9000", "model": "m"}):
            with self.assertRaises(MW.ModelError):
                MW.Model("remote", url="http://box:8000", model="m")


if __name__ == "__main__":
    unittest.main()
