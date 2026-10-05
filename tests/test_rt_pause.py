"""Live-partial cadence follows agent playback (GPU headroom for TTS)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge as B


class FakeCA:
    def __init__(self): self.playing = False
    def is_playing(self, hangover_s=0.3): return self.playing


class FakeT:
    def __init__(self): self.calls = []
    def _set_recorder_param(self, k, v): self.calls.append((k, v))


class RtPauseTests(unittest.TestCase):
    def make(self):
        b = B.CallBridge.__new__(B.CallBridge)
        b.call_audio = FakeCA(); b._transcriber = FakeT()
        return b

    def test_slows_while_agent_speaks_and_restores(self):
        b = self.make()
        b._update_rt_pause()
        b.call_audio.playing = True
        b._update_rt_pause(); b._update_rt_pause()
        b.call_audio.playing = False
        b._update_rt_pause()
        self.assertEqual([v for _, v in b._transcriber.calls],
                         [b.RT_PAUSE_IDLE, b.RT_PAUSE_SPEAKING, b.RT_PAUSE_IDLE])  # only on change
        self.assertTrue(all(k == "realtime_processing_pause" for k, _ in b._transcriber.calls))

    def test_no_transcriber_is_noop(self):
        b = self.make(); b._transcriber = None
        b._update_rt_pause()


if __name__ == "__main__":
    unittest.main()
