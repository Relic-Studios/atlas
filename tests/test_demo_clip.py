"""tools/demo/clip.py: turn pairing, hook placement and the rule pre-filter (no models)."""
import importlib.util
import shutil
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("clip", ROOT / "tools" / "demo" / "clip.py")
C = importlib.util.module_from_spec(spec)
sys.modules["clip"] = C
spec.loader.exec_module(C)


def ev():
    return [
        {"kind": "start", "t": 0.0},
        {"kind": "line_start", "line": 0, "who": "Sam", "text": "Hey Ava, co-op or competitive?", "t": 3.0},
        {"kind": "line_end", "line": 0, "t": 5.0},
        {"kind": "said_agent", "persona": "ava", "text": "Co-op. Four people, short rounds, nobody sulks.", "t": 5.4},
        {"kind": "agent_start", "t": 5.8},
        {"kind": "agent_end", "t": 9.5},
        {"kind": "line_start", "line": 1, "who": "Riley", "text": "Ha, fair.", "t": 10.0},
        {"kind": "line_end", "line": 1, "t": 10.8},
        # A long monologue setup, then a reply: the clip must open on the agent, not the speech.
        {"kind": "line_start", "line": 2, "who": "Sam", "text": "So anyway " * 12, "t": 20.0},
        {"kind": "line_end", "line": 2, "t": 27.0},
        {"kind": "said_agent", "persona": "ava", "text": "That is a long way to say you lost.", "t": 27.3},
        {"kind": "agent_start", "t": 27.6},
        {"kind": "agent_end", "t": 29.8},
    ]


def busy_rms(n=800):
    return [0.05] * n   # no silence anywhere


class Pairing(unittest.TestCase):
    def test_agent_turns_get_their_text(self):
        turns, _ = C.load_turns(ev())
        agent = [t for t in turns if t.agent]
        self.assertEqual([t.text[:6] for t in agent], ["Co-op.", "That i"])
        self.assertAlmostEqual(agent[0].start, 5.8)

    def test_text_is_not_reused(self):
        e = ev()[:6] + [{"kind": "agent_start", "t": 11.0}, {"kind": "agent_end", "t": 12.0}]
        turns, _ = C.load_turns(e)
        self.assertEqual([t.text for t in turns if t.agent][1], "")


class Planning(unittest.TestCase):
    def test_short_setup_is_the_hook(self):
        turns, to = C.load_turns(ev())
        c = C.plan(turns, to, busy_rms(), 0.05)
        first = min(c, key=lambda x: x.start)
        self.assertTrue(first.hook.startswith("Hey Ava"))
        self.assertLessEqual(first.start, 3.0)

    def test_long_setup_opens_on_agent(self):
        turns, to = C.load_turns(ev())
        late = max(C.plan(turns, to, busy_rms(), 0.05), key=lambda x: x.start)
        self.assertTrue(late.hook.startswith("That is"))
        self.assertGreater(late.start, 27.0)

    def test_dead_air_rejects(self):
        turns, to = C.load_turns(ev())
        rms = busy_rms()
        for i in range(int(6.5 / 0.05), int(9.0 / 0.05)):   # 2.5 s of silence mid-reply
            rms[i] = 0.0
        first = min(C.plan(turns, to, rms, 0.05), key=lambda x: x.start)
        self.assertFalse(first.keep)
        self.assertTrue(any("dead air" in r for r in first.reasons))

    def test_reply_timeout_rejects(self):
        e = ev() + [{"kind": "reply_timeout", "line": 0, "t": 8.0}]
        turns, to = C.load_turns(e)
        first = min(C.plan(turns, to, busy_rms(), 0.05), key=lambda x: x.start)
        self.assertFalse(first.keep)

    def test_too_short_rejects(self):
        e = [{"kind": "said_agent", "text": "Yes, that one works fine.", "t": 1.0},
             {"kind": "agent_start", "t": 1.2}, {"kind": "agent_end", "t": 2.0}]
        turns, to = C.load_turns(e)
        (c,) = C.plan(turns, to, busy_rms(), 0.05)
        self.assertFalse(c.keep)

    def test_captions_escape_braces(self):
        turns, to = C.load_turns(ev())
        c = C.plan(turns, to, busy_rms(), 0.05)[0]
        c.turns[0]["text"] = "a {weird} line"
        s = C.ass_captions(c)
        self.assertIn("a (weird) line", s)
        self.assertIn("Dialogue:", s)


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class Render(unittest.TestCase):
    def test_renders_vertical_clip(self):
        d = Path(tempfile.mkdtemp())
        src = d / "in.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1600x900:rate=30",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "14", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
        turns, to = C.load_turns(ev())
        c = min(C.plan(turns, to, busy_rms(), 0.05), key=lambda x: x.start)
        out = C.render(c, src, d, 1)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", str(out)],
                               capture_output=True, text=True).stdout.strip()
        self.assertEqual(probe, "1080,1920")
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class LateAgentText(__import__("unittest").TestCase):
    """Reel 10-05: said_agent logged 1.5s after agent_start; the caption went to the next reply."""
    def test_text_logged_during_reply(self):
        ev = [dict(kind="agent_start", t=28.76), dict(kind="said_agent", t=30.3, text="Co-op."),
              dict(kind="agent_end", t=33.0), dict(kind="agent_start", t=38.86),
              dict(kind="said_agent", t=40.96, text="Ticket to Ride."), dict(kind="agent_end", t=44.34)]
        turns, _ = C.load_turns(ev)
        self.assertEqual([t.text for t in turns], ["Co-op.", "Ticket to Ride."])
