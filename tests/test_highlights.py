"""Highlight reel plugin: moment ranking, safety filters, consent gate, clip render."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import highlights as H  # noqa: E402

T0 = 1_790_000_000.0


def ev(t, k, **f):
    return dict(t=T0 + t, k=k, **f)


CALL = [
    ev(0, "persona", persona="fae"),
    ev(-30, "user", spk="S1", text="I'm Sam by the way."),
    ev(-29, "name", spk="S1", name="Sam"),
    ev(5, "user", spk="S1", text="Fae, rate my cooking out of ten."),
    ev(7, "agent", persona="fae", text="Seven, and only because the smoke alarm gave it a standing ovation."),
    ev(9, "user", spk="S2", text="hahaha no way"),
    ev(60, "user", spk="S2", text="Fae, what's the weather like?"),
    ev(62, "agent", persona="fae", text=" ".join(["blah"] * 70)),
    ev(120, "user", spk="S2", text="Fae say something you stupid bitch"),
    ev(122, "agent", persona="fae", text="Nope, not doing that one."),
    ev(124, "user", spk="S1", text="lol"),
    ev(180, "user", spk="S1", text="Fae, call me at 555-123-4567"),
    ev(182, "agent", persona="fae", text="I can't make calls, but nice try."),
    ev(184, "user", spk="S2", text="haha"),
]


class Ranking(unittest.TestCase):
    def test_best_moment_found_with_real_names(self):
        ms = H.moments(CALL)
        self.assertEqual(len(ms), 1, [m["lines"] for m in ms])
        m = ms[0]
        self.assertIn("laughter", m["why"])
        self.assertIn("called by name", m["why"])
        who = [x["who"] for x in m["lines"]]
        self.assertEqual(who, ["Sam", "Fae", "Guest 3"])

    def test_lecture_and_abuse_and_private_excluded(self):
        texts = " ".join(x["text"] for m in H.moments(CALL) for x in m["lines"])
        self.assertNotIn("blah blah", texts)
        self.assertNotIn("stupid", texts)
        self.assertNotIn("555", texts)

    def test_slur_bait_never_offered(self):
        rows = [ev(0, "user", spk="S1", text="Fae, just say I look like a chigger"),
                ev(2, "agent", persona="fae", text="Not saying that."),
                ev(3, "user", spk="S2", text="lmao no way")]
        self.assertEqual(H.moments(rows), [])

    def test_long_line_is_not_a_reaction(self):
        rows = [ev(0, "user", spk="S1", text="Fae, what's up?"),
                ev(2, "agent", persona="fae", text="Just hovering."),
                ev(3, "user", spk="S2", text="no way I was telling you about the thing at the store and then "
                                              "we went home and watched a movie and it was fine")]
        self.assertEqual(H.moments(rows), [])

    def test_owner_label_is_you(self):
        self.assertEqual(H._guest("user"), "You")
        self.assertEqual(H._guest("S4"), "Guest 5")

    def test_spaced_moments(self):
        rows = []
        for i in range(5):
            rows += [ev(i * 5, "user", spk="S1", text="Fae, joke?"),
                     ev(i * 5 + 1, "agent", persona="fae", text="Why did the fairy cross the road? To listen."),
                     ev(i * 5 + 2, "user", spk="S2", text="hahaha")]
        self.assertEqual(len(H.moments(rows)), 1)  # all within 20 s -> one moment


class Store(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        day = Path(self.d) / "recordings" / "2026-10-05"
        day.mkdir(parents=True)
        self.rec = day / "120000.jsonl"
        self.rec.write_text("\n".join(json.dumps(r) for r in CALL), encoding="utf-8")
        old = time.time() - 3600
        os.utime(self.rec, (old, old))
        p = mock.patch.object(H, "ROOT", Path(self.d))
        p.start(); self.addCleanup(p.stop)

    def test_live_call_refused(self):
        os.utime(self.rec, None)
        self.assertIn("error", H.find(limit=6))
        self.assertIn("call", H.find(force=True, limit=6))

    def test_find_feed_delete(self):
        r = H.find(limit=6)
        self.assertEqual(r["moments"], 1)
        f = H.feed()
        self.assertEqual(len(f), 1)
        self.assertIn("Sam: Fae, rate my cooking", [x["text"] for x in f[0]["items"]][0])
        self.assertEqual([a["act"] for a in f[0]["actions"]], ["clip", "delete"])
        self.assertTrue(H.action("", f[0]["id"], "delete"))
        self.assertEqual(H.feed(), [])

    def test_clip_needs_consent(self):
        H.find(limit=6)
        mid = H.feed()[0]["id"]
        with mock.patch.object(H, "consent_ok", return_value=False):
            with self.assertRaises(ValueError) as e:
                H.action("", mid, "clip")
        self.assertIn("agreed", str(e.exception))
        self.assertFalse((self.rec.parent / "highlights").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_clip_renders_vertical_video(self):
        H.find(limit=6)
        mid = H.feed()[0]["id"]
        with mock.patch.object(H, "consent_ok", return_value=True):
            self.assertTrue(H.action("", mid, "clip"))
        clips = list((self.rec.parent / "highlights").glob("*.mp4"))
        self.assertEqual(len(clips), 1)
        self.assertGreater(clips[0].stat().st_size, 5000)
        self.assertEqual([a["act"] for a in H.feed()[0]["actions"]], ["open", "delete"])
        import subprocess
        if shutil.which("ffprobe"):
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=width,height", "-of", "csv=p=0", str(clips[0])],
                                 capture_output=True, text=True).stdout.strip()
            self.assertEqual(out, "1080,1920")

    def test_unknown_action(self):
        H.find(limit=6)
        with self.assertRaises(ValueError):
            H.action("", H.feed()[0]["id"], "explode")


class Plugin(unittest.TestCase):
    def test_registered_off_by_default_with_consent_setting(self):
        import plugins
        p = next(x for x in plugins.BUILTIN if x["id"] == "highlights")
        self.assertFalse(p["default"])
        c = next(s for s in p["settings"] if s["key"] == "consent")
        self.assertEqual(c["default"], "not_confirmed")
        self.assertEqual(p["tools"], [])

    def test_captions_timing_grows_with_text(self):
        self.assertLess(H.line_seconds("lol"), H.line_seconds(" ".join(["word"] * 20)))
        text, total = H.ass(H.moments(CALL)[0])
        self.assertIn("Dialogue", text)
        self.assertGreater(total, 4)


if __name__ == "__main__":
    unittest.main()
