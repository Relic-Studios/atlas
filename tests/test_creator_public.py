"""Public creator pieces: heart.md round trip, voice import, richer template."""
import io, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agents as A
import heart_md as H
import voice_import as VI

SAMPLE = """# Gordon
## Identity
angry celebrity chef
## Personality
You roast everyone's food but you're soft on beginners.
## Talk
Short, loud, British.
## Own stuff
You love beef wellington and think pineapple pizza is a crime.
## Interests
- cooking
- football, restaurants
## Role
loud · picky · chef
## Talkativeness
55
## Secret
You cry at Pixar films.
"""


class HeartTests(unittest.TestCase):
    def test_parse_sections_aliases_unknown(self):
        h = H.parse(SAMPLE)
        self.assertEqual(h["name"], "Gordon")
        self.assertEqual(h["fields"]["identity"], "angry celebrity chef")
        self.assertIn("soft on beginners", h["fields"]["who"])
        self.assertIn("Pixar", h["fields"]["who"])          # unknown heading folded into who
        self.assertIn("wellington", h["fields"]["canon"])
        self.assertEqual(h["interests"], ["cooking", "football", "restaurants"])
        self.assertAlmostEqual(h["talkativeness"], 0.55)

    def test_no_headings_is_raw_prompt(self):
        h = H.parse("# Bob\nYou are Bob, a sleepy wizard who mumbles.")
        self.assertEqual(h["name"], "Bob")
        self.assertIn("sleepy wizard", h["prompt"])

    def test_round_trip(self):
        h = H.parse(SAMPLE)
        h2 = H.parse(H.dump("Gordon", h["fields"], h["interests"], "loud · picky · chef", 0.55))
        self.assertEqual(h2["fields"], h["fields"]); self.assertEqual(h2["interests"], h["interests"])


class TemplateTests(unittest.TestCase):
    def test_canon_paragraph_and_interests(self):
        p = A.assemble_prompt("Gordon", {"who": "You roast food.", "canon": "You love beef wellington"})
        self.assertIn("Your tastes", p); self.assertIn("beef wellington.", p)
        self.assertNotIn("Your tastes", A.assemble_prompt("Gordon", {"who": "x"}))

    def test_public_build_without_voices_can_create(self):
        a = {"name": "Gordon", "voice": "", "prompt": "x" * 100}
        self.assertIsNone(A.validate(a, [], set()))
        self.assertEqual(A.validate(a, ["somevoice"], set()), "Pick a voice.")

    def test_llm_drafter_uses_wrapper(self):
        llm = mock.Mock()
        llm.generate.return_value = iter(['{"who": "You love soup.", "interests": ["soup", "Rain"], "canon": "Pho"}'])
        f = A.draft_fields_llm(llm, "Soupy", "a soup ghost")
        self.assertEqual(f["who"], "You love soup."); self.assertEqual(f["interests"], "soup, rain")
        self.assertFalse(llm.generate.call_args.kwargs["use_system_prompt"])


class VoiceImportTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        d = Path(self.td.name)
        self.p = [mock.patch.object(VI, "USER_DIR", d), mock.patch.object(VI, "INDEX", d / "voices.json"),
                  mock.patch("agent_registry.reload")]
        for p in self.p: p.start()

    def tearDown(self):
        for p in self.p: p.stop()
        self.td.cleanup()

    def _wav(self, seconds, sr=44100, stereo=True):
        import soundfile as sf
        t = np.arange(int(seconds * sr)) / sr
        x = 0.2 * np.sin(2 * np.pi * 180 * t)
        x = np.concatenate([np.zeros(sr), x, np.zeros(sr)])  # 1s silence each side
        if stereo: x = np.stack([x, x], 1)
        b = io.BytesIO(); sf.write(b, x, sr, format="WAV"); return b.getvalue()

    def test_import_normalises_and_registers(self):
        import soundfile as sf
        r = VI.import_voice(self._wav(30), "clip.wav", "My Voice", "hello there")
        self.assertEqual(r["key"], "my-voice")
        self.assertLessEqual(r["seconds"], VI.MAX_S)
        i = sf.info(str(VI.USER_DIR / "my-voice.wav"))
        self.assertEqual((i.samplerate, i.channels), (24000, 1))
        idx = json.loads(VI.INDEX.read_text())
        self.assertIn("text", idx["voices"]["my-voice"])

    def test_rejects_short_and_duplicate(self):
        with self.assertRaises(VI.VoiceImportError):
            VI.import_voice(self._wav(1.5), "a.wav", "short")   # silence trimmed -> too short
        VI.import_voice(self._wav(8), "a.wav", "dup")
        with self.assertRaises(VI.VoiceImportError):
            VI.import_voice(self._wav(8), "a.wav", "dup")
        with self.assertRaises(VI.VoiceImportError):
            VI.import_voice(self._wav(8), "a.wav", "voice", taken={"voice"})  # never shadow a dev voice


if __name__ == "__main__":
    unittest.main()
