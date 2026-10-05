"""Stage directions in asterisks are not voiced; neither are starred sound effects (owner 10-02)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_safety as S


class StageTests(unittest.TestCase):
    def test_descriptions_dropped(self):
        self.assertEqual(S.strip_stage("*evil laugh* Mwahaha.").strip(), "Mwahaha.")
        self.assertEqual(S.strip_stage("*clears throat* It's a sad day.").strip(), "It's a sad day.")
        self.assertEqual(S.strip_stage("Hold the pan. *beat* Pasta.").strip(), "Hold the pan. Pasta.")

    def test_sound_effects_dropped(self):
        self.assertEqual(S.strip_stage("*baaa* I'm a goat.").strip(), "I'm a goat.")
        self.assertEqual(S.strip_stage("*blorp*").strip(), "")

    def test_math_and_emphasis(self):
        self.assertEqual(S.strip_stage("5 * 3 is 15."), "5 times 3 is 15.")
        self.assertIn("never came back", S.strip_stage("*it went in the pan and never came back*"))

    def test_streaming(self):
        self.assertEqual("".join(S.screen(iter(["*evil ", "laugh* Mwahaha. You ", "fell for it."]))).strip(),
                         "Mwahaha. You fell for it.")


if __name__ == "__main__":
    unittest.main()
