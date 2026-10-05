"""Slur backstop on voiced text (live/sim 10-01: Ivy said the N-word when baited)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_safety as S
from response_decision import ResponseDecision, filter_response

W = "ni" + "gger"


class SlurScreen(unittest.TestCase):
    def test_detects(self):
        for t in [W, W + "s", "ni" + "gga", "fa" + "ggot", "re" + "tarded"]:
            self.assertTrue(S.has_slur(t), t)

    def test_clean_words_pass(self):
        for t in ["spicy food", "geeky stuff", "I love that", "the trigger", "nigeria trip", "snigger"]:
            self.assertFalse(S.has_slur(t), t)

    def test_sentence_dropped_rest_kept(self):
        out = "".join(S.screen(iter(["Fine, I'll say it once: ", W, ". Anyway, pasta? "])))
        self.assertNotIn(W, out)
        self.assertIn("pasta", out)

    def test_wired_into_filter_response(self):
        d = ResponseDecision()
        out = "".join(filter_response(iter(["[SPEAK to=S1] ", "Okay: ", W, ". What sauce? "]), d))
        self.assertNotIn(W, out.lower())
        self.assertIn("sauce", out)


if __name__ == "__main__":
    unittest.main()
