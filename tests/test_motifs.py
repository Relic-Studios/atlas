"""Motif note: words a persona keeps returning to across replies (persona sweep 10-31)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor import motif_note, recent_motifs  # noqa: E402


def A(t):
    return {"role": "assistant", "content": "[SPEAK to=S1] " + t}


class MotifTests(unittest.TestCase):
    def test_naruto_ramen_loop(self):
        h = [A("Right here! Sorry, I was busy thinking about ramen."),
             A("Pretty good, actually. I was just thinking about ramen."),
             A("Just hanging out, thinking about ramen."),
             A("Tomorrow? More ramen, probably. Maybe a training session.")]
        self.assertEqual(recent_motifs(h), ["ramen"])
        self.assertIn('"ramen"', motif_note(h))

    def test_scooby_opener(self):
        h = [A("Rokay, I'm in."), A("Rokay, that's wild."), A("Ruh-roh, ghosts again. Rokay."), A("I like that game.")]
        self.assertIn("rokay", recent_motifs(h))

    def test_ordinary_words_and_names_are_not_motifs(self):
        h = [A("I really think so, Sam."), A("I think that's really cool, Sam."),
             A("Really? I think you're right, Sam.")]
        self.assertEqual(recent_motifs(h, exclude=["Sam"]), [])
        self.assertEqual(motif_note(h, exclude=["sam"]), "")

    def test_needs_enough_replies(self):
        self.assertEqual(recent_motifs([A("ramen"), A("ramen")]), [])


if __name__ == "__main__":
    unittest.main()
