"""Stale drafts + pure reactions (live 10-02 Max 'losing his mind')."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import floor as F


class StaleDraftTests(unittest.TestCase):
    def test_fragment_draft_is_stale(self):
        self.assertTrue(F.stale_draft("[S43] Bro.", "Bro, I thought you meant your pickup game as in picking up girls not an actual game of sports"))
        self.assertTrue(F.stale_draft("[S5] Okay.", "Okay. What authority did he step on? Inspirational."))

    def test_same_turn_not_stale(self):
        self.assertFalse(F.stale_draft("Max, are you there?", "Max, are you there?"))
        self.assertFalse(F.stale_draft("I played Zelda on the Wii", "I played Zelda on the Wii, I played Zelda on the Wii"))
        self.assertFalse(F.stale_draft("hey max what's up", "hey max what's up man"))


class ReactionTests(unittest.TestCase):
    def test_reactions_are_filler(self):
        for t in ["Oh my god.", "Oh, my God.", "Holy shit.", "Bro."]:
            self.assertTrue(F.filler_only(t), t)

    def test_questions_and_address_are_not(self):
        for t in ["Oh my god?", "Max, oh my god", "Okay.", "No way, Max, really?"]:
            self.assertFalse(F.filler_only(t), t)


if __name__ == "__main__":
    unittest.main()
