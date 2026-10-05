"""Interruption policy: hold the floor through chatter, yield to real takeovers, resume."""
import sys, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import interrupts as I

N = ("ivy",)

class YieldTests(unittest.TestCase):
    def test_backchannel_talked_over(self):
        self.assertFalse(I.should_yield("[S2] yeah lol", N)[0])
    def test_named_yields(self):
        self.assertTrue(I.should_yield("[S2] Ivy wait", N)[0])
    def test_stop_yields(self):
        self.assertTrue(I.should_yield("[S2] shut up", N)[0])
    def test_short_side_remark_talked_over(self):
        self.assertFalse(I.should_yield("[S3] that's so true", N)[0])
    def test_takeover_yields(self):
        t = "[S3] okay so the thing about that whole situation is that nobody actually checked the logs"
        self.assertTrue(I.should_yield(t, N)[0])
    def test_finishing_line_holds(self):
        t = "[S3] okay so the thing about that whole situation is that nobody actually checked the logs"
        self.assertFalse(I.should_yield(t, N, agent_left_s=0.5)[0])
    def test_partner_question_yields(self):
        self.assertTrue(I.should_yield("[S1] wait, which one?", N, speaker="S1", partner="S1")[0])
    def test_partner_backchannel_holds(self):
        self.assertFalse(I.should_yield("[S1] mhm", N, speaker="S1", partner="S1")[0])
    def test_chattier_holds_longer(self):
        self.assertLess(I.takeover_words(0.0), I.takeover_words(1.0))

class LiveCutoffTests(unittest.TestCase):
    """Live call 10-05: agents cut off mid-sentence by their partner's short remarks."""
    P = dict(speaker="S1", partner="S1")
    def test_partner_remarks_talked_over(self):
        for t in ("I don't know.", "I can't do this.", "I've got a mute.", "So would I-",
                  "Still stuck in the-", "Say it hard.", "Gran Turismo 7."):
            self.assertFalse(I.should_yield("[S1] " + t, N, **self.P)[0], t)
    def test_partner_to_the_room(self):
        self.assertFalse(I.should_yield("[S1] Guys, I'm hungry.", N, **self.P)[0])
    def test_partner_pushback_and_carrying_on_yield(self):
        for t in ("no that's not what I said", "It's literally the same thing but you keep saying it"):
            self.assertTrue(I.should_yield("[S1] " + t, N, **self.P)[0], t)
    def test_finish_window(self):
        t = "[S3] okay so the thing about that whole situation is that nobody actually checked the logs"
        self.assertFalse(I.should_yield(t, N, agent_left_s=1.6)[0])
        self.assertTrue(I.should_yield(t, N, agent_left_s=3.0)[0])


class ResumeTests(unittest.TestCase):
    TXT = "So the best part of that movie is the ending because nobody saw the twist coming at all."
    def test_split_on_word_boundary(self):
        said, unsaid = I.split_spoken(self.TXT, 1.5, 6.0, True)
        self.assertTrue(self.TXT.startswith(said)); self.assertTrue(unsaid)
        self.assertEqual(" ".join((said + " " + unsaid).split()), self.TXT)
    def test_nearly_finished_no_cutoff(self):
        self.assertIsNone(I.make_cutoff(self.TXT, 5.9, 6.0, True))
    def test_cutoff_note_and_ttl(self):
        c = I.make_cutoff(self.TXT, 1.0, 6.0, True, target="S1")
        self.assertIsNotNone(c); self.assertTrue(c.live())
        self.assertIn(c.unsaid[:20], c.note())
        self.assertTrue(I.RESUME_CUE_RE.search(c.cue())); self.assertTrue(c.cue().startswith("[S1]"))
        self.assertFalse(c.live(now=c.at + I.RESUME_TTL_S + 1))

if __name__ == "__main__":
    unittest.main()
