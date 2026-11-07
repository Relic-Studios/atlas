"""Reply depth adapts to the moment (owner 11-03: "I want our agents to be more adaptive")."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor import ConversationFloor  # noqa: E402


class DepthNote(unittest.TestCase):
    def floor(self, people=("S1",)):
        t = [1000.0]
        f = ConversationFloor(clock=lambda: t[0])
        for p in people:
            f.on_user_turn(p, "hey")
        return f, t

    def test_asked_for_depth_gets_a_full_answer_and_keeps_it(self):
        f, t = self.floor(("S1", "S2"))
        n = f.depth_note("[S1] why are your replies so short? I want you to go deeper", "S1")
        self.assertIn("three to six", n)
        self.assertIn("never announce", n)
        t[0] += 60
        self.assertIn("three to six", f.depth_note("[S1] all right, what do you got?", "S1"))   # still on
        self.assertEqual(f.depth_note("[S2] lol nice", "S2"), "")                               # not S2
        t[0] += f.DEPTH_WINDOW_S + 1
        self.assertEqual(f.depth_note("[S1] lol nice", "S1"), "")                               # wore off

    def test_live_phrasings(self):
        f, _ = self.floor()
        for line in ("Are you able to talk about things that are deeper and use longer sentences?",
                     "And why are your replies so short?",
                     "I just want you to be able to hold a conversation and stimulate my mind",
                     "Ivy, what do you got?", "can you explain why", "tell me more about that"):
            f.depth_asks = {}
            self.assertIn("three to six", f.depth_note("[S1] " + line, "S1"), line)

    def test_one_on_one_open_question_gets_a_fuller_answer(self):
        f, t = self.floor(("S1",))
        self.assertEqual(f.depth_note("[S1] why do you care about me?", "S1"), "")   # call just started
        t[0] += 150
        f.on_user_turn("S1", "hmm")
        self.assertIn("two to four", f.depth_note("[S1] why do you care about me?", "S1"))
        self.assertEqual(f.depth_note("[S1] lol", "S1"), "")
        long_line = "[S1] " + "so anyway I was at the store today and the line was really long and it took forever ok"
        self.assertEqual(f.depth_note(long_line, "S1"), "")                          # long isn't deep

    def test_busy_room_stays_short(self):
        f, _ = self.floor(("S1", "S2", "S3", "S4"))
        self.assertEqual(f.depth_note("[S2] why do you think that is?", "S2"), "")
        self.assertIn("two to four", f.depth_note("[S2] go deeper on that", "S2"))   # asked: still fuller

    def test_explicit_brevity_never_activates_depth(self):
        for line in ('Please keep your answer short.', 'Make your replies brief.',
                     'Be concise.', 'Be brief.', 'Give shorter answers.', 'Keep it short.'):
            with self.subTest(line=line):
                f, _ = self.floor()
                self.assertEqual(f.depth_note('[S1] ' + line, 'S1'), '')
                self.assertNotIn('S1', f.depth_asks)

    def test_brevity_clears_prior_depth_without_changing_another_speaker(self):
        f, _ = self.floor(('S1', 'S2'))
        f.depth_note('[S1] go deeper', 'S1')
        f.depth_note('[S2] tell me more', 'S2')
        self.assertEqual(f.depth_note('[S1] Please keep your answer short.', 'S1'), '')
        self.assertEqual(f.depth_note('[S1] thanks for that', 'S1'), '')
        self.assertIn('three to six', f.depth_note('[S2] thanks for that', 'S2'))

    def test_brevity_overrides_an_open_question_in_established_one_on_one(self):
        f, t = self.floor()
        t[0] += 150
        f.on_user_turn('S1', 'still here')
        self.assertEqual(f.depth_note('[S1] Explain why, but keep your answer short.', 'S1'), '')

    def test_explicit_length_does_not_get_competing_coaching(self):
        for line in ('Write a fictional scene in at least twelve sentences.',
                     'Explain that in exactly 80 words.', 'Use at most three paragraphs.',
                     'Keep it to one short sentence.'):
            with self.subTest(line=line):
                f, _ = self.floor()
                self.assertEqual(f.depth_note('[S1] ' + line, 'S1'), '')
                self.assertEqual(f.engage_note('[S1] ' + line, 'S1', ('juniper',), False), '')


if __name__ == "__main__":
    unittest.main()
