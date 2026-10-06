"""Pacing (owner 10-06: 'our agent needs to talk less ... match healthy pacing')."""
import unittest

import numpy as np

import pacing as P
from floor import ConversationFloor


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


SENT = "this is a perfectly ordinary sentence about the game we are playing"  # 12 words


def room(agent_words_per_turn=12, rounds=8, humans=("S1", "S2", "S3")):
    """A floor where each round every human says SENT and the agent replies."""
    c = Clock()
    f = ConversationFloor(clock=c)
    for _ in range(rounds):
        for h in humans:
            c.t += 4
            f.on_user_turn(h)
            f.pacing_turn(h, SENT)
        c.t += 3
        f.on_agent_spoke(humans[0], " ".join(["word"] * agent_words_per_turn))
    return f, c


class Words(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(P.count_words("[S1] hey there, how's it going?"), 5)
        self.assertEqual(P.count_words("I'm Sam, it’s fine"), 4)
        self.assertEqual(P.count_words(""), 0)
        self.assertGreater(P.count_words("今日はいい天気ですね"), 3)


class Shares(unittest.TestCase):
    def test_fair_share_and_pressure(self):
        f, _ = room(agent_words_per_turn=12)        # agent says as much as each human
        s = f.pacing.snapshot(0.35)
        self.assertEqual(s["humans"], 3)
        self.assertAlmostEqual(s["fair_share"], 0.25, places=3)
        self.assertAlmostEqual(s["agent_share"], 0.25, places=2)
        self.assertLess(s["pressure"], P.SOFT_PRESSURE + 0.01)   # 0.25 / 0.22 = 1.14

    def test_dominating_agent_is_hard_tier(self):
        f, _ = room(agent_words_per_turn=36)        # 3x each human
        self.assertEqual(f.pacing.tier(0.35), "hard")

    def test_talkativeness_moves_the_budget(self):
        f, _ = room(agent_words_per_turn=14)
        self.assertEqual(f.pacing.tier(1.0), "ok")
        self.assertNotEqual(f.pacing.tier(0.0), "ok")

    def test_one_on_one_never_paced(self):
        f, _ = room(agent_words_per_turn=60, humans=("S1",))
        self.assertEqual(f.pacing.snapshot()["pressure"], 0.0)
        self.assertEqual(f.pacing.tier(), "ok")

    def test_window_forgets(self):
        f, c = room(agent_words_per_turn=40)
        c.t += P.WINDOW_S + 1
        self.assertEqual(f.pacing.snapshot()["total_words"], 0)
        self.assertEqual(f.pacing.tier(), "ok")

    def test_quiet_start_not_judged(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        f.pacing_turn("S1", "hi"); f.pacing_turn("S2", "yo")
        f.on_agent_spoke("S1", "hey both of you, good to hear from you tonight")
        self.assertEqual(f.pacing.snapshot()["pressure"], 0.0)

    def test_target_words_follow_the_room(self):
        f, _ = room(agent_words_per_turn=12)
        self.assertTrue(6 <= f.pacing.target_words() <= 25)
        g, _ = room(agent_words_per_turn=40)
        self.assertLessEqual(g.pacing.target_words(), f.pacing.target_words())


class Gate(unittest.TestCase):
    NAMES = ("Fae",)

    def gate(self, f, text, spk):
        return f.turn_gate(f"[{spk}] {text}", spk, self.NAMES)

    def hard_room(self):
        f, c = room(agent_words_per_turn=40)
        c.t += 40          # past the 'agent just spoke' window
        f.partner, f.partner_at = None, 0.0
        self.assertEqual(f.pacing.tier(), "hard")
        return f

    def test_named_always_passes(self):
        f = self.hard_room()
        for line in ("Fae, what time is it?", "hey Fae", "Fae you there"):
            v = self.gate(f, line, "S2")
            self.assertFalse(v and v.startswith("pacing"), (line, v))

    def test_statement_held_when_over(self):
        f = self.hard_room()
        v = self.gate(f, "yeah the boss fight was rough", "S2")
        self.assertTrue(v, v)

    def test_you_question_reaches_model(self):
        f = self.hard_room()
        v = self.gate(f, "would you leave the call if we asked?", "S2")
        self.assertFalse(v and v.startswith("pacing"), v)

    def test_answer_to_agents_question_passes(self):
        f = self.hard_room()
        f.on_agent_spoke("S2", "what game are you two playing?")
        f.last_agent_at -= 40
        f.asked = ("S2", f.clock())
        v = self.gate(f, "elden ring", "S2")
        self.assertFalse(v and v.startswith("pacing"), v)

    def test_under_budget_untouched(self):
        f, c = room(agent_words_per_turn=6)
        c.t += 40
        self.assertEqual(f.pacing.tier(), "ok")
        self.assertIsNone(f.pacing_gate("[S2] nice", "S2", self.NAMES))

    def test_last_gate_recorded(self):
        f = self.hard_room()
        self.gate(f, "yeah the boss fight was rough", "S2")
        self.assertTrue(f.last_gate and f.last_gate != "pass")
        self.gate(f, "Fae, you there?", "S2")
        self.assertEqual(f.last_gate, "pass")


class Note(unittest.TestCase):
    def test_note_is_length_only(self):
        for words in (6, 40):
            f, _ = room(agent_words_per_turn=words)
            n = f.pacing_note(lambda s: {"S1": "Sam", "S2": "Riley"}.get(s))
            self.assertTrue(n)
            self.assertNotRegex(n.lower(), r"\b(stay quiet|hold|don't answer|do not answer|silent|ignore)\b")
            self.assertIn("Sam", n)
            self.assertNotIn("S3", n)          # unknown voices are 'someone', never S-tags
            self.assertIn("you", n)

    def test_over_budget_note_asks_for_one_sentence(self):
        f, _ = room(agent_words_per_turn=40)
        self.assertIn("one short sentence", f.pacing_note())

    def test_no_note_one_on_one(self):
        f, _ = room(agent_words_per_turn=40, humans=("S1",))
        self.assertEqual(f.pacing_note(), "")


class Audio(unittest.TestCase):
    def test_features(self):
        sr = 16000
        t = np.arange(sr * 2) / sr
        x = np.zeros(sr * 3, dtype=np.float32)
        x[sr // 2: sr // 2 + sr * 2] = 0.1 * np.sin(2 * np.pi * 220 * t)   # 2 s of tone
        f = P.audio_features(x, sr)
        self.assertAlmostEqual(f["dur_s"], 3.0, places=1)
        self.assertAlmostEqual(f["voiced_s"], 2.0, delta=0.1)
        self.assertAlmostEqual(f["mean_dbfs"], -23.0, delta=1.5)   # 0.1 amp sine = -23 dBFS

    def test_garbage_is_safe(self):
        self.assertEqual(P.audio_features(None), {})
        self.assertEqual(P.audio_features([]), {})

    def test_pacing_record_has_gap_and_feats(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        f.pacing_turn("S1", SENT, {"dur_s": 3.0, "voiced_s": 2.4, "mean_dbfs": -30.0})
        c.t += 5
        rec = f.pacing_turn("S2", SENT, {"dur_s": 3.0})
        self.assertAlmostEqual(rec["gap_s"], 2.0, places=2)
        for k in ("spk", "words", "agent_share", "pressure", "gate"):
            self.assertIn(k, rec)
        self.assertEqual(f.pacing_turn("self", "echo"), {})


if __name__ == "__main__":
    unittest.main()
