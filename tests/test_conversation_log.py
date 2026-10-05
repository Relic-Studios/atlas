import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_log import ConversationLog, est_tokens  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def tick(self, s):
        self.t += s


def ramble(n, seed):
    rnd = random.Random(seed)
    vocab = "bro like the car was literally sideways and she said nah then we went to taco bell anyway".split()
    return " ".join(rnd.choice(vocab) for _ in range(n))


class ConversationLogTests(unittest.TestCase):
    def setUp(self):
        self.c = Clock()
        self.log = ConversationLog(agent_name="Max", clock=self.c)

    def test_fragments_from_same_voice_merge(self):
        self.assertEqual(self.log.add_user("S1", "so I was thinking maybe"), "added")
        self.c.tick(1.5)
        self.assertEqual(self.log.add_user("S1", "we get tacos tonight"), "merged")
        msgs, _ = self.log.messages()
        self.assertEqual(msgs, [{"role": "user", "content": "[S1] so I was thinking maybe we get tacos tonight"}])

    def test_no_merge_across_speakers_or_agent_or_gap(self):
        self.log.add_user("S1", "first")
        self.c.tick(1)
        self.log.add_user("S2", "second")
        self.c.tick(1)
        self.log.add_agent("S2", "yo")
        self.c.tick(1)
        self.log.add_user("S2", "third")
        self.c.tick(10)
        self.log.add_user("S2", "fourth")
        self.assertEqual(len(self.log.ledger), 5)

    def test_laugh_after_agent_is_reaction_not_history(self):
        self.log.add_user("S1", "Max roast S2")
        self.log.add_agent("S1", "S2 dresses like a clearance rack.")
        self.c.tick(2)
        self.assertEqual(self.log.add_user("S2", "LMAO"), "reaction")
        msgs, _ = self.log.messages()
        self.assertEqual(len(msgs), 2)
        self.assertIn("S2 laughed at your last line", self.log.state_note())
        self.c.tick(1)
        self.log.add_agent("S1", "next")
        self.assertNotIn("laughed", self.log.state_note())

    def test_cold_filler_is_ignored(self):
        self.assertEqual(self.log.add_user("S3", "hmm"), "ignored")
        self.assertEqual(len(self.log.ledger), 0)
        self.assertIn("S3", self.log.roster)

    def test_ramble_clipped_to_recent_words(self):
        self.log.add_user("S1", ramble(250, 1) + " so what do you think Max")
        (m,), _ = self.log.messages()
        self.assertLessEqual(len(m["content"].split()), 63)
        self.assertTrue(m["content"].endswith("what do you think Max"))

    def test_budget_is_flat_over_a_long_call(self):
        rnd = random.Random(7)
        sizes = []
        for turn in range(600):
            spk = rnd.choice(["S1", "S2", "S3", "S4", "S5"])
            self.log.add_user(spk, ramble(rnd.choice([3, 12, 40, 250]), turn))
            if turn % 4 == 0:
                self.log.add_agent(spk, ramble(30, -turn))
            self.c.tick(rnd.uniform(1, 9))
            if turn > 50:
                sizes.append(self.log.stats()["window_tokens"] + self.log.stats()["state_tokens"])
        self.assertLessEqual(max(sizes), self.log.budget_tokens + 400)
        self.assertLessEqual(len(self.log.ledger), self.log.ledger_cap)

    def test_evicted_speaker_keeps_last_line_in_state(self):
        self.log.add_user("S3", "my car got towed outside the gym this morning")
        for i in range(40):
            self.c.tick(5)
            self.log.add_user("S1", ramble(55, i))
            self.log.add_agent("S1", "word")
        msgs, seen = self.log.messages()
        self.assertNotIn("S3", seen)
        note = self.log.state_note("S1", seen)
        self.assertIn("S3", note)
        self.assertIn("car got towed", note)

    def test_in_flight_turn_not_duplicated(self):
        self.log.add_user("S1", "earlier line")
        self.c.tick(10)
        self.log.add_user("S1", "Max you there?")
        msgs, _ = self.log.messages("[S1] Max you there?")
        self.assertEqual([m["content"] for m in msgs], ["[S1] earlier line"])
        # merged fragment: only the in-flight tail is removed
        self.c.tick(1)
        self.log.add_user("S1", "hello?")
        msgs, _ = self.log.messages("[S1] hello?")
        self.assertEqual(msgs[-1]["content"], "[S1] Max you there?")

    def test_in_flight_dedup_ignores_punctuation(self):
        self.log.add_user("S1", "Max are you there?")
        msgs, _ = self.log.messages("[S1] Max, are you there")
        self.assertEqual(msgs, [])

    def test_assistant_uses_exact_control_header(self):
        self.log.add_agent("S2", "yo what's good")
        msgs, _ = self.log.messages()
        self.assertEqual(msgs[0]["content"], "[SPEAK to=S2] yo what's good")

    def test_self_and_empty_ignored_and_roster_capped(self):
        self.assertEqual(self.log.add_user("self", "echo of me"), "ignored")
        for i in range(20):
            self.c.tick(1)
            self.log.add_user(f"S{i}", "hey there everyone")
        self.assertLessEqual(len(self.log.roster), self.log.roster_cap)
        self.assertIn("S19", self.log.roster)

    def test_est_tokens_monotone(self):
        self.assertLess(est_tokens("a b"), est_tokens("a b c d e f"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
