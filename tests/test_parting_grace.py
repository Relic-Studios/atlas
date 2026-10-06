"""Owner 10-07: after a silence request, step_back or a closed exchange, a final
'bye Fae' must not reopen the agent (all agents). Real questions still wake it."""
import unittest
from floor import ConversationFloor

N = ("fae",)


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.c = Clock(); self.f = ConversationFloor(clock=self.c)

    def gate(self, text, spk="S1", dt=3.0):
        self.c.t += dt
        return self.f.turn_gate(f"[{spk}] {text}", spk, N)


class AfterSilence(Base):
    def test_goodbye_after_shut_up_stays_quiet(self):
        self.assertIsNotNone(self.gate("Fae, shut up for a bit"))
        self.assertTrue(self.f.is_quiet())
        self.assertEqual(self.gate("bye Fae"), "parting (grace)")
        self.assertEqual(self.gate("goodnight Fae, love you", spk="S2"), "parting (grace)")
        self.assertTrue(self.f.is_quiet())          # gate never reopened

    def test_real_question_still_wakes(self):
        self.gate("Fae, be quiet")
        self.assertIsNone(self.gate("Fae, what time is it?"))
        self.assertFalse(self.f.is_quiet())

    def test_bare_name_still_wakes(self):
        self.gate("Fae, be quiet")
        self.assertIsNone(self.gate("Fae"))

    def test_goodbye_after_quiet_expired_still_held(self):
        self.gate("Fae, shut up")
        self.c.t += self.f.quiet_ttl_s + 20          # quiet over, grace not
        self.assertFalse(self.f.is_quiet())
        self.assertEqual(self.gate("bye Fae"), "parting (grace)")

    def test_grace_ends(self):
        self.gate("Fae, shut up")
        self.c.t += self.f.quiet_ttl_s + self.f.PARTING_GRACE_S + 60
        self.assertNotEqual(self.gate("bye Fae"), "parting (grace)")


class AfterStepBack(Base):
    def test_agent_choice_respected(self):
        self.f.step_back(1.0)
        self.assertEqual(self.gate("thanks Fae"), "parting (grace)")
        self.assertTrue(self.f.is_quiet())


class AfterClose(Base):
    def test_closed_conversation_not_reopened_by_bye(self):
        self.assertIsNone(self.gate("Fae, how far is the moon"))
        self.f.on_agent_spoke("S1")
        self.assertEqual(self.gate("ok thanks"), "conversation closed")
        self.assertEqual(self.gate("bye Fae"), "parting (grace)")
        self.assertIsNone(self.f.engaged)
        self.assertIsNone(self.gate("Fae, one more thing, how far is mars?"))


class NoGraceNormally(Base):
    def test_bye_without_prior_close_reaches_model(self):
        self.assertNotEqual(self.gate("bye Fae"), "parting (grace)")


if __name__ == "__main__":
    unittest.main()
