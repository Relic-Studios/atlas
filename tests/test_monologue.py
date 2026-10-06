"""Owner 10-07: agents interrupted people mid-monologue (every breath pause ended a
'turn'). While someone holds the floor their lines are held; the last one is answered
once they really stop, if the normal gate allows it."""
import unittest
from floor import ConversationFloor

NAMES = ("Fae",)


class Mono(unittest.TestCase):
    def setUp(self):
        self.t = [1000.0]
        self.f = ConversationFloor(clock=lambda: self.t[0])

    def say(self, spk, text, dt=3.0):
        self.t[0] += dt
        g = self.f.turn_gate(f"[{spk}] {text}", spk, NAMES)
        self.f.on_user_turn(spk, text)
        return g

    def ramble(self, spk="S1"):
        self.say(spk, "So I've been playing this build for like three weeks now and honestly it's been great")
        self.say(spk, "The problem is the currency you get from the battle pass is basically useless for anything you actually want to buy in the shop")

    def test_breath_pause_held(self):
        self.ramble()
        self.assertTrue(self.f.monologue_active("S1"))
        g = self.say("S1", "Gamble which sucks because you get this currency that you can't spend")
        self.assertTrue(g and g.startswith("monologue"), g)
        self.assertTrue(self.say("S1", "What was meant to happen is that").startswith("monologue"))

    def test_name_or_request_passes(self):
        self.ramble()
        self.assertFalse((self.say("S1", "Fae, what do you think about that?") or "").startswith("monologue"))
        self.ramble()
        self.assertFalse((self.say("S1", "anyway tell me what you'd do") or "").startswith("monologue"))

    def test_listener_yeah_doesnt_break_run(self):
        self.ramble()
        self.say("S2", "Yeah.", dt=1.0)
        self.assertTrue(self.f.monologue_active("S1"))

    def test_other_speaker_takes_floor_drops_pending(self):
        self.ramble()
        self.say("S1", "and that's kind of where I'm at with it")
        self.assertIsNotNone(self.f.mono_pending)
        self.say("S2", "Dude I had the exact same problem last season with it")
        self.assertIsNone(self.f.mono_pending)
        self.assertFalse(self.f.monologue_active("S1"))

    def test_pending_holds_last_final_and_releases(self):
        self.ramble()
        self.say("S1", "and that's kind of where I'm at with it")
        self.f.on_user_turn("S1", "and that's kind of where I'm at with it, you know?")
        spk, text = self.f.take_monologue_pending()
        self.assertEqual(spk, "S1")
        self.assertIn("you know?", text)
        self.assertFalse(self.f.monologue_active("S1"))   # run ended: normal gate decides
        self.assertIsNone(self.f.take_monologue_pending())

    def test_long_pause_ends_run(self):
        self.ramble()
        self.t[0] += 20
        self.assertFalse(self.f.monologue_active())
        self.assertEqual(self.f.monologue_extra_wait(), 0.0)

    def test_agent_turn_resets(self):
        self.ramble()
        self.assertGreater(self.f.monologue_extra_wait(), 0)
        self.f.on_agent_spoke("S1", "Yeah, that's rough.")
        self.assertTrue(self.f.monologue_active())       # a reaction doesn't end their run
        self.f.on_agent_spoke("S1", "Wait, which build is that?")
        self.assertFalse(self.f.monologue_active())      # a question hands them the turn

    def test_short_back_and_forth_not_monologue(self):
        self.say("S1", "Fae, what's up?")
        self.f.on_agent_spoke("S1", "Not much!")
        self.assertFalse((self.say("S1", "Cool, what are you doing?") or "").startswith("monologue"))

    def test_pending_expires(self):
        self.ramble()
        self.say("S1", "and that's kind of where I'm at with it")
        self.t[0] += 60
        self.assertIsNone(self.f.take_monologue_pending())


class TurnWaitHook(unittest.TestCase):
    def test_hook_exists(self):
        import turndetect
        self.assertTrue(hasattr(turndetect, "EXTRA_WAIT_HOOK"))


if __name__ == "__main__":
    unittest.main()
