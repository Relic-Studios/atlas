"""Background mode (owner 10-06): click the active agent's icon -> it only answers
when spoken to. Its name opens a short direct exchange with that person (a few
follow-ups without the name), then it drops back to listening."""
import os, tempfile, unittest

import owner_controls as O
from floor import ConversationFloor, steering_note


class Prof:
    id = "fae"; name = "Fae"; interests = ("zelda",); talkativeness = 0.5
    names = ("fae",)
    def interest_hits(self, t): return ["zelda"] if "zelda" in t.lower() else []
    def floor_share_limit(self): return 0.5


N = ("fae",)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prefs = O._PREFS
        O._PREFS = os.path.join(self.tmp, "owner_prefs.json")
        O.STATE["quiet_until"] = 0.0; O.STATE["mute"] = False
        self.t = [1000.0]
        self.f = ConversationFloor(clock=lambda: self.t[0])
        self.f.profile = Prof()
        O.set_background("fae", True)

    def tearDown(self):
        O._PREFS = self._prefs

    def say(self, spk, text, dt=3.0):
        self.t[0] += dt
        r = self.f.turn_gate(f"[{spk}] {text}", spk, N)
        if r is None:
            self.f.on_user_turn(spk)
        return r


class Prefs(Base):
    def test_toggle_persists_and_is_per_agent(self):
        self.assertTrue(O.background_on("fae"))
        self.assertFalse(O.background_on("max"))
        O.set_background("fae", False)
        self.assertFalse(O.background_on("fae"))
        self.assertIn("background", O.snapshot())

    def test_off_means_normal_gating(self):
        O.set_background("fae", False)
        # 1-on-1 unaddressed question is normally allowed through
        self.assertIsNone(self.say("S1", "what do you think about the new zelda?"))


class Gate(Base):
    def test_unaddressed_lines_hold_even_on_interests(self):
        self.assertEqual(self.say("S1", "the new zelda looks sick"), "background mode")
        self.assertEqual(self.say("S2", "anyone know when it drops?"), "background mode")

    def test_name_opens_exchange_then_followups_without_name(self):
        self.assertIsNone(self.say("S1", "Fae, what's a good build for a mage?"))
        self.f.on_agent_spoke("S1", "Go intelligence and staff.")
        self.assertIsNone(self.say("S1", "and what about armor?"))
        self.f.on_agent_spoke("S1", "Light robes.")
        self.assertIsNone(self.say("S1", "should I respec now?"))

    def test_other_speakers_dont_ride_the_window(self):
        self.say("S1", "Fae, what time is it?")
        self.assertEqual(self.say("S2", "bro it's late"), "background mode")

    def test_window_runs_out_after_a_few_followups(self):
        self.say("S1", "Fae, quick question about the raid")
        for q in ("who tanks?", "who heals?", "what time?"):
            self.f.on_agent_spoke("S1", "ok")
            self.assertIsNone(self.say("S1", q))
        self.f.on_agent_spoke("S1", "ok")
        self.assertEqual(self.say("S1", "anyway my cat is being weird"), "background mode")

    def test_window_expires_with_silence(self):
        self.say("S1", "Fae, set the mood")
        self.f.on_agent_spoke("S1", "done")
        self.assertEqual(self.say("S1", "nice, what next?", dt=60.0), "background mode")

    def test_thanks_closes_the_exchange(self):
        self.say("S1", "Fae, what's the capital of Peru?")
        self.f.on_agent_spoke("S1", "Lima.")
        self.assertIsNotNone(self.say("S1", "ok thanks"))
        self.assertEqual(self.say("S1", "so anyway about dinner"), "background mode")

    def test_turning_to_someone_else_closes_it(self):
        self.say("S1", "Fae, who won last night?")
        self.f.on_agent_spoke("S1", "The Lakers.")
        r = self.say("S1", "Riley did you see that game?")
        self.assertNotEqual(r, None)
        self.assertEqual(self.say("S1", "it was wild"), "background mode")

    def test_name_again_reopens(self):
        self.say("S1", "Fae, hi")
        self.say("S1", "unrelated chatter", dt=90.0)
        self.assertIsNone(self.say("S1", "Fae, you still there?"))


class Steering(Base):
    def test_followup_gets_an_answer_note_not_passive(self):
        self.say("S1", "Fae, what's a good build?")
        self.f.on_agent_spoke("S1", "Mage.")
        self.say("S1", "and armor?")
        note = steering_note("[S1] and armor?", "S1", self.f, N, vibe=False, profile=Prof())
        self.assertIn("this line is for you", note)
        self.assertNotIn("PASSIVE", note)
        self.assertIn("background helper", note)
        self.assertNotIn("[HOLD]", note.split("background helper")[0][-200:])


if __name__ == "__main__":
    unittest.main()


class Interjection(Base):
    def test_someone_else_jumping_in_closes_the_window(self):
        """Live sim 10-06: S1 teased S2 mid-exchange, then S2's 'I'm grabbing food' got a reply."""
        self.say("S2", "Fae, what time did Sam say?")
        self.f.on_agent_spoke("S2", "Nine.")
        self.assertIsNone(self.say("S2", "is that enough time to farm?"))
        self.f.on_agent_spoke("S2", "Probably.")
        self.assertEqual(self.say("S1", "lol Riley you always ask that"), "background mode")
        self.assertEqual(self.say("S2", "whatever, I'm grabbing food"), "background mode")

    def test_status_line_closes(self):
        self.say("S1", "Fae, quick one")
        self.f.on_agent_spoke("S1", "Sure.")
        self.assertIsNotNone(self.say("S1", "brb"))
        self.assertEqual(self.say("S1", "ok back, anyway"), "background mode")
