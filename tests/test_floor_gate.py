"""Pure-logic tests for the shared pre-LLM gate (floor.turn_gate), quiet mode,
current-speaker target enforcement and the no-partner room note. No model."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floor import ConversationFloor, names_agent  # noqa: E402
from response_decision import ResponseDecision, expected_target_for, filter_response  # noqa: E402

N = ("max",)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class QuietModeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.f = ConversationFloor(clock=self.clock)

    def test_named_silence_request_persists_until_named_again(self):
        self.assertEqual(self.f.turn_gate("[S1] Max, stay quiet for a bit while we plan this", "S1", N),
                         "explicit silence request")
        self.assertTrue(self.f.is_quiet())
        self.assertEqual(self.f.turn_gate("[S2] okay so friday at eight?", "S2", N), "quiet mode")
        self.assertEqual(self.f.turn_gate("[S2] sweet, S1 bring the speaker", "S2", N), "quiet mode")
        self.assertIsNone(self.f.turn_gate("[S1] Max, okay you can talk now, thoughts?", "S1", N))
        self.assertFalse(self.f.is_quiet())

    def test_quiet_expires(self):
        self.f.turn_gate("[S1] Max, be quiet", "S1", N)
        self.clock.t += self.f.quiet_ttl_s + 1
        self.assertIsNone(self.f.turn_gate("[S2] what time is it", "S2", N))

    def test_unaimed_shut_up_between_friends_does_not_mute(self):
        self.assertEqual(self.f.turn_gate("[S2] bro shut up", "S2", N), "explicit silence request")
        self.assertFalse(self.f.is_quiet())

    def test_partner_saying_shut_up_mutes(self):
        self.f.on_agent_spoke("S2")
        self.f.turn_gate("[S2] shut up", "S2", N)
        self.assertTrue(self.f.is_quiet())

    def test_quoted_name_does_not_release(self):
        self.f.turn_gate("[S1] Max, be quiet", "S1", N)
        self.assertEqual(self.f.turn_gate('[S2] he said "max tell a joke"', "S2", N), "quiet mode")

    def test_other_addressee_comma_less(self):
        self.assertEqual(self.f.turn_gate("[S4] Maya did you bring your charger", "S4", N),
                         "addressed to Maya")
        self.assertIsNone(self.f.turn_gate("[S4] Did you see that", "S4", N))
        self.assertIsNone(self.f.turn_gate("[S4] Lmao did you see that", "S4", N))

    def test_reset_clears_quiet(self):
        self.f.turn_gate("[S1] Max, be quiet", "S1", N)
        self.f.reset()
        self.assertFalse(self.f.is_quiet())


DEV_NAMES = (__import__("pathlib").Path(__file__).resolve().parents[1] / "dev_pack").exists()  # public export renames dev agents
class NamesAgentTests(unittest.TestCase):
    @__import__("unittest").skipUnless(DEV_NAMES, "uses dev agent-name misspellings")
    def test_variants(self):
        self.assertTrue(names_agent("[S1] tomas you there", N))
        self.assertTrue(names_agent("hey bot what's up", N))
        self.assertFalse(names_agent('he said "Max" yesterday', N))
        self.assertFalse(names_agent("the kettle is boiling", N))


class RoomNoteTests(unittest.TestCase):
    def test_no_partner_multi_person_unnamed_gets_hold_note(self):
        f = ConversationFloor(clock=Clock())
        f.on_user_turn("S1")
        self.assertIn("nobody is in a conversation with you", f.room_note("S4", "[S4] my internet sucks", N))
        self.assertEqual(f.room_note("S4", "[S4] Max my internet sucks", N), "")

    def test_single_speaker_unnamed_gets_passive_note(self):
        # 09-29: Ivy assumed stray lines in a quiet room were for her.
        f = ConversationFloor(clock=Clock())
        note = f.room_note("S1", "[S1] my internet sucks", N)
        self.assertIn("PASSIVE", note)
        self.assertIn("Default [HOLD]", note)

    def test_single_speaker_named_no_note(self):
        f = ConversationFloor(clock=Clock())
        self.assertEqual(f.room_note("S1", "[S1] Max my internet sucks", N), "")


class TargetEnforcementTests(unittest.TestCase):
    def test_expected_target_for(self):
        self.assertEqual(expected_target_for("[S3] hi"), "S3")
        self.assertEqual(expected_target_for("hi"), "user")  # unknown speaker

    def test_mismatched_target_forced_to_current_speaker(self):
        d = ResponseDecision(expected_target="S1")
        body = "".join(filter_response(iter(["[SPEAK to=S2] Friday's solid."]), d))
        self.assertEqual((d.action, d.target, body), ("SPEAK", "S1", "Friday's solid."))

    def test_unlabeled_keeps_model_target(self):
        d = ResponseDecision()
        "".join(filter_response(iter(["[SPEAK to=user] yo"]), d))
        self.assertEqual(d.target, "user")


if __name__ == "__main__":
    unittest.main()


class SideExchangeBeatsPartner(unittest.TestCase):
    """first_run sim 10-04: the agent just answered Sam, then Riley asks Sam a question
    and Sam answers Riley. Sam is the agent's partner, but that exchange is theirs."""

    def setUp(self):
        self.clock = Clock()
        self.f = ConversationFloor(clock=self.clock)

    def _turn(self, spk, line):
        g = self.f.turn_gate(f"[{spk}] {line}", spk, N)
        self.f.on_user_turn(spk)
        self.clock.t += 3
        return g

    def test_partner_answering_side_question_holds(self):
        self._turn("S1", "hey Max, I'm Sam")
        self._turn("S2", "and I'm Riley")
        self.assertIsNone(self._turn("S1", "Max we're playing Mario Kart tonight"))
        self.f.on_agent_spoke("S1"); self.clock.t += 2
        self.assertIsNotNone(self._turn("S2", "S1 did you bring the extra controller"))
        self.assertEqual(self._turn("S1", "yeah it's in my bag"), "side exchange")
        # named again -> back to the agent
        self.assertIsNone(self._turn("S1", "okay Max, settle it: co-op or competitive?"))

    def test_partner_still_answered_when_no_side_question(self):
        self._turn("S1", "hey Max, I'm Sam")
        self._turn("S2", "and I'm Riley")
        self._turn("S1", "Max what's your favourite kart")
        self.f.on_agent_spoke("S1"); self.clock.t += 2
        self.assertIsNone(self._turn("S1", "really? why that one"))


class StatusOnlySkip(unittest.TestCase):
    """Demo take 4 (10-04): 'Nice. OK. Loading in.' got 'Welcome, Riley.'"""
    def test_status_lines(self):
        from floor import status_only
        N = ("Wren",)
        for t in ["[S2] Nice. OK. Loading in.", "brb", "Almost, like two more minutes.",
                  "one sec", "my internet is lagging", "ok im back"]:
            self.assertTrue(status_only(t, N), t)
        for t in ["Wren, loading in, you ready?", "are you loading in?",
                  "Okay, Wren. Settle it.", "what game are we playing",
                  "Something short though, I have work early.",
                  "loading in to the new map with the boys is gonna be sick honestly man"]:
            self.assertFalse(status_only(t, N), t)
