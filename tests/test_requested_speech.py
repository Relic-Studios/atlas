"""Requested speech survives Atlas's output and history filters, without a model."""
import unittest

from echo_reply import drop_parrots, is_parrot
from response_decision import ResponseDecision, filter_response


class RequestedSpeechTests(unittest.TestCase):
    def test_requested_words_survive_stream_and_history(self):
        for heard, reply in (("Please repeat copper.", "Copper."),
                             ("Read this phrase aloud: violet sky.", "Violet sky."),
                             ("Stop. Instead, just say twenty-three.", "Twenty-three."),
                             ("Quote hello there.", "Hello there.")):
            with self.subTest(heard=heard):
                self.assertFalse(is_parrot(reply, heard))
                history = [{"role": "user", "content": heard},
                           {"role": "assistant", "content": reply}]
                self.assertEqual(drop_parrots(history), history)
                decision = ResponseDecision()
                decision.heard = heard
                actual = "".join(filter_response(iter(["[SPEAK to=S1] " + reply]), decision))
                self.assertEqual(actual.strip(), reply)

    def test_unsolicited_mirroring_still_dropped(self):
        self.assertTrue(is_parrot("Copper.", "I found copper in the box."))

    def test_style_coaching_respects_current_reply_constraints(self):
        from floor import ConversationFloor
        floor = ConversationFloor(clock=lambda: 1000.0)
        floor.on_user_turn("S1", "hello")
        floor.depth_note("[S1] Go deeper.", "S1")
        for reply in ("Lisbon.", "Copper.", "Violet."):
            floor.on_agent_spoke("S1", reply)
        # This history normally triggers an engagement hook. The current person
        # can constrain the answer without a model-specific prompt exception.
        self.assertTrue(floor.engage_note("[S1] What next?", "S1", (), False))
        for text in ("Only answer with one word.", "Just say copper.",
                     "Please repeat violet.", "Keep your answer short."):
            with self.subTest(text=text):
                self.assertEqual(floor.depth_note("[S1] " + text, "S1"), "")
                self.assertEqual(floor.engage_note("[S1] " + text, "S1", (), False), "")
