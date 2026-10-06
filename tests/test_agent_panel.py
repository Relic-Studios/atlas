"""Agent panel (owner 10-06): two agents share a call, one speaks per turn."""
import unittest

from agent_panel import Panel, cue_text, cue_note, HANDOFF_CAP


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make():
    c = Clock()
    p = Panel(clock=c)
    p.set_members(["max", "fae"], {"max": ("Max", "tom"), "fae": ("Fae",)})
    return p, c


class Routing(unittest.TestCase):
    def test_inactive_routes_nothing(self):
        p = Panel()
        self.assertIsNone(p.route("Fae, hi"))
        p.set_members(["max"], {})
        self.assertFalse(p.active)

    def test_named_member_takes_the_line(self):
        p, _ = make()
        self.assertEqual(p.route("Fae, what's the capital of France?"), "fae")
        self.assertEqual(p.route("what do you think, Max?"), "max")
        self.assertEqual(p.route("hey tom you there"), "max")

    def test_first_named_wins(self):
        p, _ = make()
        self.assertEqual(p.route("Fae, do you agree with Max?"), "fae")

    def test_name_inside_word_doesnt_count(self):
        p, _ = make()
        self.assertEqual(p.route("I love the navigation in this game"), "max")  # lead

    def test_partner_kept_then_expires(self):
        p, c = make()
        p.on_agent_spoke("fae", "Paris is the capital.")
        self.assertEqual(p.route("oh nice, and Germany?"), "fae")
        c.t += 60
        self.assertEqual(p.route("anyway what's up"), "max")


class Handoff(unittest.TestCase):
    def test_question_to_other_agent_hands_off_once(self):
        p, _ = make()
        p.on_human_turn()
        self.assertEqual(p.on_agent_spoke("max", "Honestly no idea. Fae, what do you think?"), "fae")
        h = p.take_handoff()
        self.assertEqual((h["from"], h["to"]), ("max", "fae"))
        # Fae answers by naming Max back with a question: capped, no ping-pong.
        self.assertIsNone(p.on_agent_spoke("fae", "Max, haven't you read it?"))
        self.assertEqual(HANDOFF_CAP, 1)

    def test_mention_without_question_is_not_a_handoff(self):
        p, _ = make()
        p.on_human_turn()
        self.assertIsNone(p.on_agent_spoke("max", "Fae already said that, and she's right."))

    def test_human_turn_cancels_pending_handoff(self):
        p, _ = make()
        p.on_human_turn()
        p.on_agent_spoke("max", "Fae, any thoughts?")
        p.on_human_turn()
        self.assertIsNone(p.take_handoff())

    def test_stale_handoff_dropped(self):
        p, c = make()
        p.on_human_turn()
        p.on_agent_spoke("max", "Fae, any thoughts?")
        c.t += 60
        self.assertIsNone(p.take_handoff())

    def test_cue_is_recognised(self):
        p, _ = make()
        p.on_human_turn()
        p.on_agent_spoke("max", "Fae, any thoughts?")
        cue = cue_text(p.take_handoff(), str.title)
        self.assertIn("Max", cue)
        self.assertIn("Don't [HOLD]", cue)
        self.assertTrue(cue_note(cue))
        self.assertEqual(cue_note("[S1] hello"), "")


class Note(unittest.TestCase):
    def test_note_only_for_members_and_never_says_hold(self):
        p, _ = make()
        n = p.note("max", str.title)
        self.assertIn("Fae", n)
        self.assertNotIn("[HOLD]", n)
        self.assertEqual(p.note("pup"), "")
        p.clear()
        self.assertEqual(p.note("max"), "")


class ManagerMirror(unittest.TestCase):
    """Both members hear the room; the other member hears the speaker under its name."""
    def test_mirror(self):
        try:
            from speech_pipeline_manager import SpeechPipelineManager
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"speech stack not installed: {e}")
        import agent_panel
        m = SpeechPipelineManager.__new__(SpeechPipelineManager)
        m.current_persona = "max"
        old = agent_panel.PANEL
        try:
            agent_panel.PANEL = p = make()[0]
            m._runtime()
            m.panel_mirror_user("S1", "who wants pizza")
            m.panel_mirror_agent("I'd go pepperoni.")
            other = m.agents.get("fae").convo
            text = " ".join(str(e.__dict__) for e in other.ledger)
            self.assertIn("pizza", text)
            self.assertIn("pepperoni", text)
            mine = " ".join(str(e.__dict__) for e in m.agents.get("max").convo.ledger)
            self.assertNotIn("pizza", mine)  # own log is fed by the normal path, not the mirror
        finally:
            agent_panel.PANEL = old


if __name__ == "__main__":
    unittest.main()


class NamedTargetHeader(unittest.TestCase):
    """Live panel sim 10-06: Fae answered Max's hand-off with '[SPEAK to=Max]',
    which was dropped as INVALID (silent hand-off)."""

    def _run(self, raw, expected="user"):
        from response_decision import ResponseDecision, filter_response
        d = ResponseDecision(expected_target=expected)
        return d, "".join(filter_response(iter([raw]), d)).strip()

    def test_name_target_is_spoken(self):
        d, body = self._run("[SPEAK to=Max] Majora's Mask is darker than it looks.")
        self.assertEqual(d.action, "SPEAK")
        self.assertEqual(d.target, "user")
        self.assertIn("Majora", body)

    def test_s_tag_and_hold_unchanged(self):
        d, _ = self._run("[SPEAK to=S2] hi", expected="S2"); self.assertEqual(d.target, "S2")
        d, _ = self._run("[HOLD]"); self.assertEqual(d.action, "HOLD")

    def test_garbage_header_still_invalid(self):
        d, body = self._run("[SPEAK to=two words here] x")
        self.assertEqual(d.action, "INVALID"); self.assertEqual(body, "")
