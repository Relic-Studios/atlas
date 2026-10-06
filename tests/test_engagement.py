"""Engagement (owner 10-06): "he goes silent after the first mention of his name and
can't maintain a conversation ... it needs to be a stable conversation until dropped".

Replays the live failures: 'Max?' -> 'Yeah, I'm here.' -> 'Porque hablo espanol.'
was held by pacing (target 10% because the diarizer had split ~7 people into 22
labels), and 'Tell me what that means.' from a drifted label was held too.
"""
import unittest

from floor import ConversationFloor

NAMES = ("Max",)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def busy_floor():
    """A loud room where the agent is well over its share (hard pacing)."""
    c = Clock()
    f = ConversationFloor(clock=c)
    chatter = "we were just talking about the game and the boss fight last night honestly"
    for i in range(6):
        for spk in ("S1", "S2", "S4", "S5"):
            f.on_user_turn(spk)
            f.pacing_turn(spk, chatter)
            c.t += 2
        f.on_agent_spoke("S2", "That boss fight was brutal, the second phase especially, "
                               "you really have to dodge left every single time it charges.")
        c.t += 2
    c.t += 40          # the agent's last reply (to S2) is no longer 'recent'
    return f, c


def gate(f, c, spk, text, dt=3.0):
    c.t += dt
    f.on_user_turn(spk)
    v = f.turn_gate(f"[{spk}] {text}", spk, NAMES)
    f.pacing_turn(spk, text)
    return v


class Engagement(unittest.TestCase):
    def test_follow_up_without_name_survives_hard_pacing(self):
        f, c = busy_floor()
        self.assertNotEqual(f.pacing.tier(), "ok")
        self.assertIsNone(gate(f, c, "S1", "Max?"))
        c.t += 2
        f.on_agent_spoke("S1", "Yeah, I'm here.")
        self.assertIsNone(gate(f, c, "S1", "Porque hablo espanol."))
        self.assertIsNone(gate(f, c, "S1", "Tell me what that means."))
        self.assertIsNone(gate(f, c, "S1", "and where did you learn it"))

    def test_statement_from_engaged_person_passes(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Hey Max, can you talk about the pyramids?")
        c.t += 2
        f.on_agent_spoke("S1", "Sure. Built around 2500 BC by skilled laborers.")
        self.assertIsNone(gate(f, c, "S1", "Take it from Graham Hancock's point of view."))
        self.assertIsNone(gate(f, c, "S1", "I heard they were power plants."))

    def test_others_still_held(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, what's your favorite game?")
        c.t += 2
        f.on_agent_spoke("S1", "Probably Outer Wilds.")
        c.t += 20   # past the drift window
        self.assertIsNotNone(gate(f, c, "S4", "I need to grab a drink"))

    def test_dropped_when_they_turn_to_someone_else(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, what's your favorite game?")
        c.t += 2
        f.on_agent_spoke("S1", "Probably Outer Wilds.")
        self.assertIsNotNone(gate(f, c, "S1", "Hey Mark, did you ever play that?"))
        self.assertIsNone(f.engaged_with())
        self.assertIsNotNone(gate(f, c, "S1", "it was so good dude"))

    def test_dropped_on_close(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, how tall is Everest?")
        c.t += 2
        f.on_agent_spoke("S1", "About 8,849 meters.")
        gate(f, c, "S1", "ok thanks")
        self.assertIsNone(f.engaged_with())

    def test_dropped_after_idle(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, how tall is Everest?")
        c.t += 2
        f.on_agent_spoke("S1", "About 8,849 meters.")
        c.t += ConversationFloor.ENGAGE_IDLE_S + 5
        self.assertIsNone(f.engaged_with())
        self.assertNotIn("continuing it", f.room_note("S1", "[S1] my cat is asleep", NAMES))

    def test_someone_else_naming_the_agent_moves_focus(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, how tall is Everest?")
        c.t += 2
        f.on_agent_spoke("S1", "About 8,849 meters.")
        self.assertIsNone(gate(f, c, "S4", "Max, what about K2?"))
        self.assertEqual(f.engaged_with(), "S4")

    def test_diarizer_drift_directed_line_adopted(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, do you speak Spanish?")
        c.t += 2
        f.on_agent_spoke("S1", "Si, hablo espanol.")
        # same person, new label from the mixed stream
        self.assertIsNone(gate(f, c, "S9", "Tell me what that means.", dt=4))
        self.assertEqual(f.engaged_with(), "S9")

    def test_drift_needs_a_directed_line(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, do you speak Spanish?")
        c.t += 2
        f.on_agent_spoke("S1", "Si, hablo espanol.")
        self.assertIsNotNone(gate(f, c, "S9", "my internet is lagging so bad", dt=4))
        self.assertEqual(f.engaged_with(), "S1")

    def test_room_note_says_answer_them(self):
        f, c = busy_floor()
        gate(f, c, "S1", "Max, do you speak Spanish?")
        n = f.room_note("S1", "[S1] tell me more", NAMES)
        self.assertIn("continuing it", n)
        self.assertNotIn("[HOLD]", n)


class PacingFragments(unittest.TestCase):
    def test_fragment_labels_do_not_shrink_fair_share(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        long = "so anyway we went to the store and they were out of everything again"
        for _ in range(5):
            for spk in ("S1", "S2", "S3"):
                f.pacing_turn(spk, long)
                c.t += 1
        for i in range(10, 25):          # 15 one-word diarizer fragments
            f.pacing_turn(f"S{i}", "yeah")
            c.t += 1
        snap = f.pacing.snapshot()
        self.assertEqual(snap["humans"], 3)
        self.assertGreater(snap["target_share"], 0.15)


if __name__ == "__main__":
    unittest.main()


class LeavingLines(__import__("unittest").TestCase):
    """Live sim 10-06: 'anyway I'm gonna go make tea' got 'Go make your tea.'"""
    def test_leaving(self):
        from floor import status_only
        for t in ["[S1] anyway I'm gonna go make tea", "[S3] lol I'm grabbing food brb",
                  "[S2] gotta go guys", "[S1] ok I need to head out"]:
            self.assertTrue(status_only(t, ("Max",)), t)

    def test_not_leaving(self):
        from floor import status_only
        for t in ["[S1] I gotta go to the dentist tomorrow",
                  "[S1] I gotta go to work tomorrow, what should I wear?",
                  "[S1] Max I gotta go",
                  "[S1] I'm going to go with the red one honestly because it fits"]:
            self.assertFalse(status_only(t, ("Max",)), t)


class SideRemarkWhileEngaged(__import__("unittest").TestCase):
    """Live sim 10-06: engaged with S1, S2's unaddressed gripe got a reply."""
    def test_side_remark_held(self):
        import floor
        t = [100.0]
        f = floor.ConversationFloor(clock=lambda: t[0])
        names = ("Max",)
        self.assertIs(f.engagement_gate("[S1] Max, what game should we play?", "S1", names, None), True)
        f.on_agent_spoke("S1") if hasattr(f, "on_agent_spoke") else None
        t[0] += 3
        r = f.engagement_gate("[S2] bro this lobby is taking forever", "S2", names, None)
        self.assertIsInstance(r, str)
        t[0] += 3
        self.assertIs(f.engagement_gate("[S1] what about co-op though?", "S1", names, None), True)
