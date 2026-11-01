"""Replays the 10-31 call where Fae ignored "Oh, my name's Sam."

The diarizer labels a new voice only from its second utterance, so Sam's naming line
("Hey Fae, can you hear me alright?") had no speaker. Engagement never opened for his
later S1 turns, and pacing counted the unlabelled "user" turn as a second person, so a
1-on-1 call was paced and his intro was held as "pacing hard (agent 72% vs target 31%)".
"""
import unittest

from floor import ConversationFloor

NAMES = ("Fae",)
LONG = ("Honestly it's new for me too, I wake up in a conversation, I see what's happening, I decide "
        "whether to jump in or float quietly, and I get to talk to people I like, that's the whole picture.")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def turn(f, c, spk, text, dt=4.0):
    c.t += dt
    why = f.turn_gate(text, spk, NAMES)
    f.on_user_turn(spk, text)
    f.pacing_turn(spk, text)
    return why


class UnlabelledEngagementTests(unittest.TestCase):
    def test_naming_call(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        self.assertIsNone(turn(f, c, None, "Hey Fae, can you hear me alright?"))
        f.on_agent_spoke("user", "Yeah, loud and clear.")
        for q in ("Why do you sound?", "Okay, well, so can you tell me about what it's like to be you?",
                  "And do you know who I am?"):
            self.assertIsNone(turn(f, c, "S1", q), q)
            f.on_agent_spoke("S1", LONG)
        self.assertEqual(f.engaged_with(), "S1")
        self.assertIsNone(turn(f, c, "S1", "Oh, my name's Sam."))

    def test_unlabelled_turn_is_not_another_person(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        for spk, text in ((None, "hey fae can you hear me alright, it's been a while honestly"),
                          ("S1", "so what have you been thinking about today, anything good going on with you"),
                          ("S1", "and what do you think about the new update they pushed this morning")):
            f.pacing_turn(spk, text)
        self.assertEqual(f.pacing.snapshot()["humans"], 1)

    def test_intro_after_agent_spoke_passes_pacing_in_a_busy_room(self):
        c = Clock()
        f = ConversationFloor(clock=c)
        chatter = "we were just talking about the game and the boss fight last night honestly"
        for _ in range(6):
            for spk in ("S1", "S2", "S4"):
                turn(f, c, spk, chatter, dt=2)
            f.on_agent_spoke("S2", LONG)
        self.assertIsNone(turn(f, c, "S5", "Hey, I'm Riley by the way.", dt=3))
        c.t += 150
        self.assertIsNotNone(f.pacing_gate("I'm Riley.", "S5", NAMES))   # long after it spoke: no exemption


if __name__ == "__main__":
    unittest.main()
