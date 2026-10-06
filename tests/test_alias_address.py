"""Address to someone else by handle/alias (live 10-06: "Hey Perma, I think you
were muted before. Did you, are you back?" -> Max: "Yeah, I'm back.").

People in group calls go by gamer tags and nicknames, often after a greeting,
sometimes lower-case in the transcript. These lines must be held; lines aimed
at the agent or the whole room must still reach it.
"""
import unittest

import conversation_dynamics as C

N = ("max",)

TO_OTHER = [
    "Hey Perma, I think you were muted before. Did you, are you back?",
    "Perma, are you back?",
    "Yo Kraken did you get it?",
    "Hey Sam, are you back?",
    "Yo xXSniper, you there?",
    "Ayy JJ, did you eat?",
    "Oh Kraken you scared me",
    "I interrupted him, and now I've heard, Perma, you were talking about your brain.",
    "Okay Nerd, you good?",
    "Wait, Ghost, are you muted?",
    "Hey Relic you there",
    "Good game. Hey Perma, you coming back?",
    "Kraken, your mic is peaking.",
    "Hey Narvi",
    "So, Kraken, you coming?",
    "Yo Ivy, did you know that?",
    "Hey Nebi, your mic is shit.",
    "Hey Taco are you back",
]
# A lower-case handle can only be told from a noun with wordfreq installed
# (full install). Without it (light CI) it is deliberately treated as a noun.
if C._zipf:
    TO_OTHER.append("hey perma, you back?")
else:
    TO_AGENT_OR_ROOM_FALLBACK = ["hey perma, you back?"]

TO_AGENT_OR_ROOM = [
    "Hey Max, are you back?",
    "hey max you there",
    "Max, are you back?",
    "Are you back?",
    "You there?",
    "Hey, are you back?",
    "So pizza, you in?",
    "Okay so, are you guys ready?",
    "Hey guys, you ready?",
    "Hey man, you good?",
    "Wait what, are you serious?",
    "Okay cool, you good?",
    "I went to Paris, France, you should go sometime.",
    "Hey everyone, you hear that?",
    "Yo, what do you think?",
    "Alright alright, you win.",
    "Hey Perma and Max, you both good?",
    "Did you see what she did?",
    "Will you remember this?",
    "Yo, AI! What's up?",
    "Hey Atlas, you there?",
    "And Dobby will pass his house elf onto you, my G.",
    "Wait, so Verity is a part of a Minecraft mod?",
    "Dude, SF is like, you can find anything there.",
    "Well, voila!",
    "Yo, weee!",
]


class AliasAddress(unittest.TestCase):
    def setUp(self):
        C._ROOM_NAMES.clear()

    def tearDown(self):
        C._ROOM_NAMES.clear()

    def test_handles_to_someone_else_are_held(self):
        for t in TO_OTHER:
            with self.subTest(t=t):
                r = C.pre_llm_veto(t, N)
                self.assertTrue(r and r.startswith("addressed"), (t, r))

    def test_agent_and_room_lines_pass(self):
        for t in TO_AGENT_OR_ROOM:
            with self.subTest(t=t):
                r = C.pre_llm_veto(t, N)
                self.assertFalse(r and r.startswith("addressed"), (t, r))

    def test_room_names_trusted_even_when_common_words(self):
        # "Ghost"/"Will" are ordinary words; once the room uses them as names,
        # greeting/comma address to them is held.
        self.assertIsNone(C.pre_llm_veto("Hey will, you back?", N))
        C.note_room_name("Will")
        C.note_room_name("Ghost")
        self.assertTrue(C.pre_llm_veto("Hey Will, you back?", N))
        self.assertTrue(C.pre_llm_veto("Ghost, you there?", N))
        # ...but a comma-less "Will you ..." is still a question to the room.
        self.assertIsNone(C.pre_llm_veto("Will you remember this?", N))

    def test_namebook_feeds_room_names(self):
        import people
        nb = people.NameBook() if not hasattr(people, "NameBook") else people.NameBook()
        nb.learn("S3", "[S3] Hey, I'm Ghost.")
        self.assertIn("ghost", C._ROOM_NAMES)


if __name__ == "__main__":
    unittest.main()
