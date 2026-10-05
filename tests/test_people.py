import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from people import NameBook


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def book():
    c = Clock()
    return NameBook(agent_names=("max", "ivy"), store=None, clock=c), c


class Learn(unittest.TestCase):
    def test_intros(self):
        for line, want in [("my name is Jake", "Jake"), ("my name's jake by the way", "Jake"),
                           ("you can call me Dre", "Dre"), ("Hey guys, I'm Maya", "Maya"),
                           ("this is Chris", "Chris"), ("It's not Jake, it's Jack.", "Jack")]:
            b, _ = book()
            self.assertEqual(b.learn("S3", "[S3] " + line), want, line)

    def test_not_names(self):
        b, _ = book()
        for line in ["I'm good bro", "this is crazy", "I'm starving", "it's lit", "I'm so tired of this game",
                     "Yeah.", "What are you a woman?", "Hey Max, I'm back", "I'm Max"]:
            self.assertIsNone(b.learn("S3", line), line)
        self.assertIsNone(b.name_of("S3"))

    def test_answer_after_ask(self):
        b, c = book()
        b.learn("S4", "Max what's up")
        b.note_agent_reply("S4", "Not much. Wait, who's this?")
        c.t += 5
        self.assertEqual(b.learn("S4", "Uh it's Marcus."), "Marcus")
        self.assertEqual(b.display("S4"), "Marcus")

    def test_answer_window_expires(self):
        b, c = book()
        b.note_agent_reply("S4", "What's your name?")
        c.t += 120
        self.assertIsNone(b.learn("S4", "Pizza."))

    def test_refusal_stops_asking(self):
        b, c = book()
        b.note_agent_reply("S4", "Who am I talking to?")
        c.t += 3
        self.assertIsNone(b.learn("S4", "Not telling you"))
        self.assertIn("won't say", b.who_note("S4"))

    def test_forget(self):
        b, _ = book()
        b.learn("S2", "call me Dre")
        self.assertEqual(b.name_of("S2"), "Dre")
        b.learn("S2", "don't call me Dre")
        self.assertIsNone(b.name_of("S2"))

    def test_one_name_one_voice(self):
        b, _ = book()
        b.learn("S2", "my name is Jake")
        b.learn("S5", "no, I'm Jake")  # bare I'm-intro (<=3 words)
        self.assertEqual(b.name_of("S5"), "Jake")
        self.assertIsNone(b.name_of("S2"))

    def test_heard_vocatives(self):
        b, _ = book()
        b.learn("S1", "Maya, pass me the charger.")
        self.assertIn("Maya", b.who_note("S1"))


class Speak(unittest.TestCase):
    def test_sanitize(self):
        b, _ = book()
        b.learn("S2", "my name is Maya")
        b.learn("S3", "lol")
        self.assertEqual(b.sanitize("S2 said it needs two days", "S3"), "Maya said it needs two days")
        self.assertEqual(b.sanitize("Hey S3, relax.", "S3"), "Hey you, relax.")
        self.assertEqual(b.sanitize("Speaker S3 is wild", "S1"), "someone is wild")
        self.assertEqual(b.sanitize("the Galaxy S23 is fine", "S3"), "the Galaxy S23 is fine")

    def test_ask_cue(self):
        b, _ = book()
        class R: addressed_agent = 2; turns = 3
        note = b.who_note("S4", roster={"S4": R()})
        self.assertIn("never say an s-tag", note.lower())
        self.assertIn("ask for their name", note)
        b.note_agent_reply("S4", "and you are?")
        self.assertNotIn("ask for their name", b.who_note("S4", roster={"S4": R()}))


if __name__ == "__main__":
    unittest.main()


class LeadingIntroTest(__import__("unittest").TestCase):
    """sim_soul_conversation 10-04: 'hey Max, I'm Sam. how's it going' learned nothing,
    so Max told Sam he didn't know his name 12 turns later."""
    CASES = {"hey Max, I'm Sam. how's it going": "Sam", "I'm Sam. how's it going": "Sam",
             "hey Max, I'm Sam": "Sam", "Max, I'm Sam": "Sam",
             "I'm tired. let's go": None, "hey Max, I'm hungry": None, "I'm good. you?": None,
             "I'm playing Valorant. it's fun": None, "hey Sam, I'm Dana": None}

    def test_cases(self):
        from people import NameBook
        for text, exp in self.CASES.items():
            b = NameBook(agent_names=("max", "ivy"), store=None)
            b.learn("S1", text)
            self.assertEqual(b.name_of("S1"), exp, text)


class ClosingClauseIntroTest(__import__("unittest").TestCase):
    """Demo 10-04: 'Hey Wren, you there? It's Sam.' was never learned, so 'what's my name?' failed."""
    def test_cases(self):
        import people
        cases = {"Hey Ran, you there, it's Sam.": "Sam", "Hey Wren, you there? It's Sam.": "Sam",
                 "Hi, it's Dana.": "Dana", "what are we playing? it's Minecraft.": None,
                 "honestly, it's Tuesday.": None, "yeah it's fine.": None, "ok this is crazy": None}
        for i, (t, want) in enumerate(cases.items()):
            nb = people.NameBook()
            nb.agent_names = set(nb.agent_names) | {"wren", "ran"}
            self.assertEqual(nb.learn("S%d" % (i + 1), t), want, t)


class GreetingSentenceThenIntroTest(__import__("unittest").TestCase):
    """Demo take 5 (10-05): 'Hi, Wren. I'm Riley. We're ...' left Riley as Guest 2."""
    def check(self, text, want):
        from people import NameBook
        nb = NameBook(agent_names=("wren",), store=None)
        self.assertEqual(nb.learn("S2", text), want, text)

    def test_cases(self):
        self.check("Hi, Wren. I'm Riley. We're trying to pick a game for tonight.", "Riley")
        self.check("Hey Wren! I'm Dana.", "Dana")
        self.check("Hi, Wren. How's it going?", None)
        self.check("Hi, Wren. I'm tired. Long day.", None)
        self.check("Hi Wren. I'm so done with work today honestly.", None)


class GreetingUnknownAgentTest(__import__("unittest").TestCase):
    """Demo take 10 (10-05): agent created after startup wasn't in agent_names,
    so 'Hi Wren. I'm Riley.' learned nothing."""
    def test_lone_greeting_skipped_without_agent_names(self):
        import tempfile
        from people import NameBook
        b = NameBook(agent_names=("fae",), store=tempfile.mktemp())
        b.agent_names |= {"wren"}  # what set_persona now does for agents made after startup
        self.assertEqual(b.learn("S2", "Hi Wren, I'm Riley. We're trying to pick a game for tonight."), "Riley")
        b2 = NameBook(agent_names=(), store=tempfile.mktemp())
        self.assertEqual(b2.learn("S2", "Hi Wren. I'm Riley. We're picking a game."), "Riley")
