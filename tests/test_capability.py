"""Capability honesty (live 09-29: Max on Remote box said "I can't take screenshots")."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capability as C


class WantsLookTests(unittest.TestCase):
    def test_live_requests(self):
        for t in ["Max, can you take a screenshot?", "look at my screen", "what's on my screen",
                  "can you see my screen", "Ivy what do you see", "look at this"]:
            self.assertTrue(C.wants_look(t), t)

    def test_normal_speech(self):
        for t in ["look at this guy talking lol", "I see what you mean", "the screen door broke",
                  "can you believe that", "let me look it up", "check this out later maybe"]:
            self.assertFalse(C.wants_look(t), t)


class RefusalScrubTests(unittest.TestCase):
    H = [{"role": "user", "content": "[S1] take a screenshot"},
         {"role": "assistant", "content": "[SPEAK to=S1] No, I can't see your screen. But tell me what's up."},
         {"role": "assistant", "content": "[SPEAK to=S1] I don't have access to your screen."},
         {"role": "assistant", "content": "[SPEAK to=S1] Pizza, obviously."}]

    def test_scrub_when_on(self):
        out = C.scrub_refusals(self.H, eyes_on=True)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1]["content"], "[SPEAK to=S1] But tell me what's up.")
        self.assertEqual(out[2]["content"], "[SPEAK to=S1] Pizza, obviously.")

    def test_keep_when_off(self):
        # Eyes off: "I can't see your screen" is TRUE, so it stays.
        self.assertEqual(C.scrub_refusals(self.H, eyes_on=False, search_on=False), self.H)

    def test_tool_refusal(self):
        self.assertTrue(C.is_false_refusal("I don't have access to the internet.", False, True))
        self.assertFalse(C.is_false_refusal("I can't believe you did that.", True, True))
        self.assertFalse(C.is_false_refusal("I can't see why you'd do that.", True, True))


class NoteTests(unittest.TestCase):
    def test_look_note_modes(self):
        self.assertIn("fresh screenshot", C.look_note("take a screenshot", True))
        self.assertIn("switched off", C.look_note("take a screenshot", False))
        self.assertEqual(C.look_note("hello there", True), "")

    def test_abilities(self):
        self.assertIn("look at", C.abilities_note(True))
        self.assertNotIn("look at", C.abilities_note(False))


if __name__ == "__main__":
    unittest.main()


class SelfKnowledgeTests(unittest.TestCase):
    def test_fires_on_how_it_works(self):
        import capability as C
        for t in ["[S1] Max, how do you work?", "who made you", "what model are you",
                  "are you chatgpt", "what's your gpu", "how do you hear me"]:
            self.assertTrue(C.self_note(t), t)

    def test_quiet_on_ordinary_lines(self):
        import capability as C
        for t in ["pass the salt", "how do you know that", "how do you like pizza",
                  "I work at a model agency", "what is your favorite model car"]:
            self.assertEqual(C.self_note(t), "", t)

    def test_no_secrets_in_block(self):
        import capability as C
        s = C.SELF_KNOWLEDGE
        for bad in ("192.168", "30000", "private/", "C:/", ".key"):
            self.assertNotIn(bad, s)


class MemorySelfQuestionTest(__import__("unittest").TestCase):
    """Owner 10-03: Fae denied having a hypergraph memory because memory questions never
    triggered the HOW YOU WORK facts. Memory questions about HERSELF must; ordinary recall
    questions must not (those are handled by the never-invent-the-past rule)."""
    def test_memory_questions_trigger(self):
        import capability as C
        for t in ["Fae, do you have a hypergraph memory?", "how does your memory work",
                  "do you remember stuff between calls?", "can the other bots see your memories?",
                  "if I tell you my phone number will you remember it?", "do you have long-term memory",
                  "how do you remember things"]:
            self.assertTrue(C._SELF_Q.search(t), t)

    def test_plain_recall_does_not_trigger(self):
        import capability as C
        for t in ["do you remember my dog's name", "remember what we talked about last week?",
                  "I have a terrible memory lol", "that was a memorable game", "what did I tell you yesterday"]:
            self.assertFalse(C._SELF_Q.search(t), t)


class CharacterIdentity(__import__("unittest").TestCase):
    """Owner 10-05 option (b): character agents may play in their own world; plain agents stay strict."""
    def test_split(self):
        import capability as C
        from unittest import mock
        with mock.patch("agent_registry.traits", return_value={"character_agents": ["pup"]}):
            self.assertIn("playing a character", C.identity_truth("pup"))
            self.assertIn("no body", C.identity_truth("wren"))
            self.assertNotIn("{BODY}", C.identity_truth("pup") + C.identity_truth(""))
            self.assertIn("say plainly you're an AI", C.identity_truth("Pup"))
