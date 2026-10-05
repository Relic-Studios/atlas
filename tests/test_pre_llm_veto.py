"""Production pre-LLM gate: precision matters more than recall (a false HOLD on
a direct address is the 'he's not responding' bug). Everything ambiguous must
fall through to the LLM (None)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_dynamics import pre_llm_veto

NAMES = ("max",)
HOLD = [
    "Can you pass the salt, Jordan?", "Oliver, could you", "Maya, did you bring your charger?",
    "[S3] S2, can you send me those files?", "What do you think, Sarah?", "[S1] Mike, grab the door.",
    "Max, stop talking for a moment.", "Shut up Max.", "Be quiet.", "Alex, be quiet.",
]
PASS = [
    "Max are you there?", "Tomas, what do you think?", "Max, can you",
    "What would you choose, Max?", "Alex, be quiet. Max, keep explaining.",
    'Max, translate "please stop talking" into French.', 'She said "shut up" to me, can you believe it?',
    "Honestly, I think that's fine.", "Okay, what do you want to talk about?", "Dude, I don't want to sit.",
    "Look, it's not that deep.", "Seriously, how are you?", "Did you see what she did?",
    "Is that right, man?", "Wait, what did you say?", "Listen, can you help me?", "Can you help me, bro?",
    "Basically, I'm moving to Denver.", "So, what's the plan?", "Hey robot, can you hear me?",
    "I'm not babe.", "Friday would be best.", "Sorry, what?", "Yeah, no, she was fine.",
]

DEV_NAMES = (__import__("pathlib").Path(__file__).resolve().parents[1] / "dev_pack").exists()  # public export renames dev agents
class PreLlmVeto(unittest.TestCase):
    def test_holds(self):
        for t in HOLD:
            with self.subTest(t=t):
                self.assertIsNotNone(pre_llm_veto(t, NAMES))

    @__import__("unittest").skipUnless(DEV_NAMES, "uses dev agent-name misspellings")
    def test_never_blocks_direct_or_ambiguous(self):
        for t in PASS:
            with self.subTest(t=t):
                self.assertIsNone(pre_llm_veto(t, NAMES))

    def test_pack_nickname(self):
        # Nicknames come from the dev pack ("Tom" -> Max); public builds have none.
        from conversation_dynamics import _AGENT_NICKNAMES
        if not _AGENT_NICKNAMES.get("max", ()):
            self.skipTest("no pack nicknames in this build")
        self.assertIsNone(pre_llm_veto("Tom, you there?", NAMES))

    def test_follows_persona(self):
        self.assertIsNone(pre_llm_veto("Ivy, what do you think?", ("ivy",)))
        self.assertIsNotNone(pre_llm_veto("Ivy, what do you think?", ("max",)))

if __name__ == "__main__":
    unittest.main()
