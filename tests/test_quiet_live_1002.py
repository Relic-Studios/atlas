"""Live 10-02 Fae: 'speak when you're spoken to' ignored; long 'shut up' from the
partner never armed quiet mode; her own 'I'll stay quiet' promise wasn't kept."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import conversation_dynamics as C, floor as F

N = ("fae",)


class QuietLive1002(unittest.TestCase):
    def floor(self):
        fl = F.ConversationFloor(); fl.on_user_turn("S29"); fl.on_agent_spoke("S29")
        return fl

    def test_spoken_to_phrasings(self):
        for t in ["Speak when you're spoken to, please.", "Only talk when you're talked to.",
                  "Don't speak until you're spoken to.", "Fae, wait until you're spoken to."]:
            self.assertTrue(C.strong_silence_request(t), t)
        for t in ["My mom always said speak when you're spoken to.", "when you're spoken to it's polite"]:
            self.assertFalse(C.strong_silence_request(t), t)

    def test_partner_long_shut_up_arms_quiet(self):
        fl = self.floor()
        fl.turn_gate("[S29] Now you shut up, please until you were spoken to. Okay, thank you. Okay, Relic.", "S29", N)
        self.assertTrue(fl.is_quiet())
        self.assertEqual(fl.turn_gate("[S29] How.", "S29", N), "quiet mode")
        self.assertIsNone(fl.turn_gate("[S29] Fae, how did you start?", "S29", N))

    def test_ramble_with_shut_up_late_does_not_arm(self):
        fl = self.floor()
        fl.turn_gate("[S29] so I was at the store and my brother kept going on and on and I said bro shut up", "S29", N)
        self.assertFalse(fl.is_quiet())

    def test_agent_promise(self):
        self.assertTrue(F.promises_quiet("Noted. I'll stay quiet until someone says my name."))
        self.assertTrue(F.promises_quiet("Okay, I'll only speak when I'm spoken to."))
        for r in ["I'll stay with you.", "You'll stay quiet?", "Quiet mode is cool."]:
            self.assertFalse(F.promises_quiet(r), r)


if __name__ == "__main__":
    unittest.main()
