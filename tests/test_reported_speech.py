import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_dynamics import pre_llm_veto

N = ("max",)


class ReportedSpeechTests(unittest.TestCase):
    def test_reported_invocations_hold(self):
        for t in ["Ben said Max tell us a joke yesterday and it was bad",
                  "[S1] she was like Max shut up lol",
                  "so then Ben goes Max roast him",
                  "Ben said Max, tell us a joke"]:
            self.assertEqual(pre_llm_veto(t, N), "reported speech", t)

    def test_direct_address_still_reaches_llm(self):
        for t in ["I said Max are you there",
                  "we said Max pick a movie",
                  "Max, Ben said you were funny",
                  "you said Max was dumb",
                  "Max are you there?",
                  "my mom asked what Max thinks. Max what do you think",
                  "Ben said Max is funny, Max is that true"]:
            self.assertIsNone(pre_llm_veto(t, N), t)

    def test_reported_silence_does_not_become_silence_request(self):
        # must not trip quiet mode via the silence path
        self.assertEqual(pre_llm_veto("she was like Max shut up", N), "reported speech")
        self.assertEqual(pre_llm_veto("Max shut up", N), "explicit silence request")


if __name__ == "__main__":
    unittest.main()
