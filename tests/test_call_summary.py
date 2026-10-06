"""Call summary plugin (owner-approved 10-05): 'what did we decide?' from real lines."""
import unittest
import call_summary as C
import room_tools as R
import plugins


class Recap(unittest.TestCase):
    def setUp(self):
        C.clear()
        self.now = 10_000.0
        self.clk = lambda: self.now

    def test_recap_uses_real_lines_and_names(self):
        C.note("S1", "ok lets meet at 9 for the raid", clock=lambda: self.now - 120)
        C.note("S2", "I'll bring snacks", clock=lambda: self.now - 60)
        C.note("fae", "Got it, raid at 9.", agent=True, clock=lambda: self.now - 30)
        out = C.recap(10, name_of=lambda s: {"S1": "Sam"}.get(s), clock=self.clk)
        self.assertIn("[Sam] ok lets meet at 9", out)
        self.assertIn("[someone] I'll bring snacks", out)    # raw S-tags never reach the model
        self.assertIn("[You (fae)]", out)
        self.assertNotIn("S2", out)

    def test_window(self):
        C.note("S1", "old line", clock=lambda: self.now - 3600)
        C.note("S1", "new line", clock=lambda: self.now - 60)
        out = C.recap(10, clock=self.clk)
        self.assertIn("new line", out)
        self.assertNotIn("old line", out)

    def test_empty_is_honest(self):
        self.assertIn("No conversation recorded", C.recap(5, clock=self.clk))

    def test_long_call_keeps_decisions_and_bounds_size(self):
        for i in range(600):
            C.note("S1", f"random chatter number {i} about nothing much at all really", clock=lambda i=i: self.now - 1800 + i)
        C.note("S2", "we decided the raid is at 9 tomorrow", clock=lambda: self.now - 1500)
        out = C.recap(60, clock=self.clk)
        self.assertIn("raid is at 9", out)
        self.assertLess(len(out), C.MAX_CHARS + 600)

    def test_bad_minutes(self):
        C.note("S1", "hi", clock=lambda: self.now - 5)
        self.assertIn("hi", C.recap("abc", clock=self.clk))


class Intent(unittest.TestCase):
    def test_positive(self):
        for s, m in [("Fae, what did we decide?", 10), ("recap the last 20 minutes", 20),
                     ("what did I miss", 10), ("sum it up for me", 10), ("give us a quick recap", 10)]:
            r = R.intent(s)
            self.assertEqual(r and r[0], "recap_call", s)
            self.assertEqual(r[1]["minutes"], m, s)

    def test_focus(self):
        self.assertEqual(C.intent("what did we say about the boss fight")[1]["focus"], "the boss fight")

    def test_negative(self):
        for s in ["summarize this article for me later lol", "recap of the movie was great",
                  "what did you say?", "I love summer"]:
            self.assertIsNone(C.intent(s), s)

    def test_execute_through_room_tools(self):
        C.clear()
        C.note("S1", "plan: tacos at 7")
        out = R.execute("recap_call", {"minutes": 5}, name_of=lambda s: "Riley")
        self.assertIn("[Riley] plan: tacos at 7", out)


class Plugin(unittest.TestCase):
    def test_registered_with_switch(self):
        p = [x for x in plugins.BUILTIN if x["id"] == "call_summary"]
        self.assertEqual(len(p), 1)
        self.assertIn("recap_call", p[0]["tools"])
        self.assertIn("recap_call", R.NAMES)


if __name__ == "__main__":
    unittest.main()
