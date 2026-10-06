"""Voice mail plugin: messages addressed to a voice, passed on when that person speaks."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import voice_mail as V  # noqa: E402
import room_tools as RT  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._path = V.STATE_PATH
        V.STATE_PATH = Path(self.tmp.name) / "voice_mail.json"
        V.forget_heard()

    def tearDown(self):
        V.STATE_PATH = self._path
        V.forget_heard()
        self.tmp.cleanup()


class Intent(unittest.TestCase):
    def test_requests(self):
        cases = {
            "Fae, tell Riley the raid moved to 9 when she joins.": ("Riley", "the raid moved to 9"),
            "[S1] Tell Riley that the raid moved to 9 when she gets here": ("Riley", "the raid moved to 9"),
            "Next time you hear Jordan, tell him he owes me five bucks.": ("Jordan", "he owes me five bucks"),
            "If Riley shows up, let her know we started without her.": ("Riley", "we started without her"),
            "Leave a message for Sam: bring the snacks": ("Sam", "bring the snacks"),
            "tell Riley when she joins that we moved": ("Riley", "we moved"),
        }
        for line, (to, msg) in cases.items():
            got = V.intent(line)
            self.assertEqual(got, ("leave_message", {"to": to, "message": msg}), line)

    def test_not_requests(self):
        for line in ["Tell me a joke", "Tell Riley I said hi", "Remind me when he talks",
                     "I told Riley about it when she joined yesterday", "tell us when it's ready"]:
            self.assertIsNone(V.intent(line), line)

    def test_room_tools_routes_it(self):
        self.assertEqual(RT.intent("Fae, tell Riley the raid moved to 9 when she joins.")[0],
                         "leave_message")


class Lifecycle(Base):
    def test_leave_heard_claim(self):
        out = V.leave("Riley", "the raid moved to 9", "Sam", agent_names=("Fae",))
        self.assertIn("Saved message #1 for Riley from Sam", out)
        self.assertIsNone(V.due())                      # Riley hasn't spoken
        V.heard("Sam", "S1")
        self.assertIsNone(V.due())                      # someone else spoke
        V.heard("riley", "S4")                          # case-insensitive match
        m = V.due()
        self.assertEqual((m["to"], m["label"]), ("Riley", "S4"))
        cue = V.claim(m["id"])
        self.assertTrue(cue.startswith("[S4] (MAIL CUE #1)"))
        self.assertIn("Sam left a message for Riley", cue)
        self.assertIn("the raid moved to 9", cue)
        self.assertTrue(RT.ROOM_CUE_RE.search(cue))     # pipeline treats it as our own cue
        self.assertIn("MESSAGE", RT.cue_note(cue))
        self.assertIsNone(V.due())                      # delivered once
        self.assertEqual(V.claim(m["id"]), "")

    def test_heard_must_be_recent(self):
        V.leave("Riley", "hi from Sam", "Sam")
        V.heard("Riley", "S4", clock=lambda: 0.0)       # long ago
        self.assertIsNone(V.due())

    def test_survives_restart_and_expires(self):
        V.leave("Riley", "raid moved", "Sam", clock=lambda: 1000.0)
        V.forget_heard()                                 # new session
        V.heard("Riley", "S2", clock=lambda: 1000.0 + 3 * 86400)
        self.assertIsNotNone(V.due(clock=lambda: 1000.0 + 3 * 86400 + 5))
        self.assertIsNone(V.due(days=2, clock=lambda: 1000.0 + 3 * 86400 + 5))

    def test_refusals(self):
        self.assertIn("That's me", V.leave("Fae", "hi", "Sam", agent_names=("Fae",)))
        self.assertIn("themselves", V.leave("Sam", "hi", "Sam"))
        self.assertIn("Not saved", V.leave("Riley", "call me at 555-123-4567", "Sam"))
        self.assertIn("what's the message", V.leave("Riley", "", "Sam"))
        self.assertEqual(V.list_messages(), "No messages waiting.")

    def test_self_label_ignored(self):
        V.leave("Riley", "raid moved", "Sam")
        V.heard("Riley", "self")
        self.assertIsNone(V.due())

    def test_list_and_cancel(self):
        V.leave("Riley", "raid moved to 9", "Sam")
        V.leave("Jordan", "bring the dice", "Sam")
        self.assertIn("#2 for Jordan", V.list_messages())
        self.assertIn("Cancelled message #1", V.cancel(1))
        self.assertIn("Cancelled message #2", V.cancel(words="Jordan"))
        self.assertEqual(V.cancel(5), "No such message.")

    def test_execute_via_room_tools(self):
        out = RT.execute("leave_message", {"to": "Riley", "message": "raid moved"}, "S1", "Sam",
                         settings=lambda pid: {"days": 3}, agent_names=("Fae",))
        self.assertIn("from Sam", out)
        self.assertIn("kept 3 days", out)
        self.assertIn("#1 for Riley", RT.execute("list_messages", {}))


class Plugin(unittest.TestCase):
    def test_registered(self):
        import plugins
        p = next(x for x in plugins.BUILTIN if x["id"] == "voice_mail")
        self.assertEqual(set(p["tools"]), {"leave_message", "list_messages", "cancel_message"})
        names = {t["function"]["name"] for t in RT.TOOLS_BY_PLUGIN["voice_mail"]}
        self.assertEqual(names, set(p["tools"]))


if __name__ == "__main__":
    unittest.main()
