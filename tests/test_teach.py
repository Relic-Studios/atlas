"""Teach-me mode plugin (owner 10-05): per-agent lessons taught out loud."""
import tempfile, unittest
from pathlib import Path
import teach as T


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())


class Lessons(Base):
    def test_learn_note_recall_forget(self):
        self.assertIn("Learned", T.learn("fae", "league scoring", "win 3 points, draw 1", "Sam", root=self.root))
        self.assertIn("draw 1", T.note("fae", "[S1] how many points for a draw in our league", root=self.root))
        self.assertEqual("", T.note("fae", "[S1] what time is it", root=self.root))
        self.assertIn("win 3", T.recall("fae", "scoring", root=self.root))
        self.assertIn("Forgot", T.forget("fae", "league", root=self.root))
        self.assertIn("haven't been taught", T.recall("fae", root=self.root))

    def test_per_agent_isolation(self):
        T.learn("fae", "house rule", "no spoilers before Sunday", root=self.root)
        self.assertEqual("", T.note("pup", "any house rule about spoilers", root=self.root))
        self.assertIn("haven't been taught", T.recall("pup", root=self.root))

    def test_update_same_topic(self):
        T.learn("fae", "league scoring", "win 3", root=self.root)
        self.assertIn("Updated", T.learn("fae", "League Scoring", "win 2", root=self.root))
        self.assertIn("win 2", T.recall("fae", "league scoring", root=self.root))

    def test_private_refused(self):
        self.assertIn("Not saved", T.learn("fae", "contacts", "call me at 555-123-4567", root=self.root))
        self.assertIn("haven't been taught", T.recall("fae", root=self.root))

    def test_forget_since(self):
        T.learn("fae", "old", "old rule here", root=self.root, clock=lambda: 100.0)
        T.learn("fae", "new", "new rule here", root=self.root, clock=lambda: 200.0)
        self.assertEqual(1, T.forget_since("fae", 150.0, root=self.root))
        self.assertIn("old", T.recall("fae", root=self.root))
        self.assertEqual(1, T.forget_since("fae", 0, root=self.root))


class Intent(unittest.TestCase):
    def test_teach(self):
        i = T.intent("[S1] Fae, let me teach you how our league scoring works: a win is 3 points.")
        self.assertEqual(i[0], "learn_lesson"); self.assertEqual(i[1]["topic"], "league scoring")
        self.assertEqual(T.intent("here's how Ricochet works: two bounces max.")[1]["topic"], "Ricochet")

    def test_recall(self):
        self.assertEqual(T.intent("what did we teach you about league scoring?"),
                         ("recall_lessons", {"topic": "league scoring"}))

    def test_not_teach(self):
        for s in ("I teach math at the high school", "let me teach you a lesson buddy",
                  "here is my plan for tonight"):
            self.assertIsNone(T.intent(s), s)


class Wiring(unittest.TestCase):
    def test_tools_and_plugin(self):
        import room_tools, plugins
        self.assertTrue({"learn_lesson", "recall_lessons", "forget_lesson"} <= room_tools.NAMES)
        self.assertIn("teach", {p["id"] for p in plugins.BUILTIN})
        self.assertEqual(room_tools.intent("let me teach you the dice rule: sixes reroll")[0], "learn_lesson")


if __name__ == "__main__":
    unittest.main()
