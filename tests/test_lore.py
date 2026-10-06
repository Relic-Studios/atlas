"""Lore keeper plugin (owner 10-05): group canon, owner-approved, called back at the right moment."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lore as L  # noqa: E402

NOW = 1_800_000_000.0


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="atlas_lore_"))
        L._last_used.clear()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def add(self, agent="fae", title="Sam's roof incident",
            story="Sam fell off the roof trying to fetch a frisbee and called it a tactical retreat.",
            people=("Sam",), mode="owner"):
        return L.add(agent, title, story, people, "Riley", root=self.root, mode=mode, clock=lambda: NOW)

    def note(self, text, agent="fae", present=(), t=NOW):
        return L.note(agent, text, present, root=self.root, clock=lambda: t)


class Approval(Base):
    def test_pending_not_used_until_approved(self):
        out = self.add()
        self.assertIn("owner approves", out)
        self.assertEqual(self.note("remember when Sam was on the roof?"), "")
        self.assertIn("no lore", L.recall("fae", root=self.root).lower())
        item = L.feed(self.root)[0]
        self.assertEqual(item["agent"], "fae")
        self.assertEqual([a["act"] for a in item["actions"]], ["approve", "reject"])
        self.assertTrue(L.action("fae", item["id"], "approve", self.root))
        self.assertIn("ROOM LORE", self.note("remember when Sam was on the roof?"))

    def test_auto_mode_used_right_away(self):
        self.assertIn("Added", self.add(mode="auto"))
        self.assertIn("roof", self.note("Sam is climbing the roof again"))

    def test_rejected_hidden_and_not_used(self):
        self.add()
        L.action("fae", L.feed(self.root)[0]["id"], "reject", self.root)
        self.assertEqual(L.feed(self.root), [])
        self.assertEqual(self.note("the roof thing with Sam"), "")

    def test_bad_action_and_agent(self):
        with self.assertRaises(ValueError):
            L.action("fae", "1", "nuke", self.root)
        with self.assertRaises(ValueError):
            L.action("../x", "1", "approve", self.root)


class Callback(Base):
    def setUp(self):
        super().setUp()
        self.add(mode="auto")

    def test_unrelated_line_gets_nothing(self):
        self.assertEqual(self.note("what should we eat tonight?"), "")

    def test_cooldown_stops_repeats(self):
        self.assertTrue(self.note("Sam and that roof"))
        self.assertEqual(self.note("Sam and that roof again", t=NOW + 60), "")
        self.assertTrue(self.note("Sam and that roof once more", t=NOW + L.COOLDOWN_S + 5))

    def test_one_entry_per_turn(self):
        L.add("fae", "frisbee curse", "Every frisbee Sam touches ends up on a roof.", ["Sam"], "",
              root=self.root, mode="auto", clock=lambda: NOW)
        n = self.note("Sam frisbee roof")
        self.assertEqual(n.count("ROOM LORE"), 1)

    def test_per_agent(self):
        self.assertEqual(self.note("Sam and the roof", agent="pup"), "")

    def test_uses_counted(self):
        self.note("the roof with Sam")
        self.assertEqual(L._load("fae", self.root)[0]["uses"], 1)


class Safety(Base):
    def test_private_refused(self):
        out = L.add("fae", "Sam's number", "Call Sam at 555-123-4567 for pizza", [], "",
                    root=self.root, mode="auto")
        self.assertIn("Not kept", out)
        self.assertEqual(L._load("fae", self.root), [])

    def test_speaker_tags_not_stored_as_people(self):
        L.add("fae", "x", "the long night of the cursed dice", ["S3", "Riley"], "",
              root=self.root, mode="auto")
        self.assertEqual(L._load("fae", self.root)[0]["people"], ["Riley"])

    def test_wipe_since(self):
        self.add(mode="auto")
        L.add("fae", "later", "the cursed dice night", [], "", root=self.root, mode="auto",
              clock=lambda: NOW + 100)
        self.assertEqual(L.forget_since("fae", NOW + 50, self.root), 1)
        self.assertEqual(L.forget_since("fae", 0, self.root), 1)

    def test_same_title_updates(self):
        self.add(mode="auto")
        out = self.add(story="Sam fell off the roof twice now.", mode="auto")
        self.assertIn("Updated", out)
        self.assertEqual(len(L._load("fae", self.root)), 1)

    def test_forget(self):
        self.add(mode="auto")
        self.assertIn("Dropped", L.forget("fae", "roof", root=self.root))
        self.assertEqual(L._load("fae", self.root), [])


class Intent(unittest.TestCase):
    def test_add(self):
        self.assertEqual(L.intent("[S1] Fae, that's canon: Riley is banned from choosing the map")[0], "add_lore")
        self.assertEqual(L.intent("add that to the lore - Jordan cried at the Pixar movie")[0], "add_lore")

    def test_recall(self):
        self.assertEqual(L.intent("Fae what's our lore about Sam?"), ("recall_lore", {"topic": "Sam"}))

    def test_no_false_positives(self):
        for s in ["I love the lore in Elden Ring", "that is canon in the books", "what's the plan tonight",
                  "the lore of Zelda is deep"]:
            self.assertIsNone(L.intent(s), s)


class Wiring(unittest.TestCase):
    def test_plugin_and_tools(self):
        import plugins
        import room_tools
        p = [x for x in plugins.BUILTIN if x["id"] == "lore"][0]
        self.assertEqual(set(p["tools"]), {"add_lore", "recall_lore", "forget_lore"})
        self.assertTrue({"add_lore", "recall_lore", "forget_lore"} <= room_tools.NAMES)

    def test_execute_add(self):
        import room_tools
        root = Path(tempfile.mkdtemp(prefix="atlas_lore_x_"))
        try:
            import agent_memory
            old = agent_memory.ROOT
            agent_memory.ROOT = root
            try:
                out = room_tools.execute("add_lore", {"title": "map ban", "story": "Riley may never pick the map",
                                                      "people": "Riley, S2"}, voter_name="Sam", agent="fae")
            finally:
                agent_memory.ROOT = old
            self.assertIn("map ban", out)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
