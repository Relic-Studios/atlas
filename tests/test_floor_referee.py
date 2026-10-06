"""Floor referee plugin (owner 10-05): talk-time balance, timeboxed topics, fair sides."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import call_summary as CS  # noqa: E402
import floor_referee as F  # noqa: E402

NOW = 1_800_000_000.0
NAMES = {"S1": "Sam", "S2": "Riley", "S3": "Jordan"}


def name_of(lbl):
    return NAMES.get(lbl)


class Base(unittest.TestCase):
    def setUp(self):
        CS.clear()

    def say(self, who, text, ago_min, agent=False):
        CS.note(who, text, agent=agent, clock=lambda: NOW - ago_min * 60)

    def clk(self):
        return NOW


class Stats(Base):
    def test_shares_and_names(self):
        self.say("S1", "one two three four five six seven eight", 3)
        self.say("S2", "one two", 2)
        self.say("Fae", "agent words never count here at all", 1, agent=True)
        s = F.stats(15, name_of, self.clk)
        self.assertEqual(set(s), {"Sam", "Riley"})
        self.assertAlmostEqual(s["Sam"]["share"], 0.8)
        out = F.floor_stats(15, name_of, self.clk)
        self.assertIn("Sam 80%", out)
        self.assertIn("never shame", out)

    def test_empty(self):
        self.assertIn("No one has talked", F.floor_stats(15, name_of, self.clk))

    def test_bad_minutes(self):
        self.say("S1", "hello there", 1)
        self.assertIn("15 minutes", F.floor_stats("lots", name_of, self.clk))


class Quiet(Base):
    def fill(self):
        long = " ".join(["word"] * 60)
        self.say("S3", "I think we should go north", 9)
        self.say("S1", long, 3)
        self.say("S2", long, 1)

    def test_quiet_person_named(self):
        self.fill()
        n = F.quiet_note("S1", name_of, clock=self.clk)
        self.assertIn("Jordan", n)
        self.assertIn("Don't force", n)

    def test_not_for_the_speaker(self):
        self.fill()
        self.assertEqual(F.quiet_note("S3", name_of, clock=self.clk), "")

    def test_needs_three_people_and_real_talk(self):
        self.say("S1", "hi", 8)
        self.say("S2", "hey", 1)
        self.assertEqual(F.quiet_note("S2", name_of, clock=self.clk), "")

    def test_unnamed_labels_not_suggested(self):
        long = " ".join(["word"] * 60)
        self.say("S9", "quiet unnamed", 9)
        self.say("S1", long, 3)
        self.say("S2", long, 1)
        self.assertEqual(F.quiet_note("S1", name_of, clock=self.clk), "")


class Sides(Base):
    def test_grouped_by_person_and_focus(self):
        self.say("S1", "pineapple pizza is great", 4)
        self.say("S2", "pineapple on pizza is a crime", 3)
        self.say("S3", "what time is dinner", 2)
        out = F.debate_sides(15, "pineapple", name_of, self.clk)
        self.assertIn("Sam: pineapple pizza is great", out)
        self.assertIn("Riley: pineapple on pizza is a crime", out)
        self.assertNotIn("dinner", out)
        self.assertIn("Don't pick a winner", out)

    def test_nothing(self):
        self.assertIn("don't invent", F.debate_sides(15, "", name_of, self.clk))


class Topic(unittest.TestCase):
    def setUp(self):
        import room_tools as RT
        self.RT = RT
        self.tmp = tempfile.mkdtemp()
        self._old = RT.STATE_PATH
        RT.STATE_PATH = Path(self.tmp) / "room.json"

    def tearDown(self):
        self.RT.STATE_PATH = self._old

    def test_timebox_sets_a_timer(self):
        out = F.start_topic("the budget", 5, "S1")
        self.assertIn("Timeboxed", out)
        self.assertIn("time's up on the budget", self.RT.list_timers())

    def test_clamped(self):
        self.assertIn("60 minutes", F.start_topic("x", 999))


class Intent(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(F.intent("[S1] Max, who has been talking the most?"), ("floor_stats", {}))
        self.assertEqual(F.intent("am I talking too much?"), ("floor_stats", {}))
        self.assertEqual(F.intent("Fae, give us 5 minutes on the budget."),
                         ("start_topic", {"topic": "budget", "minutes": 5}))
        self.assertEqual(F.intent("give us ten minutes to talk about the trip"),
                         ("start_topic", {"topic": "trip", "minutes": 10}))
        self.assertEqual(F.intent("sum up both sides on pineapple pizza"),
                         ("debate_sides", {"topic": "pineapple pizza"}))

    def test_negative(self):
        for s in ["I talked to my mom", "sum up the movie", "give me five minutes to grab food",
                  "give me a minute", "I need 5 minutes for dinner"]:
            self.assertIsNone(F.intent(s), s)


class Wiring(unittest.TestCase):
    def test_plugin_and_tools(self):
        import plugins
        import room_tools
        p = next(x for x in plugins.BUILTIN if x["id"] == "floor_referee")
        self.assertFalse(p["default"])
        for t in p["tools"]:
            self.assertIn(t, room_tools.NAMES)
        self.assertEqual(room_tools.intent("who's been talking the most?")[0], "floor_stats")


if __name__ == "__main__":
    unittest.main()
