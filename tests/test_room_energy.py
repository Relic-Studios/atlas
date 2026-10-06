"""Room-energy sense: mood from the in-memory transcript; tone only, never speak/hold."""
import unittest

import call_summary
import room_energy as RE

T0 = 1_000_000.0


def feed(lines, gap=4.0):
    call_summary.clear()
    t = T0
    for who, text in lines:
        call_summary.note(who, text, agent=(who == "AGENT"), clock=lambda t=t: t)
        t += gap
    return lambda: t


class Moods(unittest.TestCase):
    def tearDown(self):
        call_summary.clear()

    def test_quiet_when_few_lines(self):
        clk = feed([("S1", "hey"), ("S2", "hi")])
        self.assertEqual(RE.read(clock=clk)["mood"], "quiet")
        self.assertEqual(RE.note(clock=clk), "")

    def test_calm_small_talk(self):
        clk = feed([("S1", "so what are you working on this week"), ("S2", "mostly the garden"),
                    ("S1", "nice, tomatoes again?"), ("S2", "yeah and some peppers"),
                    ("S3", "I tried peppers once"), ("S1", "how did that go")])
        self.assertEqual(RE.read(clock=clk)["mood"], "calm")
        self.assertEqual(RE.note(clock=clk), "")

    def test_hype_laughter(self):
        clk = feed([("S1", "hahaha no way he did that"), ("S2", "LMAO dude"), ("S1", "let's go!"),
                    ("S3", "I'm crying"), ("S2", "that was clutch"), ("S1", "lol")], gap=2.0)
        r = RE.read(clock=clk)
        self.assertEqual(r["mood"], "hype")
        self.assertIn("playful", RE.note(clock=clk))

    def test_tense(self):
        clk = feed([("S1", "you never listen"), ("S2", "shut up, seriously"), ("S1", "no YOU stop"),
                    ("S2", "I'm so done with this"), ("S1", "whatever, man"), ("S2", "fine")], gap=1.0)
        self.assertEqual(RE.read(clock=clk)["mood"], "tense")
        n = RE.note(clock=clk)
        self.assertIn("calm", n); self.assertIn("No jokes", n)

    def test_banter_with_one_shutup_and_laughs_is_not_tense(self):
        clk = feed([("S1", "shut up lol"), ("S2", "hahaha"), ("S1", "you're ridiculous"),
                    ("S3", "lmao"), ("S2", "okay okay"), ("S1", "anyway")], gap=2.0)
        self.assertNotEqual(RE.read(clock=clk)["mood"], "tense")

    def test_heavy(self):
        clk = feed([("S1", "hey sorry I've been quiet"), ("S2", "all good, what's up"),
                    ("S1", "my grandpa passed away on monday"), ("S2", "oh man I'm so sorry"),
                    ("S3", "that's really hard"), ("S1", "yeah it's been a rough week")])
        self.assertEqual(RE.read(clock=clk)["mood"], "heavy")
        self.assertIn("gentle", RE.note(clock=clk))

    def test_gaming_talk_is_not_heavy(self):
        # Real calls: "I died", "I'm scared", "hospital" are almost always about the game.
        clk = feed([("S1", "I died again to that boss"), ("S2", "I'm scared, it's dark in here"),
                    ("S1", "go to the hospital level"), ("S3", "bro we all died"),
                    ("S2", "heal up first"), ("S1", "ok going")])
        self.assertNotIn(RE.read(clock=clk)["mood"], ("heavy", "tense"))

    def test_constant_shut_up_banter_is_not_tense(self):
        clk = feed([("S1", "shut up"), ("S2", "no you shut up"), ("S1", "dude shut up"),
                    ("S3", "both of you shut up lol"), ("S2", "haha"), ("S1", "anyway")], gap=2.0)
        self.assertNotEqual(RE.read(clock=clk)["mood"], "tense")

    def test_note_never_tells_agent_to_speak_or_hold(self):
        for m, n in RE._NOTES.items():
            self.assertNotRegex(n.lower(), r"\b(stay quiet|hold|don't speak|do not speak|you must speak)\b", m)
            self.assertIn("If you speak", n)

    def test_agent_lines_not_counted_as_room(self):
        clk = feed([("AGENT", "haha"), ("AGENT", "lol"), ("AGENT", "lmao"),
                    ("S1", "ok"), ("S2", "sure"), ("S1", "right"), ("S2", "cool"), ("S1", "yep")])
        self.assertNotEqual(RE.read(clock=clk)["mood"], "hype")

    def test_feed_shape(self):
        feed([("S1", "hahaha"), ("S2", "lol"), ("S1", "no way!"), ("S2", "let's go"), ("S3", "lmao")], gap=2.0)
        items = RE.feed()
        self.assertEqual(len(items), 1); self.assertIn("title", items[0]); self.assertIn("items", items[0])


class Plugin(unittest.TestCase):
    def test_registered_with_no_tools(self):
        import plugins
        p = [x for x in plugins.BUILTIN if x["id"] == "room_energy"][0]
        self.assertEqual(p["tools"], [])


if __name__ == "__main__":
    unittest.main()
