"""Live translate plugin (owner 10-05: separate add-on on top of Languages)."""
import unittest

import translate as T


NAMES = {"S2": "Ana", "S3": "Kenji"}


def name_of(s):
    return NAMES.get(s, s)


class Base(unittest.TestCase):
    def setUp(self):
        T.reset()

    def tearDown(self):
        T.reset()


class Intent(Base):
    Q = ["Fae, what did she just say?", "what did Ana say in English?", "what does that mean in English"]

    def test_english_only_room_ignores_ambiguous_asks(self):
        T.note("S1", "okay one sec guys", "en")
        for q in self.Q + ["what did they say about the raid", "what did you say?"]:
            self.assertIsNone(T.intent(q), q)

    def test_explicit_translate_always_counts(self):
        self.assertEqual(T.intent("Max, translate that")[0], "translate_line")

    def test_mixed_room(self):
        T.note("S1", "okay one sec guys", "en")
        T.note("S2", "¿Dónde está el baño, por favor?", "es")
        for q in self.Q:
            self.assertEqual(T.intent(q)[0], "translate_line", q)
        self.assertEqual(T.intent("what did Ana say in English?")[1], {"to": "english", "who": "Ana"})
        for q in ["what did you say?", "what did they say about the raid",
                  "we need to translate that into action", "lost in translation lol"]:
            self.assertIsNone(T.intent(q), q)


class TranslateLine(Base):
    def test_finds_latest_foreign_line(self):
        T.note("S2", "¿Dónde está el baño, por favor?", "es")
        T.note("S3", "ちょっと待ってください", "ja")
        T.note("S1", "okay one sec guys", "en")
        out = T.translate_line(name_of=name_of)
        self.assertIn("Kenji", out)
        self.assertIn("ちょっと待って", out)
        self.assertIn("into English", out)

    def test_by_speaker(self):
        T.note("S2", "¿Dónde está el baño, por favor?", "es")
        T.note("S3", "ちょっと待ってください", "ja")
        out = T.translate_line(who="ana", name_of=name_of)
        self.assertIn("baño", out)

    def test_target_language(self):
        T.note("S1", "where is the bathroom please", "en")
        out = T.translate_line(to="spanish", name_of=name_of)
        self.assertIn("into Spanish", out)
        self.assertIn("bathroom", out)

    def test_nothing_to_translate_says_so(self):
        T.note("S1", "okay one sec guys", "en")
        out = T.translate_line(name_of=name_of)
        self.assertIn("don't guess", out)

    def test_old_lines_expire(self):
        T.note("S2", "¿Dónde está el baño, por favor?", "es", clock=lambda: 0.0)
        self.assertIn("No line", T.translate_line(name_of=name_of, clock=lambda: T.MAX_AGE_S + 10.0))

    def test_single_word_skipped(self):
        T.note("S2", "Hola", "es")
        self.assertIn("No line", T.translate_line(name_of=name_of))


class Captions(Base):
    def test_translates_in_background_and_shows_in_feed(self):
        seen = []

        def ask(system, user):
            seen.append(user)
            return "Where is the bathroom, please?"
        it = T.caption("Ana", "¿Dónde está el baño, por favor?", "es", 0.95, "en", ask=ask, sync=True)
        self.assertEqual(it["status"], "done")
        self.assertIn("Spanish", seen[0])
        f = T.feed()
        self.assertEqual(len(f), 1)
        self.assertIn("Where is the bathroom", f[0]["items"][0]["text"])
        self.assertIn("Ana", f[0]["title"])

    def test_skips_own_language_low_confidence_and_short(self):
        ask = lambda s, u: "x"  # noqa: E731
        self.assertIsNone(T.caption("Sam", "this is english text", "en", 0.99, "en", ask=ask, sync=True))
        self.assertIsNone(T.caption("Ana", "¿Dónde está el baño?", "es", 0.2, "en", ask=ask, sync=True))
        self.assertIsNone(T.caption("Ana", "Hola", "es", 0.99, "en", ask=ask, sync=True))

    def test_daily_cap(self):
        t = [1000.0]
        ask = lambda s, u: "ok"  # noqa: E731
        for i in range(3):
            t[0] += 5
            T.caption("Ana", f"frase numero {i} aqui", "es", 0.9, "en", ask=ask, daily=2,
                      clock=lambda: t[0], sync=True)
        self.assertEqual(len(T.feed()), 2)

    def test_model_failure_is_shown_not_raised(self):
        def ask(s, u):
            raise TimeoutError("slow")
        it = T.caption("Ana", "¿Dónde está el baño?", "es", 0.9, "en", ask=ask, sync=True)
        self.assertTrue(it["status"].startswith("failed"))
        self.assertIn("failed", T.feed()[0]["items"][0]["text"])


class Wiring(Base):
    def test_plugin_and_tool(self):
        import plugins
        import room_tools
        p = [x for x in plugins.BUILTIN if x["id"] == "live_translate"][0]
        self.assertFalse(p["default"])
        self.assertEqual(p["tools"], ["translate_line"])
        self.assertIn("translate_line", room_tools.NAMES)
        T.note("S2", "¿Dónde está el baño, por favor?", "es")
        out = room_tools.execute("translate_line", {}, "S1", "Sam", name_of=name_of)
        self.assertIn("LINE TO TRANSLATE", out)
        self.assertEqual(room_tools.intent("Fae, translate that")[0], "translate_line")


if __name__ == "__main__":
    unittest.main()
