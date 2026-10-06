"""Live 10-06: English replies were voiced with Portuguese pronunciation because the
bare word 'a' only appeared in the Portuguese stopword list."""
import unittest
import languages as L


class TtsLanguagePick(unittest.TestCase):
    def setUp(self):
        self._room = L.ROOM
        L.ROOM = L.Room("en")

    def tearDown(self):
        L.ROOM = self._room

    def test_live_english_lines_stay_english(self):
        for t in ["That's a hell of a move.", "Alright, that's a real constraint.",
                  "Make a plan.", "Sure, a d20 roll gives you a 14.", "No, a la carte is fine."]:
            self.assertEqual(L.tts_language(t, "en"), "english", t)

    def test_steady_english_room_needs_other_script(self):
        self.assertEqual(L.tts_language("La raid empieza a las nueve.", "en"), "english")
        self.assertEqual(L.tts_language("こんにちは、元気ですか", "en"), "japanese")

    def test_spanish_room_voices_spanish(self):
        L.ROOM.observe("es", 0.95, "hola que tal amigos")
        self.assertEqual(L.tts_language("Sí, es muy buena idea.", "es"), "spanish")
        self.assertEqual(L.tts_language("Okay, sounds good to me, a plan.", "es"), "english")

    def test_guess_needs_clear_margin(self):
        self.assertEqual(L.guess_text("I want a plan", "en"), "en")
        self.assertEqual(L.guess_text("el perro y la casa de la playa", "en"), "es")


if __name__ == "__main__":
    unittest.main()
