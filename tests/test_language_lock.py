"""Owner 10-06: replies (and the voice) stay in the main language; only an explicit
request unlocks another one ("speak Spanish"), and "back to English" relocks."""
import os, unittest
import languages as L


class Lock(unittest.TestCase):
    def setUp(self):
        os.environ["ATLAS_REPLY_LANGUAGE"] = "ask"
        self.r = L.Room("en")
        self._old, L.ROOM = L.ROOM, self.r

    def tearDown(self):
        L.ROOM = self._old
        os.environ.pop("ATLAS_REPLY_LANGUAGE", None)

    def test_confident_foreign_speech_does_not_switch(self):
        self.assertEqual(self.r.observe("pt", 0.95, "Isso é muito legal mesmo"), "en")
        self.assertEqual(self.r.note(), "")
        self.assertEqual(L.tts_language("Que legal, é isso mesmo", "en"), "english")

    def test_english_never_voiced_as_portuguese(self):
        for t in ["That's a hell of a move.", "A cara do jogo, a lot of it.", "Okay, a quick one."]:
            self.assertEqual(L.tts_language(t, "en"), "english")

    def test_request_unlocks_and_back_relocks(self):
        self.assertEqual(self.r.observe("en", 0.95, "Fae, speak Spanish"), "es")
        self.assertIn("Spanish", self.r.note())
        self.assertEqual(L.tts_language("Claro, hablemos en español.", self.r.current()), "spanish")
        # English speech while unlocked doesn't drop the request
        self.assertEqual(self.r.observe("en", 0.97, "how was your day though"), "es")
        self.assertEqual(self.r.observe("en", 0.95, "okay go back to English"), "en")
        self.assertIn("reply in English again", self.r.note())

    def test_requests(self):
        cases = {"speak Spanish please": "es", "can you reply in Japanese?": "ja",
                 "habla español": "es", "en español por favor": "es", "switch to German": "de",
                 "stop speaking Spanish": "__back__"}
        for t, want in cases.items():
            self.assertEqual(L.requested_language(t), want, t)

    def test_not_requests(self):
        for t in ["I speak Spanish at home sometimes", "do you speak French?",
                  "what language do you speak", "we talked to French people",
                  "That's a hell of a move."]:
            self.assertIsNone(L.requested_language(t), t)

    def test_follow_mode_still_available(self):
        os.environ["ATLAS_REPLY_LANGUAGE"] = "follow"
        self.assertEqual(self.r.observe("es", 0.95, "¿Qué tal estás hoy amigo?"), "es")

    def test_stt_retry_uses_heard_not_reply(self):
        self.r.observe("en", 0.95, "Fae, speak Spanish")
        self.assertEqual(self.r.needs_retry("pt", 0.4, "a hell of a move"), "en")


if __name__ == "__main__":
    unittest.main()
