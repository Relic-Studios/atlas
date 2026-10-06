"""Live 10-06: with Game info switched off, Fae still said "Quick check on Steam
sales for you" (asked about warranties). The abilities line only checked six
hard-coded plugins, so every newer plugin was always advertised."""
import unittest
import capability
import plugins

PHRASES = {
    "game_info": "Steam", "weather": "weather", "timers": "timers",
    "dice_polls": "roll dice", "bets": "bets", "voice_mail": "messages",
    "call_summary": "recap", "teach": "be taught the room", "lore": "lore",
    "fact_check": "fact-check", "live_translate": "translate",
    "floor_referee": "referee", "soul_reflection": "propose small changes",
}


class AbilitiesGating(unittest.TestCase):
    def setUp(self):
        self._load = plugins._load

    def tearDown(self):
        plugins._load = self._load

    def test_off_ids_covers_every_plugin(self):
        plugins._load = lambda: {d["id"]: {"enabled": False} for d in plugins.BUILTIN}
        self.assertEqual(plugins.off_ids(), {d["id"] for d in plugins.BUILTIN})

    def test_switched_off_plugins_not_advertised(self):
        for pid, phrase in PHRASES.items():
            with self.subTest(pid=pid):
                plugins._load = lambda pid=pid: {pid: {"enabled": False}}
                off = plugins.off_ids()
                self.assertIn(pid, off)
                note = capability.abilities_note(False, False, "", off)
                self.assertNotIn(phrase, note)

    def test_game_info_on_is_advertised(self):
        plugins._load = lambda: {"game_info": {"enabled": True}}
        self.assertIn("Steam", capability.abilities_note(False, False, "", plugins.off_ids()))

    def test_pipeline_uses_off_ids(self):
        src = open("speech_pipeline_manager.py", encoding="utf-8").read()
        self.assertIn("_plugins.off_ids()", src)
        self.assertNotIn("('web_search', 'eyes', 'clock', 'notes', 'step_back', 'self_check')", src)


if __name__ == "__main__":
    unittest.main()
