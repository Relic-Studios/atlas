"""Multilingual conversation (owner 10-05): transcribe real language, reply in it."""
import os, sys, types, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import languages as L


class RoomGating(unittest.TestCase):
    """Auto-follow mode (opt-in since owner 10-06; default replies stay in the main language)."""
    def setUp(self):
        self._follow = os.environ.get("ATLAS_REPLY_LANGUAGE")
        os.environ["ATLAS_REPLY_LANGUAGE"] = "follow"   # opt-in auto-follow (default is 'ask')
        self.addCleanup(self._restore_follow)
        self.r = L.Room(fallback="en")

    def _restore_follow(self):
        if self._follow is None:
            os.environ.pop("ATLAS_REPLY_LANGUAGE", None)
        else:
            os.environ["ATLAS_REPLY_LANGUAGE"] = self._follow

    def test_confident_switch(self):
        self.assertEqual(self.r.observe("es", 0.97, "¿Qué juego vamos a jugar hoy?"), "es")
        self.assertTrue(self.r.seen_other)

    def test_short_low_confidence_does_not_flip(self):
        self.assertEqual(self.r.observe("cy", 0.41, "yeah"), "en")
        self.assertEqual(self.r.observe("de", 0.95, "ja"), "en")  # one word: not enough

    def test_cjk_units(self):
        self.assertEqual(self.r.observe("ja", 0.9, "今日は何をしますか"), "ja")

    def test_retry_only_when_shaky_and_disagrees(self):
        self.assertIsNone(self.r.needs_retry("en", 0.3, "yeah"))
        self.assertEqual(self.r.needs_retry("cy", 0.4, "yeah"), "en")
        self.assertIsNone(self.r.needs_retry("es", 0.95, "vamos a jugar ahora"))

    def test_note_silent_for_english_call(self):
        self.r.observe("en", 0.99, "what are we playing tonight")
        self.assertEqual(self.r.note(), "")

    def test_note_names_language(self):
        self.r.observe("es", 0.97, "¿Qué juego vamos a jugar?")
        n = self.r.note()
        self.assertIn("Spanish", n); self.assertIn("Reply in Spanish", n)
        self.r.observe("en", 0.99, "ok back to english now")
        self.assertIn("English", self.r.note())  # stays explicit once the call went multilingual

    def test_note_unsupported_voice(self):
        self.r.observe("nl", 0.95, "wat gaan we vanavond spelen")
        self.assertIn("can't pronounce Dutch", self.r.note())


class TextLanguage(unittest.TestCase):
    def test_scripts(self):
        self.assertEqual(L.tts_language("こんにちは、元気ですか"), "japanese")
        self.assertEqual(L.tts_language("你好，我们玩什么"), "chinese")
        self.assertEqual(L.tts_language("안녕하세요 반가워요"), "korean")
        self.assertEqual(L.tts_language("Привет, как дела"), "russian")

    def test_latin_stopwords(self):
        # Once the agent is replying in another language (asked or follow mode).
        self.assertEqual(L.tts_language("Creo que es mejor jugar en equipo, pero no sé.", "es"), "spanish")
        self.assertEqual(L.tts_language("Je pense que c'est pas mal pour ce soir.", "fr"), "french")
        self.assertEqual(L.tts_language("Ich glaube, das ist nicht so schlimm.", "de"), "german")
        # Main language: Latin-script guesses never move the voice.
        self.assertEqual(L.tts_language("Creo que es mejor jugar en equipo, pero no sé.", "en"), "english")
        self.assertEqual(L.tts_language("I think that is the right call."), "english")

    def test_hint_breaks_ties_and_unsupported_falls_back(self):
        self.assertEqual(L.tts_language("OK!", "es"), "spanish")
        self.assertEqual(L.tts_language("Hallo", "nl"), "english")


class RetryWrapper(unittest.TestCase):
    def _rec(self, outputs):
        rec = types.SimpleNamespace(language="", detected_language=None,
                                    detected_language_probability=0.0,
                                    last_transcription_bytes=b"audio", calls=[])
        def pft(audio_bytes=None, use_prompt=True):
            text, det, prob = outputs.pop(0)
            rec.calls.append(rec.language)
            rec.detected_language, rec.detected_language_probability = det, prob
            return text
        rec.perform_final_transcription = pft
        return rec

    def setUp(self):
        import importlib.util
        if importlib.util.find_spec("transformers") is None:
            self.skipTest("transcribe.py needs transformers (not in the light CI deps)")
        self._room = L.ROOM
        L.ROOM = L.Room(fallback="en")
        os.environ["ATLAS_STT_LANGUAGE"] = "auto"
        self._follow = os.environ.get("ATLAS_REPLY_LANGUAGE")
        os.environ["ATLAS_REPLY_LANGUAGE"] = "follow"
        import transcribe
        self.T = transcribe

    def tearDown(self):
        L.ROOM = self._room
        os.environ.pop("ATLAS_STT_LANGUAGE", None)
        if self._follow is None:
            os.environ.pop("ATLAS_REPLY_LANGUAGE", None)
        else:
            os.environ["ATLAS_REPLY_LANGUAGE"] = self._follow

    def test_shaky_detection_retried_in_room_language(self):
        rec = self._rec([("Ja.", "cy", 0.35), ("Yeah.", "en", 0.99)])
        self.T._wrap_language_retry(rec)
        self.assertEqual(rec.perform_final_transcription(), "Yeah.")
        self.assertEqual(rec.calls, ["", "en"])
        self.assertEqual(rec.language, "")          # restored to auto
        self.assertEqual(L.ROOM.current(), "en")

    def test_confident_other_language_kept(self):
        rec = self._rec([("¿Qué vamos a jugar esta noche?", "es", 0.96)])
        self.T._wrap_language_retry(rec)
        self.assertEqual(rec.perform_final_transcription(), "¿Qué vamos a jugar esta noche?")
        self.assertEqual(rec.calls, [""])
        self.assertEqual(L.ROOM.current(), "es")

    def test_fixed_language_passes_through_live(self):
        rec = self._rec([("¿Qué tal?", "es", 0.4), ("Yeah.", "en", 0.3), ("Yeah.", "es", 0.9)])
        self.T._wrap_language_retry(rec)
        os.environ["ATLAS_STT_LANGUAGE"] = "es"        # plugin set to Spanish, no restart
        self.assertEqual(rec.perform_final_transcription(), "¿Qué tal?")
        self.assertEqual(rec.calls, ["es"])            # forced, never retried
        self.assertEqual(L.ROOM.current(), "es")
        os.environ["ATLAS_STT_LANGUAGE"] = "auto"      # back to detection
        self.assertEqual(rec.perform_final_transcription(), "Yeah.")
        self.assertEqual(rec.calls[-2:], ["", "es"])   # detection first, shaky -> retried in room language
        self.assertEqual(rec.language, "")             # restored to auto

    def test_wrap_once(self):
        rec = self._rec([])
        self.T._wrap_language_retry(rec); f = rec.perform_final_transcription
        self.T._wrap_language_retry(rec)
        self.assertIs(rec.perform_final_transcription, f)


class PluginSwitch(unittest.TestCase):
    def test_off_means_main_language(self):
        from unittest import mock
        with mock.patch.object(L, "_plugin", return_value=(False, {"primary": "de"})):
            self.assertEqual(L.setting(), "de")
        with mock.patch.object(L, "_plugin", return_value=(True, {"listen": "auto", "primary": "fr"})):
            self.assertEqual(L.setting(), "auto")
            self.assertEqual(L.primary(), "fr")

    def test_plugin_listed(self):
        import plugins
        p = {x["id"]: x for x in plugins.listing()}
        self.assertIn("languages", p)


class PluginRoundTrip(unittest.TestCase):
    """Plugins page -> plugins.json -> languages.setting()/primary(), in a temp dir.
    10-05: the 'Main language' select had lost English (a [1:] slice on the wrong list)."""
    def setUp(self):
        import tempfile, plugins
        from pathlib import Path
        from unittest import mock
        self.P = plugins
        d = Path(tempfile.mkdtemp())
        self._p = [mock.patch.object(plugins, "USER", d),
                   mock.patch.object(plugins, "STATE_PATH", d / "plugins.json")]
        for x in self._p: x.start()
        for k in ("ATLAS_STT_LANGUAGE", "ATLAS_PRIMARY_LANGUAGE"):
            os.environ.pop(k, None)

    def tearDown(self):
        for x in self._p: x.stop()

    def opts(self, key):
        p = [x for x in self.P.BUILTIN if x["id"] == "languages"][0]
        f = [f for f in p["settings"] if f["key"] == key][0]
        return f["default"], [o["value"] for o in f["options"]]

    def test_primary_options_sane(self):
        default, vals = self.opts("primary")
        self.assertIn("en", vals)
        self.assertIn(default, vals)
        self.assertNotIn("auto", vals)
        d2, v2 = self.opts("listen")
        self.assertEqual(v2[0], "auto")
        self.assertIn("en", v2)

    def test_save_then_read(self):
        self.P.save_settings("languages", {"listen": "es", "primary": "de"})
        self.assertEqual(L.setting(), "es")
        self.assertEqual(L.primary(), "de")
        self.P.save_settings("languages", {"listen": "auto"})
        self.assertEqual(L.setting(), "auto")

    def test_invalid_rejected(self):
        r = self.P.save_settings("languages", {"primary": "auto", "listen": "klingon"})
        errs = r.get("errors") if isinstance(r, dict) else None
        self.assertTrue(errs and "primary" in errs and "listen" in errs, r)
        self.assertEqual(L.primary(), "en")

    def test_off_uses_main_language(self):
        self.P.save_settings("languages", {"listen": "auto", "primary": "fr"})
        self.P.set_enabled("languages", False)
        self.assertEqual(L.setting(), "fr")


class Setting(unittest.TestCase):
    def test_env_override(self):
        os.environ["ATLAS_STT_LANGUAGE"] = "EN"
        try:
            self.assertEqual(L.setting(), "en")
        finally:
            del os.environ["ATLAS_STT_LANGUAGE"]


if __name__ == "__main__":
    unittest.main()
