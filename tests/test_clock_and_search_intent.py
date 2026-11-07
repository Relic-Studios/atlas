"""Owner 10-05: every agent can check the real date; mentioning search isn't a search request."""
import datetime as dt
import unittest

import backchannels as B
import capability
import clock


def _has_zone(name):
    try:
        import zoneinfo
        zoneinfo.ZoneInfo(name)
        return True
    except Exception:
        return False


def _has_pipeline():
    try:
        import speech_pipeline_manager  # noqa: F401  (needs the full install)
        return True
    except Exception:
        return False


class Clock(unittest.TestCase):
    NOW = dt.datetime(2026, 10, 5, 8, 24, tzinfo=dt.timezone(dt.timedelta(hours=-7)))

    def test_now_note_has_real_date(self):
        n = clock.now_note(self.NOW)
        # now_note shows the machine's local time, so compare against NOW in this
        # machine's zone (CI runners are UTC; the dev box is Pacific).
        self.assertIn(clock._fmt(self.NOW.astimezone()), n)

    @unittest.skipUnless(_has_zone("Asia/Tokyo"), "no IANA time zone data (pip install tzdata)")
    def test_check_other_zone(self):
        self.assertIn("Tuesday, 6 October 2026", clock.check("Asia/Tokyo", self.NOW))

    def test_bad_zone_falls_back(self):
        self.assertIn("Unknown time zone", clock.check("Mars/Base", self.NOW))

    def test_explicit_clock_commands_have_real_arguments(self):
        self.assertEqual(clock.intent('Juniper, use your date and time tool to check today.'),
                         ('check_date_time', {'timezone': ''}))
        self.assertEqual(clock.intent('Juniper, call check_date_time for Asia/Tokyo.'),
                         ('check_date_time', {'timezone': 'Asia/Tokyo'}))
        for text in ('What time is it?', 'She said use your clock tool.',
                     'Repeat "use your clock tool".', 'Juniper, use your time tool in Paris.',
                     'Juniper, do not use your clock tool.', 'Juniper, repeat use your clock tool.',
                     'Juniper, say “use your clock tool”.'):
            self.assertIsNone(clock.intent(text), text)

    def test_explicit_clock_prefetch_requires_direct_address(self):
        import ast,re
        from pathlib import Path
        from types import SimpleNamespace
        from floor import names_agent
        path=Path(__file__).resolve().parents[1]/'speech_pipeline_manager.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SpeechPipelineManager')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_room_prefetch')
        import room_tools,logging
        ns={'_clock':clock,'_room':room_tools,'re':re,'logger':logging.getLogger('test')}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        mgr=SimpleNamespace(current_persona='juniper',floor=SimpleNamespace(active_partner=lambda:'S1'))
        run=ns['_room_prefetch']
        self.assertEqual(run(mgr,'[S1] Juniper, use your clock tool.'),('check_date_time',{'timezone':''}))
        self.assertIsNone(run(mgr,'[S1] Alex, use your clock tool.'))
        self.assertIsNone(run(mgr,'[S1] Juniper, use your clock tool.',is_cue=True))

    def test_every_agent_sees_date_and_tool(self):
        for persona in ("", "fae", "wren"):
            note = capability.abilities_note(False, True, persona)
            self.assertIn("Right now it is", note)
            self.assertIn("check the real date", note)

    @unittest.skipUnless(_has_pipeline(), "needs the full install")
    def test_tool_registered_and_dispatched(self):
        import speech_pipeline_manager as S
        names = [t["function"]["name"] for t in S.AGENT_TOOLS]
        self.assertIn("check_date_time", names)
        self.assertIn("local time", S._execute_tool("check_date_time", {}))


class SearchIntent(unittest.TestCase):
    def test_mentions_are_not_requests(self):
        for t in ("I can ask Fae to search the internet.", "she can search stuff too",
                  "we could ask her to google it", "she googles things for us",
                  "current research in voice-to-voice is cool", "I did some research on it"):
            self.assertFalse(B.wants_search(t), t)

    def test_requests_still_search(self):
        for t in ("Can you search fusion research?", "[S1] Fae, search the latest fusion news",
                  "Fae, could you look up the weather in Tokyo", "look it up",
                  "Can you research Chinese fusion?", "Fae, research voice models",
                  # live 11-01: read as a mention, nothing searched, "I'll look into it"
                  "What kind of wings do you want? Do you think you could search up some bodies for yourself?",
                  "you should look that up", "maybe you could google it", "you can just search it"):
            self.assertTrue(B.wants_search(t), t)

    def test_other_people_searching_is_still_a_mention(self):
        for t in ("Fae can look things up for us", "my brother could search it later", "he can google it"):
            self.assertFalse(B.wants_search(t), t)


if __name__ == "__main__":
    unittest.main()
