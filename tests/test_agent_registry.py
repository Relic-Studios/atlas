"""Agent registry: dev pack vs public (no pack), user agents, placeholder boot."""
import json, os, sys, tempfile, unittest, wave
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_registry as R


def pipeline_namespace():
    """Execute the real registry consumers without importing the speech stack."""
    import ast, logging
    source = Path(__file__).resolve().parents[1] / 'speech_pipeline_manager.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in ('_replace_in_place', '_rebuild_personas')) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PLACEHOLDER_ID'
                                              for t in n.targets))]
    ns = dict(_registry=R, PERSONAS={}, PERSONA_FILES={}, PERSONA_META={},
              logger=logging.getLogger('test'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return ns


class _Iso(unittest.TestCase):
    """Point the registry at temp user dirs; restore afterwards."""
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = os.environ.get("ATLAS_DEV_PACK")
        self.paths = patch.multiple(R, ROOT=self.tmp, DEV_PACK=self.tmp / 'dev_pack/pack.json',
            BUNDLED_PACK=self.tmp / 'starter_pack/pack.json', USER_AGENTS=self.tmp / 'personas/agents.json',
            USER_VOICES=self.tmp / 'voices/user/voices.json')
        self.paths.start()
        import user_settings
        self.settings = patch.object(user_settings, 'PATH', self.tmp / 'user/settings.json')
        self.settings.start()
        os.environ['ATLAS_DEV_PACK'] = '1'

    def tearDown(self):
        dev = self._saved
        if dev is None:
            os.environ.pop("ATLAS_DEV_PACK", None)
        else:
            os.environ["ATLAS_DEV_PACK"] = dev
        self.settings.stop(); self.paths.stop(); R.reload()
        import voices; voices.refresh()
        self._tmp.cleanup()

    def wav(self, path, rate=24000, channels=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), 'wb') as audio:
            audio.setnchannels(channels); audio.setsampwidth(2); audio.setframerate(rate)
            audio.writeframes(b'\x01\x00' * 2400 * channels)
        return path

    def bundle(self):
        self.wav(self.tmp / 'starter_pack/juniper-natural.wav')
        (self.tmp / 'starter_pack/juniper-natural.txt').write_text('Hello there.', encoding='utf-8')
        R.BUNDLED_PACK.write_text(json.dumps({'default':'juniper', 'voices':{'juniper-natural':{
            'wav':'starter_pack/juniper-natural.wav', 'text':'starter_pack/juniper-natural.txt'}},
            'agents':[{'id':'juniper', 'name':'Juniper', 'voice':'juniper-natural', 'prompt':'You are Juniper.'}]}), encoding='utf-8')

    def public(self):
        os.environ["ATLAS_DEV_PACK"] = "0"; R.reload()

    def add_user_agent(self, aid="nova", voice="myvoice"):
        (self.tmp / "personas").mkdir(parents=True, exist_ok=True)
        (self.tmp / "voices" / "user").mkdir(parents=True, exist_ok=True)
        wav = self.wav(self.tmp / "voices" / "user" / "myvoice.wav", rate=16000, channels=2)
        R.USER_VOICES.write_text(json.dumps({"voices": {voice: {"wav": str(wav)}}}))
        p = self.tmp / "personas" / f"{aid}.txt"; p.write_text("You're Nova. " * 20)
        R.USER_AGENTS.write_text(json.dumps([{"id": aid, "name": "Nova", "voice": voice,
            "prompt_file": str(p), "interests": ["space", "synths"], "talkativeness": 0.6,
            "nicknames": ["novs"], "gender": "female", "what": "a space pirate"}]))


class DevPackTests(_Iso):
    def test_dev_pack_agents_and_voices(self):
        wav = self.wav(self.tmp / 'dev_pack/dev.wav')
        R.DEV_PACK.write_text(json.dumps({'default':'dev-agent', 'voices':{'dev-voice':{'wav':str(wav)}},
            'agents':[{'id':'dev-agent', 'voice':'dev-voice', 'prompt':'You are a dev fixture.'}]}))
        R.reload()
        self.assertTrue(len(R.agent_ids()) >= 1)
        d = R.default_agent()
        self.assertTrue(R.prompt_text(d))
        self.assertTrue(R.agent_voice(d))
        self.assertTrue(Path(R.voice_path(R.agent_voice(d))).is_file())
        self.assertTrue(R.is_dev(d))


class PublicModeTests(_Iso):
    def test_public_has_no_agents_or_voices(self):
        self.public()
        self.assertEqual(R.agent_ids(), [])
        self.assertEqual(R.voices(), {})
        self.assertIsNone(R.default_agent())
        self.assertIsNone(R.default_voice())

    def test_public_placeholder_persona(self):
        self.public()
        ns = pipeline_namespace()
        ns['_rebuild_personas']()
        self.assertEqual(list(ns['PERSONAS']), [ns['PLACEHOLDER_ID']])
        self.assertTrue(ns['PERSONA_META'][ns['PLACEHOLDER_ID']]['placeholder'])
        self.assertEqual(ns['DEFAULT_PERSONA'], ns['PLACEHOLDER_ID'])

    def test_voices_facade_empty(self):
        self.public()
        import voices; voices.refresh()
        self.assertEqual(voices.voice_names(), [])
        self.assertIsNone(voices.voice_wav(None))
        self.assertIsNone(voices.DEFAULT_VOICE)

    def test_user_agent_everywhere(self):
        self.public(); self.add_user_agent(); R.reload()
        self.assertEqual(R.agent_ids(), ["nova"])
        self.assertEqual(R.default_agent(), "nova")
        self.assertEqual(R.agent_voice("nova"), "myvoice")
        self.assertEqual(R.profile("nova")["interests"], ("space", "synths"))
        self.assertEqual(R.identity("nova")["gender"], "female")
        self.assertIn("novs", R.all_agent_words())
        import identity, conversation_dynamics as C, people
        self.assertTrue(identity.note("nova", "Nova"))
        self.assertEqual(C._AGENT_NICKNAMES.get("nova"), ("novs",))
        self.assertIn("nova", people._agent_words())
        import voices; voices.refresh()
        self.assertEqual(voices.PERSONA_VOICES, {"nova": "myvoice"})

    def test_custom_agent_without_identity_gets_no_note(self):
        self.public(); self.add_user_agent(); R.reload()
        a = json.loads(R.USER_AGENTS.read_text()); a[0].pop("gender"); a[0].pop("what")
        R.USER_AGENTS.write_text(json.dumps(a)); R.reload()
        import identity
        self.assertEqual(identity.note("nova", "Nova"), "")


class BundledDefaultTests(_Iso):
    def test_empty_public_install_has_one_immutable_ready_default(self):
        self.public(); self.bundle()
        for _ in range(3):
            R.reload()
            self.assertEqual(R.agent_ids(), ['juniper'])
            self.assertEqual(R.default_agent(), 'juniper')
            self.assertEqual(R.default_voice(), 'juniper-natural')
            self.assertEqual(R.ready_agent(), {'id':'juniper', 'name':'Juniper',
                'voice':'juniper-natural', 'bundled':True})
            self.assertTrue(R.agent('juniper')['bundled'])
            self.assertFalse(R.agent('juniper')['custom'])
            self.assertFalse(R.agent('juniper')['dev'])
            self.assertIsNone(R.bundle_error())
        self.assertFalse(R.USER_AGENTS.exists())
        self.assertFalse(R.USER_VOICES.exists())
        self.assertEqual(Path(R.voice_text_path('juniper-natural')).read_text(), 'Hello there.')

    def test_existing_user_and_saved_nondefault_selection_win(self):
        self.public(); self.bundle(); self.add_user_agent()
        records = json.loads(R.USER_AGENTS.read_text())
        records.append({'id':'moon', 'name':'Moon', 'voice':'moon-voice', 'prompt':'You are Moon.'})
        R.USER_AGENTS.write_text(json.dumps(records))
        voices = json.loads(R.USER_VOICES.read_text())
        voices['voices']['moon-voice'] = {'wav':str(self.wav(self.tmp / 'other/moon.wav', rate=48000, channels=2))}
        R.USER_VOICES.write_text(json.dumps(voices))
        before = (R.USER_AGENTS.read_bytes(), R.USER_VOICES.read_bytes())
        R.reload()
        self.assertEqual(R.default_agent(), 'nova')
        import user_settings as US
        US.update({'agent':'moon'})
        R.reload()
        self.assertEqual(R.default_agent(), 'moon')
        self.assertEqual(R.default_voice(), 'moon-voice')
        self.assertEqual(before, (R.USER_AGENTS.read_bytes(), R.USER_VOICES.read_bytes()))

    def test_user_agent_collision_and_effective_voice_collision_are_preserved(self):
        self.public(); self.bundle(); self.add_user_agent(aid='juniper')
        R.reload()
        self.assertEqual(R.ready_agent()['name'], 'Nova')
        self.assertEqual(R.ready_agent()['voice'], 'myvoice')
        self.assertFalse(R.ready_agent()['bundled'])
        self.assertTrue(R.agent('juniper')['custom'])
        # Only the voice key collides: the effective user reference may be
        # stereo/16kHz, outside starter_pack, and without a transcript.
        R.USER_AGENTS.write_text('[]')
        own_wav = self.wav(self.tmp / 'personal.wav', rate=16000, channels=2)
        R.USER_VOICES.write_text(json.dumps({'voices':{'juniper-natural':{'wav':str(own_wav)}}}))
        R.reload()
        self.assertTrue(R.ready_agent()['bundled'])
        self.assertEqual(R.voice_path('juniper-natural'), str(own_wav))
        self.assertTrue(R.voices()['juniper-natural']['user'])

    def test_dev_collision_remains_authoritative(self):
        self.bundle(); self.add_user_agent(aid='juniper', voice='juniper-natural')
        dev_wav = self.wav(self.tmp / 'dev_pack/dev.wav', rate=48000, channels=2)
        R.DEV_PACK.write_text(json.dumps({'default':'juniper',
            'voices':{'juniper-natural':{'wav':str(dev_wav)}},
            'agents':[{'id':'juniper', 'name':'Dev fixture', 'voice':'juniper-natural', 'prompt':'Dev prompt.'}]}))
        R.reload()
        self.assertEqual(R.ready_agent()['name'], 'Dev fixture')
        self.assertEqual(R.voice_path('juniper-natural'), str(dev_wav))
        self.assertTrue(R.is_dev('juniper'))

    def test_bad_bundle_is_not_exposed_as_an_agent(self):
        self.public()
        for fault in ('missing', 'corrupt', 'stereo', 'rate', 'empty-text', 'empty-prompt', 'escape', 'bad-json'):
            with self.subTest(fault=fault):
                self.bundle()
                wav = self.tmp / 'starter_pack/juniper-natural.wav'
                if fault == 'missing': wav.unlink()
                if fault == 'corrupt': wav.write_bytes(b'not a wave')
                if fault == 'stereo': self.wav(wav, channels=2)
                if fault == 'rate': self.wav(wav, rate=16000)
                if fault == 'empty-text': (self.tmp / 'starter_pack/juniper-natural.txt').write_text(' ')
                if fault in ('empty-prompt', 'escape'):
                    pack = json.loads(R.BUNDLED_PACK.read_text())
                    if fault == 'empty-prompt': pack['agents'][0]['prompt'] = ' '
                    else: pack['voices']['juniper-natural']['wav'] = str(self.wav(self.tmp / 'outside.wav'))
                    R.BUNDLED_PACK.write_text(json.dumps(pack))
                if fault == 'bad-json': R.BUNDLED_PACK.write_text('{')
                R.reload()
                self.assertEqual(R.agent_ids(), [])
                self.assertIsNone(R.ready_agent())
                self.assertTrue(R.bundle_error())

    def test_invalid_saved_voice_cannot_override_startup_persona(self):
        self.public(); self.bundle(); self.add_user_agent(aid='broken', voice='gone')
        (self.tmp / 'voices/user/myvoice.wav').unlink()
        import user_settings as US
        US.update({'agent':'broken'})
        R.reload()
        self.assertEqual(R.default_agent(), 'juniper')
        self.assertEqual(R.default_voice(), 'juniper-natural')
        ns = pipeline_namespace()
        ns['_rebuild_personas']()
        self.assertEqual(ns['DEFAULT_PERSONA'], 'juniper')
        self.assertIn('broken', ns['PERSONAS'])  # Existing record is preserved, never silently rewritten.

    def test_nonscalar_bundle_identifiers_are_skipped_without_losing_valid_records(self):
        self.public()
        for field in ('id', 'voice', 'default'):
            for bad in (['bad'], {'bad':'value'}):
                with self.subTest(field=field, bad=bad):
                    self.bundle()
                    pack = json.loads(R.BUNDLED_PACK.read_text())
                    if field == 'default':
                        pack['default'] = bad
                    else:
                        invalid = dict(pack['agents'][0], id='invalid-agent')
                        invalid[field] = bad
                        pack['agents'].insert(0, invalid)
                    R.BUNDLED_PACK.write_text(json.dumps(pack))
                    R.reload()
                    self.assertEqual(R.agent_ids(), ['juniper'])
                    self.assertEqual(R.ready_agent()['id'], 'juniper')
                    self.assertTrue(R.bundle_error())


if __name__ == "__main__":
    unittest.main()


class InPlaceRebuildTest(_Iso):
    """Agent and voice tables are rebuilt in place (agent switches read them on a worker thread since 11-04): a reader
    never sees them empty while a rebuild runs."""

    def test_tables_never_empty_during_rebuilds(self):
        import threading
        import voices
        ns = pipeline_namespace()
        self.add_user_agent()
        R.reload(); voices.refresh(); ns['_rebuild_personas']()
        stop, empties = threading.Event(), []

        def reader():
            while not stop.is_set():
                if not ns['PERSONAS'] or not voices.VOICES or not R._state:
                    empties.append(1)
                list(ns['PERSONAS'])
        t = threading.Thread(target=reader)
        t.start()
        try:
            for _ in range(60):
                R.reload(); voices.refresh(); ns['_rebuild_personas']()
        finally:
            stop.set(); t.join()
        self.assertEqual(empties, [])
        self.assertIn("nova", ns['PERSONAS'])

    def test_removed_entries_go(self):
        ns = pipeline_namespace()
        live = {"a": 1, "b": 2}
        ns['_replace_in_place'](live, {"b": 3, "c": 4})
        self.assertEqual(live, {"b": 3, "c": 4})


class DanglingVoiceTest(__import__("unittest").TestCase):
    """Remote box clean-user e2e 10-03: a persona whose voice was deleted left
    default_voice() = None and QwenVoice crashed at boot with no reference."""

    def test_default_voice_falls_back_to_installed_voice(self):
        import agent_registry as R
        from unittest import mock
        state = {"agents": {"mira": {"id": "mira", "voice": "gone"}},
                 "voices": {"starter": {"wav": "x.wav"}}, "default": "mira", "traits": {}}
        with mock.patch.object(R, "_s", return_value=state):
            self.assertIsNone(R.agent_voice("mira"))
            self.assertEqual(R.default_voice(), "starter")

    def test_default_voice_none_when_nothing_installed(self):
        import agent_registry as R
        from unittest import mock
        state = {"agents": {"mira": {"id": "mira", "voice": "gone"}},
                 "voices": {}, "default": "mira", "traits": {}}
        with mock.patch.object(R, "_s", return_value=state):
            self.assertIsNone(R.default_voice())
