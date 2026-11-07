"""Dev and packaged launches honor the same explicit Settings selection."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import remote_llm as R
import user_settings as U


class BuildParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.settings = root / 'settings.json'
        private = root / 'private'
        private.mkdir()
        (private / 'llm_mode.txt').write_text('remote', encoding='utf-8')
        legacy = private / 'remote_llm.json'
        legacy.write_text(json.dumps({'url':'https://legacy.invalid', 'model':'legacy'}), encoding='utf-8')
        for target, value in (('user_settings.PATH', self.settings),
                              ('remote_llm.ROOT', root), ('remote_llm.CONFIG_PATH', legacy)):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {'ATLAS_LLM':''})
        env.start()
        self.addCleanup(env.stop)

    def save(self, mode, cloud=None):
        self.settings.write_text(json.dumps({'llm':{'mode':mode,
            'local':{'model':'chosen-local', 'url':'http://127.0.0.1:11434'},
            'cloud':cloud or {}}}), encoding='utf-8')

    def test_explicit_local_wins_over_old_remote_mode_in_both_builds(self):
        self.save('local')
        for public in (True, False):
            with self.subTest(public=public), patch.object(U, 'is_public', return_value=public):
                self.assertFalse(R.remote_selected())
                self.assertEqual(U.startup_local_config()['model'], 'chosen-local')
                with patch.object(R, 'activate') as activate:
                    R.restore_at_startup(Mock())
                    activate.assert_not_called()

    def test_explicit_cloud_wins_over_private_host_in_both_builds(self):
        self.save('cloud', {'provider':'custom', 'base_url':'https://chosen.invalid/v1',
                            'model':'chosen-api'})
        for public in (True, False):
            with self.subTest(public=public), patch.object(U, 'is_public', return_value=public):
                self.assertTrue(R.remote_selected())
                self.assertEqual(R.load_config()['model'], 'chosen-api')
                self.assertEqual(R.load_config()['url'], 'https://chosen.invalid')
                with patch.object(R, 'activate') as activate:
                    R.restore_at_startup(Mock())
                    self.assertEqual(activate.call_args.args[1]['model'], 'chosen-api')

    def test_incomplete_cloud_choice_cannot_fall_back_to_private_host(self):
        self.save('cloud')
        self.assertIsNone(R.load_config())
        with patch.object(R, 'activate') as activate:
            with self.assertRaises(Exception):
                R.restore_at_startup(Mock())
            activate.assert_not_called()

    def test_explicit_custom_vllm_is_available_to_plugins_in_both_builds(self):
        self.save('cloud', {'provider':'custom', 'base_url':'http://my-server/v1',
                            'model':'vision-model', 'vllm':True, 'vision':True})
        for public in (True, False):
            with self.subTest(public=public), patch.object(U, 'is_public', return_value=public):
                cfg = R.load_config()
                self.assertTrue(cfg['vllm'])
                self.assertFalse(cfg['cloud'])
        self.save('cloud', {'provider':'deepseek', 'base_url':'https://api.deepseek.com/v1',
                            'model':'deepseek-flash', 'vllm':True})
        self.assertFalse(R.load_config()['vllm'])
        self.assertTrue(R.load_config()['cloud'])

    def test_unconfigured_legacy_profile_is_preserved(self):
        with patch.object(U, 'is_public', return_value=False):
            self.assertIsNone(U.startup_local_config())
        self.assertTrue(R.remote_selected())
        self.assertEqual(R.load_config()['model'], 'legacy')

    def test_explicit_legacy_remote_switch_survives_restart(self):
        self.save('local')
        cfg = R.load_config()
        U.remember_llm_backend('remote', cfg)
        self.assertTrue(R.remote_selected())
        saved = R.load_config()
        self.assertEqual(saved['url'], cfg['url'])
        self.assertEqual(saved['model'], cfg['model'])
        self.assertEqual(saved['vllm'], cfg['vllm'])
        self.assertEqual(saved['allow_local_fallback'], cfg['allow_local_fallback'])
