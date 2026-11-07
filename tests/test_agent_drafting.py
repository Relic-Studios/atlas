"""Creator runtime selection regressions; all inference is scripted, never network/GPU."""
import ast
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests
from fastapi import Request
from fastapi.responses import JSONResponse
import agents as A
from llm_module import LLM
import remote_llm as R

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"who": "You love soup and quietly argue for rainy afternoons.", "interests": ["soup", "rain"]}


class Response:
    status_code = 200
    def raise_for_status(self): pass
    def iter_content(self, chunk_size=None):
        yield (json.dumps({"message": {"content": json.dumps(FIELDS)}, "done": False}) + '\n').encode()
        yield (json.dumps({"message": {"content": ""}, "done": True}) + '\n').encode()
    def iter_lines(self, decode_unicode=False):
        yield ('data: ' + json.dumps({"choices": [{"delta": {"content": json.dumps(FIELDS)}}]})).encode()
        yield b'data: [DONE]'
    def close(self): pass


def local_llm():
    llm = LLM("ollama", "actually-loaded")
    llm._client_initialized = llm._ollama_connection_ok = True
    llm.ollama_session = Mock(post=Mock(return_value=Response()))
    llm.system_prompt = 'call persona unchanged'
    llm.system_prompt_message = {"role": "system", "content": llm.system_prompt}
    return llm


def remote_llm():
    llm = local_llm()
    local = llm.ollama_session
    rs = R.RemoteSession({"url": "https://api.example.test", "model": "api-current", "label": "chosen", "vllm": False}, local)
    rs.ping = Mock(return_value=True)
    rs._http.post = Mock(return_value=Response())
    llm.remote = llm.ollama_session = rs
    return llm, rs, local


class CurrentDraftTests(unittest.TestCase):
    def test_native_draft_is_bound_to_account_and_preserves_safe_provider_errors(self):
        from native_providers import NativeSession
        from provider_errors import failure
        llm=local_llm()
        session=NativeSession({"provider":"codex","model":"native-current","url":"","label":"Account",
                               "account_binding":"first-account","instruction_fingerprint":"instructions"})
        llm.remote=llm.ollama_session=session
        self.assertEqual(A.runtime_model_info(llm)["account_binding"],"first-account")
        llm.generate=Mock(side_effect=failure("codex","usageLimitExceeded"))
        with self.assertRaisesRegex(A.DraftError,"usage limit"):
            A.build_runtime_draft(llm,'Soupy','A quiet soup ghost.','')
        def changed(*args,**kwargs):
            session.cfg["account_binding"]="second-account"
            yield json.dumps(FIELDS)
        llm.generate=changed
        with self.assertRaisesRegex(A.DraftError,"changed"):
            A.build_runtime_draft(llm,'Soupy','A quiet soup ghost.','')
        session.close()

    def test_local_draft_isolated_system_and_native_budget(self):
        llm = local_llm()
        before = copy.deepcopy(llm.system_prompt_message)
        draft = A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', 'voice')
        self.assertEqual(draft['model_info']['model'], 'actually-loaded')
        self.assertEqual(draft['source'], 'llm')
        call = llm.ollama_session.post.call_args.kwargs
        self.assertNotIn('strict_backend', call)
        self.assertNotIn('allow_local_fallback', call)
        payload = call['json']
        self.assertEqual(payload['options']['num_predict'], 600)
        self.assertNotIn('max_tokens', payload['options'])
        self.assertEqual(len(payload['messages']), 2)
        self.assertTrue(payload['messages'][0]['content'].startswith(A.DRAFT_SYSTEM))
        self.assertNotIn('tools', payload)
        self.assertNotIn('call persona unchanged', str(payload))
        self.assertEqual(llm.system_prompt_message, before)
        self.assertEqual(llm.system_prompt, 'call persona unchanged')
        self.assertFalse(llm._active_requests)

    def test_remote_draft_uses_live_model_and_no_local_transport(self):
        llm, rs, local = remote_llm()
        draft = A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
        self.assertEqual(draft['model_info'], {'backend': 'remote', 'model': 'api-current', 'endpoint': 'https://api.example.test'})
        body = rs._http.post.call_args.kwargs['json']
        self.assertEqual(body['model'], 'api-current')
        self.assertEqual(body['max_tokens'], 600)
        self.assertNotIn('strict_backend', body)
        self.assertNotIn('allow_local_fallback', body)
        local.post.assert_not_called()
        rs.close()

    def test_strict_remote_outage_and_cooldown_never_fall_back(self):
        llm, rs, local = remote_llm()
        rs._http.post.side_effect = requests.ConnectionError('key=secret-provider-details')
        for _ in range(2):
            with self.assertRaises(A.DraftError) as caught:
                A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
            self.assertIn('retry', str(caught.exception))
            self.assertNotIn('secret-provider-details', str(caught.exception))
        local.post.assert_not_called()
        self.assertEqual(rs.fallbacks, 0)
        self.assertTrue(rs.cfg.get('allow_local_fallback', True))
        rs.close()

    def test_strict_remote_init_never_probes_or_starts_local(self):
        llm, rs, local = remote_llm()
        llm._client_initialized = False
        rs.ping.return_value = False
        with patch('llm_module._check_ollama_connection') as probe, patch('llm_module._run_ollama_ps') as start:
            with self.assertRaises(A.DraftError):
                A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
        probe.assert_not_called(); start.assert_not_called(); local.post.assert_not_called()
        rs.close()

    def test_direct_api_native_budget_and_no_internal_option(self):
        for backend in ('openai', 'lmstudio'):
            # The scripted client below exercises Atlas's API options without
            # requiring the optional SDK in the light public CI environment.
            with patch('llm_module.OPENAI_AVAILABLE', True):
                llm = LLM(backend, 'api-live')
            llm._client_initialized = True
            chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=json.dumps(FIELDS)))])
            llm.client = Mock()
            llm.client.chat.completions.create.return_value = iter([chunk])
            draft = A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
            self.assertEqual(draft['model_info']['model'], 'api-live')
            args = llm.client.chat.completions.create.call_args.kwargs
            self.assertEqual(args['max_tokens'], 600)
            self.assertNotIn('num_predict', args)
            self.assertNotIn('strict_backend', args)
            self.assertFalse(llm._active_requests)

    def test_unusable_output_is_error_instead_of_template(self):
        for text in ('', 'not JSON', '{}', '{"who": ""}', '[]'):
            llm = local_llm()
            llm.generate = Mock(return_value=iter([text]))
            with self.assertRaises(A.DraftError):
                A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')

    def test_model_change_discards_result(self):
        llm = local_llm()
        def changed(*args, **kwargs):
            llm.model = 'other-model'
            yield json.dumps(FIELDS)
        llm.generate = changed
        with self.assertRaisesRegex(A.DraftError, 'changed'):
            A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')

    def test_nonstring_personality_is_error_before_coercion(self):
        for who in (None, True, False, 42, 0, [], ['unexpected'], {}, {'unexpected': 'object'}):
            with self.subTest(who=who):
                llm = local_llm()
                llm.generate = Mock(return_value=iter([json.dumps({'who': who})]))
                with self.assertRaisesRegex(A.DraftError, 'usable personality'):
                    A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')

    def test_draft_validation_preserves_optional_and_manual_coercion(self):
        llm = local_llm()
        llm.generate = Mock(return_value=iter([json.dumps({**FIELDS, 'identity': 42})]))
        result = A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
        self.assertEqual(result['fields']['identity'], '42')
        self.assertEqual(A.coerce_fields({'who': ['manual']})['who'], "['manual']")

    def test_runtime_info_never_exposes_endpoint_credentials(self):
        llm, rs, _ = remote_llm()
        rs.cfg['url'] = 'https://user:secret@api.example.test:8443/v1?api_key=secret#secret'
        self.assertEqual(A.runtime_model_info(llm)['endpoint'], 'https://api.example.test:8443/v1')
        rs.close()

    def test_unavailable_runtime_never_uses_saved_default(self):
        with self.assertRaises(A.DraftError): A.runtime_model_info(None)
        llm = local_llm(); llm._remote_startup_error = 'missing remote config'
        with self.assertRaises(A.DraftError): A.build_runtime_draft(llm, 'Soupy', 'A quiet soup ghost.', '')
        llm.ollama_session.post.assert_not_called()

    def test_ordinary_remote_requests_keep_existing_fallback_policy(self):
        llm, rs, local = remote_llm()
        local.post.return_value = 'LOCAL'
        rs._http.post.side_effect = requests.ConnectionError('down')
        self.assertEqual(rs.post('http://local/api/chat', json={'messages': []}), 'LOCAL')
        self.assertEqual(rs.fallbacks, 1)
        self.assertNotIn('allow_local_fallback', local.post.call_args.kwargs)
        rs.close()


class CreatorRouteTests(unittest.IsolatedAsyncioTestCase):
    def route(self, manager, switching=False):
        tree = ast.parse((ROOT / 'server.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'agents_draft')
        node.decorator_list = []
        ns = {'Request': Request, 'JSONResponse': JSONResponse, 'asyncio': asyncio,
              '_is_owner': lambda request: True,
              'app': SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=manager, llm_switching=switching))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'server.py', 'exec'), ns)
        return ns['agents_draft']

    def request(self):
        request = Mock()
        async def body(): return {'name': 'Soupy', 'description': 'A quiet soup ghost.', 'voice': 'voice'}
        request.json = body
        return request

    async def test_route_uses_current_instance_not_legacy_model_field(self):
        llm = local_llm()
        result = await self.route(SimpleNamespace(llm=llm, llm_model='stale-default'))(self.request())
        self.assertEqual(result['model_info']['model'], 'actually-loaded')

    async def test_route_reports_model_failure_and_busy_as_errors(self):
        for manager, switching, status in ((SimpleNamespace(llm=None), False, 503), (SimpleNamespace(llm=local_llm()), True, 409)):
            response = await self.route(manager, switching)(self.request())
            self.assertEqual(response.status_code, status)
            self.assertIn('retry', json.loads(response.body)['error'])

    async def test_route_rejects_manager_replacement_during_draft(self):
        manager = SimpleNamespace(llm=local_llm())
        def replaced(*args, **kwargs):
            manager.llm = local_llm()
            return {'source': 'llm'}
        with patch.object(A, 'build_runtime_draft', side_effect=replaced):
            response = await self.route(manager)(self.request())
        self.assertEqual(response.status_code, 503)
        self.assertIn('changed', json.loads(response.body)['error'])

    async def test_route_reports_nonstring_personality_as_failure(self):
        llm = local_llm()
        llm.generate = Mock(return_value=iter(['{"who": {"unexpected": "object"}}']))
        response = await self.route(SimpleNamespace(llm=llm))(self.request())
        self.assertEqual(response.status_code, 503)
        self.assertIn('usable personality', json.loads(response.body)['error'])


if __name__ == '__main__': unittest.main()
