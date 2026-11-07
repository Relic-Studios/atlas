"""remote_llm translation + fallback, CPU only (no network)."""
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from unittest.mock import patch, Mock
import remote_llm as R


class FakeResp:
    def __init__(self, lines, status=200):
        self._lines, self.status_code = lines, status
    def iter_lines(self, decode_unicode=False):
        for l in self._lines:
            yield l.encode()
    def raise_for_status(self): pass
    def close(self): pass


def sse(obj):
    return "data: " + json.dumps(obj)


def ndjson(resp):
    return [json.loads(b) for b in resp.iter_content()]


class TranslateTests(unittest.TestCase):
    def test_payload_options_and_thinking(self):
        b = R.translate_payload({"messages": [{"role": "user", "content": "hi"}], "stream": True,
                                 "think": False, "options": {"temperature": .6, "top_p": .9, "num_predict": 80, "num_ctx": 4096},
                                 "tools": [{"type": "function"}]}, "local-model")
        self.assertEqual(b["model"], "local-model")
        self.assertEqual((b["temperature"], b["top_p"], b["max_tokens"]), (.6, .9, 80))
        self.assertNotIn("num_ctx", b)
        self.assertFalse(b["chat_template_kwargs"]["enable_thinking"])
        self.assertTrue(b["tools"])

    def test_provider_option_is_exact_host_and_explicit_false_only(self):
        original = R.translate_payload({'messages':[{'role':'user','content':'ordinary input'}],
            'think':False, 'format':'json', 'options':{'top_k':20}}, 'any-model', vllm=False)
        adapted = R.provider_request_body(original, 'https://api.deepseek.com/v1', think=False)
        self.assertEqual(adapted, {**original, 'thinking':{'type':'disabled'}})
        self.assertNotIn('thinking', original)
        self.assertNotIn('chat_template_kwargs', adapted)
        for endpoint in ('https://api.openai.com/v1', 'https://api.deepseek.com.evil.test/v1',
                         'https://evil.test/api.deepseek.com', 'http://localhost:8001'):
            self.assertEqual(R.provider_request_body(original, endpoint, think=False), original)
        for think in (None, True, 0):
            self.assertEqual(R.provider_request_body(original, 'https://api.deepseek.com', think=think), original)

    def test_actual_remote_requests_share_provider_option_for_voice_creator_and_warmup(self):
        requests = [
            ('voice', {'think':False, 'messages':[{'role':'user','content':'ordinary voice input'}],
                       'options':{'temperature':0.6}}, True),
            ('creator', {'think':False, 'format':'json', 'messages':[{'role':'user','content':'character fields'}]}, False),
            ('warmup', {'think':False, 'messages':[{'role':'user','content':'warm up'}]}, False),
            ('tool round', {'think':False, 'messages':[
                {'role':'assistant','tool_calls':[{'function':{'name':'lookup','arguments':{}}}]},
                {'role':'tool','content':'lookup result'}], 'tools':[{'type':'function','function':{'name':'lookup'}}]}, True),
        ]
        for name, payload, stream in requests:
            with self.subTest(request=name):
                session = R.RemoteSession({'url':'https://api.deepseek.com', 'model':'deepseek-flash',
                    'label':'DeepSeek', 'vllm':False, 'allow_local_fallback':False}, None)
                session._http.post = Mock(return_value=FakeResp([]))
                session.post('http://localhost/api/chat', json=payload, stream=stream)
                sent = session._http.post.call_args.kwargs['json']
                self.assertEqual(sent['thinking'], {'type':'disabled'})
                self.assertNotIn('chat_template_kwargs', sent)
                self.assertNotIn('top_k', sent)
                self.assertNotIn('thinking', payload)
                if name == 'creator':
                    self.assertEqual(sent['response_format'], {'type':'json_object'})
                if name == 'tool round':
                    self.assertEqual(sent['messages'][-1]['role'], 'tool')
                    self.assertEqual(sent['messages'][0]['tool_calls'][0]['id'], sent['messages'][1]['tool_call_id'])

    def test_tool_roundtrip_ids_and_image(self):
        msgs = R.translate_messages([
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "look_at_screen", "arguments": {"reason": "x"}}}]},
            {"role": "tool", "content": "screenshot", "images": ["iVBORabc"]}])
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], '{"reason": "x"}')
        self.assertEqual(msgs[1]["tool_call_id"], msgs[0]["tool_calls"][0]["id"])
        self.assertEqual(msgs[2]["role"], "user")
        self.assertTrue(msgs[2]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))


class StreamTests(unittest.TestCase):
    def test_content_stream_to_ndjson(self):
        r = R.NdjsonResponse(FakeResp([sse({"choices": [{"delta": {"content": "[SPEAK to=S1]"}}]}),
                                       sse({"choices": [{"delta": {"content": " yo"}, "finish_reason": "stop"}]}),
                                       "data: [DONE]"]), "m")
        out = ndjson(r)
        self.assertEqual("".join(o["message"]["content"] for o in out), "[SPEAK to=S1] yo")
        self.assertTrue(out[-1]["done"])

    def test_tool_call_fragments_are_assembled(self):
        r = R.NdjsonResponse(FakeResp([
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "web_search", "arguments": '{"que'}}]}}]}),
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ry": "btc"}'}}]}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]"]), "m")
        calls = [o for o in ndjson(r) if o["message"].get("tool_calls")]
        self.assertEqual(calls[0]["message"]["tool_calls"][0]["function"], {"name": "web_search", "arguments": {"query": "btc"}})

    def test_error_event_surfaces(self):
        out = ndjson(R.NdjsonResponse(FakeResp([sse({"error": "boom"})]), "m"))
        self.assertIn("boom", out[0]["error"])


class FallbackTests(unittest.TestCase):
    def test_private_provider_preserves_declared_image_support(self):
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'remote.json'
            for vision in (True, False, None):
                path.write_text(json.dumps({'url': 'http://chosen', 'model': 'chosen-model',
                                            'vision': vision}), encoding='utf-8')
                with patch.object(R, 'CONFIG_PATH', path):
                    self.assertIs(R.load_config()['vision'], vision)

    def test_remote_only_blocks_generate_except_explicit_unload(self):
        local = Mock()
        rs = R.RemoteSession({"url": "http://x", "model": "m", "label": "L",
                              "allow_local_fallback": False}, local)
        for body in ({"model": "local"}, {"model": "local", "keep_alive": -1},
                     {"model": "local", "keep_alive": 0, "prompt": "warm"}):
            with self.assertRaises(requests.ConnectionError):
                rs.post("http://localhost/api/generate", json=body)
        local.post.assert_not_called()
        rs.post("http://localhost/api/generate", json={"model": "local", "keep_alive": 0})
        local.post.assert_called_once()

    def test_remote_only_never_calls_local_on_request_failure_or_cooldown(self):
        local = Mock()
        rs = R.RemoteSession({"url": "http://x", "model": "m", "label": "L",
                              "allow_local_fallback": False}, local)
        rs._http.post = Mock(side_effect=requests.ConnectionError("down"))
        for _ in range(2):
            with self.assertRaises(requests.ConnectionError):
                rs.post("http://localhost/api/chat", json={"messages": []})
        local.post.assert_not_called()
        self.assertEqual(rs.fallbacks, 0)

    def test_remote_only_connection_check_never_calls_local(self):
        local = Mock()
        rs = R.RemoteSession({"url": "http://x", "model": "m", "label": "L",
                              "allow_local_fallback": False}, local)
        rs.ping = Mock(return_value=False)
        with self.assertRaises(requests.ConnectionError):
            rs.get("http://localhost/")
        local.get.assert_not_called()

    def test_unreachable_remote_falls_back_to_local_and_cools_down(self):
        class Local:
            calls = 0
            def post(self, url, **kw):
                Local.calls += 1; return "LOCAL"
        rs = R.RemoteSession({"url": "http://x", "model": "m", "key": "", "label": "L"}, Local())
        def boom(*a, **k): raise requests.ConnectionError("down")
        rs._http.post = boom
        self.assertEqual(rs.post("http://127.0.0.1:11434/api/chat", json={"messages": []}, stream=True, timeout=(1, 5)), "LOCAL")
        self.assertFalse(rs.status()["healthy"])
        rs._http.post = lambda *a, **k: self.fail("should not retry remote during cooldown")
        self.assertEqual(rs.post("http://127.0.0.1:11434/api/chat", json={"messages": []}, stream=True, timeout=(1, 5)), "LOCAL")
        self.assertEqual(rs.fallbacks, 1)

    def test_no_config_means_public_build(self):
        old = R.CONFIG_PATH
        R.CONFIG_PATH = Path("does/not/exist.json")
        try:
            self.assertIsNone(R.load_config())
        finally:
            R.CONFIG_PATH = old


if __name__ == "__main__":
    unittest.main()


class WatchdogTests(unittest.TestCase):
    def test_remote_only_outage_never_warms_local(self):
        rs = self.rs()
        rs.cfg["allow_local_fallback"] = False
        for now in (100, 110, 120):
            self.assertIsNone(rs.tick(False, now=now))
        self.assertFalse(rs.local_warm)
        self.assertIsNone(rs.tick(True, now=130))
        self.assertTrue(rs._remote_ok())

    def rs(self):
        return R.RemoteSession({"url": "http://x", "model": "m", "key": "", "label": "L"}, None)

    def test_one_blip_does_not_warm_local(self):
        rs = self.rs()
        self.assertIsNone(rs.tick(False, now=100))
        self.assertIsNone(rs.tick(True, now=110))
        self.assertFalse(rs.local_warm)

    def test_sustained_outage_warms_once_then_routes_local(self):
        rs = self.rs()
        rs.tick(False, now=100)
        self.assertEqual(rs.tick(False, now=110), "warm_local")
        self.assertIsNone(rs.tick(False, now=120))   # only once
        self.assertGreater(rs.down_until, 120)        # turns go local meanwhile

    def test_recovery_routes_back_immediately_unloads_after_stable(self):
        rs = self.rs()
        rs.tick(False, now=100); rs.tick(False, now=110)
        self.assertIsNone(rs.tick(True, now=120))
        self.assertTrue(rs._remote_ok())              # remote used again right away
        self.assertTrue(rs.local_warm)                # standby kept for a while
        self.assertIsNone(rs.tick(True, now=150))
        self.assertEqual(rs.tick(True, now=181), "unload_local")
        self.assertFalse(rs.local_warm)

    def test_failed_request_counts_as_down(self):
        rs = self.rs()
        rs.down_until = 10**12   # a live request just failed
        self.assertEqual(rs.tick(False, now=100), "warm_local")

    def test_flap_resets_stability_clock(self):
        rs = self.rs()
        rs.tick(False, now=100); rs.tick(False, now=110)
        rs.tick(True, now=120); rs.tick(False, now=150)
        self.assertIsNone(rs.tick(True, now=181))     # up_since restarted at 181
        self.assertTrue(rs.local_warm)


class StartupTests(unittest.TestCase):
    def test_selected_remote_is_activated_before_caller_prewarms(self):
        llm = Mock()
        cfg = {"url": "http://x", "model": "m", "label": "Remote"}
        with patch.object(R.Path, "read_text", return_value="remote"), \
                patch.object(R, "load_config", return_value=cfg), \
                patch.object(R, "activate") as activate:
            R.restore_at_startup(llm)
        activate.assert_called_once_with(llm, cfg)
        llm.prewarm.assert_not_called()

    def test_missing_remote_config_does_not_silently_use_local(self):
        with patch.object(R.Path, "read_text", return_value="remote"), \
                patch.object(R, "load_config", return_value=None):
            with self.assertRaises(requests.ConnectionError):
                R.restore_at_startup(Mock())
