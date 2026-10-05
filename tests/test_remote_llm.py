"""remote_llm translation + fallback, CPU only (no network)."""
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
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
