"""Remote two-stage sampling: greedy header/tool decision, sampled body."""
import json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remote_llm as R


class _Resp:
    def __init__(self, obj=None, lines=None, code=200):
        self.status_code, self._obj, self._lines, self.text = code, obj, lines or [], ""
    def json(self): return self._obj
    def iter_lines(self, decode_unicode=False): return iter(self._lines)
    def raise_for_status(self): pass
    def close(self): pass


class _Http:
    def __init__(self, first, second=None):
        self.first, self.second, self.bodies = first, second, []
    def post(self, url, json=None, stream=False, timeout=None):
        self.bodies.append(json)
        return self.first if len(self.bodies) == 1 else self.second


def _sse(*chunks):
    out = [("data: " + json.dumps({"choices": [{"delta": {"content": c}}]})).encode() for c in chunks]
    return out + [b"data: [DONE]"]


def _body():
    return {"model": "m", "stream": True, "temperature": 0.6, "top_p": 0.9,
            "messages": [{"role": "system", "content": "persona ... If silent, output only [HOLD]."},
                         {"role": "user", "content": "[S1] Ivy what should we play"}]}


def _text(resp):
    out = []
    for line in resp.iter_content():
        m = json.loads(line).get("message") or {}
        out.append(m.get("content") or "")
    return "".join(out)


class TwoStageTests(unittest.TestCase):
    def sess(self, http):
        s = R.RemoteSession({"url": "http://x", "model": "m", "label": "t"}, local=None)
        s._http = http
        return s

    def test_speak_header_greedy_body_sampled(self):
        http = _Http(_Resp({"choices": [{"message": {"content": "[SPEAK to=S1"}}]}),
                     _Resp(lines=_sse(" Deep", " Rock.")))
        r = self.sess(http)._two_stage(_body(), (4, 60))
        self.assertEqual(_text(r), "[SPEAK to=S1] Deep Rock.")
        s1, s2 = http.bodies
        self.assertEqual(s1["temperature"], 0.0)
        self.assertEqual(s1["stop"], ["]"])
        self.assertEqual(s2["temperature"], 0.6)
        self.assertTrue(s2["continue_final_message"])
        self.assertEqual(s2["messages"][-1], {"role": "assistant", "content": "[SPEAK to=S1]"})

    def test_hold_needs_one_request(self):
        http = _Http(_Resp({"choices": [{"message": {"content": "[HOLD"}}]}))
        r = self.sess(http)._two_stage(_body(), (4, 60))
        self.assertEqual(_text(r), "[HOLD]")
        self.assertEqual(len(http.bodies), 1)

    def test_tool_call_decided_greedily(self):
        # Stage 1 (16 tokens) truncates a tool call to '{}'; it must be redone
        # with room to finish, and the FULL arguments must reach the executor.
        cut = {"function": {"name": "web_search", "arguments": "{}"}}
        full = {"function": {"name": "web_search", "arguments": '{"query": "Max Didymus"}'}}
        http = _Http(_Resp({"choices": [{"message": {"content": "", "tool_calls": [cut]},
                                         "finish_reason": "tool_calls"}]}),
                     _Resp({"choices": [{"message": {"content": "", "tool_calls": [full]}}]}))
        r = self.sess(http)._two_stage(_body(), (4, 60))
        self.assertEqual(len(http.bodies), 2)
        self.assertEqual(http.bodies[1]["max_tokens"], 256)
        self.assertNotIn("stop", http.bodies[1])
        lines = [json.loads(l) for l in r.iter_content()]
        calls = [l["message"]["tool_calls"] for l in lines if l.get("message", {}).get("tool_calls")]
        self.assertEqual(calls[0][0]["function"],
                         {"name": "web_search", "arguments": {"query": "Max Didymus"}})

    def test_broken_tool_json_falls_back(self):
        bad = {"function": {"name": "web_search", "arguments": '{"query": "Tho'}}
        http = _Http(_Resp({"choices": [{"message": {"content": "", "tool_calls": [bad]}}]}),
                     _Resp({"choices": [{"message": {"content": "", "tool_calls": [bad]}}]}))
        self.assertIsNone(self.sess(http)._two_stage(_body(), (4, 60)))

    def test_garbage_header_falls_back(self):
        http = _Http(_Resp({"choices": [{"message": {"content": "Sure, I think"}}]}))
        self.assertIsNone(self.sess(http)._two_stage(_body(), (4, 60)))

    def test_only_participation_calls(self):
        b = _body(); b["messages"][0]["content"] = "summarize this"
        self.assertFalse(R._is_participation(b))
        self.assertTrue(R._is_participation(_body()))


if __name__ == "__main__":
    unittest.main()
