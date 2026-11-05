"""Remote two-stage sampling: greedy header/tool decision, sampled body."""
import json, math, sys, unittest
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
    def setUp(self):
        _old_one = R.ONE_REQUEST
        R.ONE_REQUEST = False          # these pin today's two-request pattern
        self.addCleanup(setattr, R, "ONE_REQUEST", _old_one)

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


    def _with_search_tool(self, user_line):
        b = _body()
        b["tools"] = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
        b["messages"][-1]["content"] = user_line
        return b

    def test_spoken_search_command_forces_the_tool(self):
        http = _Http(_Resp({"choices": [{"message": {"content": "[HOLD"}}]}))
        self.sess(http)._two_stage(self._with_search_tool("[S1] Ivy, search up the patch notes"), (4, 60))
        self.assertEqual(http.bodies[0]["tool_choice"]["function"]["name"], "web_search")

    def test_task_cue_never_forces_a_new_search(self):
        from tasks import cue_text
        cue = cue_text({"id": 3, "asker": "S1", "text": "patch notes"})
        http = _Http(_Resp({"choices": [{"message": {"content": "[HOLD"}}]}))
        self.sess(http)._two_stage(self._with_search_tool(cue), (4, 60))
        self.assertNotIn("tool_choice", http.bodies[0])





def _sse_lp(*pieces, top_first=None):
    """Chat SSE chunks with per-token logprobs. pieces: (text, [(token, top_token, ...)]) where each token
    tuple is (sampled, argmax); top_first: the first token's top_logprobs ({token: logprob})."""
    import math
    out = []
    first = True
    for text, toks in pieces:
        content = []
        for sampled, best in toks:
            top = [{"token": best, "logprob": -0.01}]
            if sampled != best:
                top.append({"token": sampled, "logprob": -4.0})
            if first and top_first:
                top = [{"token": k, "logprob": v} for k, v in top_first.items()]
            first = False
            content.append({"token": sampled, "logprob": -0.01 if sampled == best else -4.0, "top_logprobs": top})
        out.append(("data: " + json.dumps({"choices": [{"delta": {"content": text},
                                                        "logprobs": {"content": content}}]})).encode())
    return out + [b"data: [DONE]"]


_SPEAK = [("[S", [("[S", "[S")]), ("PEAK to", [("PE", "PE"), ("AK", "AK"), (" to", " to")]),
          ("=S1] Deep", [("=S", "=S"), ("1", "1"), ("]", "]"), (" Deep", " Deep")]), (" Rock.", [(" Rock", " Rock"), (".", ".")])]


class OneRequestTests(unittest.TestCase):
    """11-03: today's decision and reply in one sampled request; the header counts only when every header
    token was the model's top choice (what the greedy header request would have produced)."""

    def setUp(self):
        old = R.ONE_REQUEST
        R.ONE_REQUEST = True
        self.addCleanup(setattr, R, "ONE_REQUEST", old)

    def sess(self, http):
        s = R.RemoteSession({"url": "http://x", "model": "m", "label": "t"}, local=None)
        s._http = http
        return s

    def test_speak_and_reply_in_one_request(self):
        http = _Http(_Resp(lines=_sse_lp(*_SPEAK)))
        R.TURN_LOG.clear()
        r = self.sess(http)._two_stage(_body(), (4, 60))
        self.assertEqual(_text(r), "[SPEAK to=S1] Deep Rock.")
        self.assertEqual(len(http.bodies), 1)
        b = http.bodies[0]
        self.assertEqual((b["temperature"], b["top_p"], b["stream"]), (0.6, 0.9, True))   # the reply's own sampling
        self.assertTrue(b["logprobs"])
        self.assertNotIn("stop", b)
        self.assertEqual((R.TURN_LOG[-1]["decision"], R.TURN_LOG[-1]["one"]), ("speak", "ok"))

    def test_header_off_the_top_choice_falls_back_to_two_requests(self):
        off = [("[S", [("[S", "[S")]), ("PEAK to=S2]", [("PE", "PE"), ("AK", "AK"), (" to", " to"), ("=S", "=S"),
                                                         ("2", "1"), ("]", "]")]), (" Hi.", [(" Hi", " Hi")])]
        seq = [_Resp(lines=_sse_lp(*off)), _Resp({"choices": [{"message": {"content": "[SPEAK to=S1"}}]}),
               _Resp(lines=_sse(" Deep", " Rock."))]
        http = _Http(None)
        http.post = lambda url, json=None, stream=False, timeout=None: (http.bodies.append(json), seq.pop(0))[1]
        r = self.sess(http)._two_stage(_body(), (4, 60))
        self.assertEqual(_text(r), "[SPEAK to=S1] Deep Rock.")       # the greedy header, as before
        self.assertEqual(len(http.bodies), 3)
        self.assertEqual(http.bodies[1]["temperature"], 0.0)

    def test_reply_tokens_in_the_header_chunk_are_not_checked(self):
        # Speculative decoding sends "=S1] Deep" as one chunk; " Deep" is a sampled reply token.
        lines = _sse_lp(("[SPEAK to", [("[S", "[S"), ("PE", "PE"), ("AK", "AK"), (" to", " to")]),
                        ("=S1] Oh", [("=S", "=S"), ("1", "1"), ("]", "]"), (" Oh", " Deep")]), (" no.", [(" no", " so")]))
        http = _Http(_Resp(lines=lines))
        self.assertEqual(_text(self.sess(http)._two_stage(_body(), (4, 60))), "[SPEAK to=S1] Oh no.")
        self.assertEqual(len(http.bodies), 1)

    def test_near_tie_counts_as_the_top_choice(self):
        lines = _sse_lp(*_SPEAK)
        ev = json.loads(lines[2][5:])                          # "=S1] Deep": make "1" a 0.50/0.50 near tie
        lp = ev["choices"][0]["logprobs"]["content"][1]
        lp["logprob"] = -0.70
        lp["top_logprobs"] = [{"token": "2", "logprob": -0.69}, {"token": "1", "logprob": -0.70}]
        lines[2] = ("data: " + json.dumps(ev)).encode()
        http = _Http(_Resp(lines=lines))
        self.assertEqual(_text(self.sess(http)._two_stage(_body(), (4, 60))), "[SPEAK to=S1] Deep Rock.")
        self.assertEqual(len(http.bodies), 1)

    def test_first_token_must_be_the_exact_top_choice(self):
        # SPEAK vs HOLD near tie: a sampled [H when the top choice is [S must not HOLD a directed line.
        tie = [("[H", [("[H", "[S")]), ("OLD]", [("OLD", "OLD"), ("]", "]")])]
        lines = _sse_lp(*tie)
        ev = json.loads(lines[0][5:])
        ev["choices"][0]["logprobs"]["content"][0]["logprob"] = -0.70
        ev["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [{"token": "[S", "logprob": -0.69},
                                                                       {"token": "[H", "logprob": -0.70}]
        lines[0] = ("data: " + json.dumps(ev)).encode()
        seq = [_Resp(lines=lines), _Resp({"choices": [{"message": {"content": "[SPEAK to=S1"}}]}),
               _Resp(lines=_sse(" Hi."))]
        http = _Http(None)
        http.post = lambda url, json=None, stream=False, timeout=None: (http.bodies.append(json), seq.pop(0))[1]
        self.assertEqual(_text(self.sess(http)._two_stage(_body(), (4, 60))), "[SPEAK to=S1] Hi.")
        self.assertEqual(len(http.bodies), 3)

    def test_calibrated_hold_never_replaces_a_greedy_tool_call(self):
        # Review 11-03: top first token <tool_call>, sampled [S; on a line that asks nothing (thresh > 0)
        # the decision must still be today's greedy one (the tool), not a HOLD from P(HOLD).
        lines = _sse_lp(*_SPEAK, top_first={"<tool_call>": math.log(0.6), "[S": math.log(0.35),
                                            "[H": math.log(0.05)})
        call = {"function": {"name": "web_search", "arguments": '{"query": "x"}'}}
        seq = [_Resp(lines=lines), _Resp({"choices": [{"message": {"content": "", "tool_calls": [call]}}]}),
               _Resp({"choices": [{"message": {"content": "", "tool_calls": [call]}}]})]   # greedy + its redo
        http = _Http(None)
        http.post = lambda url, json=None, stream=False, timeout=None: (http.bodies.append(json), seq.pop(0))[1]
        s = self.sess(http)
        s.next_hold_thresh = 0.03
        r = s._two_stage(_body(), (4, 60))
        self.assertIn("tool_calls", json.loads(next(r.iter_content()))["message"])

    def test_hold_by_threshold_and_greedy_hold(self):
        lines = _sse_lp(*_SPEAK, top_first={"[S": math.log(0.9), "[H": math.log(0.1)})
        s = self.sess(_Http(_Resp(lines=lines)))
        s.next_hold_thresh = 0.05
        self.assertEqual(_text(s._two_stage(_body(), (4, 60))), "[HOLD]")
        self.assertAlmostEqual(R.LAST_HOLD_SHARE, 0.1, places=3)
        hold = [("[H", [("[H", "[H")]), ("OLD]", [("OLD", "OLD"), ("]", "]")])]
        http = _Http(_Resp(lines=_sse_lp(*hold)))
        self.assertEqual(_text(self.sess(http)._two_stage(_body(), (4, 60))), "[HOLD]")
        self.assertEqual(len(http.bodies), 1)

    def test_tool_call_goes_to_the_two_requests(self):
        tool = [("data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
            "name": "web_search", "arguments": ""}}]}}]})).encode()]
        full = {"function": {"name": "web_search", "arguments": '{"query": "x"}'}}
        cut = {"function": {"name": "web_search", "arguments": "{}"}}
        seq = [_Resp(lines=tool), _Resp({"choices": [{"message": {"content": "", "tool_calls": [cut]}}]}),
               _Resp({"choices": [{"message": {"content": "", "tool_calls": [full]}}]})]
        http = _Http(None)
        http.post = lambda url, json=None, stream=False, timeout=None: (http.bodies.append(json), seq.pop(0))[1]
        r = self.sess(http)._two_stage(_body(), (4, 60))
        lines = [json.loads(l) for l in r.iter_content()]
        self.assertEqual(lines[0]["message"]["tool_calls"][0]["function"]["arguments"], {"query": "x"})
        self.assertEqual(len(http.bodies), 3)          # one request, then today's greedy decision + its redo

    def test_switch_off_and_forced_search_use_two_requests(self):
        R.ONE_REQUEST = False
        http = _Http(_Resp({"choices": [{"message": {"content": "[SPEAK to=S1"}}]}), _Resp(lines=_sse(" Hi.")))
        self.assertEqual(_text(self.sess(http)._two_stage(_body(), (4, 60))), "[SPEAK to=S1] Hi.")
        self.assertEqual(http.bodies[0]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
