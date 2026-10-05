"""CPU-only lifecycle regressions; every HTTP request uses a scripted transport."""
import copy
import json
import unittest
from unittest.mock import patch

from llm_module import LLM


def packet(content="", tools=None, done=False):
    message = {"content": content}
    if tools:
        message["tool_calls"] = tools
    return (json.dumps({"message": message, "done": done}) + "\n").encode()


SEARCH = {"function": {"name": "web_search", "arguments": {"query": "weather"}}}


class Response:
    def __init__(self, chunks=(), error=None, on_read=None):
        self.chunks = chunks
        self.error = error
        self.on_read = on_read
        self.closes = 0

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        if self.on_read:
            self.on_read()
        yield from self.chunks
        if self.error:
            raise self.error

    def close(self):
        self.closes += 1


class Transport:
    def __init__(self, responses, on_post=None):
        self.responses = iter(responses)
        self.on_post = on_post
        self.payloads = []

    def post(self, url, **kwargs):
        self.payloads.append(copy.deepcopy(kwargs["json"]))
        if self.on_post:
            self.on_post()
        return next(self.responses)


class LifecycleTests(unittest.TestCase):
    def make_llm(self, responses=(), agentic=False, on_post=None):
        llm = LLM("ollama", "unit-test-model", no_think=True,
                  tools=[{"type": "function", "function": {"name": "web_search"}}]
                  if agentic else None,
                  tool_executor=(lambda name, args: "search result") if agentic else None)
        llm.ollama_session.close()
        llm.ollama_session = Transport(responses, on_post)
        llm._client_initialized = True
        llm._ollama_connection_ok = True
        return llm

    def test_cancel_during_post_does_not_register_late_response(self):
        for agentic in (False, True):
            with self.subTest(agentic=agentic):
                response = Response([packet("must not escape", done=True)])
                llm = self.make_llm([response], agentic=agentic)
                cancelled = []
                llm.ollama_session.on_post = lambda: cancelled.append(llm.cancel_generation("r"))
                self.assertEqual(list(llm.generate("hello", request_id="r")), [])
                self.assertEqual(cancelled, [True])
                self.assertEqual(response.closes, 1)
                self.assertEqual(llm._active_requests, {})

    def test_uncancelled_nonetype_attribute_error_is_not_swallowed(self):
        for agentic in (False, True):
            with self.subTest(agentic=agentic):
                error = AttributeError("'NoneType' object has no attribute 'readline'")
                response = Response(error=error)
                llm = self.make_llm([response], agentic=agentic)
                with self.assertRaises(AttributeError):
                    list(llm.generate("hello", request_id="r"))
                self.assertEqual(response.closes, 1)
                self.assertEqual(llm._active_requests, {})

    def test_cancel_from_tool_callback_stops_executor_and_followup(self):
        response = Response([packet(tools=[SEARCH], done=True)])
        llm = self.make_llm([response, Response([packet("late", done=True)])], agentic=True)
        executed = []
        llm.tool_executor = lambda *args: executed.append(args) or "result"
        llm.on_tool_call = lambda calls: llm.cancel_generation("r")
        self.assertEqual(list(llm.generate("hello", request_id="r")), [])
        self.assertEqual(executed, [])
        self.assertEqual(len(llm.ollama_session.payloads), 1)
        self.assertEqual(response.closes, 1)
        self.assertEqual(llm._active_requests, {})

    def test_tool_round_content_is_hidden_until_final_answer(self):
        first = Response([packet("[SPEAK to=S1] Searching."), packet(tools=[SEARCH], done=True)])
        second = Response([packet("[SPEAK to=S1] Found "), packet("it.", done=True)])
        llm = self.make_llm([first, second], agentic=True)
        output = list(llm.generate("[S1] weather?", request_id="r", participation=True))
        self.assertEqual(output, ["[SPEAK to=S1] Found ", "it."])
        self.assertEqual([first.closes, second.closes], [1, 1])
        followup = llm.ollama_session.payloads[1]
        self.assertEqual(followup["messages"][-1]["role"], "tool")
        self.assertTrue(followup["messages"][-1]["content"].startswith("search result"))
        self.assertEqual(followup["messages"][-2]["tool_calls"], [SEARCH])
        self.assertIn("CURRENT TURN: speaker=S1", followup["messages"][0]["content"])
        import llm_module as _L
        self.assertEqual(followup["options"]["temperature"], _L.PARTICIPATION_SAMPLING["temperature"])
        self.assertNotIn("tools", followup)  # budget spent: final round must answer, not search


if __name__ == "__main__":
    unittest.main()
