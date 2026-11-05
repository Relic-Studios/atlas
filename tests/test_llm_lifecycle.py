"""CPU-only lifecycle regressions; every HTTP request uses a scripted transport."""
import copy
import json
import unittest
from unittest.mock import patch, Mock

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
    def test_remote_prewarm_reports_actual_model_and_never_checks_local(self):
        llm = LLM("ollama", "local-model")
        llm.remote = Mock(cfg={"url": "http://remote:30000", "model": "model", "allow_local_fallback": False})
        llm.remote.ping.return_value = True
        with patch("llm_module._check_ollama_connection") as local_check, \
                patch("llm_module._run_ollama_ps") as local_ps, \
                patch.object(llm, "generate", return_value=iter(["OK"])), \
                self.assertLogs("llm_module", level="INFO") as logs:
            self.assertTrue(llm.prewarm(max_retries=0))
        local_check.assert_not_called()
        local_ps.assert_not_called()
        self.assertTrue(any("remote model at http://remote:30000" in line for line in logs.output))
        self.assertFalse(any("prewarm for 'local-model'" in line for line in logs.output))

    def test_remote_outage_does_not_start_or_probe_ollama(self):
        llm = LLM("ollama", "local-model")
        llm.remote = Mock(cfg={"url": "http://remote:30000", "model": "model", "allow_local_fallback": False})
        llm.remote.ping.return_value = False
        with patch("llm_module._check_ollama_connection") as local_check, \
                patch("llm_module._run_ollama_ps") as local_ps:
            self.assertFalse(llm._lazy_initialize_clients())
        local_check.assert_not_called()
        local_ps.assert_not_called()

    def test_remote_recovers_after_transient_startup_outage(self):
        llm = LLM("ollama", "local-model")
        llm.remote = Mock(cfg={"url": "http://remote:30000", "model": "model", "allow_local_fallback": False})
        llm.remote.ping.side_effect = [False, True]
        with patch("llm_module._run_ollama_ps") as local_ps:
            self.assertFalse(llm._lazy_initialize_clients())
            self.assertTrue(llm._lazy_initialize_clients())
        local_ps.assert_not_called()
        self.assertEqual(llm.remote.ping.call_count, 2)

    def test_missing_selected_remote_config_blocks_later_generation(self):
        llm = LLM("ollama", "local-model")
        llm._remote_startup_error = "Missing remote configuration"
        llm.ollama_session = Mock()
        with self.assertRaisesRegex(ConnectionError, "local inference remains disabled"):
            list(llm.generate("hello"))
        llm.ollama_session.get.assert_not_called()
        llm.ollama_session.post.assert_not_called()

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

    def test_cancel_during_post_interrupts_the_remote_session(self):
        """A session still inside post() (think_listen's thought finish) gets interrupt()."""
        response = Response([packet("must not escape", done=True)])
        llm = self.make_llm([response])
        interrupts = []
        llm.ollama_session.interrupt = lambda: interrupts.append(1)
        llm.ollama_session.on_post = lambda: llm.cancel_generation("r")
        self.assertEqual(list(llm.generate("hello", request_id="r")), [])
        self.assertEqual(interrupts, [1])
        self.assertFalse(llm.cancel_generation("r"))          # nothing active: no interrupt
        self.assertEqual(interrupts, [1])

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
