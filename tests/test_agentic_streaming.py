"""Tool-enabled turns speak as they stream when no tool call can follow (10-31).

Every round of the tool loop used to be buffered whole, so a normal reply waited for the
model to finish and the answer after a search waited ~1.2 s more. A remote two-stage SPEAK
body (decided before its first word) and the forced final round (no tools offered) now
stream; any other round is still held in case a tool call follows its text."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_module import LLM  # noqa: E402
from test_llm_lifecycle import SEARCH, Transport, packet  # noqa: E402


class Live:
    """A response that records how far it has been read."""

    def __init__(self, chunks, speak_body=False):
        self.chunks, self.read = list(chunks), 0
        if speak_body:
            self.speak_body = True

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=None):
        for c in self.chunks:
            self.read += 1
            yield c

    def close(self):
        pass


def make_llm(responses, executed):
    llm = LLM("ollama", "unit-test-model", no_think=True,
              tools=[{"type": "function", "function": {"name": "web_search"}}],
              tool_executor=lambda name, args: executed.append(name) or "result")
    llm.ollama_session.close()
    llm.ollama_session = Transport(responses)
    llm._client_initialized = True
    llm._ollama_connection_ok = True
    return llm


class AgenticStreamingTests(unittest.TestCase):
    def test_speak_body_is_voiced_as_it_streams(self):
        r = Live([packet("[SPEAK to=S1] Yeah,"), packet(" I'm here."), packet(" Loud and clear.", done=True)],
                 speak_body=True)
        gen = make_llm([r], []).generate("[S1] hey", request_id="a")
        self.assertEqual(next(gen), "[SPEAK to=S1] Yeah,")
        self.assertEqual(r.read, 1)                      # first words out before the rest arrived
        self.assertEqual("".join(gen), " I'm here. Loud and clear.")

    def test_ordinary_round_is_still_buffered(self):
        r = Live([packet("[SPEAK to=S1] Sure,"), packet(" one sec.", done=True)])
        gen = make_llm([r], []).generate("[S1] hey", request_id="b")
        first = next(gen)
        self.assertEqual(r.read, 2)                      # whole round read before anything is said
        self.assertEqual(first + "".join(gen), "[SPEAK to=S1] Sure, one sec.")

    def test_forced_final_round_after_a_search_streams(self):
        executed = []
        search = Live([packet(tools=[SEARCH], done=True)])
        answer = Live([packet("[SPEAK to=S1] It's 18"), packet(" degrees.", done=True)])
        gen = make_llm([search, answer], executed).generate("[S1] weather?", request_id="c")
        self.assertEqual(next(gen), "[SPEAK to=S1] It's 18")
        self.assertEqual(answer.read, 1)
        self.assertEqual(executed, ["web_search"])

    def test_tool_call_after_spoken_text_is_ignored(self):
        executed = []
        r = Live([packet("[SPEAK to=S1] Let me think."), packet(tools=[SEARCH], done=True)], speak_body=True)
        llm = make_llm([r], executed)
        self.assertEqual("".join(llm.generate("[S1] hm", request_id="d")), "[SPEAK to=S1] Let me think.")
        self.assertEqual(executed, [])
        self.assertEqual(len(llm.ollama_session.payloads), 1)


if __name__ == "__main__":
    unittest.main()
