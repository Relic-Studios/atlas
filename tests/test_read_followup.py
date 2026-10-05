"""read_page follow-up round after a figure-seeking search (live 10-01 'no live numbers')."""
import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_llm_lifecycle import Response, Transport, packet, SEARCH
from llm_module import LLM

READ = {"function": {"name": "read_page", "arguments": {"url": "https://x.example/w"}}}
TOOLS = [{"type": "function", "function": {"name": "web_search"}},
         {"type": "function", "function": {"name": "read_page"}}]


def make(responses, calls):
    def ex(name, args):
        calls.append(name)
        return "page text 23 degrees" if name == "read_page" else "links only"
    llm = LLM("ollama", "m", no_think=True, tools=TOOLS, tool_executor=ex)
    llm.ollama_session.close(); llm.ollama_session = Transport(responses)
    llm._client_initialized = True; llm._ollama_connection_ok = True
    return llm


def names(p):
    return [t["function"]["name"] for t in p.get("tools", [])]


class ReadFollowup(unittest.TestCase):
    def test_figure_question_gets_read_only_round(self):
        calls = []
        rs = [Response([packet(tools=[SEARCH], done=True)]),
              Response([packet(tools=[READ], done=True)]),
              Response([packet("[SPEAK to=S1] It's 23.", done=True)])]
        llm = make(rs, calls)
        out = "".join(llm.generate("[S1] what's the weather in paris", request_id="r", participation=True))
        self.assertEqual(out, "[SPEAK to=S1] It's 23.")
        self.assertEqual(calls, ["web_search", "read_page"])
        p = llm.ollama_session.payloads
        self.assertEqual(names(p[0]), ["web_search", "read_page"])
        self.assertEqual(names(p[1]), ["read_page"])        # bonus round: read only
        self.assertNotIn("tools", p[2])                       # then forced answer

    def test_model_may_skip_reading(self):
        calls = []
        rs = [Response([packet(tools=[SEARCH], done=True)]),
              Response([packet("[SPEAK to=S1] Bitcoin's at 84k.", done=True)])]
        llm = make(rs, calls)
        out = "".join(llm.generate("[S1] bitcoin price today", request_id="r", participation=True))
        self.assertEqual(out, "[SPEAK to=S1] Bitcoin's at 84k.")
        self.assertEqual(calls, ["web_search"])

    def test_opinion_question_no_bonus(self):
        calls = []
        rs = [Response([packet(tools=[SEARCH], done=True)]),
              Response([packet("[SPEAK to=S1] Kendrick.", done=True)])]
        llm = make(rs, calls)
        list(llm.generate("[S1] search who the best rapper is", request_id="r", participation=True))
        self.assertNotIn("tools", llm.ollama_session.payloads[1])

    def test_can_disable(self):
        os.environ["ATLAS_READ_FOLLOWUP"] = "0"
        try:
            calls = []
            rs = [Response([packet(tools=[SEARCH], done=True)]),
                  Response([packet("[SPEAK to=S1] Dunno.", done=True)])]
            llm = make(rs, calls)
            list(llm.generate("[S1] weather in paris", request_id="r", participation=True))
            self.assertNotIn("tools", llm.ollama_session.payloads[1])
        finally:
            os.environ.pop("ATLAS_READ_FOLLOWUP")


if __name__ == "__main__":
    unittest.main()
