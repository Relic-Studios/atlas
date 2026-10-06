"""Live call 10-06 ('he's remembering too much'): recall echoed lines from the last few
minutes (already in the conversation log), recalled the very line being answered, filled
slots with bare questions people asked the agent, and matched anything containing its name."""
import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hypergraph_memory as H  # noqa: E402
from test_hypergraph_memory import FakeEmb  # noqa: E402


class Freshness(unittest.TestCase):
    def setUp(self):
        self.now = [1_000_000.0]
        self.td = Path(tempfile.mkdtemp())
        self.g = H.HyperMemory("max", root=self.td, embedder=FakeEmb(), clock=lambda: self.now[0])
        self._age = H.RECALL_MIN_AGE_S
        H.RECALL_MIN_AGE_S = 900.0

    def tearDown(self):
        H.RECALL_MIN_AGE_S = self._age

    def add(self, text, who="Corona", ago=0.0):
        self.now[0] -= ago
        self.g.add(text, who=who)
        self.now[0] += ago

    def live(self, q):
        return [h["text"] for h in self.g.retrieve(q, k=3, min_age_s=H.RECALL_MIN_AGE_S, mark_active=False)]

    def test_recent_lines_are_not_recalled(self):
        self.add("Corona filmed the stop motion canadian music video in the garage", ago=3600)
        self.add("why did he turn canadian in the stop motion video", ago=60)
        got = self.live("canadian stop motion video")
        self.assertIn("Corona filmed the stop motion canadian music video in the garage", got)
        self.assertNotIn("why did he turn canadian in the stop motion video", got)
        # benchmarks / non-live callers still see everything
        self.assertEqual(len(self.g.retrieve("canadian stop motion video", k=3, mark_active=False)), 2)

    def test_bare_questions_skipped_live(self):
        self.add("Max do you have hair under that canadian hat?", ago=3600)
        self.add("Corona owns a canadian hat from the stop motion shoot.", ago=3600)
        got = self.live("canadian hat stop motion")
        self.assertNotIn("Max do you have hair under that canadian hat?", got)
        self.assertIn("Corona owns a canadian hat from the stop motion shoot.", got)
        self.assertTrue(H._only_questions("Where is this film from?"))
        self.assertFalse(H._only_questions("Moving on. How are we tonight? I'm great."))

    def test_agent_name_stripped(self):
        self.assertEqual(H.strip_agent_name("max", "Max, what's the meaning of life?"),
                         "what's the meaning of life?")
        self.assertEqual(H.strip_agent_name("max", "thomasina rocks"), "thomasina rocks")
        svc = H.MemoryService(root=self.td, embedder=FakeEmb())
        self.assertEqual(svc.recall_note("max", "Max?"), "")


if __name__ == "__main__":
    unittest.main()
