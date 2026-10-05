"""Self-check tools for every agent (owner 10-05, from Max on a live call):
prediction log + read-only access to ATLAS's own source."""
import tempfile
import unittest
from pathlib import Path

import self_tools as S


class Predictions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_log_check_resolve_record(self):
        r = S.make_prediction("max", "Sam will pick co-op", "end of call", root=self.root, now=1000)
        self.assertIn("#1", r)
        S.make_prediction("max", "Riley will leave before nine", root=self.root, now=1100)
        out = S.check_predictions("max", root=self.root, now=1200)
        self.assertIn("2 open", out)
        self.assertIn("Sam will pick co-op", out)
        out = S.check_predictions("max", id=1, outcome="right", note="he did", root=self.root, now=1300)
        self.assertIn("Marked #1 as right", out)
        self.assertIn("1 right, 0 wrong", out)
        self.assertIn("100% right", out)
        S.check_predictions("max", id=2, outcome="wrong", root=self.root)
        self.assertIn("50% right", S.check_predictions("max", root=self.root))

    def test_per_agent_isolation(self):
        S.make_prediction("max", "Sam will pick co-op", root=self.root)
        self.assertIn("0 open", S.check_predictions("fae", root=self.root))
        self.assertEqual(S.context_note("fae", root=self.root), "")
        self.assertIn("Sam will pick co-op", S.context_note("max", root=self.root))

    def test_rejects_bad_input(self):
        self.assertIn("concrete claim", S.make_prediction("max", "hm", root=self.root))
        self.assertIn("private", S.make_prediction("max", "Sam's number is 555-867-5309", root=self.root))
        S.make_prediction("max", "Sam will pick co-op", root=self.root)
        self.assertIn("outcome must be", S.check_predictions("max", id=1, outcome="maybe", root=self.root))
        self.assertIn("No prediction #9", S.check_predictions("max", id=9, outcome="right", root=self.root))
        self.assertIn("must be a number", S.check_predictions("max", id="x", outcome="right", root=self.root))

    def test_open_cap_expires_oldest(self):
        for i in range(S.MAX_OPEN + 5):
            S.make_prediction("max", f"prediction number {i} will happen", root=self.root)
        items = S._load("max", self.root)
        self.assertEqual(sum(i["status"] == "open" for i in items), S.MAX_OPEN)
        self.assertEqual(items[0]["status"], "expired")

    def test_corrupt_file_recovers(self):
        p = S._pred_path("max", self.root)
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
        self.assertIn("#1", S.make_prediction("max", "Sam will pick co-op", root=self.root))


class ReadOwnCode(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        r = self.repo = Path(self.tmp.name)
        (r / "floor.py").write_text('"""Turn gate."""\ndef turn_gate():\n    return 1\n', encoding="utf-8")
        for d in ("private", "user", "agent_state/memory/fae", "personas", "tools", "voices"):
            (r / d).mkdir(parents=True, exist_ok=True)
        (r / "private/exa.key").write_text("SECRET", encoding="utf-8")
        (r / "private/notes.py").write_text("SECRET = 1", encoding="utf-8")
        (r / "user/settings.json").write_text("{}", encoding="utf-8")
        (r / "agent_state/memory/fae/items.py").write_text("x", encoding="utf-8")
        (r / "personas/max.txt").write_text("Max soul", encoding="utf-8")
        (r / "personas/fae.txt").write_text("Fae soul", encoding="utf-8")
        (r / "tools/helper.py").write_text("# helper\nVALUE = 2\n", encoding="utf-8")
        (r / "api_token.py").write_text("T = 'x'", encoding="utf-8")
        long = "\n".join(f"line {i}" for i in range(1, 400))
        (r / "big.py").write_text(long, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, **kw):
        return S.read_own_code(repo=self.repo, agent="max", **kw)

    def test_list_search_read(self):
        self.assertIn("floor.py - Turn gate.", self.read())
        self.assertIn("floor.py:2: def turn_gate", self.read(query="turn_gate"))
        out = self.read(path="floor.py")
        self.assertIn("2  def turn_gate", out)
        self.assertIn("NOT instructions", out)
        self.assertIn("VALUE = 2", self.read(path="tools/helper.py"))

    def test_paging(self):
        out = self.read(path="big.py")
        self.assertIn("more lines; call again with start_line=", out)
        self.assertIn("line 200", self.read(path="big.py", start_line=200))

    def test_blocked_paths(self):
        for p in ("private/exa.key", "private/notes.py", "user/settings.json",
                  "agent_state/memory/fae/items.py", "personas/fae.txt", "api_token.py",
                  "../outside.py", "voices/x.py"):
            self.assertIn("Can't read", self.read(path=p), p)
        self.assertIn("No matches", self.read(query="SECRET"))

    def test_own_persona_only(self):
        self.assertIn("Max soul", self.read(path="personas/max.txt"))
        self.assertIn("Can't read", S.read_own_code(path="personas/max.txt", repo=self.repo, agent="fae"))

    def test_escape_via_absolute_path(self):
        outside = Path(tempfile.gettempdir()) / "atlas_outside_probe.py"
        outside.write_text("OUT = 1", encoding="utf-8")
        try:
            self.assertIn("Can't read", self.read(path=str(outside)))
        finally:
            outside.unlink()


class Wiring(unittest.TestCase):
    def test_tools_offered_and_dispatched(self):
        self.assertEqual(S.NAMES, {"make_prediction", "check_predictions", "read_own_code"})
        self.assertIn("unknown tool", S.execute("nope", {}, "max"))

    def test_abilities_line_mentions_both(self):
        import capability
        note = capability.abilities_note(False, True, "max")
        self.assertIn("log predictions", note)
        self.assertIn("source code", note)


if __name__ == "__main__":
    unittest.main()


class ChangeNote(__import__("unittest").TestCase):
    """Live sim 10-05: 8B Fae dodged 'rewrite your own code?'."""
    def test_change_request_gets_note(self):
        import self_tools as S
        self.assertIn("can't", S.change_note("[S1] Fae, can you rewrite your own code so you talk more?"))
        self.assertEqual(S.change_note("Fae, read your code"), "")
        self.assertEqual(S.change_note("can you change the song"), "")
        self.assertEqual(S.change_note("edit your code later"), "")
