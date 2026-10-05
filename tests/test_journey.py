"""Guide strip (journey.py): Set up -> Make an agent -> Join a call -> Review."""
import json, os, tempfile, time, unittest
from pathlib import Path
from unittest import mock

import journey


def _rec(d: Path, name: str, lines, mtime=None):
    p = d / "2026-10-04" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    if mtime:
        os.utime(p, (mtime, mtime))
    return p


class Journey(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.p_agents = mock.patch.object(journey, "_user_agents", return_value=["mira"])
        self.p_pend = mock.patch.object(journey, "_pending_candidates", return_value=0)
        self.p_pub = mock.patch("user_settings.is_public", return_value=True)
        for p in (self.p_agents, self.p_pend, self.p_pub):
            p.start()
            self.addCleanup(p.stop)

    def st(self, **s):
        return journey.state(settings=s, rec_dir=self.tmp)

    def test_fresh_install_starts_at_setup(self):
        self.assertEqual(self.st()["current"], "setup")

    def test_no_agent_points_at_plus(self):
        self.p_agents.stop(); self.addCleanup(lambda: None)
        with mock.patch.object(journey, "_user_agents", return_value=[]):
            j = self.st(setup_complete=True)
        self.p_agents.start()
        self.assertEqual(j["current"], "agent")

    def test_real_recording_format_counts_as_call(self):
        # recordings use "k", not "kind" (live bug: calls counted 0)
        _rec(self.tmp, "a.jsonl", [{"t": 1, "k": "user", "spk": "user", "text": "hi"}])
        _rec(self.tmp, "b.jsonl", [{"t": 1, "k": "agent", "text": "only the agent"}])
        j = self.st(setup_complete=True)
        self.assertEqual(j["calls"], 1)
        self.assertEqual(j["current"], "review")

    def test_reviewed_then_new_call_points_back_at_review(self):
        _rec(self.tmp, "a.jsonl", [{"k": "user", "text": "hi"}], mtime=time.time() - 100)
        j = self.st(setup_complete=True, reviewed_at=time.time())
        self.assertTrue(all(s["done"] for s in j["steps"]))
        _rec(self.tmp, "b.jsonl", [{"k": "user", "text": "again"}], mtime=time.time() + 5)
        self.assertEqual(self.st(setup_complete=True, reviewed_at=time.time())["current"], "review")

    def test_pending_memories_keep_review_open(self):
        _rec(self.tmp, "a.jsonl", [{"k": "user", "text": "hi"}], mtime=time.time() - 100)
        with mock.patch.object(journey, "_pending_candidates", return_value=2):
            j = self.st(setup_complete=True, reviewed_at=time.time())
        self.assertEqual(j["current"], "review")
        self.assertIn("2 memory", j["hint"])

    def test_garbage_lines_dont_crash(self):
        p = self.tmp / "x" / "bad.jsonl"; p.parent.mkdir(parents=True)
        p.write_bytes(b'{"user" not json\n\xff\xfe')
        self.assertEqual(self.st(setup_complete=True)["calls"], 0)


if __name__ == "__main__":
    unittest.main()


class FirstRunOnly(Journey):
    """Owner 10-04: no 'Your loop' recap for an established user. The card is
    first-run help and disappears for good after the first conversation."""

    def test_new_user_sees_card(self):
        self.assertTrue(self.st()["show"])

    def test_set_up_with_agent_but_no_call_still_sees_card(self):
        self.assertTrue(self.st(setup_complete=True)["show"])

    def test_returning_user_never_sees_card(self):
        _rec(self.tmp, "a.jsonl", [{"k": "user", "text": "hi"}])
        self.assertFalse(self.st(setup_complete=True)["show"])
        with mock.patch.object(journey, "_pending_candidates", return_value=3):
            self.assertFalse(self.st(setup_complete=True)["show"])

    def test_dismiss_hides_it(self):
        self.assertFalse(self.st(journey_hidden=True)["show"])
