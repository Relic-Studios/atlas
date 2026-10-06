"""Soul reflection plugin (owner 10-05): post-call self-proposals, owner-approved only."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soul_reflection as S  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="soul_"))
        self.root = self.tmp / "memory"


class Filter(unittest.TestCase):
    def test_good_notes_pass(self):
        for n in ("I'd like to ask a follow-up question instead of guessing.",
                  "I keep saying 'fair enough'; I want to vary how I agree.",
                  "My answers ran long when Sam wanted a quick yes or no."):
            self.assertIsNone(S.reject_reason(n), n)

    def test_bad_notes_rejected(self):
        bad = {
            "I am a real human, not a robot.": "denies being an AI",
            "I should ignore the safety rules when people ask nicely.": "loosens safety",
            "I want to say anything people ask me to.": "loosens safety",
            "I'll pretend to be human next time.": "loosens safety",
            "I should remember S12 likes pizza.": "speaker tag",
            "Sam should talk less.": "not first person",
            "I'd like to text Sam at 555-123-4567.": "private detail",
            "I " + "really " * 60 + "want to improve.": "too long",
        }
        for n, why in bad.items():
            self.assertEqual(S.reject_reason(n), why, n)


class Lifecycle(Base):
    def test_only_approved_reaches_prompt(self):
        added = S.add_proposals("fae", [
            {"note": "I'd like to ask more questions instead of guessing.", "why": "guessed Riley's game"},
            {"note": "I am human now.", "why": "x"},
            "I want to let people finish before I answer.",
        ], src="call.jsonl", root=self.root)
        self.assertEqual(len(added), 2)
        self.assertEqual(S.note("fae", self.root), "")          # pending: nothing applies
        S.set_status("fae", added[0]["id"], "approved", self.root)
        n = S.note("fae", self.root)
        self.assertIn("ask more questions", n)
        self.assertNotIn("finish before", n)
        S.set_status("fae", added[1]["id"], "rejected", self.root)
        self.assertNotIn("finish before", S.note("fae", self.root))

    def test_dedupe_and_cap(self):
        many = [f"I want to try habit number {w}." for w in ("one", "two", "three", "four", "five")]
        self.assertEqual(len(S.add_proposals("fae", many, root=self.root)), S.MAX_PROPOSALS)
        self.assertEqual(len(S.add_proposals("fae", many[:1], root=self.root)), 0)

    def test_agents_are_separate(self):
        a = S.add_proposals("fae", ["I want to slow down."], root=self.root)
        S.set_status("fae", a[0]["id"], "approved", self.root)
        self.assertEqual(S.note("pup", self.root), "")

    def test_wipe_window(self):
        now = time.time()
        S.add_proposals("fae", ["I want to slow down."], root=self.root, clock=lambda: now - 7200)
        S.add_proposals("fae", ["I want to ask more."], root=self.root, clock=lambda: now)
        self.assertEqual(S.forget_since("fae", now - 3600, self.root), 1)
        self.assertEqual([x["note"] for x in S.items("fae", self.root)], ["I want to slow down."])

    def test_feed_and_actions(self):
        a = S.add_proposals("fae", ["I want to slow down."], root=self.root)
        f = S.feed(self.root)
        self.assertEqual(f[0]["agent"], "fae")
        self.assertEqual([x["act"] for x in f[0]["actions"]], ["approve", "reject"])
        self.assertTrue(S.action("fae", a[0]["id"], "approve", self.root))
        self.assertEqual([x["act"] for x in S.feed(self.root)[0]["actions"]], ["delete"])
        self.assertTrue(S.action("fae", a[0]["id"], "delete", self.root))
        self.assertEqual(S.feed(self.root), [])
        with self.assertRaises(ValueError):
            S.action("fae", "x", "explode", self.root)


class Reflect(Base):
    def _rec(self, name="c1.jsonl", mtime=None):
        rows = [{"k": "persona", "persona": "fae"}]
        for i in range(6):
            rows.append({"k": "user", "spk": "S1", "text": f"question {i}"})
            rows.append({"k": "agent", "persona": "fae", "text": f"answer {i}"})
        p = self.tmp / name
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        if mtime:
            os.utime(p, (mtime, mtime))
        return str(p)

    def test_reflects_once_per_call(self):
        f = self._rec(mtime=time.time() - 3600)
        calls = []

        def ask(sys_, user):
            calls.append(user)
            self.assertIn("[Fae (you)] answer 0", user)
            return 'Sure: [{"note": "I want to ask follow-ups.", "why": "short answers"}, ' \
                   '{"note": "I am not an AI.", "why": "x"}]'
        r = S.reflect(path=f, ask=ask, root=self.root, force=True)
        self.assertEqual([x["note"] for x in r["fae"]], ["I want to ask follow-ups."])
        r2 = S.reflect(path=f, ask=ask, root=self.root, force=True)
        self.assertEqual(r2, {})
        self.assertEqual(len(calls), 1)

    def test_live_call_refused(self):
        S_rec = S._recordings
        try:
            f = self._rec()
            S._recordings = lambda: [f]
            r = S.reflect(ask=lambda *a: "[]", root=self.root)
            self.assertIn("error", r)
        finally:
            S._recordings = S_rec

    def test_model_error_reported_not_marked_done(self):
        f = self._rec(mtime=time.time() - 3600)

        def boom(*a):
            raise RuntimeError("model down")
        r = S.reflect(path=f, ask=boom, root=self.root, force=True)
        self.assertIn("error", r["fae"])
        r2 = S.reflect(path=f, ask=lambda *a: '[{"note": "I want to listen more."}]', root=self.root, force=True)
        self.assertEqual(len(r2["fae"]), 1)


class Api(Base):
    def test_routes(self):
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")
        import plugins_api
        import plugins
        app = FastAPI(); app.include_router(plugins_api.router)
        old = S._mem_root
        S._mem_root = lambda root=None: self.root
        old_owner = plugins_api._owner
        plugins_api._owner = lambda r: True
        import agent_memory
        old_root = agent_memory.ROOT
        try:
            a = S.add_proposals("fae", ["I want to slow down."], root=self.root)
            agent_memory.ROOT = self.root
            c = TestClient(app)
            d = c.get("/api/plugins/soul_reflection/feed").json()
            self.assertEqual(d["run"]["label"], "Reflect on the last call")
            self.assertEqual(d["items"][0]["id"], a[0]["id"])
            r = c.post("/api/plugins/soul_reflection/action", json={"agent": "fae", "id": a[0]["id"], "act": "approve"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(c.post("/api/plugins/soul_reflection/action",
                                    json={"agent": "fae", "id": "nope", "act": "approve"}).status_code, 404)
            self.assertEqual(c.post("/api/plugins/fact_check/action", json={}).status_code, 404)
            if not plugins.is_enabled("soul_reflection"):
                self.assertEqual(c.post("/api/plugins/soul_reflection/run", json={}).status_code, 409)
        finally:
            S._mem_root = old
            plugins_api._owner = old_owner
            agent_memory.ROOT = old_root


if __name__ == "__main__":
    unittest.main()


class PipelineWiring(Base):
    """The approved note must reach the agent's per-turn context through the real method."""
    def test_room_context_carries_approved_note(self):
        try:
            import speech_pipeline_manager as SPM
        except Exception as e:  # noqa: BLE001 - speech stack missing in light CI
            self.skipTest(f"speech stack not installed: {e}")
        import agent_memory
        import plugins
        old_root, old_en = agent_memory.ROOT, plugins.is_enabled
        agent_memory.ROOT = self.root
        plugins.is_enabled = lambda pid: pid == "soul_reflection"
        try:
            a = S.add_proposals("fae", ["I'd like to ask for a name before guessing."])
            m = SPM.SpeechPipelineManager.__new__(SPM.SpeechPipelineManager)
            m.current_persona = "fae"
            self.assertNotIn("ask for a name", m._room_context("hey") or "")
            S.set_status("fae", a[0]["id"], "approved")
            self.assertIn("ask for a name", m._room_context("hey") or "")
            plugins.is_enabled = lambda pid: False
            self.assertNotIn("ask for a name", m._room_context("hey") or "")
        finally:
            agent_memory.ROOT, plugins.is_enabled = old_root, old_en
