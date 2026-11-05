"""The memory viewer's API (design docs/designs/experience_memory.md, D7, D10; acceptance A5 viewer cases)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory as AM  # noqa: E402
import experience_memory as EM  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import memory_api as MA
        self.tmp = tempfile.TemporaryDirectory()
        self._root = AM.ROOT
        AM.ROOT = Path(self.tmp.name) / "memory"
        AM._migrated.clear()
        AM.ensure("ivy")
        self.MA = MA
        app = FastAPI()
        app.include_router(MA.router)
        self.runner = SimpleNamespace(running=lambda: False, last_reason=None, start_now=lambda: "started")
        app.state.MemoryRunner = self.runner
        self.c = TestClient(app)
        self._own = MA._owner
        MA._owner = lambda req: True
        self._shipped = MA.SHIPPED_REPORT          # a public build ships a report next to the code; none here
        MA.SHIPPED_REPORT = Path(self.tmp.name) / "shipped_report.json"

    def tearDown(self):
        self.MA.SHIPPED_REPORT = self._shipped
        self.MA._owner = self._own
        AM.ROOT = self._root
        AM._migrated.clear()
        self.tmp.cleanup()


class Viewer(Base):
    def test_remote_address_is_refused_everywhere(self):
        self.MA._owner = self._own
        for method, url in (("get", "/api/memory_state"), ("post", "/api/memory_state/settings"),
                            ("post", "/api/memory_state/facts"), ("post", "/api/memory_state/process"),
                            ("post", "/api/memory_state/old_data"), ("post", "/api/memory/ivy/episodes"),
                            ("get", "/api/memory_state/log"), ("post", "/api/memory_state/allow")):
            r = getattr(self.c, method)(url, **({"json": {}} if method == "post" else {}))
            self.assertEqual(r.status_code, 403, url)

    def test_episodes_listed_and_deleted(self):
        EM.put_episode("ivy", {"id": "c1.ivy", "date": "2026-11-03", "start": 1, "end": 2,
                                "summary": ["the group beat the boss"], "moments": ["someone fell"]})
        d = self.c.get("/api/memory/ivy").json()
        self.assertEqual(d["episodes"][0]["summary"], "the group beat the boss")
        self.assertTrue(self.c.post("/api/memory/ivy/episodes", json={"id": "c1.ivy", "action": "delete"}).json()["ok"])
        self.assertEqual(self.c.get("/api/memory/ivy").json()["episodes"], [])

    def test_facts_confirm_delete_clear(self):
        a = EM.add_owner_fact("The owner likes pizza", "I like pizza", "said", False)
        b = EM.add_owner_fact("The owner mains support", "I main support", "said", True)
        s = self.c.get("/api/memory_state").json()
        self.assertEqual([f["id"] for f in s["facts"]], [b["id"], a["id"]])
        self.c.post("/api/memory_state/facts", json={"id": a["id"], "action": "confirm"})
        self.assertTrue(all(f["confirmed"] for f in EM.owner_facts()))
        self.c.post("/api/memory_state/facts", json={"id": b["id"], "action": "delete"})
        self.assertEqual(len(EM.owner_facts()), 1)
        self.c.post("/api/memory_state/facts", json={"action": "clear"})
        self.assertEqual(EM.owner_facts(), [])

    def test_a_shipped_report_counts_only_for_its_code(self):
        EM.write_json(self.MA.SHIPPED_REPORT, {"all_zero": True, "code": "older", "date": "x"})
        self.assertEqual(self.c.post("/api/memory_state/settings", json={"key": "use_memories", "value": True})
                         .status_code, 409)
        EM.write_json(self.MA.SHIPPED_REPORT, {"all_zero": True, "code": EM.code_hash(), "date": "x"})
        r = self.c.post("/api/memory_state/settings", json={"key": "use_memories", "value": True})
        self.assertTrue(r.json()["settings"]["use_memories"])

    def test_use_memories_waits_for_a_passing_report(self):
        r = self.c.post("/api/memory_state/settings", json={"key": "use_memories", "value": True})
        self.assertEqual(r.status_code, 409)
        EM.write_json(EM.root() / "_eval_report.json", {"all_zero": True, "code": "stale", "date": "x"})
        self.assertEqual(self.c.post("/api/memory_state/settings", json={"key": "use_memories", "value": True})
                         .status_code, 409)                                     # measured on other code
        EM.write_json(EM.root() / "_eval_report.json", {"all_zero": True, "code": EM.code_hash(), "date": "x"})
        r = self.c.post("/api/memory_state/settings", json={"key": "use_memories", "value": True})
        self.assertTrue(r.json()["settings"]["use_memories"])
        self.assertTrue(self.c.post("/api/memory_state/settings", json={"key": "paused", "value": True})
                        .json()["settings"]["paused"])

    def test_process_now_is_refused_during_a_call(self):
        with mock.patch("call_boundary.in_progress", return_value=True):
            self.assertEqual(self.c.post("/api/memory_state/process").status_code, 409)
        self.assertTrue(self.c.post("/api/memory_state/process").json()["ok"])
        self.runner.start_now = lambda: "nothing"
        self.assertEqual(self.c.post("/api/memory_state/process").status_code, 409)

    def test_allow_list_and_log(self):
        r = self.c.post("/api/memory_state/allow", json={"names": ["Valorant", " Elden  Ring ", "", "Valorant"]})
        self.assertEqual(r.json()["allow"], ["Valorant", "Elden Ring"])
        self.assertEqual(self.c.get("/api/memory_state").json()["allow"], ["Valorant", "Elden Ring"])
        self.assertEqual(self.c.get("/api/memory_state/log").json()["lines"], [])

    def test_review_items(self):
        import lore
        lore._save("ivy", [{"id": 1, "title": "x", "story": "the long night", "status": "review", "at": 1}])
        d = self.c.get("/api/memory/ivy").json()
        self.assertEqual(d["review"][0]["kind"], "lore")
        self.c.post("/api/memory/ivy/review", json={"kind": "lore", "id": "1", "action": "approve"})
        self.assertEqual(lore._load("ivy")[0]["status"], "approved")

    def test_old_data_deleted_only_from_the_list(self):
        d = AM.folder("ivy")
        (d / "graph.json").write_text("{}", encoding="utf-8")
        (d / "episodes.json").write_text(json.dumps({"episodes": []}), encoding="utf-8")
        self.assertIn(d / "graph.json", self.MA._old_data_files())
        self.assertNotIn(d / "episodes.json", self.MA._old_data_files())
        with mock.patch.object(self.MA, "_old_data_files", lambda: [d / "graph.json"]):
            self.assertEqual(self.c.post("/api/memory_state/old_data").json()["deleted"], 1)
        self.assertFalse((d / "graph.json").exists())
        self.assertTrue((d / "episodes.json").exists())


if __name__ == "__main__":
    unittest.main()
