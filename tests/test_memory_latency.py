"""Recall latency work (10-04): incremental hub cache + recall prefetch from partials.

Owner asked what delay memory adds per reply: ~140 ms, ~85% of it embedding the line.
Prefetching from partial transcripts moves that off the turn; the hub cache keeps a
full store (4000 memories, 136 ms O(n^2) rebuild after every live add) from regressing.
These tests pin that both are exact (same results as the slow path) and safe (owner
forget/wipe/delete never leaves a stale prefetched memory behind).
"""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_memory as AM  # noqa: E402
import call_memory as CM  # noqa: E402
import hypergraph_memory as H  # noqa: E402
from test_hypergraph_memory import FakeEmb, W  # noqa: E402

FACTS = ["Relic keeps a lizard named Pickle in a glass tank",
         "Nina works night shifts as a paramedic downtown",
         "Sam is learning bass guitar for a punk band",
         "the water temple puzzle took Relic three whole evenings",
         "Jo baked sourdough bread that collapsed in the oven"]


def full_hub(vecs):
    n = len(vecs)
    if n < 8:
        return np.zeros(n, np.float32)
    m = vecs @ vecs.T
    np.fill_diagonal(m, np.nan)
    c = np.nanmean(m, axis=1)
    return np.clip(c - float(np.mean(c)), 0.0, None).astype(np.float32)


class SlowEmb(FakeEmb):
    """Fake embedder with a real-ish encode cost so in-flight waits are exercised."""
    def __init__(self, delay=0.05):
        self.delay, self.calls = delay, 0

    def encode(self, texts):
        self.calls += len(texts)
        time.sleep(self.delay)
        return super().encode(texts)


class Base(unittest.TestCase):
    emb_cls = FakeEmb

    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        self.root = self.td / "memory"
        self._saved = (AM.ROOT, CM.ROOT, H.ROOT)
        AM.ROOT = CM.ROOT = H.ROOT = self.root
        AM._migrated.clear()
        self.emb = self.emb_cls()
        self.svc = H.MemoryService(root=self.root, embedder=self.emb)
        g = self.svc.graph("fae")
        for i, f in enumerate(FACTS):
            g.add(f, who="")
        for i in range(12):   # filler so the hub penalty is live (n >= 8)
            g.add(f"{W(i)} {W(i + 50)} {W(i + 100)} chatter", who="")

    def tearDown(self):
        self.svc.close()
        AM.ROOT, CM.ROOT, H.ROOT = self._saved
        AM._migrated.clear()

    def wait_pf(self, agent="fae", timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            with self.svc._pf_cv:
                if self.svc._pf_running is None and self.svc._pf_pending is None:
                    return
            time.sleep(0.01)


class HubCache(Base):
    def test_incremental_hub_equals_full_rebuild(self):
        g = self.svc.graph("fae")
        g._hub_scores()                                  # build once
        for i in range(30):                              # live adds: incremental path
            g.add(f"{W(i + 200)} {W(i + 300)} {W(i % 5)} more talk", who="")
            np.testing.assert_allclose(g._hub_scores(), full_hub(g.vecs), atol=1e-5)
        self.assertIs(g._hub_src, g.vecs, "stayed on the incremental path")

    def test_hub_rebuilds_after_forget_and_reload(self):
        g = self.svc.graph("fae")
        g._hub_scores()
        g.forget([g.edges[0]["id"], g.edges[3]["id"]])
        np.testing.assert_allclose(g._hub_scores(), full_hub(g.vecs), atol=1e-5)
        g.save()
        g2 = H.HyperMemory("fae", root=self.root, embedder=self.emb)
        np.testing.assert_allclose(g2._hub_scores(), full_hub(g2.vecs), atol=1e-5)

    def test_duplicate_add_keeps_cache_valid(self):
        g = self.svc.graph("fae")
        g._hub_scores()
        g.add(FACTS[0], who="")                          # near-duplicate: reinforce, no new row
        np.testing.assert_allclose(g._hub_scores(), full_hub(g.vecs), atol=1e-5)


class Prefetch(Base):
    def test_qkey_ignores_tags_case_punctuation(self):
        self.assertEqual(H.qkey("[S3] Hey, what's Nina's JOB?"), H.qkey("hey what's nina's job"))
        self.assertNotEqual(H.qkey("what's Nina's job"), H.qkey("what's Nina's dog"))

    def test_prefetch_hit_returns_same_hits_as_slow_path(self):
        q = "what does Nina do for work, the paramedic thing"
        slow = [h["id"] for h in self.svc.graph("fae").retrieve(q, mark_active=False)]
        self.svc.prefetch("fae", q)
        self.wait_pf()
        note = self.svc.recall_note("fae", "[S2] " + q + "?")
        self.assertEqual(self.svc.pf_stats["hit"], 1)
        self.assertEqual([h["id"] for h in self.svc._last_hits["fae"]], slow)
        self.assertIn("paramedic", note)

    def test_prefetch_hit_marks_active_only_when_used(self):
        g = self.svc.graph("fae")
        g.active = []
        self.svc.prefetch("fae", "tell me about the lizard Pickle in the glass tank")
        self.wait_pf()
        self.assertEqual(g.active, [], "a partial is not a turn: no Hebbian activity yet")
        self.svc.recall_note("fae", "tell me about the lizard Pickle in the glass tank")
        self.assertTrue(g.active)

    def test_different_final_text_misses_and_stays_correct(self):
        self.svc.prefetch("fae", "tell me about the lizard")
        self.wait_pf()
        note = self.svc.recall_note("fae", "how is Sam doing with the bass guitar punk band")
        self.assertEqual(self.svc.pf_stats["miss"], 1)
        self.assertIn("bass", note)

    def test_latest_partial_wins(self):
        for p in ["Sam is", "Sam is learning", "Sam is learning bass guitar", "Sam is learning bass guitar punk"]:
            self.svc.prefetch("fae", p)
        self.wait_pf()
        self.assertEqual(self.svc._pf["fae"]["key"], H.qkey("Sam is learning bass guitar punk"))

    def test_other_agent_never_gets_the_prefetch(self):
        self.svc.graph("pup").add("Pup hid snacks under the porch steps", who="")
        self.svc.prefetch("fae", "Relic keeps a lizard named Pickle")
        self.wait_pf()
        self.svc.recall_note("pup", "Relic keeps a lizard named Pickle")
        self.assertEqual(self.svc.pf_stats["hit"], 0)
        self.assertNotIn("Pickle", " ".join(h["text"] for h in self.svc._last_hits.get("pup", [])))

    def test_owner_forget_and_wipe_invalidate_prefetch(self):
        q = "Relic keeps a lizard named Pickle in a glass tank"
        self.svc.prefetch("fae", q)
        self.wait_pf()
        eid = self.svc._pf["fae"]["hits"][0]["id"]
        self.svc.forget("fae", [eid])
        note = self.svc.recall_note("fae", q)
        self.assertNotIn("Pickle", note, "forgotten memory came back from the prefetch cache")
        self.svc.prefetch("fae", "Jo baked sourdough bread that collapsed")
        self.wait_pf()
        self.svc.wipe("fae", 0)
        self.assertEqual(self.svc.recall_note("fae", "Jo baked sourdough bread that collapsed"), "")

    def test_drop_agent_invalidates_prefetch(self):
        q = "Nina works night shifts as a paramedic"
        self.svc.prefetch("fae", q)
        self.wait_pf()
        self.svc.drop_agent("fae")
        self.assertNotIn("fae", self.svc._pf)


class InFlight(Base):
    emb_cls = SlowEmb

    def test_recall_waits_for_matching_inflight_prefetch_instead_of_recomputing(self):
        q = "Nina works night shifts as a paramedic downtown, right"
        self.svc.prefetch("fae", q)
        time.sleep(0.01)                     # prefetch is mid-encode
        calls = self.emb.calls
        t = time.perf_counter()
        note = self.svc.recall_note("fae", q)
        dt = time.perf_counter() - t
        self.assertIn("paramedic", note)
        self.assertEqual(self.svc.pf_stats["waited"], 1)
        self.assertLessEqual(self.emb.calls - calls, 1, "recall re-embedded instead of waiting")
        self.assertLess(dt, 0.35)

    def test_budget_still_caps_a_slow_inflight(self):
        self.emb.delay = 0.6
        q = "Sam is learning bass guitar for a punk band"
        self.svc.prefetch("fae", q)
        time.sleep(0.01)
        t = time.perf_counter()
        self.svc.recall_note("fae", q, budget_s=0.2)
        self.assertLess(time.perf_counter() - t, 0.5)

    def test_concurrent_prefetch_and_recall_stress(self):
        errs = []

        def talk():
            try:
                for i in range(40):
                    self.svc.prefetch("fae", f"Sam is learning bass {W(i)}")
            except Exception as e:  # noqa: BLE001
                errs.append(e)
        self.emb.delay = 0.002
        th = threading.Thread(target=talk)
        th.start()
        for i in range(20):
            self.svc.recall_note("fae", f"Sam is learning bass {W(i)}", budget_s=0.3)
        th.join()
        self.assertEqual(errs, [])


if __name__ == "__main__":
    unittest.main()


class SameQueryRule(__import__("unittest").TestCase):
    """bench_recall_latency 10-04: finals often differ from the last partial by one word."""
    def test_rules(self):
        import hypergraph_memory as H
        q = H.qkey
        self.assertTrue(H.same_query(q("my sister moved to denver last spring"),
                                     q("[S2] My sister moved to Denver last spring, yeah")))
        self.assertTrue(H.same_query(q("I think we should play chess tonight ok"),
                                     q("I think we should play chess tonight")))
        # short lines: one word is the meaning
        self.assertFalse(H.same_query(q("what's Nina's job"), q("what's Sam's job")))
        # two words changed in a long line: different question
        self.assertFalse(H.same_query(q("did Nina ever get that job at the bakery"),
                                      q("did Sam ever get that job at the garage")))
