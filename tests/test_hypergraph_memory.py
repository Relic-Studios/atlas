"""Hypergraph memory: Hebbian learning, decay, isolation, persistence, concurrency (CPU, fake embedder)."""
import hashlib, json, os, sys, tempfile, threading, time, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hypergraph_memory as H


def W(i):
    """A distinct letters-only token per integer (content_nodes ignores digits)."""
    a = "bcdfghjklmnpqrstvwxz"
    out = ""
    i += 1
    while i:
        i, r = divmod(i, 20)
        out += a[r] + "o"
    return "zq" + out


DEV_NAMES = (__import__("pathlib").Path(__file__).resolve().parents[1] / "dev_pack").exists()  # public export renames dev agents
class FakeEmb:
    model_name = "fake"
    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(256, np.float32)
            for w in H.content_nodes(t):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.stack(out) if out else np.zeros((0, 256), np.float32)


class Clock:
    def __init__(self): self.t = 1_000_000.0
    def __call__(self): return self.t


def mem(td, agent="a", clock=None):
    return H.HyperMemory(agent, root=Path(td), embedder=FakeEmb(), clock=clock or Clock())


class Basics(unittest.TestCase):
    def test_add_retrieve_and_gate(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td)
            e = g.add("Maya adopted a grey kitten named Pickle", who="Maya")
            g.add("Jake plays bass in a ska band", who="Jake")
            r = g.retrieve("grey kitten named Pickle")
            self.assertEqual(r[0]["id"], e)
            self.assertEqual(r[0]["who"], "Maya")
            self.assertEqual(g.retrieve("quantum chromodynamics lecture notes"), [])

    def test_scrubs_tags_numbers_slurs_and_thin_lines(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td)
            # contact details: the whole line is skipped (10-03), not stored redacted
            self.assertIsNone(g.add("S12 text me at 555 123 4567 about the guitar lessons", who="S12"))
            e = g.add("S12 wants to start guitar lessons with a jazz teacher downtown", who="S12")
            self.assertIsNotNone(e)
            stored = g.edges[0]
            self.assertNotIn("S12", stored["text"])
            self.assertEqual(stored["who"], "")                 # S-tags never become names
            self.assertIsNone(g.add("lol yeah"))                # too thin to be a memory

    def test_near_duplicate_reinforces_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td)
            a = g.add("Leo builds mechanical keyboards with lavender switches")
            b = g.add("Leo builds mechanical keyboards with lavender switches")
            self.assertEqual(a, b); self.assertEqual(len(g.edges), 1)
            self.assertGreater(g.edges[0]["w"], H.INIT_W)


class Hebbian(unittest.TestCase):
    def test_coactivation_links_and_strengthens(self):
        with tempfile.TemporaryDirectory() as td:
            c = Clock(); g = mem(td, clock=c)
            a = g.add("Tess collects vintage polaroid cameras")
            b = g.add("lighthouse keeper fantasy novel draft chapters")
            w0 = g.edges[0]["w"]
            g.reinforce([a, b], used_ids=[a, b])
            self.assertGreater(g.edges[0]["w"], w0)
            self.assertGreater(g._h(g._pk(a, b), c()), 0.25)
            # the linked memory now rides along when its partner is recalled
            got = [x["id"] for x in g.retrieve("vintage polaroid cameras", k=3)]
            self.assertIn(a, got)

    def test_unused_retrieval_does_not_strengthen(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td)
            a = g.add("Omar prints little dragons on his printer")
            w0 = g.edges[0]["w"]
            g.reinforce([a], used_ids=[])
            self.assertEqual(g.edges[0]["w"], w0)

    def test_used_by_reply_overlap(self):
        hits = [{"id": "x", "text": "Rufus the dog had leg surgery"}, {"id": "y", "text": "Liverpool won again"}]
        self.assertEqual(H.used_by("How's Rufus doing after the surgery?", hits), ["x"])

    def test_temporal_link_to_active_memories(self):
        with tempfile.TemporaryDirectory() as td:
            c = Clock(); g = mem(td, clock=c)
            a = g.add("Nina trains for the Chicago marathon")
            g.retrieve("Chicago marathon training")          # a is now active
            b = g.add("Nina hurt her knee running hills")
            self.assertGreater(g._h(g._pk(a, b), c()), 0.0)


class Decay(unittest.TestCase):
    def test_half_life_and_pin_floor(self):
        with tempfile.TemporaryDirectory() as td:
            c = Clock(); g = mem(td, clock=c)
            g.add("Carlos is learning Japanese every day")
            g.add("The owner built these agents himself", pinned=True)
            w = g._w(g.edges[0], c())
            c.t += H.HALF_LIFE_S
            self.assertAlmostEqual(g._w(g.edges[0], c()), w / 2, places=5)
            c.t += 50 * H.HALF_LIFE_S
            self.assertEqual(g._w(g.edges[1], c()), H.PIN_FLOOR)

    def test_prune_old_weak_keeps_pinned_and_fresh(self):
        with tempfile.TemporaryDirectory() as td:
            c = Clock(); g = mem(td, clock=c)
            old = g.add("Theo failed parallel parking on his driving test")
            pin = g.add("The owner built these agents himself", pinned=True)
            c.t += 30 * 86400
            fresh = g.add("Rosa plants tomatoes on the balcony garden")
            g.maintain()
            ids = {e["id"] for e in g.edges}
            self.assertNotIn(old, ids); self.assertIn(pin, ids); self.assertIn(fresh, ids)

    def test_use_resets_decay(self):
        with tempfile.TemporaryDirectory() as td:
            c = Clock(); g = mem(td, clock=c)
            a = g.add("Ben supports Liverpool since age six")
            c.t += 2 * H.HALF_LIFE_S
            g.reinforce([a], used_ids=[a])
            self.assertGreater(g._w(g.edges[0], c()), H.INIT_W / 4)

    def test_size_cap(self):
        with tempfile.TemporaryDirectory() as td:
            old = H.MAX_EDGES; H.MAX_EDGES = 20
            try:
                g = mem(td)
                for i in range(40):
                    g.add(f"topic {W(i)} alpha {W(i+500)}")
                self.assertLessEqual(len(g.edges), 20)
                self.assertEqual(len(g.vecs), len(g.edges))
            finally:
                H.MAX_EDGES = old


class Isolation(unittest.TestCase):
    def test_agents_never_share_memories(self):
        with tempfile.TemporaryDirectory() as td:
            ivy, fae = mem(td, "ivy"), mem(td, "fae")
            ivy.add("Maya adopted a grey kitten named Pickle")
            self.assertEqual(fae.retrieve("grey kitten named Pickle"), [])
            ivy.save(); fae.save()
            self.assertTrue((Path(td) / "ivy" / "graph.json").exists())
            self.assertNotIn("Pickle", (Path(td) / "fae" / "graph.json").read_text())

    @__import__("unittest").skipUnless(DEV_NAMES, "reads the dev-only export tool")
    def test_export_excludes_memory_stores(self):
        src = (Path(__file__).resolve().parents[1] / "tools" / "export_public.py").read_text(encoding="utf-8")
        self.assertIn('"agent_state/*"', src)
        self.assertTrue(str(H.ROOT).replace("\\", "/").endswith("agent_state/memory"))


class Persistence(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td)
            a = g.add("Dana's bakery sells sourdough loaves daily"); b = g.add("Dana hosts a haunted house party")
            g.reinforce([a, b]); g.save()
            g2 = mem(td)
            self.assertEqual(len(g2.edges), 2); self.assertEqual(len(g2.hebb), 1)
            self.assertEqual(g2.retrieve("sourdough loaves bakery")[0]["id"], a)
            self.assertNotEqual(g2.add("Priya studies for the bar exam soon"), a)   # ids keep counting

    def test_corrupt_store_recovers_and_keeps_copy(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td); g.add("Ivy quit to become a tattoo artist"); g.save()
            (Path(td) / "a" / "graph.json").write_text("{not json")
            g2 = mem(td)
            self.assertEqual(g2.edges, [])
            self.assertTrue(any(p.name.startswith("corrupt_") for p in (Path(td) / "a").iterdir()))
            g2.add("Kai speedruns Celeste under thirty minutes"); g2.save()
            self.assertEqual(len(mem(td).edges), 1)

    def test_mismatched_embeddings_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td); g.add("Sam broke a wrist skateboarding badly"); g.save()
            np.save(Path(td) / "a" / "emb.npy", np.zeros((3, 256), np.float32))
            self.assertEqual(mem(td).edges, [])


class Concurrency(unittest.TestCase):
    def test_parallel_adds_and_reads_stay_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            g = mem(td); errs = []
            def writer(n):
                try:
                    for i in range(40): g.add(f"writer {W(1000+n)} item {W(i)} fact {W(100+n*40+i)}")
                except Exception as e: errs.append(e)
            def reader():
                try:
                    for i in range(80): g.retrieve(f"item {W(i % 40)}")
                except Exception as e: errs.append(e)
            ts = [threading.Thread(target=writer, args=(n,)) for n in range(4)] + \
                 [threading.Thread(target=reader) for _ in range(4)]
            [t.start() for t in ts]; [t.join() for t in ts]
            self.assertEqual(errs, [])
            self.assertEqual(len(g.vecs), len(g.edges)); self.assertEqual(len(g.edges), 160)
            self.assertEqual(sum(len(v) for v in g.node_index.values()),
                             sum(len(e["nodes"]) for e in g.edges))

    def test_service_queue_never_blocks_and_flushes(self):
        with tempfile.TemporaryDirectory() as td:
            s = H.MemoryService(root=Path(td), embedder=FakeEmb())
            t = time.perf_counter()
            for i in range(50):
                s.observe_user("fae", "Maya", f"Maya mentioned guitar lessons {W(i)} on {W(i+300)}")
            self.assertLess(time.perf_counter() - t, 0.05)
            s.flush()
            self.assertEqual(len(s.graph("fae").edges), 50)
            note = s.recall_note("fae", f"guitar lessons {W(3)} {W(303)}")
            self.assertIn("Maya", note)
            s.observe_reply("fae", f"How are the guitar lessons going {W(3)}?")
            s.flush(); s.close()
            self.assertTrue((Path(td) / "fae" / "graph.json").exists())

    def test_recall_budget_times_out_cleanly(self):
        class Slow(FakeEmb):
            def encode(self, texts):
                time.sleep(0.5); return super().encode(texts)
        with tempfile.TemporaryDirectory() as td:
            s = H.MemoryService(root=Path(td), embedder=Slow())
            g = s.graph("a"); g.add("Omar prints little dragons", vec=FakeEmb().encode(["Omar prints little dragons"])[0])
            t = time.perf_counter()
            self.assertEqual(s.recall_note("a", "little dragons", budget_s=0.1), "")
            self.assertLess(time.perf_counter() - t, 0.3)
            s.close()


if __name__ == "__main__":
    unittest.main()
