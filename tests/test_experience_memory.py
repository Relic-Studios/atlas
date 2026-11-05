"""Experience memory stores (design docs/designs/experience_memory.md, D1, D10; acceptance A8 store cases)."""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent_memory  # noqa: E402
import experience_memory as EM  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        agent_memory._migrated.clear()

    def tearDown(self):
        agent_memory.ROOT = self._root
        agent_memory._migrated.clear()
        self.tmp.cleanup()

    def ep(self, eid, start=100.0, end=200.0, summary=("the group cleared the dungeon boss",)):
        return {"id": eid, "call": eid.split(".")[0], "agent": "ivy", "date": "2026-11-04", "start": start,
                "end": end, "summary": list(summary), "topics": ["dungeon"], "games": ["Elden Ring"],
                "decisions": [], "jokes": ["the cursed dice"], "moments": ["someone fell off the map three times"]}


class Settings(Base):
    def test_defaults_and_set(self):
        self.assertEqual(EM.settings(), {"use_memories": False, "paused": False, "cloud_allowed": False})
        self.assertTrue(EM.set_setting("paused", True)["paused"])
        self.assertTrue(EM.settings()["paused"])
        with self.assertRaises(KeyError):
            EM.set_setting("nonsense", True)


class OwnerFacts(Base):
    def test_add_confirm_delete_and_usable(self):
        a = EM.add_owner_fact("The owner likes pizza", "I like pizza", "said", True, now=10)
        b = EM.add_owner_fact("The owner mains support", "I main support", "extracted", True, call="c1", now=20)
        c = EM.add_owner_fact("The owner is a night owl", "I'm a night owl", "said", False, now=30)
        self.assertEqual([f["id"] for f in EM.owner_facts()], [c["id"], b["id"], a["id"]])      # newest first
        self.assertEqual([f["id"] for f in EM.usable_owner_facts(False)], [a["id"]])            # trial
        self.assertEqual([f["id"] for f in EM.usable_owner_facts(True)], [b["id"], a["id"]])
        self.assertEqual(EM.review_count(), 1)
        EM.confirm_fact(c["id"])
        self.assertEqual(EM.review_count(), 0)
        EM.to_confirm([a["id"]])
        self.assertNotIn(a["id"], [f["id"] for f in EM.usable_owner_facts(True)])
        self.assertTrue(EM.delete_fact(b["id"]))
        self.assertIsNone(EM.add_owner_fact("again", "I", "extracted", True, fid=b["id"]))      # a re-run can't undo it

    def test_clear(self):
        EM.add_owner_fact("The owner likes pizza", "I like pizza", "said", True)
        self.assertEqual(EM.clear_owner_facts(now=50), 1)
        self.assertEqual(EM.owner_facts(), [])
        self.assertTrue(EM.wiped("_owner", 10, 20))


class SameFactOnce(Base):
    def test_the_same_fact_is_kept_once_and_the_stronger_source_confirms_it(self):
        """A "remember that" fact the pass also extracts from the same owner line is not stored twice."""
        a = EM.add_owner_fact("The owner loves curry", "I love curry", "said", False)
        b = EM.add_owner_fact("the owner loves curry.", "I love curry", "extracted", True, call="c1")
        self.assertEqual(b["id"], a["id"])
        facts = EM.owner_facts()
        self.assertEqual(len(facts), 1)
        self.assertTrue(facts[0]["confirmed"])
        EM.add_owner_fact("The owner loves pizza", "I love pizza", "said", True)
        self.assertEqual(len(EM.owner_facts()), 2)

    def test_a_deleted_fact_does_not_come_back_from_the_pass(self):
        f = EM.add_owner_fact("The owner loves curry", "I love curry", "said", True)
        self.assertTrue(EM.delete_fact(f["id"]))
        self.assertIsNone(EM.add_owner_fact("The owner loves curry", "I love curry", "extracted", True, call="c1"))
        self.assertEqual(EM.owner_facts(), [])
        self.assertNotIn("curry", (EM.root() / "_wipes.json").read_text(encoding="utf-8"))
        again = EM.add_owner_fact("The owner loves curry", "I love curry", "said", True)   # said again: a new request
        self.assertIsNotNone(again)


class Episodes(Base):
    def test_replace_by_id_delete_and_wipe(self):
        self.assertTrue(EM.put_episode("ivy", self.ep("c1.ivy")))
        self.assertTrue(EM.put_episode("ivy", self.ep("c1.ivy", summary=("a rerun of the same call",))))
        self.assertEqual([e["summary"] for e in EM.episodes("ivy")], [["a rerun of the same call"]])
        self.assertEqual(EM.episodes("max"), [])                                     # agents never share
        self.assertTrue(EM.delete_episode("ivy", "c1.ivy"))
        self.assertFalse(EM.put_episode("ivy", self.ep("c1.ivy")))                    # deleted id stays deleted
        EM.put_episode("ivy", self.ep("c2.ivy", start=300, end=400))
        EM.put_episode("ivy", self.ep("c3.ivy", start=500, end=600))
        self.assertEqual(EM.wipe_episodes("ivy", cutoff=450, now=700), 1)
        self.assertEqual([e["id"] for e in EM.episodes("ivy")], ["c2.ivy"])
        self.assertFalse(EM.put_episode("ivy", self.ep("c4.ivy", start=460, end=650)))  # overlaps the wipe
        self.assertTrue(EM.put_episode("ivy", self.ep("c5.ivy", start=800, end=900)))

    def test_folder_identity(self):
        agent_memory.ensure("ivy")
        ident = agent_memory.ident("ivy")
        agent_memory.archive("ivy")
        agent_memory.ensure("ivy")
        self.assertFalse(EM.put_episode("ivy", self.ep("c1.ivy"), ident=ident))     # re-created folder
        self.assertEqual(EM.episodes("ivy"), [])

    def test_items_have_stable_ids(self):
        items = EM.episode_items(self.ep("c1.ivy"))
        self.assertIn(("c1.ivy/summary/0", "summary", "the group cleared the dungeon boss"), items)
        self.assertIn(("c1.ivy/moments/0", "moments", "someone fell off the map three times"), items)


class Writes(Base):
    def test_failed_replace_leaves_the_old_file(self):
        EM.add_owner_fact("The owner likes pizza", "I like pizza", "said", True)
        before = EM._facts_path().read_text(encoding="utf-8")
        with mock.patch("experience_memory.os.replace", side_effect=PermissionError("in use")):
            with self.assertRaises(PermissionError):
                EM.write_json(EM._facts_path(), {"facts": []}, retry_s=0.1)
        self.assertEqual(EM._facts_path().read_text(encoding="utf-8"), before)
        self.assertEqual(list(EM._facts_path().parent.glob("*.tmp")), [])

    def test_reply_path_lock_never_waits(self):
        import threading
        held, release = threading.Event(), threading.Event()

        def holder():
            with EM.store_lock():
                held.set()
                release.wait(5)
        t = threading.Thread(target=holder)
        t.start()
        held.wait(5)
        try:
            with self.assertRaises(EM.StoreBusy):
                with EM.store_lock(timeout=0):
                    pass
        finally:
            release.set()
            t.join(5)

    def test_two_processes_lose_no_record(self):
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(ROOT)!r})
            import agent_memory, experience_memory as EM
            from pathlib import Path
            agent_memory.ROOT = Path({str(agent_memory.ROOT)!r})
            for i in range(40):     # production writers retry a busy store; a loaded test machine needs the wait
                EM.add_owner_fact("The owner fact %s %d" % (sys.argv[1], i), "I", "said", True, timeout=60)
        """)
        f = Path(self.tmp.name) / "writer.py"
        f.write_text(script, encoding="utf-8")
        ps = [subprocess.Popen([sys.executable, str(f), tag]) for tag in ("a", "b")]
        for p in ps:
            self.assertEqual(p.wait(120), 0)
        self.assertEqual(len(EM.owner_facts()), 80)


class Recalling(Base):
    def test_episode_and_fact_hits(self):
        EM.put_episode("ivy", self.ep("c1.ivy"))
        r = EM.Recall()
        hits = r.episode_hits("ivy", "that dungeon boss was brutal")
        self.assertEqual(hits[0][0], "c1.ivy/summary/0")
        self.assertEqual(r.episode_hits("max", "that dungeon boss was brutal"), [])
        EM.add_owner_fact("The owner collects vintage synthesizers", "I collect vintage synthesizers", "said", True)
        EM.add_owner_fact("The owner restores vintage pinball machines", "I restore vintage pinball machines",
                          "extracted", True)
        self.assertEqual(len(r.fact_hits("any vintage synthesizers lately?", use_memories=False)), 1)
        r.invalidate("owner_facts")
        self.assertEqual([t for _, _, t in r.fact_hits("vintage pinball machines", use_memories=False)], [])
        self.assertEqual(len(r.fact_hits("vintage pinball machines", use_memories=True)), 1)

    def test_a_delete_during_an_index_build_is_not_lost(self):
        """Bug review #5: an invalidation while the index reads the store must not leave the old data cached."""
        from unittest import mock
        f = EM.add_owner_fact("The owner collects vintage synthesizers", "I collect vintage synthesizers", "said",
                              True)
        r = EM.Recall()
        real = EM.owner_facts
        calls = []

        def racing():
            out = real()
            if not calls:                                  # the first read: the owner deletes meanwhile
                calls.append(1)
                EM.delete_fact(f["id"])
                r.invalidate("owner_facts")
            return out
        with mock.patch.object(EM, "owner_facts", racing):
            r.fact_hits("vintage synthesizers", use_memories=False)
        self.assertEqual(r.fact_hits("any vintage synthesizers lately?", use_memories=False), [])


if __name__ == "__main__":
    unittest.main()
