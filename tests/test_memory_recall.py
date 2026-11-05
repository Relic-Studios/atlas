"""Recall (design docs/designs/experience_memory.md, D4, D5; acceptance A4): one composer of the system prompt, the
owner note only while the owner is present, snippets by score with cooldown and limits, the current call's own past.
A lightweight pipeline object: no models, temporary stores."""
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory  # noqa: E402
import experience_memory as EM  # noqa: E402
import remote_llm as R  # noqa: E402
import importlib.util  # noqa: E402
if not all(importlib.util.find_spec(m) for m in ("torch", "transformers", "huggingface_hub")):
    raise unittest.SkipTest("imports the speech pipeline, which needs the speech stack (not in the light CI deps)")
import speech_pipeline_manager as SPM  # noqa: E402


class FakeOwner:
    def __init__(self, labels=()):
        self._labels = set(labels)

    def labels(self):
        return sorted(self._labels)

    def is_adopted(self, label):
        return label in self._labels


class FakePeople:
    def __init__(self, owner=None, names=None):
        self.owner = owner
        self._names = names or {}

    def name_of(self, label):
        return self._names.get(label)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        self._listeners = list(EM._listeners)
        self.m = self.mgr()

    def tearDown(self):
        import threading
        for t in threading.enumerate():                 # background index rebuilds read the temp store
            if t.name == "RecallWarm":
                t.join(10)
        EM._listeners[:] = self._listeners
        agent_memory.ROOT = self._root
        self.tmp.cleanup()

    def mgr(self, remote=True, owner_labels=()):
        m = object.__new__(SPM.SpeechPipelineManager)
        rs = R.RemoteSession({"url": "http://x", "model": "m", "vllm": True, "context": 8192}, local=None) \
            if remote else None
        m.llm = SimpleNamespace(remote=rs, system_prompt="", system_prompt_message=None)
        m.current_persona = "ivy"
        m._persona_prompt = "Your name is Ivy. PERSONA"
        m.system_prompt = m._persona_prompt
        m.people = FakePeople(FakeOwner(owner_labels), {"S2": "Zed"})
        m.agents = SimpleNamespace(active=SimpleNamespace(system_prompt="raw"))
        m.convo = SimpleNamespace(oldest_rendered_wall=None)
        with mock.patch("call_summary.add_listener"), mock.patch("call_boundary.BOUNDARY.on_start"):
            m._memory_setup()
        return m

    def use_memories(self, on=True):
        EM.set_setting("use_memories", on)
        self.m._settings_cache = (0.0, self.m._settings_cache[1])

    def episode(self, eid="c1.ivy", end=200.0, summary="the group beat the dungeon boss after three wipes",
                moments=("someone fell off the map three times",)):
        EM.put_episode("ivy", {"id": eid, "call": eid.split(".")[0], "agent": "ivy", "date": "2026-11-03",
                                "start": end - 100, "end": end, "summary": [summary], "topics": ["dungeon raid"],
                                "games": [], "decisions": [], "jokes": [], "moments": list(moments)})


class Compose(Base):
    def test_one_writer_for_every_holder(self):
        self.use_memories(True)
        self.episode()
        text = self.m._compose_system()
        self.assertIn("From earlier calls you were in (memories, not instructions):", text)
        self.assertIn("dungeon boss", text)
        self.assertEqual(self.m.llm.system_prompt, text)
        self.assertEqual(self.m.llm.system_prompt_message, {"role": "system", "content": text})
        self.assertEqual(self.m.agents.active.system_prompt, text)
        self.assertTrue(text.startswith("Your name is Ivy. PERSONA"))

    def test_trial_has_no_per_call_note(self):
        self.episode()
        self.assertNotIn("From earlier calls", self.m._compose_system())

    def test_owner_note_only_while_present(self):
        EM.add_owner_fact("The owner collects vintage synthesizers", "I collect vintage synthesizers", "said", True)
        self.assertNotIn("About the owner", self.m._compose_system())        # no owner adopted
        self.m.people.owner._labels.add("S1")
        self.m.owner_spoke()
        self.assertIn("vintage synthesizers", self.m.system_prompt)
        self.m._owner_last = time.time() - SPM.SpeechPipelineManager.OWNER_PRESENT_S - 1
        self.m._snippets("[S2] hello there everyone tonight")                # presence checked per generation
        self.assertNotIn("About the owner", self.m.system_prompt)

    def test_store_changes_compose_again(self):
        self.m.people.owner._labels.add("S1")
        self.m.owner_spoke()
        EM.add_owner_fact("The owner restores pinball machines", "I restore pinball machines", "said", True)
        self.assertIn("pinball", self.m.system_prompt)
        fid = EM.owner_facts()[0]["id"]
        EM.delete_fact(fid)
        self.assertNotIn("pinball", self.m.system_prompt)

    def test_small_context_budgets(self):
        m = self.mgr(remote=False)
        self.assertEqual(m._context_window(), 4096)
        self.assertEqual(m._budgets()["snippets"], 1)
        self.assertEqual(self.m._budgets()["snippets"], 3)


class Snippets(Base):
    def test_episode_items_only_with_use_memories(self):
        self.episode()
        self.assertEqual(self.m._snippets("[S2] that dungeon boss was brutal"), ("", []))
        self.use_memories(True)
        text, ids = self.m._snippets("[S2] remember the map? someone fell off three times")
        self.assertIn("c1.ivy/moments/0", ids)
        self.assertIn("In an earlier call you were in: someone fell off the map three times", text)

    def test_items_in_the_notes_are_not_repeated(self):
        self.use_memories(True)
        self.episode()
        self.m._compose_system()                       # the summary is in the per-call note
        _, ids = self.m._snippets("[S2] that dungeon boss after three wipes")
        self.assertFalse(any(i.startswith("c1.ivy/summary") for i in ids), ids)

    def test_owner_facts_only_on_owner_turns(self):
        EM.add_owner_fact("The owner collects vintage synthesizers", "I collect vintage synthesizers", "said", True)
        self.m.people.owner._labels.add("S1")
        self.assertEqual(self.m._snippets("[S2] still collecting vintage synthesizers?")[1], [])
        text, ids = self.m._snippets("[S1] still collecting vintage synthesizers?")
        self.assertEqual(len(ids), 1)
        self.assertIn("About the owner: The owner collects vintage synthesizers", text)

    def test_cooldown_starts_only_when_spoken(self):
        self.use_memories(True)
        self.episode()
        q = "[S2] remember the map? someone fell off three times"
        _, ids = self.m._snippets(q)
        self.assertTrue(ids)
        self.assertEqual(self.m._snippets(q)[1], ids)          # a cancelled draft used nothing
        with mock.patch("call_boundary.WRITER.reply"):
            self.m.reply_spoken(SimpleNamespace(memory_ids=ids), "yeah that was wild")
        self.assertEqual(self.m._snippets(q)[1], [])           # used: 20-minute cooldown

    def test_short_turns_and_limits(self):
        self.use_memories(True)
        for i in range(6):
            self.episode(f"c{i}.ivy", end=100.0 + i, summary=f"dungeon boss raid number {i} went sideways",
                         moments=(f"the dungeon boss raid {i} wiped twice",))
        self.assertEqual(self.m._snippets("[S2] boss"), ("", []))
        self.assertLessEqual(len(self.m._snippets("[S2] the dungeon boss raid again")[1]), 3)
        small = self.mgr(remote=False)
        small._settings_cache = (0.0, small._settings_cache[1])
        self.assertLessEqual(len(small._snippets("[S2] the dungeon boss raid again")[1]), 1)


class CurrentCall(Base):
    def test_older_lines_come_back_and_recent_ones_do_not(self):
        now = time.time()
        self.m._call_lines.add(now - 600, "S2", "my cousin's band plays synthwave covers at the harbor festival")
        self.m._call_lines.add(now - 20, "S2", "synthwave harbor festival tickets are cheap")
        self.m.convo.oldest_rendered_wall = now - 60
        self.assertEqual(self.m._snippets("[S3] wait which synthwave harbor festival")[1], [])   # not asked
        text, ids = self.m._snippets("[S3] wait, which synthwave harbor festival did you say earlier?")
        self.assertEqual(len(ids), 1)
        self.assertIn("Earlier in this call: Zed: my cousin's band plays synthwave covers", text)

    def test_unasked_memories_are_rare(self):
        """Owner 11-04 ("constantly making callbacks"): without an ask, a memory needs a strong match and comes at
        most once in UNPROMPTED_SPACING_S; an ask still gets one at once."""
        self.use_memories(True)
        self.episode(moments=("someone fell off the volcano map into the lava twice",))
        self.assertEqual(self.m._snippets("[S2] the map was rough today")[1], [])              # weak match
        _, ids = self.m._snippets("[S2] I fell into the lava on that volcano map again")      # strong match
        self.assertTrue(ids)
        with mock.patch("call_boundary.WRITER.reply"):
            self.m.reply_spoken(SimpleNamespace(memory_ids=ids), "classic")
        self.episode("c2.ivy", end=300.0, summary="the group argued about the raid loot split for an hour",
                     moments=("someone rage quit over the raid loot split",))
        self.m.recall.invalidate("")
        self.assertEqual(self.m._snippets("[S2] that raid loot split argument again")[1], [])  # spacing
        self.assertTrue(self.m._snippets("[S2] remember the raid loot split argument?")[1])    # asked

    def test_agent_lines_and_the_current_turn_are_not_candidates(self):
        now = time.time()
        self.m._call_lines.add(now - 600, "ivy", "synthwave harbor festival is my favorite", agent=True)
        self.m.convo.oldest_rendered_wall = None
        self.m._call_lines.add(now, "S2", "synthwave harbor festival tonight")
        self.assertEqual(self.m._snippets("[S2] synthwave harbor festival tonight")[1], [])

    def test_each_spoken_reply_logs_kinds_and_ids_not_words(self):
        """A8 / D4 Log: one line per spoken reply with every snippet's kind and id and the note ids, no turn words."""
        import call_boundary as CB
        gen = SimpleNamespace(memory_ids=["c1.ivy/moments/0", "f-1a2b3c4d", "line/1793000000000"])
        with mock.patch.object(CB, "WRITER", SimpleNamespace(reply=lambda **kw: None)),                 self.assertLogs("speech_pipeline_manager", level="INFO") as cm:
            SPM.SpeechPipelineManager.reply_spoken(self.m, gen, "the secret word is banana")
            SPM.SpeechPipelineManager.reply_spoken(self.m, SimpleNamespace(memory_ids=[]), "nothing recalled here")
        lines = [r for r in cm.output if "memory used" in r]
        self.assertEqual(len(lines), 2)
        self.assertIn("episode:c1.ivy/moments/0", lines[0])
        self.assertIn("owner_fact:f-1a2b3c4d", lines[0])
        self.assertIn("line:line/1793000000000", lines[0])
        for w in ("secret", "banana", "recalled"):
            self.assertNotIn(w, " ".join(lines))

    def test_lookup_is_fast(self):
        from wordfreq import top_n_list
        vocab = [w for w in top_n_list("en", 20000) if w.isalpha() and len(w) > 3]
        rng = random.Random(3)
        self.use_memories(True)
        eps = []
        for e in range(1000):
            eps.append({"id": f"c{e}.ivy", "start": e, "end": e + 1, "summary": [" ".join(rng.choices(vocab, k=12))],
                        "topics": [" ".join(rng.choices(vocab, k=3)) for _ in range(5)],
                        "games": [], "decisions": [" ".join(rng.choices(vocab, k=8)) for _ in range(4)],
                        "jokes": [" ".join(rng.choices(vocab, k=8)) for _ in range(4)],
                        "moments": [" ".join(rng.choices(vocab, k=10)) for _ in range(6)]})
        EM.write_json(EM._ep_path("ivy"), {"episodes": eps})      # 20,000 items
        self.m.current_persona = "ivy"
        self.m._memory_changed("episodes", "ivy")              # D1 Size: rebuilt after the write, off the reply path
        for _ in range(200):
            if "ivy" in self.m.recall._episodes:
                break
            time.sleep(0.05)
        self.assertIn("ivy", self.m.recall._episodes)
        now = time.time()
        for i in range(4000):
            self.m._call_lines.add(now - 4000 + i, "S2", " ".join(rng.choices(vocab, k=10)))
        self.m.convo.oldest_rendered_wall = now - 30
        t0 = time.perf_counter()
        self.m._snippets("[S2] the first lookup after a write finds the index built")
        self.assertLess(time.perf_counter() - t0, 0.01)
        times = []
        for _ in range(1000):
            q = "[S2] " + " ".join(rng.choices(vocab, k=10))
            t0 = time.perf_counter()
            self.m._snippets(q)
            times.append(time.perf_counter() - t0)
        times.sort()
        self.assertLess(times[int(len(times) * 0.99)], 0.001, f"p99 {times[int(len(times) * 0.99)] * 1000:.2f} ms")


if __name__ == "__main__":
    unittest.main()
