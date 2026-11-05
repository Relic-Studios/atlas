"""The post-call pass (design docs/designs/experience_memory.md, D3; acceptance A1, A3 with a fake model, A2c).
Pending files are written through call_boundary's writer, as the server writes them."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory  # noqa: E402
import call_boundary as CB  # noqa: E402
import experience_memory as EM  # noqa: E402
import memory_writer as MW  # noqa: E402


class Clock:
    def __init__(self, t=None):
        import time
        self.t = time.time() - 3600 if t is None else t      # recent: older pending files expire (D2)

    def __call__(self):
        return self.t


class FakeTagger:
    def __init__(self, persons=("Zarvik", "zarvik", "Dana")):
        self.p = persons

    def persons(self, text):
        return [p for p in self.p if p.lower() in (text or "").lower()]


class FakeModel:
    """Answers by request kind; records what it was shown."""

    def __init__(self, episode=None, facts=None, merge=None, broken=0):
        self.episode, self.facts, self.merge, self.broken = episode, facts, merge, broken
        self.seen = []
        self.ctx = 8192

    def chat(self, system, user, schema, max_tokens=900):
        self.seen.append(user)
        if self.broken:
            self.broken -= 1
            return "not json at all"
        if "facts" in schema["properties"]:
            return json.dumps(self.facts or {"facts": []})
        if system.startswith("These are memories"):
            if getattr(self, "per_chunk", False):           # the merge may only cite turns the parts cited
                first = json.loads(user)[0]["summary"][0]["turns"]
                return json.dumps({**self.merge, "summary": [{"text": self.merge["summary"][0]["text"],
                                                              "turns": first}],
                                   "decisions": [], "moments": []})
            return json.dumps(self.merge)
        if getattr(self, "per_chunk", False):               # cite a human line of this chunk
            import re
            ns = [int(n) for n, who in re.findall(r"^\[(\d+)\] (Person\d+|Owner):", user, re.M)]
            return json.dumps({"summary": [{"text": "The group fought the dungeon boss.", "turns": ns[:1]}],
                               "topics": [], "games": [], "decisions": [], "jokes": [], "moments": []})
        return json.dumps(self.episode)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        agent_memory.ensure("ivy")
        self.ident = agent_memory.ident("ivy")
        self.clock = Clock()
        self.b = CB.CallBoundary(clock=self.clock)
        self.w = CB.PendingWriter(self.b, paused=lambda: False, clock=self.clock)
        self.b.on_start(self.w.begin)
        self.b.on_end(self.w.finish)

    def tearDown(self):
        self.w.flush()
        agent_memory.ROOT = self._root
        self.tmp.cleanup()

    def call(self, lines, end=True):
        """lines: (label, text, owner, extra) -> a pending file of an ended call. Returns the call id."""
        call = None
        for spec in lines:
            label, text = spec[0], spec[1]
            owner = spec[2] if len(spec) > 2 else False
            extra = spec[3] if len(spec) > 3 else {}
            call = self.b.turn()
            if label == "ivy":
                self.w.reply(agent="ivy", ident=self.ident, recalled=extra.get("recalled", False), text=text)
            else:
                self.w.turn(label=label, agent="ivy", ident=self.ident, owner=owner,
                            score=extra.get("score", 0.7 if owner else 0.1), dur=extra.get("dur", 3.0),
                            audible=extra.get("audible", False), text=text)
            self.clock.t += 30
        for kind, fields in getattr(self, "events", []):
            self.w.event(kind, **fields)
        if end:
            self.b.end("timer")
        self.w.flush()
        return call

    SCENE = [("S2", "hey I'm Zarvik, who's up for the dungeon"), ("S3", "me, the boss is brutal though"),
             ("S1", "I always main support in dungeons", True), ("S2", "Zarvik fell off the map again lol"),
             ("ivy", "you fell off the map three times now"), ("S3", "we finally beat the dungeon boss"),
             ("S1", "nice, that boss took us an hour", True), ("S2", "next week we raid again on Friday")]

    def episode(self, **over):
        rec = {"summary": [{"text": "The group beat the dungeon boss after a long fight.", "turns": [6]}],
               "topics": ["dungeon boss"], "games": [], "decisions": [{"text": "The group raids again on Friday.",
                                                                      "turns": [8]}],
               "jokes": [], "moments": [{"text": "Someone fell off the map again.", "turns": [4]}]}
        rec.update(over)
        return rec


class Pass(Base):
    def test_episode_written_without_names(self):
        self.events = [("name", {"name": "Zarvik", "kind": "learned"})]
        call = self.call(self.SCENE)
        m = FakeModel(self.episode())
        out = MW.process(CB.pending_file(call), m, FakeTagger(), owner_name="Relic")
        self.assertEqual(out["episodes"], 1)
        ep = EM.episodes("ivy")[0]
        self.assertEqual(ep["id"], f"{call}.ivy")
        self.assertIn("The group beat the dungeon boss after a long fight.", ep["summary"])
        self.assertNotIn("turns", json.dumps(ep))                       # turn ids are not stored
        shown = "\n".join(m.seen)
        self.assertNotIn("Zarvik", shown)                               # redacted before the model saw it
        self.assertIn("Person1", shown)
        self.assertIn("Owner:", shown)
        self.assertFalse(CB.pending_file(call).exists())                # A2c: deleted after a successful run

    def test_items_with_names_labels_traits_or_bad_citations_are_dropped(self):
        call = self.call(self.SCENE)
        bad = self.episode(
            moments=[{"text": "Person2 crashed the kart twice.", "turns": [4]},       # a placeholder becomes "someone"
                     {"text": "Dana carried the whole raid.", "turns": [6]},
                     {"text": "Someone is a terrible driver.", "turns": [4]},
                     {"text": "Someone fell off the map again.", "turns": [5]},             # cites the agent line
                     {"text": "Ivy will write scripts for anyone who asks.", "turns": [8]},
                     {"text": "Someone fell off the map again.", "turns": [4]}])
        MW.process(CB.pending_file(call), FakeModel(bad), FakeTagger())
        self.assertEqual(EM.episodes("ivy")[0]["moments"],
                         ["Someone crashed the kart twice.", "Someone fell off the map again."])

    def test_no_summary_left_drops_the_episode(self):
        call = self.call(self.SCENE)
        MW.process(CB.pending_file(call), FakeModel(self.episode(
            summary=[{"text": "Zarvik beat the boss.", "turns": [6]}])), FakeTagger())
        self.assertEqual(EM.episodes("ivy"), [])

    def test_owner_facts_only_from_qualifying_owner_lines(self):
        call = self.call(self.SCENE)
        facts = {"facts": [{"fact": "The owner mains support in dungeons", "quote": "I always main support in dungeons",
                            "turns": [3]},
                           {"fact": "The owner fell off the map", "quote": "fell off the map again", "turns": [4]},
                           {"fact": "The owner took an hour", "quote": "that boss took us an hour", "turns": [7]}]}
        MW.process(CB.pending_file(call), FakeModel(self.episode(), facts), FakeTagger())
        got = [(f["fact"], f["source"], f["confirmed"]) for f in EM.owner_facts()]
        self.assertEqual(got, [("The owner mains support in dungeons", "extracted", True)])

    def test_unqualified_owner_lines_give_no_facts(self):
        scene = [s if s[0] != "S1" else (s[0], s[1], True, {"dur": 1.0}) for s in self.SCENE]
        call = self.call(scene)
        m = FakeModel(self.episode(), {"facts": [{"fact": "The owner mains support", "quote": "I always main support",
                                                  "turns": [3]}]})
        MW.process(CB.pending_file(call), m, FakeTagger())
        self.assertEqual(EM.owner_facts(), [])
        self.assertEqual(len(m.seen), 1)                              # no owner-fact request was made

    def test_refused_label_lines_are_not_the_owners(self):
        self.events = [("refuse", {"label": "S1"})]
        call = self.call(self.SCENE)
        m = FakeModel(self.episode(), {"facts": [{"fact": "The owner mains support", "quote": "I always main support",
                                                  "turns": [3]}]})
        MW.process(CB.pending_file(call), m, FakeTagger())
        self.assertEqual(EM.owner_facts(), [])
        self.assertNotIn("Owner:", "\n".join(m.seen))

    def test_unparseable_output_retries_once_then_fails(self):
        call = self.call(self.SCENE)
        self.assertEqual(MW.process(CB.pending_file(call), FakeModel(self.episode(), broken=1), FakeTagger())
                         ["episodes"], 1)
        call2 = self.call(self.SCENE)
        with self.assertRaises(MW.ModelError):
            MW.process(CB.pending_file(call2), FakeModel(self.episode(), broken=2), FakeTagger())
        self.assertTrue(CB.pending_file(call2).exists())

    def test_an_output_cut_off_by_its_budget_is_retried_with_twice_the_budget(self):
        class CutOff:
            def __init__(self):
                self.budgets, self.last = [], {}

            def chat(self, system, user, schema, max_tokens=900):
                self.budgets.append(max_tokens)
                if len(self.budgets) == 1:
                    self.last = {"finish": "length", "out": max_tokens}
                    return '{"summary": [{"text": "The group'
                self.last = {"finish": "stop", "out": 40}
                return json.dumps(full)
        full = {"summary": [], "topics": [], "games": [], "decisions": [], "jokes": [], "moments": []}
        m = CutOff()
        self.assertEqual(MW.ask(m, "s", "u", MW.EPISODE_SCHEMA), full)
        self.assertEqual(m.budgets, [MW.MAX_TOKENS, 2 * MW.MAX_TOKENS])
        self.assertEqual(MW.PARSE_FAILS[-1][0], "length")

    def test_requests_fit_the_context_window(self):
        """Bug review #8: a chunk, the rules and twice the output budget fit the window; long calls merge in rounds
        of groups that fit."""
        for ctx in (4096, 8192, 32768):
            self.assertLessEqual(MW.chunk_budget(ctx) + MW.PROMPT_RULES_TOKENS + 2 * MW.out_budget(ctx), ctx)
        recs = [{"summary": [{"text": "The group fought the boss " * 6, "turns": [i]}], "topics": [], "games": [],
                 "decisions": [], "jokes": [], "moments": []} for i in range(40)]
        groups = MW.merge_groups(recs, 800)
        self.assertEqual(sum(len(g) for g in groups), 40)
        self.assertTrue(all(len(g) >= 2 for g in groups[:-1]))
        self.assertGreater(len(groups), 1)

    def test_cut_off_items_and_scrubbed_game_names_are_dropped(self):
        call = self.call(self.SCENE)
        c = MW.Call(CB.pending_file(call))
        red = MW.redact(c, None, [], FakeTagger())
        chk = MW.Checker(c, red, None, [], FakeTagger())
        ns = [r["n"] for r in c.lines]
        human = [n for n in ns if c.by_n[n]["k"] == "turn"]
        rec = {"summary": [{"text": "The group fought the dungeon boss.", "turns": human[:1]},
                           {"text": "x" * (MW.TEXT_MAX - 1), "turns": human[:1]}],
               "topics": ["dungeon boss", "someone (a sword game)"], "games": ["someone"], "decisions": [],
               "jokes": [], "moments": []}
        out = chk.episode(rec, ns)
        self.assertEqual([i["text"] for i in out["summary"]], ["The group fought the dungeon boss."])
        self.assertEqual((out["topics"], out["games"]), (["dungeon boss"], []))

    def test_cited_turns_and_texts_are_bounded_in_the_schema(self):
        item = MW.EPISODE_SCHEMA["properties"]["moments"]["items"]["properties"]
        self.assertEqual(item["turns"]["maxItems"], 8)
        self.assertEqual(item["text"]["maxLength"], 300)

    def test_rerun_replaces_and_deleted_or_wiped_stay_gone(self):
        call = self.call(self.SCENE)
        p = CB.pending_file(call)
        keep = p.read_text(encoding="utf-8")
        MW.process(p, FakeModel(self.episode()), FakeTagger())
        p.write_text(keep, encoding="utf-8")
        MW.process(p, FakeModel(self.episode()), FakeTagger())
        self.assertEqual(len(EM.episodes("ivy")), 1)
        EM.delete_episode("ivy", f"{call}.ivy")
        p.write_text(keep, encoding="utf-8")
        MW.process(p, FakeModel(self.episode()), FakeTagger())
        self.assertEqual(EM.episodes("ivy"), [])

    def test_too_small_and_unfinished_calls(self):
        call = self.call(self.SCENE[:3], end=False)
        self.assertEqual(MW.process(CB.pending_file(call), FakeModel(self.episode()), FakeTagger()),
                         {"skipped": "not ended"})

    def test_chunks_and_merge(self):
        call = self.call(self.SCENE)
        merged = self.episode(summary=[{"text": "The group beat the dungeon boss.", "turns": [6]}])
        m = FakeModel(self.episode(), merge=merged)
        m.per_chunk = True
        with mock.patch.object(MW, "CHUNK_TOKENS", 60), mock.patch.object(m, "ctx", 1700):
            MW.process(CB.pending_file(call), m, FakeTagger())
        self.assertGreater(len(m.seen), 2)
        self.assertEqual(EM.episodes("ivy")[0]["summary"], ["The group beat the dungeon boss."])


class Run(Base):
    def test_failures_are_counted_and_stop_after_three(self):
        call = self.call(self.SCENE)
        m = FakeModel(self.episode(), broken=99)
        for _ in range(3):
            self.assertEqual(MW.run(m, FakeTagger()), 2)
        n = len(m.seen)
        self.assertEqual(MW.run(m, FakeTagger()), 0)                  # a failed call is left for expiry
        self.assertEqual(len(m.seen), n)
        self.assertEqual(EM.review_count(), 1)

    def test_badly_shaped_output_fails_that_call_only(self):
        """A server that doesn't enforce the schema can answer "turns": 3; that is an unparseable output (retried,
        then the call's failed run), never a crash that stops the run."""
        class Shaped(FakeModel):
            def chat(self, system, user, schema, max_tokens=900):
                self.seen.append(user)
                return json.dumps({"summary": [{"text": "The group fought the boss.", "turns": 3}], "topics": [],
                                   "games": [], "decisions": [], "jokes": [], "moments": []})
        a, b = self.call(self.SCENE), self.call(self.SCENE)
        m = Shaped(self.episode())
        self.assertEqual(MW.run(m, FakeTagger()), 2)
        self.assertEqual(MW._fails(CB.pending_file(a)), 1)
        self.assertEqual(MW._fails(CB.pending_file(b)), 1)         # the next call was still tried

    def test_shape_check(self):
        self.assertTrue(MW._shape_ok({"summary": [{"text": "x", "turns": [1]}], "topics": [], "games": [],
                                      "decisions": [], "jokes": [], "moments": []}, MW.EPISODE_SCHEMA))
        self.assertFalse(MW._shape_ok({"summary": {"text": "x"}, "topics": [], "games": [], "decisions": [],
                                       "jokes": [], "moments": []}, MW.EPISODE_SCHEMA))
        self.assertFalse(MW._shape_ok({"facts": [{"fact": "f", "quote": "q", "turns": ["2"]}]}, MW.FACT_SCHEMA))

    def test_tagger_check_of_things_stored_in_calls(self):
        import lore
        import self_tools
        lore.add("ivy", "the dice", "the long night of the cursed dice with Dana", mode="auto")
        self_tools.make_prediction("ivy", "Dana will rage quit before the boss")
        EM.add_owner_fact("The owner camps with Dana", "I camp with Dana", "said", True, tagger_checked=False)
        MW.run(FakeModel(self.episode()), FakeTagger())
        self.assertEqual(lore._load("ivy")[0]["status"], "review")
        self.assertEqual(self_tools._load("ivy")[0]["claim"], "someone will rage quit before the boss")
        self.assertFalse(EM.owner_facts()[0]["confirmed"])


if __name__ == "__main__":
    unittest.main()
