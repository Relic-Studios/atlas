"""Owner voice recognition (design docs/designs/owner_voice.md): the stored voice (A1) and adoption (A2).
Voiceprints are injected as numpy vectors; no audio model, no files outside a temporary folder."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import owner_voice as ov  # noqa: E402

RNG = np.random.default_rng(7)


def unit(v):
    return v / np.linalg.norm(v)


OWNER = unit(RNG.normal(size=192)).astype(np.float32)


def voice_at(sim, base=OWNER, seed=None):
    """A unit vector whose cosine with `base` is `sim`."""
    r = np.random.default_rng(seed)
    o = r.normal(size=192)
    o -= (o @ base) * base
    o = unit(o)
    return (sim * base + np.sqrt(max(0.0, 1 - sim * sim)) * o).astype(np.float32)


def turns(n, sim, seed=0):
    return [voice_at(sim, seed=seed + i) for i in range(n)]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "owner_voice.json"

    def tearDown(self):
        self.dir.cleanup()

    def store(self, **kw):
        return ov.Store(self.path, **kw)

    def test_round_trip(self):
        s = self.store()
        s.enroll("Sam", OWNER, 12, "2026-11-04T10:12")
        t = self.store()
        self.assertTrue(t.load())
        self.assertEqual((t.name, len(t.samples)), ("Sam", 1))
        self.assertGreater(float(t.voice @ OWNER), 0.999)

    def test_bad_files_load_as_nothing_without_numbers_in_the_log(self):
        good = {"version": 1, "name": "Sam", "embedder": "ecapa", "center": None,
                "samples": [{"centroid": [0.1] * 192, "turns": 5, "at": "x"}]}
        bad = [
            "not json",
            dict(good, version=2),
            dict(good, embedder="titanet"),
            dict(good, name=""),
            dict(good, samples=[]),
            dict(good, samples=[{"centroid": [0.1] * 191, "turns": 5, "at": "x"}]),
            dict(good, samples=[{"centroid": [float("nan")] * 192, "turns": 5, "at": "x"}]),
            dict(good, samples=[{"centroid": [0.0] * 192, "turns": 5, "at": "x"}]),
        ]
        for b in bad:
            self.path.write_text(b if isinstance(b, str) else json.dumps(b), encoding="utf-8")
            with self.assertLogs("owner_voice", level="WARNING") as logs:
                self.assertFalse(self.store().load(), b if isinstance(b, str) else list(b))
            self.assertNotRegex(" ".join(logs.output), r"\d\.\d")
        self.path.write_text(json.dumps(good), encoding="utf-8")
        self.assertTrue(self.store().load())
        self.assertFalse(self.store(uses_center=True).load())  # a diarizer that centers its voiceprints

    def test_a_failed_write_keeps_the_old_file_and_voice(self):
        s = self.store()
        s.enroll("Sam", OWNER, 12, "a")
        before = self.path.read_bytes()
        with mock.patch("owner_voice.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                s.enroll("Other", voice_at(0.1, seed=1), 12, "b")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual((s.name, len(s.samples)), ("Sam", 1))
        self.assertGreater(float(s.voice @ OWNER), 0.999)

    def test_one_sample_per_session_and_five_at_most(self):
        s = self.store()
        s.enroll("Sam", OWNER, 12, "a")
        s.enroll("Sam", voice_at(0.9, seed=2), 12, "a")         # same session: replaced
        self.assertEqual(len(s.samples), 1)
        for i in range(6):
            s.enroll("Sam", voice_at(0.9, seed=10 + i), 12, f"s{i}")
        self.assertEqual(len(s.samples), ov.MAX_SAMPLES)
        self.assertEqual([x["at"] for x in s.samples], ["s1", "s2", "s3", "s4", "s5"])

    def test_forget_deletes_the_file(self):
        s = self.store()
        s.enroll("Sam", OWNER, 12, "a")
        s.forget()
        self.assertFalse(self.path.exists())
        self.assertEqual(s.status(), {"enrolled": False, "name": None, "samples": 0})

    def test_status_has_no_numbers_about_the_voice(self):
        s = self.store()
        s.enroll("Sam", OWNER, 12, "a")
        self.assertEqual(s.status(), {"enrolled": True, "name": "Sam", "samples": 1})


class SampleTests(unittest.TestCase):
    def test_too_few_turns(self):
        self.assertIsNone(ov.sample(turns(ov.ENROLL_TURNS - 1, 0.8)))

    def test_consistent_turns_give_a_sample(self):
        s = ov.sample(turns(6, 0.8))
        self.assertIsNotNone(s)
        self.assertGreater(float(s @ OWNER), 0.9)

    def test_a_row_mixing_two_voices_is_refused(self):
        other = unit(RNG.normal(size=192)).astype(np.float32)
        mixed = turns(3, 0.7) + [voice_at(0.7, base=other, seed=50 + i) for i in range(3)]
        self.assertIsNone(ov.sample(mixed))


class AdopterTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.store = ov.Store(Path(self.dir.name) / "v.json")
        self.store.enroll("Sam", OWNER, 12, "a")
        self.a = ov.Adopter(self.store)

    def tearDown(self):
        self.dir.cleanup()

    def feed(self, label, sims, seed=0):
        return [self.a.observe(label, voice_at(x, seed=seed + i)) for i, x in enumerate(sims)]

    def test_never_before_the_fourth_turn(self):
        self.assertEqual(self.feed("S1", [0.8, 0.8, 0.8, 0.8]), [False, False, False, True])
        self.assertEqual(self.a.labels(), ["S1"])

    def test_threshold_is_on_the_running_median(self):
        th = ov.ADOPT_SIM
        self.assertEqual(self.feed("S2", [th - 0.03] * 6), [False] * 6)
        self.assertEqual(self.feed("S3", [th + 0.02] * 4, seed=20), [False, False, False, True])
        # one odd turn does not decide it: the median ignores it
        self.assertEqual(self.feed("S4", [0.9, 0.9, -0.2, 0.9], seed=40), [False, False, False, True])

    def test_refused_and_self_labels_are_never_adopted(self):
        self.a.refuse("S5")
        self.assertEqual(self.feed("S5", [0.9] * 6), [False] * 6)
        self.assertEqual(self.feed("self", [0.9] * 6), [False] * 6)
        self.assertFalse(self.a.observe("S6", None))

    def test_several_labels_can_be_the_owner(self):
        self.feed("S1", [0.8] * 4)
        self.feed("S7", [0.8] * 4, seed=30)
        self.assertEqual(self.a.labels(), ["S1", "S7"])

    def test_merges_carry_adoption_and_refusal(self):
        self.feed("S1", [0.8] * 4)
        self.a.merge("S1", "S9")
        self.assertEqual(self.a.labels(), ["S9"])
        self.a.refuse("S10")
        self.a.merge("S10", "S11")
        self.assertEqual(self.feed("S11", [0.9] * 6, seed=60), [False] * 6)
        self.a.refuse("S12")
        self.feed("S13", [0.8] * 4, seed=80)
        self.a.merge("S12", "S13")                 # an adopted survivor stays adopted
        self.assertIn("S13", self.a.labels())

    def test_seen_counts_turns_for_the_name_ask_hold(self):
        self.feed("S1", [0.1, 0.1])
        self.assertEqual(self.a.seen("S1"), 2)
        self.assertEqual(self.a.seen("S99"), 0)

    def test_enrollment_sample_from_kept_turns(self):
        self.feed("S3", [0.8] * 5)
        s, n = self.a.sample_for("S3")
        self.assertEqual(n, 5)
        self.assertGreater(float(s @ OWNER), 0.9)

    def test_nothing_is_adopted_without_a_stored_voice(self):
        self.store.forget()
        self.a.clear()
        self.assertEqual(self.feed("S1", [0.9] * 6), [False] * 6)
        self.assertEqual(self.a.seen("S1"), 6)          # turns still kept for an enrollment

    def test_owner_name_hook(self):
        self.feed("S1", [0.8] * 4)
        self.assertEqual(self.a.owner_name("S1"), "Sam")
        self.assertIsNone(self.a.owner_name("S2"))
        self.a.refuse("S1")
        self.assertIsNone(self.a.owner_name("S1"))



class NameBookOwnerTests(unittest.TestCase):
    """D3: the owner's name is read through the hook; no other voice may take it; nothing said renames the owner."""

    def setUp(self):
        from people import NameBook
        self.dir = tempfile.TemporaryDirectory()
        self.store = ov.Store(Path(self.dir.name) / "v.json")
        self.store.enroll("Sam Rivera", OWNER, 12, "a")
        self.a = ov.Adopter(self.store, session="b")
        self.t = [1000.0]
        self.b = NameBook(agent_names=("max", "ivy"), store=None, clock=lambda: self.t[0])
        self.b.owner = self.a
        for i in range(4):
            self.a.observe("S1", voice_at(0.8, seed=i))      # S1 adopted

    def tearDown(self):
        self.dir.cleanup()

    def test_an_adopted_label_reads_the_owner_name_everywhere(self):
        b = self.b
        self.assertEqual((b.name_of("S1"), b.display("S1")), ("Sam Rivera", "Sam Rivera"))
        self.assertEqual(b.sanitize("tell S1 hi", target="S2"), "tell Sam Rivera hi")
        note = b.who_note("S1")
        self.assertIn("you recognize this voice: it's Sam Rivera", note)
        self.assertNotIn("name", note.split("Sam Rivera", 1)[1].lower())    # no name-ask cue for the owner
        for word in ("file", "stored", "computer", "owner", "voiceprint"):
            self.assertNotIn(word, note.lower())
        self.assertIn("Sam Rivera", b.known_names())

    def test_refuse_and_forget_take_effect_on_the_next_read(self):
        self.a.refuse("S1")
        self.assertIsNone(self.b.name_of("S1"))
        self.a.adopt("S1")
        self.store.forget()
        self.assertIsNone(self.b.name_of("S1"))

    def test_merged_labels_follow_the_owner(self):
        self.b.merge("S1", "S5")
        self.a.merge("S1", "S5")
        self.assertEqual(self.b.name_of("S1"), "Sam Rivera")      # alias resolves to S5, which is adopted

    def test_speech_never_renames_the_owner(self):
        self.assertIsNone(self.b.learn("S1", "[S1] my name is Bob"))
        self.assertIsNone(self.b.learn("S1", "[S1] don't call me Sam"))
        self.assertEqual(self.b.name_of("S1"), "Sam Rivera")

    def test_no_other_voice_takes_the_owner_name(self):
        b = self.b
        self.assertIsNone(b.learn("S2", "[S2] hey guys, I'm Sam"))                 # introduction
        self.assertIsNone(b.name_of("S2"))
        b.note_agent_reply("S3", "Wait, who's this?")
        self.t[0] += 3
        self.assertIsNone(b.learn("S3", "[S3] Sam."))                               # answer to the agent
        self.assertIsNone(b.name_of("S3"))
        self.assertEqual(b.people["S3"].asks, b.max_asks)                           # not asked again ...
        self.assertFalse(b.people["S3"].declined)                                   # ... and not "won't say"
        self.assertIsNone(b.peek_intro("[S4] hi, I'm Sam"))                         # the dashboard's label
        self.assertEqual(b.learn("S6", "[S6] hey guys, I'm Riley"), "Riley")       # anyone else still works

    def test_the_name_guess_never_offers_the_owner(self):
        from people import NameBook
        b = NameBook(agent_names=("max",), store=None)
        b.owner = self.a
        b._prints = {"Sam": list(map(float, OWNER)), "Riley": list(map(float, voice_at(0.95, seed=99)))}
        b.voice_lookup = lambda label: OWNER
        b.learn("S7", "[S7] hello there")
        self.assertEqual(b.people["S7"].probable, "Riley")          # skipped Sam, kept looking

    def test_no_name_asking_until_the_check_had_its_turns(self):
        class R:
            addressed_agent = 2
            turns = 3
        self.a.observe("S8", voice_at(0.1, seed=70))
        self.assertNotIn("ask for their name", self.b.who_note("S8", roster={"S8": R()}))
        for i in range(3):
            self.a.observe("S8", voice_at(0.1, seed=71 + i))
        self.assertIn("ask for their name", self.b.who_note("S8", roster={"S8": R()}))
        self.b.owner = None                                          # no hook: exactly today's behavior
        self.assertIn("ask for their name", self.b.who_note("S9", roster={"S9": R()}))

    def test_owner_adopted_clears_the_unmatched_name(self):
        self.b.heard["sam"] = self.t[0]
        self.b.owner_adopted("S1")
        self.assertNotIn("sam", self.b.heard)

    def test_drop_prints_removes_named_and_similar_entries(self):
        from people import NameBook
        path = Path(self.dir.name) / "voiceprints.json"
        b = NameBook(agent_names=(), store=path)
        b._prints = {"Sam": [0.1] * 192, "sam rivera": [0.2] * 192, "Relic": list(map(float, voice_at(0.9, seed=5))),
                     "Riley": list(map(float, voice_at(0.1, seed=6)))}
        self.assertEqual(b.drop_prints("Sam Rivera", OWNER, 0.70), 3)
        self.assertEqual(list(b._prints), ["Riley"])
        self.assertEqual(list(json.loads(path.read_text(encoding="utf-8"))), ["Riley"])


if __name__ == "__main__":
    unittest.main()
