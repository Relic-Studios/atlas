"""Off the record and "remember that" (design docs/designs/experience_memory.md, D7, S3, S7; acceptance A5)."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory  # noqa: E402
import call_boundary as CB  # noqa: E402
import experience_memory as EM  # noqa: E402
import owner_voice as ov  # noqa: E402
import importlib.util  # noqa: E402
SPEECH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "huggingface_hub"))
NEEDS_SPEECH = unittest.skipUnless(SPEECH, "imports the server, which needs the speech stack "
                                   "(not in the light CI deps)")

OWNER = (lambda v: v / np.linalg.norm(v))(np.random.default_rng(5).normal(size=192)).astype(np.float32)


def voice_at(sim, seed):
    o = np.random.default_rng(seed).normal(size=192)
    o -= (o @ OWNER) * OWNER
    o /= np.linalg.norm(o)
    return (sim * OWNER + np.sqrt(1 - sim * sim) * o).astype(np.float32)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        self.b = CB.CallBoundary()
        self.w = CB.PendingWriter(self.b, paused=lambda: False)
        self.b.on_start(self.w.begin)
        self.b.on_end(self.w.finish)
        self.off = CB.OffRecord()
        CB._MARKED.clear()                  # marked windows are per process; one test's must not refuse another's writes
        CB._REPLY_OFF_UNTIL[0] = 0.0
        self.patches = [mock.patch.object(CB, "BOUNDARY", self.b), mock.patch.object(CB, "WRITER", self.w),
                        mock.patch.object(CB, "OFF", self.off)]
        for p in self.patches:
            p.start()
        self.store = ov.Store(Path(self.tmp.name) / "owner_voice.json")
        self.store.enroll("Relic", OWNER, 12, "earlier")
        self.adopter = ov.Adopter(self.store, session="s1")
        self.mgr = SimpleNamespace(people=SimpleNamespace(agent_names={"ivy", "max"}), held_facts={},
                                   said_facts={}, current_persona="ivy", owner_spoke=lambda: None)
        import server
        self.server = server
        self.cb = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(OwnerAdopter=self.adopter)),
                                  _turn_meta=None)
        self.cb._remember_that = lambda *a: server.TranscriptionCallbacks._remember_that(self.cb, *a)

    def tearDown(self):
        self.w.flush()
        CB._MARKED.clear()
        CB._REPLY_OFF_UNTIL[0] = 0.0
        for p in self.patches:
            p.stop()
        agent_memory.ROOT = self._root
        self.tmp.cleanup()

    def adopt(self, label="S1"):
        for i in range(5):
            self.adopter.observe(label, voice_at(0.85, 100 + i))
        self.assertTrue(self.adopter.is_adopted(label))

    def say(self, label, text, sim=0.85, dur=3.0, audible=False):
        self.b.turn()
        self.adopter.observe(label, voice_at(sim, int(time.time() * 1000) % 100000))
        self.cb._turn_meta = {"dur": dur, "audible": audible}
        self.server.TranscriptionCallbacks._memory_line(self.cb, label, text, self.mgr)
        self.w.flush()

    def lines(self):
        self.w.flush()
        return [r.get("text") for r in CB.read_records(CB.pending_file(self.b.call_id)) if r["k"] == "turn"]


@NEEDS_SPEECH
class OffTheRecord(Base):
    def test_backward_trigger_removes_the_last_minute(self):
        self.say("S2", "we finally cleared the raid")
        with mock.patch("time.time", return_value=time.time() + 120):
            pass
        self.say("S2", "my sister is pregnant again")
        self.say("S2", "forget that")
        self.assertNotIn("my sister is pregnant again", self.lines())
        self.assertNotIn("forget that", self.lines())
        self.assertFalse(self.off.is_open())

    def test_span_marks_until_back_on_the_record(self):
        self.say("S2", "ok this is off the record")
        self.assertTrue(self.off.is_open())
        self.say("S2", "I'm quitting my job on Friday")
        self.say("S2", "back on the record")
        self.say("S2", "anyway the boss fight was rough")
        lines = self.lines()
        self.assertNotIn("I'm quitting my job on Friday", lines)
        self.assertNotIn("ok this is off the record", lines)
        self.assertIn("anyway the boss fight was rough", lines)
        self.assertFalse(self.off.is_open())

    def test_an_unplaced_voice_can_go_off_the_record(self):
        """D7: anyone can; a voice the diarizer hasn't placed yet (no label) included."""
        self.say(None, "ok this is off the record, my cousin works at the plant")
        self.assertTrue(self.off.is_open())
        self.say("S2", "no way, they're closing it?")
        self.assertEqual(self.lines(), [])
        self.say(None, "back on the record")
        self.say(None, "anyway that boss was rough")
        recs = [r for r in CB.read_records(CB.pending_file(self.b.call_id)) if r["k"] == "turn"]
        self.assertEqual((recs[-1]["text"], recs[-1]["label"]), ("anyway that boss was rough", None))

    def test_ordinary_speech_triggers_nothing(self):
        self.say("S2", "I don't remember his name")
        self.assertEqual(self.lines(), ["I don't remember his name"])

    def test_nothing_is_stored_inside_a_span(self):
        import lore
        import self_tools
        import teach
        self.off.open()
        self.assertIn("off the record", lore.add("ivy", "x", "the roof incident", root=agent_memory.ROOT))
        self.assertIn("off the record", teach.learn("ivy", "rule", "a win is 3 points", root=agent_memory.ROOT))
        self.assertIn("off the record", self_tools.make_prediction("ivy", "the raid fails tonight",
                                                                   root=agent_memory.ROOT))
        self.adopt()
        self.say("S1", "remember that I like pizza")
        self.assertEqual(EM.owner_facts(), [])

    def test_backward_trigger_deletes_what_those_lines_stored(self):
        import teach
        self.adopt()
        self.say("S1", "remember that I like pizza")
        teach.learn("ivy", "dice rule", "doubles roll again", root=None)
        self.assertEqual(len(EM.owner_facts()), 1)
        self.say("S1", "actually, strike that")
        self.w.flush()
        self.assertEqual(EM.owner_facts(), [])
        self.assertEqual(teach._load("ivy"), [])

    def test_writes_from_marked_lines_after_the_delete_are_refused(self):
        """A tool write whose source line is in a marked window is refused even when it arrives after the delete, and
        a prediction resolved from a marked line goes back to open (D7)."""
        import lore
        import self_tools
        self_tools.make_prediction("ivy", "the raid fails tonight", root=None, now=time.time() - 600)  # earlier
        self.say("S2", "we cleared the raid")
        self_tools.check_predictions("ivy", id=1, outcome="wrong", note="they cleared it", root=None)
        self.say("S2", "forget that")
        self.w.flush()
        self.assertIn("off the record", lore.add("ivy", "x", "the raid got cleared", mode="auto"))
        items = self_tools._load("ivy")
        self.assertEqual(items[0]["status"], "open")
        self.assertNotIn("note", items[0])

    def test_the_reply_to_forget_that_is_not_kept(self):
        """Bug review #10: the reply to a backward trigger (finished after its window) may echo what was retracted."""
        import speech_pipeline_manager as SPM
        mgr = object.__new__(SPM.SpeechPipelineManager)
        mgr.current_persona = "ivy"
        self.say("S2", "my sister is pregnant again")
        self.say("S2", "actually forget that")
        SPM.SpeechPipelineManager.reply_spoken(mgr, SimpleNamespace(memory_ids=[]), "forgotten, your sister's news stays")
        self.w.flush()
        replies = [r["text"] for r in CB.read_records(CB.pending_file(self.b.call_id)) if r["k"] == "reply"]
        self.assertEqual(replies, [])

    def test_backward_trigger_drops_held_remember_that_turns(self):
        import speech_pipeline_manager as SPM
        fn = lambda a, b: SPM.SpeechPipelineManager._forget_held(self.mgr, a, b)  # noqa: E731
        CB.on_forget_window(fn)
        try:
            self.say("S1", "remember that I collect vintage synthesizers")
            self.assertIn("S1", self.mgr.held_facts)
            self.say("S1", "forget that")
            self.assertEqual(self.mgr.held_facts, {})
        finally:
            CB._WINDOW_LISTENERS.remove(fn)

    def test_a_reply_inside_a_span_is_not_written(self):
        import speech_pipeline_manager as SPM
        mgr = object.__new__(SPM.SpeechPipelineManager)
        mgr.current_persona = "ivy"
        self.say("S2", "ok this is off the record")
        SPM.SpeechPipelineManager.reply_spoken(mgr, SimpleNamespace(memory_ids=[]), "wow, quitting on Friday?")
        self.say("S2", "back on the record")
        SPM.SpeechPipelineManager.reply_spoken(mgr, SimpleNamespace(memory_ids=[]), "welcome back")
        self.w.flush()
        replies = [r["text"] for r in CB.read_records(CB.pending_file(self.b.call_id)) if r["k"] == "reply"]
        self.assertEqual(replies, ["welcome back"])


class Recorder(unittest.TestCase):
    def test_mask_since_and_live_masking_across_a_roll(self):
        import call_recorder as CR
        with tempfile.TemporaryDirectory() as d:
            rec = CR.CallRecorder(root=Path(d))
            rec.enabled = True
            rec.event("user", spk="S2", text="before the window")
            time.sleep(0.1)
            t0 = time.time()
            rec.event("user", spk="S2", text="my sister is pregnant")
            rec.event("retry", dropped="my sister is pregnant, wow")
            rec.close()                                   # the next event opens a new file (a roll)
            time.sleep(1.1)
            rec.event("agent", persona="ivy", text="congrats to her")
            try:
                masked = rec.mask_since(t0 - 0.01, time.time() + 1)
            finally:
                rec.close()
            self.assertEqual(masked, 3)
            off = CB.OffRecord()
            off.open()
            with mock.patch.object(CB, "OFF", off):
                rec.event("user", spk="S2", text="inside the span")
            rec.close()
            texts = []
            for f in sorted(Path(d).rglob("*.jsonl")):
                for raw in f.read_text(encoding="utf-8").splitlines():
                    e = json.loads(raw)
                    texts.append(e.get("text") or e.get("dropped"))
            self.assertEqual(texts.count("[off the record]"), 4)
            self.assertIn("before the window", texts)

    def test_masking_keeps_the_audio_recording_and_a_span_records_silence(self):
        import numpy as _np
        import call_recorder as CR
        with tempfile.TemporaryDirectory() as d:
            rec = CR.CallRecorder(root=Path(d))
            rec.enabled, rec.audio_enabled = True, True
            rec.event("user", spk="S2", text="my sister is pregnant")
            rec.audio(_np.ones(1600, dtype=_np.float32) * 0.1)
            self.assertIsNotNone(rec._wav)
            rec.mask_since(time.time() - 5, time.time() + 1)
            self.assertIsNotNone(rec._wav)                     # bug review #13: audio kept recording
            off = CB.OffRecord()
            off.open()
            with mock.patch.object(CB, "OFF", off):
                rec.audio(_np.ones(1600, dtype=_np.float32) * 0.1)
            rec.close()
            import soundfile as sf
            wav = sf.read(str(next(Path(d).rglob("*.wav"))))[0]
            self.assertGreater(abs(wav[:1600]).max(), 0.05)
            self.assertEqual(abs(wav[1600:]).max(), 0.0)          # the span's audio is silence

    def test_masking_rewrites_only_the_tail_of_a_large_file(self):
        """D7 masking holds the recorder lock (shared with the audio recorder) for milliseconds, not a 50 MB rewrite."""
        import call_recorder as CR
        with tempfile.TemporaryDirectory() as d:
            rec = CR.CallRecorder(root=Path(d))
            rec.enabled = True
            rec.event("user", spk="S1", text="opening line")
            path = rec.path
            now = time.time()
            with open(path, "a", encoding="utf-8") as fh:
                for i in range(150_000):                          # ~15 MB of older events
                    fh.write(json.dumps({"t": now - 3600 + i * 0.01, "k": "user", "spk": "S2", "text": f"old {i}"})
                             + "\n")
            rec.event("user", spk="S2", text="my sister is pregnant")
            read = [0]
            real_open = open

            class Counting:                               # how much of the file masking reads (not how long it took)
                def __init__(self, f):
                    self.f = f

                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    self.f.close()

                def read(self, *a):
                    data = self.f.read(*a)
                    read[0] += len(data)
                    return data

                def __getattr__(self, k):
                    return getattr(self.f, k)
            with mock.patch.object(CR, "open", lambda *a, **kw: Counting(real_open(*a, **kw)), create=True):
                masked = rec.mask_since(time.time() - 5, time.time() + 1)
            rec.close()
            self.assertEqual(masked, 1)
            self.assertLess(read[0], 1_000_000)                   # the file is about 15 MB
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[-1])["text"], "[off the record]")
            self.assertEqual(json.loads(lines[-2])["text"], "old 149999")
            self.assertEqual(json.loads(lines[0])["text"], "opening line")


@NEEDS_SPEECH
class RememberThat(Base):
    def test_outcome_1_confirmed_and_used_at_once(self):
        self.adopt()
        self.say("S1", "Ivy, remember that I like pizza")
        f = EM.owner_facts()[0]
        self.assertEqual((f["fact"], f["quote"], f["source"], f["confirmed"]),
                         ("The owner likes pizza", "I like pizza", "said", True))
        self.assertEqual(EM.usable_owner_facts(False)[0]["id"], f["id"])

    def test_outcome_2_to_confirm(self):
        self.adopt()
        self.say("S1", "remember that I main support", dur=1.0)          # too short to qualify
        f = EM.owner_facts()[0]
        self.assertFalse(f["confirmed"])
        self.assertEqual(EM.usable_owner_facts(True), [])

    def test_outcome_3_held_until_adopted(self):
        self.say("S1", "remember that I collect vintage synthesizers")
        self.assertEqual(EM.owner_facts(), [])
        self.assertIn("S1", self.mgr.held_facts)
        self.server._release_held_facts(self.mgr, self.adopter, "S1")
        self.w.flush()
        self.assertEqual(EM.owner_facts()[0]["fact"], "The owner collects vintage synthesizers")

    def test_outcome_4_anyone_else(self):
        self.adopt("S1")
        self.say("S2", "remember that I hate pizza", sim=0.1)
        self.assertEqual(EM.owner_facts(), [])
        self.assertEqual(self.mgr.held_facts, {})

    def test_outcome_5_failing_checks(self):
        self.adopt()
        self.say("S1", "remember that my mom is sick")                      # health
        self.say("S1", "remember that I go camping with Jake")             # another person's name
        self.assertEqual(EM.owner_facts(), [])

    def test_allow_list_words_and_later_capitals(self):
        """D7: an allow-listed game name is kept; a capital later in the fact (even at a sentence start) fails."""
        EM.root().mkdir(parents=True, exist_ok=True)
        (EM.root() / "_allow_names.txt").write_text("Zorbatron\n", encoding="utf-8")
        self.adopt()
        self.say("S1", "remember that I main Zorbatron")
        self.say("S1", "remember that I like pizza. Jake hates it")
        self.assertEqual([f["fact"] for f in EM.owner_facts()], ["The owner mains Zorbatron"])

    def test_questions_and_past_events_store_nothing(self):
        self.adopt()
        self.say("S1", "remember that time we wiped five times")
        self.say("S1", "do you remember that?")
        self.assertEqual(EM.owner_facts(), [])

    def test_later_not_me_sends_facts_back_to_confirm(self):
        self.adopt()
        self.say("S1", "remember that I like pizza")
        self.adopter.refuse("S1")
        self.server._said_facts_to_confirm(self.mgr, "S1")
        self.w.flush()
        self.assertFalse(EM.owner_facts()[0]["confirmed"])


@NEEDS_SPEECH
class Cues(Base):
    def mgr2(self, owner):
        import speech_pipeline_manager as SPM
        m = object.__new__(SPM.SpeechPipelineManager)
        m.people = SimpleNamespace(agent_names={"ivy"}, owner=owner)
        return m

    def test_cues_from_the_draft(self):
        m = self.mgr2(self.adopter)
        self.assertEqual(m.memory_trigger("[S2] ok off the record, I hate my job"), "off")
        self.assertEqual(m.memory_trigger("[S1] remember that I like pizza"), "held")      # nobody adopted yet
        self.adopt()
        self.assertEqual(m.memory_trigger("[S1] remember that I like pizza"), "noted")
        self.assertEqual(m.memory_trigger("[S1] remember that my mom is sick"), "rejected")
        self.assertEqual(m.memory_trigger("[S2] remember that I like pizza"), "not_owner")
        self.assertIsNone(m.memory_trigger("[S2] I don't remember his name"))
        self.assertIsNone(m.memory_trigger("[S2] yesterday I went to the store"))
        self.assertEqual(m.memory_trigger("[S2] do you remember me?"), "past_other")
        self.assertEqual(m.memory_trigger("[S1] remember when we played chess?"), "past_owner")
        self.assertEqual(self.mgr2(None).memory_trigger("[user] remember that I like pizza"), "not_owner")
        self.off.open()
        self.assertEqual(m.memory_trigger("[S1] remember that I like pizza"), "off")


if __name__ == "__main__":
    unittest.main()
