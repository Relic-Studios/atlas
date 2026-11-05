"""The pending-file writer (design docs/designs/experience_memory.md, D2, D7, S1; acceptance A2b)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent_memory  # noqa: E402
import call_boundary as CB  # noqa: E402


class Clock:
    def __init__(self, t=1_700_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        self.clock = Clock()
        self.paused = False
        self.b = CB.CallBoundary(clock=self.clock)
        self.w = CB.PendingWriter(self.b, paused=lambda: self.paused, clock=self.clock)
        self.b.on_start(self.w.begin)
        self.b.on_end(self.w.finish)

    def tearDown(self):
        self.w.flush()
        agent_memory.ROOT = self._root
        self.tmp.cleanup()

    def turn(self, text="we cleared the dungeon", label="S2", owner=False, write=True):
        call = self.b.turn()
        n = self.w.turn(label=label, agent="ivy", ident="id1", owner=owner, score=0.6 if owner else 0.1, dur=2.5,
                        audible=False, text=text, write=write)
        self.clock.t += 30
        return call, n

    def records(self, call):
        self.w.flush()
        return CB.read_records(CB.pending_file(call))


class Records(Base):
    def test_turns_replies_and_events(self):
        call, n1 = self.turn("Zarvik and xX_sniper_99 joined", owner=True)
        n2 = self.w.reply(agent="ivy", ident="id1", recalled=True, text="welcome back")
        self.w.event("adopt", label="S2", how="auto")
        CB_name = self.w.event
        CB_name("name", name="Zarvik", kind="learned")
        recs = self.records(call)
        kinds = [r["k"] for r in recs]
        self.assertEqual(kinds[0], "start")
        turn = next(r for r in recs if r["k"] == "turn")
        self.assertEqual((turn["n"], turn["label"], turn["owner"], turn["dur"], turn["audible"]),
                         (1, "S2", True, 2.5, False))
        self.assertEqual(next(r for r in recs if r["k"] == "reply")["n"], n2)
        self.assertIn({"k": "name", "name": "xX_sniper_99", "kind": "handle"}, recs)   # handles from the line
        self.assertIn({"k": "name", "name": "Zarvik", "kind": "learned"}, recs)
        self.assertIn({"k": "adopt", "label": "S2", "how": "auto"}, recs)
        self.assertEqual((n1, n2), (1, 2))

    def test_span_lines_are_numbered_but_never_written(self):
        call, _ = self.turn("first")
        _, n = self.turn("this is off the record", write=False)
        self.turn("third")
        texts = [r.get("text") for r in self.records(call) if r["k"] == "turn"]
        self.assertEqual(texts, ["first", "third"])
        self.assertEqual(n, 2)

    def test_paused_writes_only_start_and_end(self):
        self.paused = True
        call, _ = self.turn()
        for _ in range(7):
            self.turn()
        self.w.event("name", name="Zarvik", kind="learned")
        self.b.end("api")
        self.assertEqual([r["k"] for r in self.records(call)], ["start", "end"])

    def test_too_small_call_leaves_no_file(self):
        call, _ = self.turn()
        self.b.end("api")
        self.w.flush()
        self.assertFalse(CB.pending_file(call).exists())

    def test_end_record(self):
        call, _ = self.turn()
        for _ in range(6):
            self.turn()
        self.b.end("timer")
        recs = self.records(call)
        self.assertEqual(recs[-1]["k"], "end")
        self.assertEqual(recs[-1]["how"], "timer")

    def test_backward_rewrite(self):
        call, _ = self.turn("keep me")
        t0 = self.clock.t
        self.turn("my sister is pregnant")
        self.turn("forget that")
        self.w.drop_since(t0 - 1)
        texts = [r.get("text") for r in self.records(call) if r["k"] == "turn"]
        self.assertEqual(texts, ["keep me"])

    def test_store_writes_retry_until_they_succeed(self):
        tries = []

        def flaky():
            tries.append(1)
            if len(tries) < 3:
                raise OSError("busy")
        self.w.submit(flaky, retry_s=0.01)
        self.w.flush()
        self.assertEqual(len(tries), 3)


class Startup(Base):
    def _file(self, call, turns, end=False, t0=1_700_000_000.0):
        p = CB.pending_file(call)
        p.parent.mkdir(parents=True, exist_ok=True)
        recs = [{"k": "start", "call": call, "t": t0}]
        recs += [{"k": "turn", "n": i + 1, "t": t0 + 30 * i, "text": "x"} for i in range(turns)]
        if end:
            recs.append({"k": "end", "t": t0 + 30 * turns, "how": "timer"})
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        return p

    def test_leftovers_get_an_end_record_or_go(self):
        big = self._file("c-big", 8)
        small = self._file("c-small", 2)
        done = self._file("c-done", 8, end=True)
        closed = CB.close_leftovers(current="c-now")
        self.assertEqual(sorted(closed), ["c-big", "c-small"])
        self.assertEqual(CB.read_records(big)[-1]["how"], "startup")
        self.assertFalse(small.exists())
        self.assertEqual(CB.read_records(done)[-1]["how"], "timer")

    def test_expiry_after_two_days(self):
        old = self._file("c-old", 8, end=True, t0=1_700_000_000.0)
        new = self._file("c-new", 8, end=True, t0=1_700_000_000.0 + 3 * 86400)
        gone = CB.expire(now=1_700_000_000.0 + 2 * 86400 + 400)
        self.assertEqual(gone, ["c-old"])
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())


if __name__ == "__main__":
    unittest.main()
