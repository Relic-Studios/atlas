"""Hybrid diarizer decisions with controlled voice vectors (no model, no GPU)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diarize  # noqa: E402
from diarize_hybrid import HybridDiarizer  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def time(self):
        return self.t


def unit(*xs):
    v = np.zeros(16, dtype=np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v / np.linalg.norm(v)


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self._real = diarize.time
        diarize.time = self.clock
        self.d = diarize.Diarizer()
        self.d.MATCH_MATURE, self.d.MATCH_YOUNG, self.d.MATCH_MARGIN = 0.40, 0.30, 0.0
        self.next = None
        self.d._vec = lambda audio, sr: self.next
        self.h = HybridDiarizer(self.d)

    def tearDown(self):
        diarize.time = self._real

    def say(self, vec, slot, prob=0.9, seconds=2.0):
        self.next = vec
        self.clock.t += 3
        return self.h.process(np.zeros(int(16000 * seconds), dtype=np.float32), 16000, slot, prob,
                              agent_audible=False).speaker

    def test_unbound_slot_founds_a_person_on_the_first_turn(self):
        a = self.say(unit(1, 0), slot=0)
        self.assertIsNotNone(a)                       # voiceprints alone would wait for a second turn
        self.assertEqual(self.h.bind[0], a)

    def test_unsure_voice_takes_its_slots_person(self):
        a = self.say(unit(1, 0), slot=0)
        for _ in range(3):
            self.say(unit(1, 0.05), slot=0)
        self.assertEqual(self.say(unit(1, 2.2), slot=0), a)    # cos ~0.41 with A: weak, slot says A

    def test_reused_slot_with_a_clearly_new_voice_founds_someone_new(self):
        a = self.say(unit(1, 0, 0), slot=2)
        b = self.say(unit(0, 0, 1), slot=2)            # same slot, voice unlike A at all
        self.assertNotEqual(a, b)
        self.assertEqual(self.h.bind[2], b)

    def test_voice_that_clearly_belongs_to_someone_else_moves_the_slot(self):
        a = self.say(unit(1, 0, 0), slot=0)
        b = self.say(unit(0, 1, 0), slot=1)
        for _ in range(3):
            self.say(unit(1, 0.02, 0), slot=0)
            self.say(unit(0.02, 1, 0), slot=1)
        self.assertEqual(self.say(unit(0.05, 1, 0), slot=0), b)   # slot 0 now carries B's voice
        self.assertEqual(self.h.bind[0], b)

    def test_low_slot_activity_falls_back_to_voiceprints(self):
        self.say(unit(1, 0), slot=0)
        self.assertIsNone(self.say(unit(0, 1), slot=0, prob=0.1))  # slot unsure, voice unknown: no label


if __name__ == "__main__":
    unittest.main()
