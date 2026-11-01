"""Hybrid diarization: frame-level speaker slots + persistent voiceprints.

A frame-level streaming diarizer (NVIDIA Nemotron-3-Diarization) is good at "which voice is
talking in this stretch of the call": it labels nearly every turn and sees overlap. But its
8 slots are anonymous, reset per session, and get reused once more than 8 people have passed
through (churn eval 10-31: 47-58% of turns wrong). The voiceprint Diarizer is the reverse:
it knows who a person is, for any number of people over hours, but leaves ~20% of short
turns unlabelled rather than guess.

Each slot is bound to a voiceprint person. A turn first goes through the Diarizer as usual;
when that is unsure, the turn takes its slot's person unless the voice contradicts them,
and when the voice clearly belongs to someone else the slot is rebound (slot reused by a
newcomer). An unbound slot with a voice nobody matches founds a new person at once: the slot
model vouches that it is one consistent voice, which the voiceprints alone wait two turns for.
"""
from __future__ import annotations

import numpy as np

import diarize as _d
from diarize import SpeakerDecision, _cosine


def slot_from_window(probs, active_on: float = 0.5, min_active: float = 0.3):
    """(slot, prob, active_fraction) for a turn from its window of frame activities.

    probs: (frames, slots) sigmoid activities, 10 ms each. Live turn windows carry trailing
    silence and lose their last frames to the model's look-ahead, so only frames where some
    slot is clearly on count; a window with too few of them gives no slot."""
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2 or len(p) < 10:
        return None, 0.0, 0.0
    on = p.max(axis=1) >= active_on
    frac = float(on.mean())
    if frac < min_active or on.sum() < 10:
        return None, 0.0, frac
    act = p[on]
    slot = int(np.argmax(act.sum(axis=0)))
    return slot, float(act[:, slot].mean()), frac


class HybridDiarizer:
    def __init__(self, diarizer: "_d.Diarizer", slot_on: float = 0.3, keep_floor: float = 0.05,
                 rebind_margin: float = 0.12, found_new: bool = True, min_found_s: float = 0.8):
        self.d = diarizer
        self.slot_on = slot_on              # slot activity needed to trust the slot at all
        self.keep_floor = keep_floor        # voice-vs-bound-person score below this: slot was reused
        self.rebind_margin = rebind_margin  # voice must favour another person by this much to move the slot
        self.found_new = found_new
        self.min_found_s = min_found_s
        self.bind: dict = {}                # slot -> person label

    def _profile(self, label):
        label = self.d._resolve(label)
        for p in self.d._profiles:
            if p.label == label:
                return p
        return None

    def _sim(self, emb, label) -> float:
        p = self._profile(label)
        return p.score(emb, self.d.EXEMPLARS) if p is not None else -1.0

    def _bound(self, slot):
        label = self.bind.get(slot)
        if label is None:
            return None
        if self._profile(label) is None:     # the voiceprint was forgotten or merged away
            self.bind.pop(slot, None)
            return None
        return self.d._resolve(label)

    def process(self, audio: np.ndarray, sample_rate: int, slot, slot_prob: float,
                agent_audible: bool | None = None) -> SpeakerDecision:
        dec = self.d.process(audio, sample_rate, agent_audible=agent_audible)
        if dec.speaker == "self" or slot is None or slot_prob < self.slot_on:
            return dec
        emb = self.d.last_embedding
        bound = self._bound(slot)
        if dec.speaker is not None:
            if bound is None or bound == dec.speaker:
                self.bind[slot] = dec.speaker
                return dec
            # the voiceprints and the slot disagree
            if self._sim(emb, dec.speaker) - self._sim(emb, bound) >= self.rebind_margin:
                self.bind[slot] = dec.speaker          # the slot now holds someone else
                return dec
            return SpeakerDecision(speaker=bound, confidence=0.5, unknown_probability=0.5)
        # voiceprints unsure: lean on the slot
        if bound is not None and self._sim(emb, bound) >= self.keep_floor:
            p = self._profile(bound)
            p.last_seen_at = _d.time.time()
            return SpeakerDecision(speaker=bound, confidence=0.5, unknown_probability=0.5)
        duration = len(audio) / float(sample_rate)
        if not self.found_new or duration < self.min_found_s or emb is None:
            return dec
        # unbound (or reused) slot with a voice nobody matches: a new person, vouched by the slot
        with self.d._lock:
            self.d._pending = [q for q in self.d._pending if _cosine(q[0], emb) < 0.999]
            p = self.d._add_profile(emb, duration, 1, _d.time.time())
        self.bind[slot] = p.label
        return SpeakerDecision(speaker=p.label, confidence=0.5, unknown_probability=0.5)
