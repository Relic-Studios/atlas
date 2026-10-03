"""Speaker diarization for a Discord-call-resident agent.

Identifies who is speaking so the agent can treat each person in the call as a
distinct participant and respond appropriately. Implements WhoSpeaksLive's
embedding-scoring algorithm (cosine similarity + sigmoid/softmax), with two
additions the agent needs:

1. **Self-profile** — the agent's own cloned voice embedding is registered up
   front. Any utterance matching it is tagged "self" and ignored, so the agent
   never mistakes its own echo (bleed-through from VAIO3) for a participant.

2. **Time-decay** — speaker profiles fade with inactivity and are pruned after
   a max idle time, so the agent does not keep stale identities from earlier
   calls/people and confuse them with current participants.

Embeddings are computed in-process with SpeechBrain ECAPA-TDNN on CUDA.
"""
from __future__ import annotations
import os

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v))))


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp((v - m) / temperature) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 0.0 else float(np.dot(a, b) / denom)


def _normalize(v) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float32).reshape(-1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(vec))
    return (vec / n).astype(np.float32) if n > 0 else vec


@dataclass
class SpeakerProfile:
    label: str
    centroid: np.ndarray
    speech_seconds: float = 0.0
    sentence_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)

    def update(self, embedding: np.ndarray, duration: float, weight: float) -> None:
        weight = max(0.0, min(1.0, weight))
        self.centroid = _normalize(self.centroid * (1.0 - weight) + embedding * weight)
        self.sentence_count += 1
        self.speech_seconds += max(0.0, duration)
        self.last_seen_at = time.time()


@dataclass
class SpeakerDecision:
    speaker: str | None
    is_self: bool = False
    confidence: float = 0.0
    unknown_probability: float = 1.0


class Diarizer:
    """Online speaker diarization with self-filtering and time-decay."""

    def __init__(
        self,
        *,
        self_reference_audio: str | None = None,
        self_similarity: float = 0.62,
        same_speaker_similarity: float = 0.60,
        similarity_temperature: float = 0.10,
        softmax_temperature: float = 0.12,
        new_speaker_threshold: float = 0.80,
        min_new_speaker_seconds: float = 1.2,
        max_speakers: int = 16,
        max_idle_seconds: float = 1800.0,    # a lull must not re-label everyone
        decay_half_life_seconds: float = 60.0,  # fade speech_seconds weight
        device: str = "cuda",
    ):
        self.self_similarity = self_similarity
        self.same_speaker_similarity = same_speaker_similarity
        self.similarity_temperature = similarity_temperature
        self.softmax_temperature = softmax_temperature
        self.new_speaker_threshold = new_speaker_threshold
        self.min_new_speaker_seconds = min_new_speaker_seconds
        self.max_speakers = max_speakers
        self.max_idle_seconds = max_idle_seconds
        self.decay_half_life_seconds = decay_half_life_seconds
        self.device = device

        self._profiles: list[SpeakerProfile] = []
        # Monotonic label counter: S{len+1} reused a live label after a profile
        # decayed (S2 idles out -> next new voice became a second "S3").
        self._next_label = 1
        self._pending: list = []          # (embedding, seconds, time) not yet a voice
        self._aliases: dict = {}          # merged label -> surviving label
        self._lock = threading.Lock()
        self._embedder = None
        self._self_embedding: np.ndarray | None = None

        if self_reference_audio:
            import soundfile as sf
            audio, sr = sf.read(self_reference_audio, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._self_embedding = _normalize(self._embed(audio, sr))

    def set_self_reference(self, audio_path: str | None = None) -> None:
        """Re-point the self-profile at a new voice (e.g. after a voice swap)."""
        if audio_path is None:
            with self._lock:
                self._self_embedding = None
            return
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        with self._lock:
            self._self_embedding = _normalize(self._embed(audio, sr))

    def _embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        if self._embedder is None:
            self._embedder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "atlas", "speechbrain_ecapa"),
                run_opts={"device": self.device},
            )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // g, sample_rate // g).astype(np.float32)
        wave = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            emb = self._embedder.encode_batch(wave, normalize=True)
        return _normalize(emb.detach().cpu().numpy())

    def _decay(self, now: float) -> None:
        """Prune idle profiles and fade their accumulated speech weight."""
        keep = []
        for p in self._profiles:
            idle = now - p.last_seen_at
            if idle > self.max_idle_seconds:
                continue
            # exponential fade of speech_seconds so stale profiles lose maturity
            if self.decay_half_life_seconds > 0:
                factor = 0.5 ** (idle / self.decay_half_life_seconds)
                p.speech_seconds *= factor
            keep.append(p)
        self._profiles = keep

    # --- Thresholds measured on Discord-like audio (Opus 24k, 16 kHz, ECAPA);
    # see tests/sim_diarize_fragment.py. Same-speaker utterance-vs-utterance
    # cosine is only ~0.3-0.5 there, utterance-vs-mature-centroid ~0.55-0.8,
    # different speakers <= ~0.3 (utt) / ~0.4 (centroid) at the 95th pct.
    MATCH_MATURE = 0.40      # profile with >= 3 utterances
    MATCH_YOUNG = 0.30       # profile built from 1-2 utterances
    PENDING_PAIR = 0.30      # two unmatched utterances agree -> confirm a voice
    SOLO_NEW_SECONDS = 2.5   # one long, clearly-foreign utterance may found a voice
    SOLO_NEW_MAX_SIM = 0.30
    MERGE_SIM = 0.60         # two profiles this close are one person split in two
    PENDING_TTL = 180.0
    SELF_MATCH = float(os.environ.get("ATLAS_SELF_MATCH", "0.40"))   # own voice must be the
    SELF_MARGIN = float(os.environ.get("ATLAS_SELF_MARGIN", "0.05")) # closest voice by this margin
    AUDIBLE_RELAX = float(os.environ.get("ATLAS_SELF_RELAX", "0.10"))

    def _match_threshold(self, p: "SpeakerProfile") -> float:
        return self.MATCH_MATURE if p.sentence_count >= 3 else self.MATCH_YOUNG

    def _resolve(self, label):
        seen = set()
        while label in self._aliases and label not in seen:
            seen.add(label)
            label = self._aliases[label]
        return label

    def _merge_close_profiles(self) -> None:
        """Fold fragments of one person back together (keeps the older label)."""
        merged = True
        while merged and len(self._profiles) > 1:
            merged = False
            for i in range(len(self._profiles)):
                for j in range(i + 1, len(self._profiles)):
                    a, b = self._profiles[i], self._profiles[j]
                    if _cosine(a.centroid, b.centroid) < self.MERGE_SIM:
                        continue
                    keep, drop = (a, b) if a.created_at <= b.created_at else (b, a)
                    wk = keep.sentence_count / max(1, keep.sentence_count + drop.sentence_count)
                    keep.centroid = _normalize(keep.centroid * wk + drop.centroid * (1 - wk))
                    keep.sentence_count += drop.sentence_count
                    keep.speech_seconds += drop.speech_seconds
                    keep.last_seen_at = max(keep.last_seen_at, drop.last_seen_at)
                    self._aliases[drop.label] = keep.label
                    self._profiles.remove(drop)
                    merged = True
                    break
                if merged:
                    break

    def _add_profile(self, centroid, duration, count, now) -> "SpeakerProfile":
        if len(self._profiles) >= self.max_speakers:
            # evict the least recently heard voice rather than refusing new people
            self._profiles.remove(min(self._profiles, key=lambda p: p.last_seen_at))
        p = SpeakerProfile(label=self._new_label(), centroid=_normalize(centroid),
                           speech_seconds=duration, sentence_count=count,
                           created_at=now, last_seen_at=now)
        self._profiles.append(p)
        return p

    def process(self, audio: np.ndarray, sample_rate: int = 16000,
                agent_audible: bool | None = None) -> SpeakerDecision:
        """Classify one finished utterance (float32 mono, any sample rate).

        Returns an existing label, a newly CONFIRMED label, "self", or None
        (a voice we can't place yet). A label is only minted once a voice is
        confirmed: two unmatched utterances that agree, or one long clearly
        foreign one. Single short unmatched lines stay anonymous instead of
        each becoming S44, S45, ...
        """
        embedding = _normalize(self._embed(audio, sample_rate))
        duration = float(len(audio)) / float(sample_rate)
        now = time.time()

        with self._lock:
            self._decay(now)
            self._pending = [q for q in self._pending if now - q[2] <= self.PENDING_TTL]

            top_sim, top = -1.0, None
            if self._profiles:
                sims = [_cosine(embedding, p.centroid) for p in self._profiles]
                i = max(range(len(sims)), key=lambda k: sims[k])
                top, top_sim = self._profiles[i], sims[i]

            # Own voice (echo past AEC). Treated like a mature profile: it wins
            # when it's the closest voice by a margin, or is very close outright.
            # agent_audible=False means the bridge KNOWS the agent was silent
            # for this whole utterance, so it cannot be our echo. A human whose
            # voice resembles the TTS voice must then be free to get a profile.
            self_sim = -1.0
            if self._self_embedding is not None and agent_audible is not False:
                self_sim = _cosine(embedding, self._self_embedding)
                is_self = self_sim >= self.self_similarity or (
                    self_sim >= self.SELF_MATCH and self_sim >= top_sim + self.SELF_MARGIN)
                if not is_self and agent_audible is True:
                    # the bridge KNOWS we were playing: our voice only has to be
                    # the closest one, with a lower floor (short TTS lines are noisy)
                    is_self = self_sim >= self.SELF_MATCH - self.AUDIBLE_RELAX and self_sim >= top_sim
                if is_self:
                    if duration >= 1.0 and self_sim >= self.SELF_MATCH + 0.1:
                        # adapt toward how the TTS actually sounds on this line
                        self._self_embedding = _normalize(
                            self._self_embedding * 0.9 + embedding * 0.1)
                    return SpeakerDecision(speaker="self", is_self=True,
                                           confidence=1.0, unknown_probability=0.0)

            if top is not None:
                if top_sim >= self._match_threshold(top):
                    # running mean while young, slow EMA once established
                    w = max(0.08, 1.0 / (top.sentence_count + 1)) * \
                        max(0.25, min(1.0, duration / 2.0))
                    top.update(embedding, duration, weight=min(0.5, w))
                    self._merge_close_profiles()
                    label = self._resolve(top.label)
                    conf = max(0.0, min(1.0, (top_sim - 0.2) / 0.6))
                    return SpeakerDecision(speaker=label, confidence=conf,
                                           unknown_probability=1.0 - conf)

            # Unmatched. Can it found a new voice? Never from something that
            # sounds more like our own voice than anyone else.
            if duration < 0.6 or (self_sim >= self.SELF_MATCH - 0.08 and self_sim >= top_sim):
                return SpeakerDecision(speaker=None)
            if agent_audible is True and duration < self.SOLO_NEW_SECONDS:
                # overlapped by our own voice: too contaminated to found a new person
                return SpeakerDecision(speaker=None, unknown_probability=1.0)
            best_j, best = None, -1.0
            for j, (e, d, _t) in enumerate(self._pending):
                c = _cosine(embedding, e)
                if c > best:
                    best_j, best = j, c
            if best_j is not None and best >= self.PENDING_PAIR:
                e, d, _t = self._pending.pop(best_j)
                p = self._add_profile(embedding * duration + e * d, duration + d, 2, now)
                return SpeakerDecision(speaker=p.label, confidence=0.6,
                                       unknown_probability=0.4)
            if duration >= self.SOLO_NEW_SECONDS and top_sim < self.SOLO_NEW_MAX_SIM:
                p = self._add_profile(embedding, duration, 1, now)
                return SpeakerDecision(speaker=p.label, confidence=0.5,
                                       unknown_probability=0.5)
            self._pending.append((embedding, duration, now))
            self._pending = self._pending[-12:]
            return SpeakerDecision(speaker=None, unknown_probability=1.0)

    def _new_label(self) -> str:
        label = f"S{self._next_label}"
        self._next_label += 1
        return label

    def centroid(self, label: str):
        """Current voice centroid for a label (for cross-call name memory)."""
        with self._lock:
            for p in self._profiles:
                if p.label == label:
                    return p.centroid.copy()
        return None

    def identify(self, audio: np.ndarray, sample_rate: int = 16000,
                 tail_seconds: float = 4.0,
                 agent_audible: bool | None = None) -> str | None:
        """Guess who is speaking in an utterance that is STILL IN PROGRESS.

        Read-only: never creates or updates a profile (process() does that once
        per finished turn). Uses only the most recent tail_seconds so that in
        overlapping speech the label follows whoever is talking *now*.
        Returns "self", an existing label, the label a new speaker WOULD get,
        or None when the audio is too short / ambiguous.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        tail = int(tail_seconds * sample_rate)
        if len(audio) > tail:
            audio = audio[-tail:]
        duration = len(audio) / float(sample_rate)
        if duration < 0.6:
            return None
        embedding = _normalize(self._embed(audio, sample_rate))
        with self._lock:
            live = [p for p in self._profiles
                    if time.time() - p.last_seen_at <= self.max_idle_seconds]
            best = max(live, key=lambda p: _cosine(embedding, p.centroid)) if live else None
            best_sim = _cosine(embedding, best.centroid) if best is not None else -1.0
            if self._self_embedding is not None and agent_audible is not False:
                self_sim = _cosine(embedding, self._self_embedding)
                if self_sim >= self.self_similarity or (
                        self_sim >= self.SELF_MATCH and self_sim >= best_sim + self.SELF_MARGIN):
                    return "self"
            if best is None:
                return None
            if _cosine(embedding, best.centroid) >= self._match_threshold(best):
                return best.label
            return None

    def live_state(self) -> dict:
        """Snapshot for the agent's prompt: who's in the call right now."""
        with self._lock:
            self._decay(time.time())
            now = time.time()
            participants = [
                {
                    "speaker": p.label,
                    "recent": now - p.last_seen_at < 15.0,
                    "idle_seconds": round(now - p.last_seen_at, 1),
                    "turns": p.sentence_count,
                }
                for p in sorted(self._profiles, key=lambda p: -p.last_seen_at)
            ]
        return {"participant_count": len(participants), "participants": participants}

    def summary(self) -> str:
        state = self.live_state()
        if state["participant_count"] == 0:
            return "no distinct speakers yet"
        parts = []
        for p in state["participants"]:
            tag = "speaking now" if p["recent"] else f"idle {p['idle_seconds']:.0f}s"
            parts.append(f"{p['speaker']} ({tag})")
        return "; ".join(parts)
