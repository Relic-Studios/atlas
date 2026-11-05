"""The owner's voice: recognized in every session, by every agent (design docs/designs/owner_voice.md).

The owner enrolls from the dashboard ("This is me" on their roster row). Each enrollment stores one sample: the mean
voiceprint of that row's turns in the current session. The stored voice is the mean of the latest MAX_SAMPLES samples,
so enrolling again in a later call (another day, another headset) makes recognition steadier.

In each session, every committed turn's voiceprint (the diarizer's `last_embedding`, the same vector dev builds record
as `vp`) is compared with the stored voice. A label is adopted as the owner once it has ADOPT_TURNS turns and the median
of its turns' similarities reaches ADOPT_SIM. Both constants come from tools/voice_id_eval.py on recorded calls with
marks made from the transcript text alone (work/egm/voice-id/eval.md): the threshold was set on the October sessions
and, on the November ones, no marked friend's label was adopted in any enrollment case.

The stored voice is never logged, put in a prompt, sent anywhere, or returned by an endpoint.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PATH = Path(__file__).resolve().parent / "memory_db" / "owner_voice.json"
VERSION = 1
DIM = 192
EMBEDDER = "ecapa"

ENROLL_TURNS = 4        # a sample needs this many of the row's turns in this session
CONSISTENCY = 0.60      # ... and those turns must agree: median similarity to their own mean (mixed rows fail)
MAX_SAMPLES = 5
ADOPT_TURNS = 4         # tools/voice_id_eval.py, 2026-11-04 (work/egm/voice-id/eval.md)
ADOPT_SIM = 0.46
TURN_SIM = 0.529          # T1: a single turn's owner-voice score for a qualifying owner line (experience memory
                         # D3/D7; calibrated with tools/voice_id_eval.py --turn, work/egm/experience-memory/
                         # owner_turn_eval.md: at most 0.6% of non-owner turns reach it, a median 72% of the owner's)
QUALIFY_MIN_S = 2.0      # a qualifying owner line lasts at least 2 seconds
KEEP_TURNS = 200        # voiceprints kept per label for an enrollment sample
NAME_MAX = 40
_NAME_CHARS = re.compile(r"^[\w '\-.]+$", re.UNICODE)
_LABEL = re.compile(r"^S\d+$")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def is_label(label) -> bool:
    return isinstance(label, str) and bool(_LABEL.match(label))


def first_word(name) -> str:
    w = str(name or "").strip().split()
    return w[0].casefold() if w else ""


def valid_name(name, agent_words=()) -> str | None:
    """The owner's name as typed: trimmed, 1-40 characters of letters (any script), digits, spaces, ' - . and no word
    that is an agent's name (the agents would take it for their own)."""
    name = " ".join(str(name or "").split())
    if not name or len(name) > NAME_MAX or not _NAME_CHARS.match(name) or "_" in name:
        return None
    agents = {str(a).casefold() for a in agent_words}
    if any(w.casefold() in agents for w in re.findall(r"[^\s'\-.]+", name)):
        return None
    return name


def sample(vecs) -> np.ndarray | None:
    """One enrollment sample from a row's turn voiceprints, or None if there are too few or they disagree (a row the
    diarizer mixed with someone else is no voice to remember)."""
    vs = [_unit(v) for v in list(vecs)[-KEEP_TURNS:]]
    if len(vs) < ENROLL_TURNS:
        return None
    c = _unit(np.mean(vs, axis=0))
    if float(np.median([float(v @ c) for v in vs])) < CONSISTENCY:
        return None
    return c


def combine(samples) -> np.ndarray:
    return _unit(np.mean([_unit(s) for s in samples], axis=0))


def session_id() -> str:
    """This process's key for its enrollment sample (one sample per session)."""
    return _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------- the stored voice (memory_db/owner_voice.json)
class Store:
    """{"version", "name", "embedder", "center": null, "samples": [{"centroid": [192], "turns": n, "at": session}]}"""

    def __init__(self, path: Path = PATH, embedder: str = EMBEDDER, uses_center: bool = False):
        self.path = Path(path)
        self.embedder, self.uses_center = embedder, bool(uses_center)
        self.name: str | None = None
        self.samples: list[dict] = []
        self.voice: np.ndarray | None = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        """Read the file. Anything wrong (unreadable, another version or embedder, a diarizer that centers its
        voiceprints, a bad name, bad numbers) leaves no stored voice and logs one warning; the owner enrolls again."""
        with self._lock:
            self.name, self.samples, self.voice = None, [], None
            if not self.path.exists():
                return False
            try:
                if self.uses_center:
                    raise ValueError("the diarizer centers its voiceprints")
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(d, dict) or d.get("version") != VERSION or d.get("embedder") != self.embedder:
                    raise ValueError("another version or embedder")
                if d.get("center") is not None:
                    raise ValueError("a centered voice")
                name = valid_name(d.get("name"))
                if not name:
                    raise ValueError("no valid name")
                samples = []
                for s in d.get("samples") or []:
                    c = np.asarray(s.get("centroid"), dtype=np.float32)
                    if c.shape != (DIM,) or not np.isfinite(c).all() or float(np.linalg.norm(c)) == 0:
                        raise ValueError("a bad sample")
                    samples.append({"centroid": _unit(c), "turns": int(s.get("turns") or 0), "at": str(s.get("at") or "")})
                if not samples:
                    raise ValueError("no samples")
            except Exception as e:  # noqa: BLE001
                why = str(e) if isinstance(e, ValueError) and not isinstance(e, json.JSONDecodeError) else type(e).__name__
                log.warning("owner voice not loaded (%s); enroll again from the dashboard", why)
                return False
            self.name, self.samples = name, samples[-MAX_SAMPLES:]
            self.voice = combine([s["centroid"] for s in self.samples])
            return True

    def _write(self, name, samples) -> None:
        d = {"version": VERSION, "name": name, "embedder": self.embedder, "center": None,
             "samples": [{"centroid": [round(float(x), 5) for x in s["centroid"]], "turns": s["turns"], "at": s["at"]}
                         for s in samples]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(d, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def enroll(self, name: str, centroid, turns: int, session: str) -> None:
        """Add (or, within one session, replace) a sample and set the name. Written atomically; on a failed write the
        file and the voice in memory stay as they were and the error propagates."""
        with self._lock:
            entry = {"centroid": _unit(centroid), "turns": int(turns), "at": session}
            samples = ([s for s in self.samples if s["at"] != session] + [entry])[-MAX_SAMPLES:]
            self._write(name, samples)
            self.name, self.samples = name, samples
            self.voice = combine([s["centroid"] for s in samples])

    def forget(self) -> None:
        """Delete the file, then the voice in memory. If the file cannot be deleted the error propagates and nothing
        changes (it would load again at the next start)."""
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.name, self.samples, self.voice = None, [], None

    def status(self) -> dict:
        return {"enrolled": self.voice is not None, "name": self.name, "samples": len(self.samples)}


# ---------------------------------------------------------------- per-session adoption
class Adopter:
    """This session's evidence per diarizer label: its turn voiceprints (for an enrollment sample) and their
    similarities to the stored voice (for adoption). Labels are per process, so this state is too. It is also the
    NameBook's owner-name hook (`people.NameBook.owner`)."""

    def __init__(self, store: Store, session: str | None = None):
        self.store = store
        self.session = session or session_id()
        self.vecs: dict[str, list] = {}
        self.sims: dict[str, list] = {}
        self.adopted: set[str] = set()
        self.refused: set[str] = set()
        self._lock = threading.Lock()

    # -- evidence and the rule
    def observe(self, label: str | None, vec) -> bool:
        """Record one committed, labeled turn. True if this turn adopts the label as the owner."""
        if not is_label(label) or vec is None:
            return False
        v = _unit(vec)
        if v.shape != (DIM,) or not np.isfinite(v).all():
            return False
        with self._lock:
            vs = self.vecs.setdefault(label, [])
            vs.append(v)
            del vs[:-KEEP_TURNS]
            voice = self.store.voice
            if voice is None:
                return False
            sims = self.sims.setdefault(label, [])
            sims.append(float(v @ voice))
            if label in self.adopted or label in self.refused or len(sims) < ADOPT_TURNS:
                return False
            if float(np.median(sims)) >= ADOPT_SIM:
                self.adopted.add(label)
                return True
            return False

    def sample_for(self, label: str) -> tuple[np.ndarray | None, int]:
        with self._lock:
            vs = list(self.vecs.get(label, ()))
        return sample(vs), len(vs)

    def restart_scores(self) -> None:
        """After an enrollment changed the stored voice: score the labels' kept turns against the new voice."""
        with self._lock:
            voice = self.store.voice
            self.sims = ({l: [float(v @ voice) for v in vs] for l, vs in self.vecs.items()}
                         if voice is not None else {})

    # -- owner actions
    def adopt(self, label: str) -> None:
        with self._lock:
            self.adopted.add(label)
            self.refused.discard(label)

    def refuse(self, label: str) -> None:
        with self._lock:
            self.adopted.discard(label)
            self.refused.add(label)

    def last_score(self, label: str | None) -> float | None:
        """The owner-voice score of the label's most recent observed turn (None without a stored voice)."""
        with self._lock:
            s = self.sims.get(label or "")
            return s[-1] if s else None

    def end_call(self) -> None:
        """Call end (docs/designs/experience_memory.md, S4): labels die with the call, so adoptions, refusals,
        scores and the per-label voiceprints go; the stored owner voice stays."""
        with self._lock:
            self.vecs.clear()
            self.sims.clear()
            self.adopted.clear()
            self.refused.clear()

    def clear(self) -> None:
        """After "Forget my voice": nothing is adopted; similarities to the old voice mean nothing."""
        with self._lock:
            self.adopted.clear()
            self.sims.clear()

    def merge(self, old: str, new: str) -> None:
        """The diarizer folded `old` into `new` (one person split in two): their evidence is one person's."""
        if not old or not new or old == new:
            return
        with self._lock:
            self.vecs[new] = (self.vecs.pop(old, []) + self.vecs.get(new, []))[-KEEP_TURNS:]
            self.sims[new] = self.sims.pop(old, []) + self.sims.get(new, [])
            if old in self.adopted:
                self.adopted.discard(old)
                if new not in self.refused:
                    self.adopted.add(new)
            if old in self.refused:
                self.refused.discard(old)
                if new not in self.adopted:
                    self.refused.add(new)

    # -- reading (also the NameBook's hook)
    def labels(self) -> list[str]:
        with self._lock:
            return sorted(self.adopted)

    def seen(self, label: str) -> int:
        with self._lock:
            return len(self.vecs.get(label, ()))

    def is_adopted(self, label: str) -> bool:
        with self._lock:
            return label in self.adopted

    def has_voice(self) -> bool:
        return self.store.voice is not None

    def stored_name(self) -> str | None:
        return self.store.name if self.store.voice is not None else None

    def owner_name(self, label: str | None) -> str | None:
        """The owner's name for an adopted label while a voice is stored, else None."""
        name = self.stored_name()
        if not name or not label:
            return None
        with self._lock:
            return name if label in self.adopted else None


def voice_status(adopter: Adopter | None) -> dict | None:
    """The `voice` object of the owner status: None when recognition is not active (browser mode). No voice numbers."""
    if adopter is None:
        return None
    st = adopter.store.status()
    st["labels"] = adopter.labels() if st["enrolled"] else []
    return st
