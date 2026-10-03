"""Canonical voice cues: real recorded words/sounds spliced in place of TTS.

A voice can ship voices/cues/<voice>/ with 24 kHz mono int16 WAV clips plus a
cues.json describing them (no character data lives in code):

  {"words": [[regex, clip_name], ...],     # leading cue words -> clip
   "aliases": {cue_name: file_stem},       # optional
   "sound_only": [cue_name, ...],          # never voiced by TTS, even mid-line
   "attention": [cue_name, ...],           # may get an auto sparkle (off by default)
   "cooldown_s": {cue_name: seconds},      # rate limits (anti-bait)
   "lead_cut_s": {cue_name: seconds},      # long clip cut short when speech follows
   "asks": [{"pattern", "cue", "note", "cooldown_note"}]}  # prompt nudge when asked

When a reply segment STARTS with one of the cue words ("Hey!", "Listen!"), the
real recording plays instead of the clone. Only LEADING cues are swapped: a cue
mid-reply would jump ahead of audio already queued.

Pure text logic (split_leading) is unit-testable without audio.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

CUE_DIR = Path(__file__).resolve().parent / "voices" / "cues"
MAX_CUES = 3
SPARKLE: dict = {}  # automatic sparkle off: voices choose their sounds
SPARKLE_EVERY_S = 25.0


def _load_specs() -> dict:
    specs = {}
    if CUE_DIR.is_dir():
        for f in CUE_DIR.glob("*/cues.json"):
            try:
                specs[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("cue spec %s unreadable: %s", f, e)
    return specs


_SPECS = _load_specs()
CUE_WORDS = {v: [tuple(w) for w in s.get("words", [])] for v, s in _SPECS.items()}
CLIP_FILE = {v: s.get("aliases", {}) for v, s in _SPECS.items()}
SOUND_ONLY = {v: set(s.get("sound_only", [])) for v, s in _SPECS.items()}
ATTENTION = {v: set(s.get("attention", [])) for v, s in _SPECS.items()}
LEAD_CUT_S = {v: s.get("lead_cut_s", {}) for v, s in _SPECS.items()}
COOLDOWN_S: dict = {}
for _v, _s in _SPECS.items():
    for _n, _sec in _s.get("cooldown_s", {}).items():
        COOLDOWN_S[_n] = float(os.environ.get(f"ATLAS_{_n.upper()}_COOLDOWN_S", _sec))


def split_leading(text: str, voice: str) -> tuple[list[str], str]:
    """('Hey! Listen! The boss...', voice) -> (['hey', 'listen'], 'The boss...').

    A cue only counts when it stands alone as an exclamation/greeting
    ("Hey!", "Hey,", "Listen!"); "Hey there buddy" or "Look at this" stay TTS.
    """
    words = CUE_WORDS.get(voice)
    if not words or not text:
        return [], text
    rest = text.lstrip()
    out: list[str] = []
    while len(out) < MAX_CUES:
        for pat, name in words:
            m = re.match(rf"(?i){pat}\s*([!?.,]+)\s*", rest)
            if m:
                out.append(name)
                rest = rest[m.end():]
                break
        else:
            break
    return out, rest.strip()


def strip_sound_words(text: str, voice: str) -> str:
    """Remove chosen-sound words that aren't leading so TTS never says them."""
    snd = SOUND_ONLY.get(voice)
    if not snd or not text:
        return text
    pats = [p for p, n in CUE_WORDS.get(voice, []) if n in snd]
    if not pats:
        return text
    out = re.sub(rf"(?i)(?<![\w'])(?:{'|'.join(pats)})\s*[!?.,]+\s*", "", text)
    return re.sub(r"\s{2,}", " ", out).strip()


def _fade_cut(pcm: bytes, seconds: float, sr: int = 24000, fade_s: float = 0.15) -> bytes:
    import numpy as np
    x = np.frombuffer(pcm, dtype=np.int16)[: int(seconds * sr)].astype(np.float32)
    k = min(len(x), int(fade_s * sr))
    if k:
        x[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return x.astype(np.int16).tobytes()


class VoiceCues:
    def __init__(self, clock=time.monotonic):
        self._clips: dict[str, dict[str, bytes]] = {}
        self._lock = threading.Lock()
        self._last_sparkle: dict[str, float] = {}
        self._last_cue: dict = {}
        self._clock = clock

    def _load(self, voice: str) -> dict[str, bytes]:
        out = {}
        d = CUE_DIR / voice
        if d.is_dir():
            for f in d.glob("*.wav"):
                try:
                    with wave.open(str(f), "rb") as w:
                        if (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (24000, 2, 1):
                            out[f.stem] = w.readframes(w.getnframes())
                except Exception as e:  # noqa: BLE001
                    logger.warning("cue %s unreadable: %s", f, e)
        return out

    def clips(self, voice: str) -> dict[str, bytes]:
        with self._lock:
            if voice not in self._clips:
                self._clips[voice] = self._load(voice)
            return self._clips[voice]

    def has(self, voice: str) -> bool:
        return bool(CUE_WORDS.get(voice)) and bool(self.clips(voice))

    def cooling(self, voice: str, name: str) -> bool:
        return self._clock() - self._last_cue.get((voice, name), -1e9) < COOLDOWN_S.get(name, 0)

    def render(self, text: str, voice: str, allow_sparkle: bool = True) -> tuple[list[bytes], str, list[str]]:
        """Return (pcm chunks to play first, remaining text for TTS, cue names)."""
        if not self.has(voice):
            return [], text, []
        names, rest = split_leading(text, voice)
        lib = self.clips(voice)
        alias = CLIP_FILE.get(voice, {})
        now = self._clock()
        names = [n for n in names if alias.get(n, n) in lib and not self.cooling(voice, n)]
        for n in names:
            if n in COOLDOWN_S:
                self._last_cue[(voice, n)] = now
        rest = strip_sound_words(rest, voice)
        if not names:
            # A sound word with no clip is dropped, never spoken by the clone.
            return [], strip_sound_words(text, voice), []
        pcm = []
        sp = SPARKLE.get(voice)
        if (allow_sparkle and sp in lib and names[0] in ATTENTION.get(voice, ())
                and now - self._last_sparkle.get(voice, -1e9) >= SPARKLE_EVERY_S):
            pcm.append(lib[sp])
            self._last_sparkle[voice] = now
        cuts = LEAD_CUT_S.get(voice, {})
        for n in names:
            clip = lib[alias.get(n, n)]
            if n in cuts and rest:
                clip = _fade_cut(clip, float(cuts[n]))
            pcm.append(clip)
        return pcm, rest, names


CUES = VoiceCues()


def cue_note(voice: str, text: str) -> str:
    """One-line nudge when someone asks for a cue the voice can play."""
    if not text:
        return ""
    for ask in _SPECS.get(voice, {}).get("asks", []):
        try:
            if re.search(ask["pattern"], text, re.I):
                if CUES.cooling(voice, ask.get("cue", "")):
                    return ask.get("cooldown_note", "")
                return ask.get("note", "")
        except (re.error, KeyError):
            continue
    return ""
