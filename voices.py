"""Voice registry facade: available reference voices and which agent uses which.

Backed by agent_registry (dev pack + user-imported voices). Each voice is a
reference WAV; QwenEngine caches its latent to disk keyed by the audio hash, so
the first hot-swap costs a one-time precompute and later switches are instant.

VOICES / PERSONA_VOICES / DEFAULT_VOICE keep their old shapes for callers; call
refresh() after the registry changes (agent created, voice imported).
"""
from __future__ import annotations

from pathlib import Path

import agent_registry as _reg

VOICES: dict[str, dict[str, str]] = {}
PERSONA_VOICES: dict[str, str] = {}
DEFAULT_VOICE: str | None = None


def _replace_in_place(live: dict, new: dict) -> None:
    """Never empty a table another thread may be reading (agent switches run off the event loop since 11-04)."""
    live.update(new)
    for k in [k for k in list(live) if k not in new]:
        live.pop(k, None)


def refresh() -> None:
    global DEFAULT_VOICE
    _replace_in_place(VOICES, {k: {"wav": v["wav"], **({"text": v["text"]} if v.get("text") else {})}
                               for k, v in _reg.voices().items() if v.get("wav")})
    _replace_in_place(PERSONA_VOICES, {aid: v for aid in _reg.agent_ids() if (v := _reg.agent_voice(aid))})
    DEFAULT_VOICE = _reg.default_voice()


def voice_names() -> list[str]:
    return list(VOICES)


def voice_wav(name: str | None) -> str | None:
    """Absolute path of a voice's reference WAV, or None (engine default voice)."""
    if not name or name not in VOICES:
        return None
    return VOICES[name]["wav"]


def voice_text(name: str) -> str | None:
    """Transcript of the reference audio (for ICL cloning), or None if absent."""
    p = (VOICES.get(name) or {}).get("text")
    if not p:
        return None
    q = Path(p)
    return q.read_text(encoding="utf-8").strip() if q.is_file() else None


refresh()
