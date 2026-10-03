"""Single source of truth for which agents and voices exist.

Two sources, one schema:
  * dev pack   -- dev_pack/pack.json (owner's hand-tuned agents + voice clones).
                  Loaded only when present. The public export omits dev_pack/,
                  personas/ and voices/, so none of it ships. Disable locally with
                  ATLAS_DEV_PACK=0 to run exactly as the public build does.
  * user data  -- personas/agents.json (agents made in the creator / heart.md) and
                  voices/user/voices.json (imported reference audio).

Every other module asks this registry (agents, voices, profiles, identity,
nicknames, backchannel lines) instead of hard-coding names, so the code itself
carries no built-in characters.

Agent record fields (all optional except id):
  id, name, prompt_file | prompt, voice, role, accent, gender ('male'|'female'|None),
  what, interests[list], talkativeness[0..1], nicknames[list], backchannels[list],
  clip_key (backchannel clip dir; defaults to voice), dev (bool)
"""
from __future__ import annotations

import json
import re
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEV_PACK = ROOT / "dev_pack" / "pack.json"
USER_AGENTS = ROOT / "personas" / "agents.json"
USER_VOICES = ROOT / "voices" / "user" / "voices.json"
GENERIC_PROMPT = ("You're a friendly voice in a Discord call. No character has been set up yet, "
                  "so keep replies short, and if asked, say the owner can create an agent in ATLAS.")

_lock = threading.RLock()
_state: dict = {}


def dev_enabled() -> bool:
    return os.environ.get("ATLAS_DEV_PACK", "1") != "0" and DEV_PACK.is_file()


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except Exception as e:  # noqa: BLE001
        logger.warning("registry: %s unreadable: %s", p, e)
        return default


def _abs(p: str | None) -> str | None:
    if not p:
        return None
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def reload() -> dict:
    """(Re)read every source. Cheap; called at import and after any create/delete."""
    agents: dict[str, dict] = {}
    voices: dict[str, dict] = {}
    default = None
    traits: dict = {}
    if dev_enabled():
        pack = _read_json(DEV_PACK, {})
        traits = dict(pack.get("traits") or {})
        for k, v in (pack.get("voices") or {}).items():
            voices[k] = {"wav": _abs(v.get("wav")), "text": _abs(v.get("text")), "dev": True}
        for a in pack.get("agents") or []:
            if a.get("id"):
                agents[a["id"]] = {**a, "dev": True}
        default = pack.get("default")
    uv = _read_json(USER_VOICES, {})
    for k, v in ((uv.get("voices") or {}) if isinstance(uv, dict) else {}).items():
        voices.setdefault(k, {"wav": _abs(v.get("wav")), "text": _abs(v.get("text")),
                              "label": v.get("label"), "user": True})
    reg = _read_json(USER_AGENTS, [])
    for a in reg if isinstance(reg, list) else []:
        if isinstance(a, dict) and a.get("id") and a["id"] not in agents:
            rec = {**a, "custom": True}
            rec.setdefault("prompt_file", f"personas/{a['id']}.txt")
            agents[a["id"]] = rec
    if default not in agents:
        default = next(iter(agents), None)
    with _lock:
        _state.clear()
        _state.update(agents=agents, voices=voices, default=default, traits=traits)
    return _state


def _s() -> dict:
    with _lock:
        if not _state:
            reload()
        return _state


def traits() -> dict:
    """Pack-supplied engine data (interjections, narrator names, ICL voices, audio
    bus names, self-knowledge). Empty in the public build: code has no characters."""
    return dict(_s().get("traits") or {})


def _phrase_re(p: str) -> str:
    return r"[\s-]*".join(re.escape(w) for w in re.split(r"[\s-]+", p.strip().lower()) if w)


def interjection_words() -> set[str]:
    out = set()
    for p in traits().get("interjections") or []:
        out.update(w for w in re.split(r"\s+", str(p).lower()) if w)
    return out


def opener_alt(key: str) -> str:
    """Regex alternation of a pack opener list ('' when none)."""
    ps = sorted({_phrase_re(str(p)) for p in traits().get(key) or [] if str(p).strip()},
                key=len, reverse=True)
    return "|".join(ps)


# ---- agents -------------------------------------------------------------------
def agents() -> dict[str, dict]:
    return dict(_s()["agents"])


def agent(aid: str) -> dict:
    return _s()["agents"].get(aid, {})


def agent_ids() -> list[str]:
    return list(_s()["agents"])


def default_agent() -> str | None:
    return _s()["default"]


def is_dev(aid: str) -> bool:
    return bool(agent(aid).get("dev"))


def prompt_text(aid: str) -> str | None:
    a = agent(aid)
    if a.get("prompt"):
        return str(a["prompt"]).strip()
    p = _abs(a.get("prompt_file"))
    try:
        return Path(p).read_text(encoding="utf-8").strip() if p else None
    except FileNotFoundError:
        logger.warning("registry: prompt file for %s missing: %s", aid, p)
        return None


def display_name(aid: str) -> str:
    return agent(aid).get("name") or aid.title()


def meta(aid: str) -> dict:
    a = agent(aid)
    out = {"role": a.get("role", ""), "accent": a.get("accent", "#22d3ee")}
    if a.get("custom"):
        out.update(name=a.get("name"), custom=True)
    return out


def agent_voice(aid: str) -> str | None:
    v = agent(aid).get("voice")
    return v if v in _s()["voices"] else None


def profile(aid: str) -> dict:
    a = agent(aid)
    return {"interests": tuple(a.get("interests") or ()),
            "talkativeness": a.get("talkativeness", 0.35)}


def identity(aid: str) -> dict:
    a = agent(aid)
    return {"gender": a.get("gender"), "what": a.get("what") or ""}


def nicknames(aid: str) -> tuple[str, ...]:
    return tuple(n.lower() for n in agent(aid).get("nicknames") or ())


def all_agent_words() -> set[str]:
    """Every name/nickname any agent answers to (people.py must never learn these as human names)."""
    out = set()
    for aid, a in _s()["agents"].items():
        out.add(aid.lower())
        out.update(w.lower() for w in str(a.get("name") or "").split())
        out.update(n.lower() for n in a.get("nicknames") or ())
    return out


def backchannel_lines(aid: str) -> list[str]:
    return list(agent(aid).get("backchannels") or [])


def clip_key(aid: str) -> str | None:
    a = agent(aid)
    return a.get("clip_key") or a.get("voice")


def agent_for_clip_key(key: str) -> str | None:
    for aid, a in _s()["agents"].items():
        if (a.get("clip_key") or a.get("voice")) == key and a.get("backchannels"):
            return aid
    return None


# ---- voices -------------------------------------------------------------------
def voices() -> dict[str, dict]:
    return dict(_s()["voices"])


def voice_path(name: str) -> str | None:
    return (_s()["voices"].get(name) or {}).get("wav")


def voice_text_path(name: str) -> str | None:
    return (_s()["voices"].get(name) or {}).get("text")


def default_voice() -> str | None:
    # The default agent's voice may have been deleted (or never imported); fall
    # back to any installed voice rather than leaving TTS with no reference.
    d = default_agent()
    return (agent_voice(d) if d else None) or next(iter(_s()["voices"]), None)


reload()
