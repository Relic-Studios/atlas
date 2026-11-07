"""Single source of truth for which agents and voices exist.

Three sources, one schema:
  * dev pack   -- dev_pack/pack.json (owner's hand-tuned agents + voice clones).
                  Loaded only when present. The public export omits dev_pack/,
                  personas/ and voices/, so none of it ships. Disable locally with
                  ATLAS_DEV_PACK=0 to run exactly as the public build does.
  * user data  -- personas/agents.json (agents made in the creator / heart.md) and
                  voices/user/voices.json (imported reference audio).
  * bundle     -- starter_pack/pack.json, immutable public defaults. Existing
                  dev and user IDs/voice keys always take precedence.

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
BUNDLED_PACK = ROOT / "starter_pack" / "pack.json"
GENERIC_PROMPT = ("You're a friendly voice in a group conversation. No character has been set up yet, "
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
    if not p or not isinstance(p, (str, os.PathLike)):
        return None
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)


def _prompt_for(a: dict) -> str | None:
    if a.get("prompt"):
        return str(a["prompt"]).strip()
    p = _abs(a.get("prompt_file"))
    try:
        return Path(p).read_text(encoding="utf-8").strip() if p else None
    except (OSError, UnicodeError):
        return None


def _voice_usable(v: dict, *, bundled=False) -> bool:
    try:
        path = Path(v.get("wav") or "")
        if bundled:
            import wave
            text = Path(v.get("text") or "")
            base = BUNDLED_PACK.parent.resolve()
            if not (path.resolve().is_relative_to(base) and text.resolve().is_relative_to(base)):
                return False
            if not text.read_text(encoding="utf-8").strip():
                return False
            # PCM samples are finite by construction. Bound decoding to a 30s
            # reference and require the declared frames to actually be readable.
            with wave.open(str(path), "rb") as audio:
                n = audio.getnframes()
                if (audio.getnchannels() != 1 or audio.getframerate() != 24000
                        or audio.getcomptype() != "NONE" or not 0 < n <= 30 * 24000):
                    return False
                return len(audio.readframes(n)) == n * audio.getsampwidth()
        # Existing user/dev references may be stereo, another rate/format, and
        # need not have an ICL transcript. Do not impose the bundle's format.
        import soundfile as sf
        with sf.SoundFile(str(path)) as audio:
            return audio.frames > 0 and audio.samplerate > 0 and audio.channels > 0
    except Exception:  # noqa: BLE001 - corrupt references must not make setup fail
        return False


def reload() -> dict:
    """(Re)read every source. Cheap; called at import and after any create/delete."""
    agents: dict[str, dict] = {}
    voices: dict[str, dict] = {}
    default = None
    traits: dict = {}
    dev = dev_enabled()
    if dev:
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
        if isinstance(a, dict) and isinstance(a.get("id"), str) and a["id"] and a["id"] not in agents:
            rec = {**a, "custom": True}
            rec.setdefault("prompt_file", f"personas/{a['id']}.txt")
            agents[a["id"]] = rec
    bundle_error = None
    pack = _read_json(BUNDLED_PACK, {})
    if not isinstance(pack, dict):
        pack = {}
    bundle_voices = pack.get("voices") or {}
    bundle_agents = pack.get("agents") or []
    bundle_default = pack.get("default")
    if not isinstance(bundle_default, str) or not bundle_default.strip():
        bundle_default = None
        if pack:
            bundle_error = "The default agent ID is invalid. Repair ATLAS, or choose another agent."
    if not isinstance(bundle_voices, dict):
        bundle_voices = {}
        bundle_error = "The default agent bundle is invalid. Repair ATLAS, or create an agent."
    if not isinstance(bundle_agents, list):
        bundle_agents = []
        bundle_error = "The default agent bundle is invalid. Repair ATLAS, or create an agent."
    for k, v in bundle_voices.items():
        if k in voices:
            continue
        if not isinstance(v, dict):
            bundle_error = "The bundled voice record is invalid. Repair ATLAS, or import a voice."
            continue
        rec = {"wav": _abs(v.get("wav")), "text": _abs(v.get("text")),
               "label": v.get("label"), "bundled": True}
        if _voice_usable(rec, bundled=True):
            voices[k] = rec
        else:
            bundle_error = "The bundled voice is missing or damaged. Repair ATLAS, or import a voice and create an agent."
    for a in bundle_agents:
        if (not isinstance(a, dict) or not isinstance(a.get("id"), str) or not a["id"].strip()
                or not isinstance(a.get("voice"), str) or not a["voice"].strip()):
            bundle_error = "A bundled agent ID or voice is invalid. Repair ATLAS, or choose another agent."
            continue
        if a["id"] not in agents:
            prompt_path = _abs(a.get("prompt_file"))
            contained_prompt = (bool(a.get("prompt")) or
                                (prompt_path and Path(prompt_path).resolve().is_relative_to(BUNDLED_PACK.parent.resolve())))
            if (contained_prompt and _prompt_for(a)
                    and _voice_usable(voices.get(a.get("voice"), {}),
                                      bundled=bool(voices.get(a.get("voice"), {}).get("bundled")))):
                agents[a["id"]] = {**a, "bundled": True, "custom": False, "dev": False}
    usable_voices = {k for k, v in voices.items() if _voice_usable(v, bundled=bool(v.get("bundled")))}
    usable = [aid for aid, a in agents.items() if isinstance(a.get("voice"), str)
              and a["voice"] in usable_voices and _prompt_for(a)]
    if not pack and not dev:
        bundle_error = "The default agent bundle is missing or unreadable. Repair ATLAS, or import a voice and create an agent."
    elif bundle_default and bundle_default not in usable:
        bundle_error = bundle_error or "The default agent has no usable voice or personality. Repair ATLAS, or choose another agent."
    # Settings I/O stays outside the registry lock. A public saved selection
    # determines both persona and voice, rather than two independent defaults.
    selected = None
    if not dev:
        import user_settings
        selected = user_settings.load().get("agent")
    if dev and default in usable:
        pass
    elif selected in usable:
        default = selected
    else:
        default = (next((aid for aid in usable if agents[aid].get("custom")), None)
                   or (bundle_default if bundle_default in usable else None)
                   or next(iter(usable), None))
    with _lock:   # update in place, so a reader without the lock never sees it empty
        _state.update(agents=agents, voices=voices, default=default, traits=traits,
                      usable=usable, bundle_error=bundle_error)
    return _state


def _s() -> dict:
    with _lock:
        initialized = bool(_state)
    if not initialized:
        reload()  # settings I/O and asset decoding never run under the registry lock
    with _lock:
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


def ready_agents() -> list[dict]:
    state = _s()
    return [{"id": aid, "name": state["agents"][aid].get("name") or aid.title(),
             "voice": state["agents"][aid]["voice"], "bundled": bool(state["agents"][aid].get("bundled"))}
            for aid in state.get("usable", [])]


def ready_agent() -> dict | None:
    return next((a for a in ready_agents() if a["id"] == default_agent()), None)


def bundle_error() -> str | None:
    return _s().get("bundle_error")


def is_dev(aid: str) -> bool:
    return bool(agent(aid).get("dev"))


def prompt_text(aid: str) -> str | None:
    return _prompt_for(agent(aid))


def display_name(aid: str) -> str:
    return agent(aid).get("name") or aid.title()


def meta(aid: str) -> dict:
    a = agent(aid)
    out = {"role": a.get("role", ""), "accent": a.get("accent", "#22d3ee")}
    if a.get("custom"):
        out.update(name=a.get("name"), custom=True)
    if a.get("bundled"):
        out.update(name=a.get("name"), bundled=True)
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
