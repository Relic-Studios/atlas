"""Owner controls (ATLAS dashboard + hotkey): hard mute, quiet mode, per-agent
talkativeness, muted speakers.

Deterministic, no model calls. Shared by the live server and the sims through
floor.turn_gate / floor.is_quiet, so a control can't be bypassed by any path
that starts a generation (live turn, held fragment, task delivery, resume).

Talkativeness overrides persist to private/owner_prefs.json (dev-only, excluded
from the public export with the rest of private/).
"""
from __future__ import annotations

import json
import os
import threading
import time

_PREFS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "private", "owner_prefs.json")
_lock = threading.Lock()

STATE = {
    "mute": False,            # hard mute: nothing is generated or played
    "muted_speakers": set(),  # labels whose lines are never answered
    "quiet_until": 0.0,       # owner quiet mode (name still wakes the agent)
}
QUIET_DEFAULT_MIN = 10.0


def _load_prefs() -> dict:
    try:
        with open(_PREFS, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_prefs(d: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_PREFS), exist_ok=True)
        tmp = _PREFS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, _PREFS)
    except Exception:  # noqa: BLE001
        pass


def talkativeness_override(pid: str):
    v = _load_prefs().get("talkativeness", {}).get(pid)
    try:
        return None if v is None else max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def set_talkativeness(pid: str, value: float) -> float:
    v = max(0.0, min(1.0, float(value)))
    with _lock:
        d = _load_prefs()
        d.setdefault("talkativeness", {})[pid] = round(v, 2)
        _save_prefs(d)
    return v


def set_mute(on: bool) -> None:
    STATE["mute"] = bool(on)


def set_quiet(on: bool, minutes: float = QUIET_DEFAULT_MIN, clock=time.monotonic) -> None:
    STATE["quiet_until"] = clock() + max(0.5, min(120.0, float(minutes))) * 60 if on else 0.0


def quiet_active(clock=time.monotonic) -> bool:
    return clock() < STATE["quiet_until"]


def set_speaker_muted(label: str, on: bool) -> None:
    lab = (label or "").strip().upper()
    if not lab:
        return
    with _lock:
        (STATE["muted_speakers"].add if on else STATE["muted_speakers"].discard)(lab)


def gate(speaker, names_agent: bool, clock=time.monotonic):
    """HOLD reason from owner controls, or None. Quiet mode yields to the agent's
    name (like the agent's own quiet mode); hard mute and muted speakers don't."""
    if STATE["mute"]:
        return "owner mute"
    if speaker and speaker.upper() in STATE["muted_speakers"]:
        return "owner muted speaker"
    if quiet_active(clock) and not names_agent:
        return "owner quiet mode"
    return None


def snapshot(clock=time.monotonic) -> dict:
    return {"mute": STATE["mute"],
            "quiet_s": max(0.0, round(STATE["quiet_until"] - clock(), 1)),
            "muted_speakers": sorted(STATE["muted_speakers"]),
            "talkativeness": _load_prefs().get("talkativeness", {})}
