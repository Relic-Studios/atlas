"""Which game is on screen, and may the agents give tips in it (docs/designs/live_screen.md, section 7).

A profile names a game, the process names (and window titles) that identify it, whether it is competitive (no tips
then: no live advantage in ranked play), and the spoiler level for tips (none: nothing about what the player hasn't
seen; light: point the way without naming outcomes; full: anything). Built-in profiles live in
live_screen/game_profiles/*.json; the owner's own go in private/live_screen_profiles/*.json and win on the same name.
An unknown game gets no tips."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILTIN = HERE / "game_profiles"
OWNER = HERE.parent / "private" / "live_screen_profiles"
SPOILERS = ("none", "light", "full")

_lock = threading.Lock()
_cache: dict = {"profiles": None}


def _load_dir(d: Path) -> list:
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(p, dict) and p.get("name"):
            p.setdefault("processes", [])
            p.setdefault("titles", [])
            p["competitive"] = bool(p.get("competitive", False))
            p["spoilers"] = p.get("spoilers") if p.get("spoilers") in SPOILERS else "light"
            out.append(p)
    return out


def profiles(reload: bool = False) -> list:
    with _lock:
        if _cache["profiles"] is None or reload:
            by_name = {p["name"].lower(): p for p in _load_dir(BUILTIN)}
            by_name.update({p["name"].lower(): p for p in _load_dir(OWNER)})
            _cache["profiles"] = list(by_name.values())
        return list(_cache["profiles"])


def match(title: str, process: str) -> dict | None:
    """The profile of the game in front, by process name first, then window title; None if unknown."""
    proc = (process or "").lower()
    for p in profiles():
        if proc and proc in [x.lower() for x in p["processes"]]:
            return p
    for p in profiles():
        if any(re.search(t, title or "", re.IGNORECASE) for t in p["titles"]):
            return p
    return None
