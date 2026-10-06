"""Room-energy sense (plugin "room_energy").

Reads the mood of the room from this session's in-memory transcript
(call_summary's buffer): laughter, exclamations, pace of turn-taking,
pile-ups (people talking over each other), hostility and heaviness.
It adds no tool and no action; it only shapes HOW the agent talks
(tone and length), never WHETHER it talks.

Signals are text + timing only (no audio features yet). Pure and fast:
one pass over the last few minutes of lines.
"""
from __future__ import annotations

import re
import time
from typing import Callable

import call_summary

WINDOW_MIN = 3.0
MIN_LINES = 5            # don't judge a room from two lines
PILEUP_S = 1.2           # a different speaker within this gap = talking over

_LAUGH = re.compile(r"\b(?:ha(?:ha)+h?|he(?:he)+|lo+l|lmf?ao+|rofl|dead+|i'?m\s+crying|"
                    r"that'?s\s+(?:so\s+)?funny)\b|\(laugh\w*\)|\[laugh\w*\]", re.I)
_HYPE = re.compile(r"\b(?:let'?s\s+go+|no\s+way|yo+|holy\s+\w+|insane|crazy|clutch|"
                   r"poggers|pog|w+\b|hype|sick)\b", re.I)
# Strong hostility vs. weak (bare "shut up" is constant banter in gaming groups).
_HOSTILE = re.compile(r"\b(?:shut\s+the\s+\w+\s+up|f+u+c+k\s+(?:you|off)|screw\s+you|"
                      r"you'?re\s+(?:so\s+)?(?:stupid|an?\s+idiot|a\s+\w*hole|pathetic|useless)|"
                      r"i'?m\s+(?:so\s+)?(?:done\s+with\s+(?:you|this)|pissed)|leave\s+me\s+alone|"
                      r"that'?s\s+not\s+(?:funny|cool|okay)|i\s+hate\s+(?:you|this)\s+(?:so\s+much|seriously))\b", re.I)
_WEAK_HOSTILE = re.compile(r"\b(?:shut\s+up|stfu|stop\s+(?:it|talking)|fuck\s+this|whatever,?\s+man|"
                           r"i'?m\s+(?:so\s+)?done)\b", re.I)
# Heavy needs unmistakable real-life phrasing ("I died" / "I'm scared" are usually games).
_HEAVY = re.compile(r"\b(?:passed\s+away|funeral|(?:my|our)\s+(?:mom|dad|mother|father|grandma|grandpa|grandmother|grandfather|brother|sister|"
                    r"uncle|aunt|cousin|friend|wife|husband|girlfriend|boyfriend|son|daughter|dog|cat|baby)\s+"
                    r"(?:died|is\s+in\s+(?:the\s+)?hospital|is\s+sick)|"
                    r"i'?m\s+(?:really\s+|so\s+)?(?:depressed|struggling|not\s+okay)|"
                    r"(?:my|been\s+having)\s+(?:depression|anxiety|panic\s+attacks?)|"
                    r"rough\s+(?:day|week|time)\s+(?:lately|honestly|at\s+home)|"
                    r"(?:she|he|they)\s+broke\s+up\s+with\s+me|lost\s+my\s+(?:job|mom|dad|mother|father|"
                    r"grandma|grandpa|brother|sister|dog|cat)|cancer|miscarriage)\b", re.I)

MOODS = ("quiet", "calm", "hype", "tense", "heavy")

_NOTES = {
    "hype": ("ROOM ENERGY: the room is lively and laughing. If you speak, match it: "
             "quick, playful, short. Don't slow it down with long explanations."),
    "tense": ("ROOM ENERGY: the room is tense (people are frustrated or snapping at each other). "
              "If you speak, be calm, kind and brief. No jokes at anyone's expense, don't take sides, "
              "don't lecture."),
    "heavy": ("ROOM ENERGY: someone is going through something hard. If you speak, be gentle and "
              "warm, slow down, no jokes, and listen more than you advise."),
}


def read(minutes: float = WINDOW_MIN, clock: Callable[[], float] = time.time) -> dict:
    """{"mood": str, "lines": n, "scores": {...}} for the last few minutes."""
    rows = [r for r in call_summary.lines(minutes, clock) if r[2]]
    humans = [r for r in rows if not r[3]]
    out = {"mood": "quiet", "lines": len(humans), "scores": {}}
    if len(humans) < MIN_LINES:
        return out
    laugh = sum(1 for r in humans if _LAUGH.search(r[2]))
    hype = sum(1 for r in humans if _HYPE.search(r[2]) or r[2].rstrip().endswith("!"))
    hostile = sum(1.0 if _HOSTILE.search(r[2]) else 0.4 if _WEAK_HOSTILE.search(r[2]) else 0.0
                  for r in humans)
    heavy = sum(1 for r in humans if _HEAVY.search(r[2]))
    pile = 0
    for a, b in zip(rows, rows[1:]):
        if b[1] != a[1] and (b[0] - a[0]) < PILEUP_S:
            pile += 1
    span = max(rows[-1][0] - rows[0][0], 30.0)
    pace = len(humans) / (span / 60.0)  # human turns per minute
    n = len(humans)
    s = {"laugh": round(laugh / n, 2), "hype": round(hype / n, 2), "hostile": round(min(hostile / n, 1.0), 2),
         "heavy": round(heavy / n, 2), "pileup": round(pile / max(len(rows) - 1, 1), 2),
         "pace": round(pace, 1)}
    out["scores"] = s
    # Priority: hurt beats tension beats fun. Two signals needed for "tense" so one
    # heated "stop it" in banter doesn't flip the whole room.
    if heavy >= 1 and hostile < 1 and laugh <= 1:
        mood = "heavy"
    elif laugh <= 1 and (hostile >= 2 or (hostile >= 1.4 and s["pileup"] >= 0.35)):
        mood = "tense"
    elif laugh >= 2 or (s["hype"] >= 0.3 and pace >= 6):
        mood = "hype"
    else:
        mood = "calm"
    out["mood"] = mood
    return out


def note(clock: Callable[[], float] = time.time) -> str:
    """Per-turn tone hint, or "" for calm/quiet rooms. Never says speak or hold."""
    try:
        return _NOTES.get(read(clock=clock)["mood"], "")
    except Exception:  # noqa: BLE001 - a mood read must never cost a turn
        return ""


def feed() -> list:
    """Owner panel: the current read (same item shape as the other plugin feeds)."""
    r = read()
    sc = r.get("scores") or {}
    meta = (f"{r['lines']} lines in the last {int(WINDOW_MIN)} min"
            + (f" · laugh {sc.get('laugh', 0):.0%} · tense {sc.get('hostile', 0):.0%} · "
               f"pace {sc.get('pace', 0)}/min" if sc else ""))
    hint = _NOTES.get(r["mood"], "No tone change: the agent talks normally.")
    return [{"title": f'Room: {r["mood"].capitalize()}', "meta": meta,
             "items": [{"text": hint.replace("ROOM ENERGY: ", "")}]}]
