"""Floor referee plugin: talk-time balance for debates and meetings.

- floor_stats: who has talked how much in the last N minutes (word share; words are
  a fair proxy for speaking time at conversational pace).
- start_topic: timebox a topic. Reuses the room timer, so the "time's up" line is
  delivered at a gap like any reminder.
- debate_sides: the real lines grouped by speaker, so the model can summarise each
  side fairly from what was actually said.
- quiet_note: an optional per-turn hint ("Riley hasn't spoken in 6 minutes") that the
  agent may use when it is already talking. It never makes the agent speak on its own.

Everything comes from call_summary's in-memory transcript; nothing is written to disk.
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from typing import Callable, Optional

import call_summary

DEFAULT_MIN = 15
QUIET_MIN = 5.0            # silent this long while others talk -> "quiet"
MIN_ROOM_WORDS = 120       # don't comment on balance until there's real talk
MAX_SIDE_CHARS = 2600


def _resolve(who: str, name_of: Optional[Callable]) -> str:
    if name_of and who:
        try:
            n = name_of(who)
            if n:
                return n
        except Exception:  # noqa: BLE001
            pass
    return who or "someone"


def _humans(minutes: float, clock: Callable[[], float]):
    return [r for r in call_summary.lines(minutes, clock) if not r[3] and r[1]]


def stats(minutes: float = DEFAULT_MIN, name_of: Optional[Callable] = None,
          clock: Callable[[], float] = time.time) -> dict:
    """{name: {"words": n, "share": 0..1, "last_ago_min": m}} for humans only."""
    now = clock()
    out: "OrderedDict[str, dict]" = OrderedDict()
    for ts, who, text, _a in _humans(minutes, clock):
        n = _resolve(who, name_of)
        d = out.setdefault(n, {"words": 0, "last": 0.0})
        d["words"] += len(text.split())
        d["last"] = max(d["last"], ts)
    total = sum(d["words"] for d in out.values()) or 1
    for d in out.values():
        d["share"] = d["words"] / total
        d["last_ago_min"] = round((now - d.pop("last")) / 60, 1)
    return dict(out)


def floor_stats(minutes=DEFAULT_MIN, name_of: Optional[Callable] = None,
                clock: Callable[[], float] = time.time) -> str:
    try:
        minutes = float(minutes or DEFAULT_MIN)
    except (TypeError, ValueError):
        minutes = DEFAULT_MIN
    s = stats(minutes, name_of, clock)
    if not s:
        return (f"No one has talked in the last {minutes:g} minutes of this session "
                "(I only hold what was said since ATLAS started). Say so plainly.")
    parts = [f"{n} {round(d['share'] * 100)}%" + (f" (quiet for {d['last_ago_min']:g} min)"
             if d["last_ago_min"] >= QUIET_MIN else "")
             for n, d in sorted(s.items(), key=lambda kv: -kv[1]["words"])]
    return (f"Share of talking in the last {minutes:g} minutes (by words, humans only): "
            + ", ".join(parts) + ". Report it lightly and kindly; never shame anyone for talking a lot.")


def start_topic(topic: str, minutes, asker: str = "") -> str:
    import room_tools
    topic = re.sub(r"\s+", " ", str(topic or "")).strip()[:120] or "this topic"
    try:
        minutes = float(minutes)
    except (TypeError, ValueError):
        minutes = 5.0
    minutes = max(1.0, min(60.0, minutes))
    out = room_tools.set_timer(minutes, f"time's up on {topic}: wrap it up and sum up where each "
                                        f"person landed", asker)
    if not out.startswith("Timer #"):
        return out
    return f"Timeboxed \"{topic}\" to {minutes:g} minutes. I'll call time at the next gap after that."


def debate_sides(minutes=DEFAULT_MIN, topic: str = "", name_of: Optional[Callable] = None,
                 clock: Callable[[], float] = time.time) -> str:
    try:
        minutes = float(minutes or DEFAULT_MIN)
    except (TypeError, ValueError):
        minutes = DEFAULT_MIN
    rows = _humans(minutes, clock)
    if topic:
        keys = [w for w in re.findall(r"[a-z0-9']{4,}", topic.lower())]
        if keys:
            hit = [r for r in rows if any(k in r[2].lower() for k in keys)]
            rows = hit or rows
    if not rows:
        return (f"Nothing was said in the last {minutes:g} minutes that I can summarise. "
                "Say so plainly; don't invent positions.")
    by: "OrderedDict[str, list]" = OrderedDict()
    for _ts, who, text, _a in rows:
        by.setdefault(_resolve(who, name_of), []).append(text)
    budget = MAX_SIDE_CHARS // max(1, len(by))
    blocks = []
    for n, ts in by.items():
        body = " | ".join(ts)
        if len(body) > budget:
            body = "... " + body[-budget:]
        blocks.append(f"{n}: {body}")
    return ("Real lines grouped by person (" + (f"about '{topic}', " if topic else "")
            + f"last {minutes:g} min). Summarise each person's position in one fair sentence, "
            "using only what's here; if someone didn't take a side, say so. Don't pick a winner "
            "unless asked.\n" + "\n".join(blocks))


def quiet_note(speaker_label: Optional[str], name_of: Optional[Callable] = None,
               minutes: float = DEFAULT_MIN, clock: Callable[[], float] = time.time) -> str:
    """One-line hint when someone has gone quiet while others talk. '' otherwise."""
    s = stats(minutes, name_of, clock)
    if len(s) < 3 or sum(d["words"] for d in s.values()) < MIN_ROOM_WORDS:
        return ""
    cur = _resolve(speaker_label, name_of) if speaker_label else ""
    quiet = [n for n, d in s.items() if d["last_ago_min"] >= QUIET_MIN and n != cur
             and not re.fullmatch(r"S\d+|someone", n)]
    if not quiet:
        return ""
    q = quiet[0]
    return (f"FLOOR: {q} hasn't spoken in about {s[q]['last_ago_min']:.0f} minutes while others "
            f"talk. If you're answering anyway and it fits naturally, you may invite {q} in with a "
            "light question. Don't force it or call them out.")


# ------------------------------------------------------------------ intent
_STATS_RE = re.compile(r"\b(?:who(?:'s| has| is)?\s+(?:been\s+)?(?:talk(?:ed|ing)|speak(?:ing)?)\s+"
                       r"(?:the\s+)?most|talk(?:ing)?\s+time|who\s+(?:hasn'?t|has\s+not)\s+"
                       r"(?:talked|spoken|said\s+anything)|am\s+i\s+talking\s+too\s+much)\b", re.I)
_TOPIC_RE = re.compile(r"\b(?:give|timebox|time-box|allow|limit)\b.{0,30}?\b(\d+|one|two|three|five|ten|"
                       r"fifteen|twenty)\s*(?:min(?:ute)?s?)\b.{0,12}?\b(?:on|for|to\s+(?:talk|argue|debate)\s+about)\s+"
                       r"(?:the\s+)?(.{2,60}?)[.?!]*$", re.I)
_SIDES_RE = re.compile(r"\b(?:sum(?:marise|marize|\s+up)|recap|break\s+down)\b.{0,20}?\b(?:both|each|"
                       r"the\s+two|everyone'?s?|all)\s+(?:sides?|positions?|arguments?|takes?)"
                       r"(?:\s+(?:on|about)\s+(.{2,60}?))?[.?!]*$", re.I)
_W = {"one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20}


def intent(text: str):
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "").strip()
    m = _SIDES_RE.search(t)
    if m:
        return ("debate_sides", {"topic": (m[1] or "").strip()})
    m = _TOPIC_RE.search(t)
    if m:
        n = int(m[1]) if m[1].isdigit() else _W.get(m[1].lower(), 5)
        return ("start_topic", {"topic": m[2].strip(" ,"), "minutes": n})
    if _STATS_RE.search(t):
        return ("floor_stats", {})
    return None
