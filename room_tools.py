"""Room tools for group calls (owner-approved plugins, 10-05):

  dice_polls  roll_dice, flip_coin, pick_one, start_poll, cast_vote, poll_results
  timers      set_timer, list_timers, cancel_timer   (fires through a ROOM CUE at a gap)
  weather     check_weather                          (Open-Meteo, free, no key)

Everything is local except check_weather, which sends only the place name to
open-meteo.com. Randomness comes from the OS (secrets), so rolls are fair.
State (timers, the open poll) lives in memory_db/room.json and is shared by all
agents: a timer set while Fae was active still fires if Pup is active later.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "memory_db" / "room.json"
_rng = secrets.SystemRandom()
_lock = threading.RLock()

MAX_TIMERS = 12
MAX_TIMER_MIN = 24 * 60
MAX_DICE = 50
MAX_SIDES = 1000
TIMER_STALE_S = 15 * 60      # a reminder that couldn't be said for 15 min is dropped

# Our own "a timer went off" cue. Never a real user line (floor.steering_note and
# the pipeline treat it like a TASK CUE: no passivity HOLD, no bait checks).
ROOM_CUE_RE = re.compile(r"\((?:ROOM|MAIL|BET) CUE #(\d+)\)")


# ------------------------------------------------------------------ state
def _load() -> dict:
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _next_id(d: dict) -> int:
    d["seq"] = int(d.get("seq", 0)) + 1
    return d["seq"]


# ------------------------------------------------------------------ dice & coins
_DICE_RE = re.compile(r"^\s*(\d*)\s*d\s*(\d+)\s*(?:([+-])\s*(\d+))?\s*$", re.I)


def roll_dice(spec: str = "d20") -> str:
    spec = (spec or "d20").strip().lower().replace(" ", "") or "d20"
    if re.fullmatch(r"\d+", spec):
        spec = "d" + spec
    m = _DICE_RE.match(spec)
    if not m:
        return f"Couldn't read '{spec}'. Use dice notation like d20, 2d6 or 3d8+2."
    n = int(m[1] or 1)
    sides = int(m[2])
    mod = int(m[4] or 0) * (-1 if m[3] == "-" else 1)
    if not (1 <= n <= MAX_DICE) or not (2 <= sides <= MAX_SIDES):
        return f"Keep it to 1-{MAX_DICE} dice with 2-{MAX_SIDES} sides."
    rolls = [_rng.randint(1, sides) for _ in range(n)]
    total = sum(rolls) + mod
    shown = f"{n}d{sides}" + (f"{'+' if mod > 0 else '-'}{abs(mod)}" if mod else "")
    detail = "" if n == 1 and not mod else f" (rolls: {', '.join(map(str, rolls))}" + \
        (f", {'+' if mod > 0 else '-'}{abs(mod)}" if mod else "") + ")"
    extra = ""
    if n == 1 and sides == 20:
        extra = " Natural 20!" if rolls[0] == 20 else (" Natural 1." if rolls[0] == 1 else "")
    return f"Rolled {shown}: {total}{detail}.{extra} Announce the result exactly."


def flip_coin(times: int = 1) -> str:
    try:
        times = max(1, min(10, int(times or 1)))
    except (TypeError, ValueError):
        times = 1
    flips = [_rng.choice(("heads", "tails")) for _ in range(times)]
    if times == 1:
        return f"The coin landed on {flips[0]}. Announce it exactly."
    return f"Flipped {times} coins: {', '.join(flips)}. Announce them exactly."


def pick_one(options) -> str:
    opts = _options(options)
    if len(opts) < 2:
        return "Give me at least two options to pick from."
    return f"Picked at random: {_rng.choice(opts)} (from {', '.join(opts)}). Announce it exactly."


def _options(options) -> List[str]:
    if isinstance(options, str):
        parts = re.split(r"\s*(?:,|/|\bor\b|\n)\s*", options)
    elif isinstance(options, (list, tuple)):
        parts = [str(x) for x in options]
    else:
        parts = []
    out, seen = [], set()
    for p in parts:
        p = p.strip(" .!?\"'")[:60]
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out[:10]


# ------------------------------------------------------------------ polls
def start_poll(question: str, options) -> str:
    opts = _options(options)
    q = (question or "").strip()[:140]
    if len(opts) < 2:
        return "A poll needs at least two options."
    with _lock:
        d = _load()
        d["poll"] = {"question": q or "Quick vote", "options": opts, "votes": {}, "opened": time.time()}
        _save(d)
    return (f"Poll open: {q or 'Quick vote'} Options: {', '.join(opts)}. Tell the room the options "
            "and that they can just say their pick. Record each pick with cast_vote.")


def _match_option(choice: str, opts: List[str]) -> Optional[str]:
    c = (choice or "").strip().lower()
    if not c:
        return None
    for o in opts:
        if c == o.lower():
            return o
    hits = [o for o in opts if c in o.lower() or o.lower() in c]
    if len(hits) == 1:
        return hits[0]
    if re.fullmatch(r"\d+", c) and 1 <= int(c) <= len(opts):
        return opts[int(c) - 1]
    return None


def cast_vote(choice: str, voter: str = "") -> str:
    with _lock:
        d = _load()
        p = d.get("poll")
        if not p:
            return "There's no poll open. Start one with start_poll first."
        o = _match_option(choice, p["options"])
        if o is None:
            return f"'{choice}' isn't one of the options ({', '.join(p['options'])})."
        who = (voter or "").strip() or f"voter{len(p['votes']) + 1}"
        changed = who in p["votes"]
        p["votes"][who] = o
        _save(d)
    return (f"{'Changed' if changed else 'Counted'}: {o}. "
            + _tally_text(p) + " Acknowledge briefly; don't read the whole tally unless asked.")


def _tally_text(p: dict) -> str:
    counts = {o: 0 for o in p["options"]}
    for v in p["votes"].values():
        counts[v] = counts.get(v, 0) + 1
    return "Tally: " + ", ".join(f"{o} {n}" for o, n in counts.items()) + "."


def poll_results(close: bool = False) -> str:
    with _lock:
        d = _load()
        p = d.get("poll")
        if not p:
            return "No poll is open."
        counts = {o: 0 for o in p["options"]}
        for v in p["votes"].values():
            counts[v] = counts.get(v, 0) + 1
        top = max(counts.values()) if counts else 0
        leaders = [o for o, n in counts.items() if n == top and top > 0]
        if close:
            d.pop("poll", None)
            _save(d)
    if not leaders:
        head = "No votes yet."
    elif len(leaders) == 1:
        head = f"{leaders[0]} wins with {top}." if close else f"{leaders[0]} leads with {top}."
    else:
        head = f"Tie between {' and '.join(leaders)} at {top} each."
    return f"{p['question']} -- {head} {_tally_text(p)}" + (" Poll closed." if close else "")


# ------------------------------------------------------------------ timers
_UNIT_S = {"s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1, "m": 60, "min": 60, "mins": 60,
           "minute": 60, "minutes": 60, "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600}


def _minutes(minutes) -> Optional[float]:
    if isinstance(minutes, (int, float)):
        return float(minutes)
    s = str(minutes or "").strip().lower()
    try:
        return float(s)
    except ValueError:
        pass
    total, ok = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+)", s):
        if unit in _UNIT_S:
            total += float(num) * _UNIT_S[unit] / 60.0
            ok = True
    return total if ok else None


def set_timer(minutes, message: str = "", asker: str = "", clock: Callable[[], float] = time.time) -> str:
    mins = _minutes(minutes)
    if mins is None or mins <= 0:
        return "How long? Give the timer a duration like 10 (minutes) or '90 seconds'."
    if mins > MAX_TIMER_MIN:
        return "Timers can be at most 24 hours."
    msg = re.sub(r"\s+", " ", (message or "").strip())[:160] or "time's up"
    now = clock()
    with _lock:
        d = _load()
        ts = [t for t in d.get("timers", []) if not t.get("done")]
        if len(ts) >= MAX_TIMERS:
            return f"There are already {MAX_TIMERS} timers running. Cancel one first."
        t = {"id": _next_id(d), "due": now + mins * 60, "set": now, "message": msg,
             "asker": asker if re.fullmatch(r"S\d+", asker or "") else "", "done": False, "tries": 0}
        ts.append(t)
        d["timers"] = ts
        _save(d)
    return (f"Timer #{t['id']} set for {_human(mins * 60)}: \"{msg}\". It goes off on its own; "
            "you'll get a cue to announce it. Confirm briefly.")


def _human(sec: float) -> str:
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} seconds"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''}" + (f" {s} seconds" if s and m < 5 else "")
    h, m = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {m} minutes" if m else "")


def list_timers(clock: Callable[[], float] = time.time) -> str:
    now = clock()
    with _lock:
        ts = [t for t in _load().get("timers", []) if not t.get("done")]
    if not ts:
        return "No timers running."
    return "Running timers: " + "; ".join(
        f"#{t['id']} \"{t['message']}\" in {_human(max(0, t['due'] - now))}" for t in sorted(ts, key=lambda x: x["due"]))


def cancel_timer(id=None, message: str = "") -> str:  # noqa: A002 (tool arg name)
    with _lock:
        d = _load()
        ts = [t for t in d.get("timers", []) if not t.get("done")]
        hit = None
        try:
            tid = int(id) if id not in (None, "") else None
        except (TypeError, ValueError):
            tid = None
        if tid is not None:
            hit = next((t for t in ts if t["id"] == tid), None)
        elif message:
            ms = [t for t in ts if message.lower() in t["message"].lower()]
            hit = ms[0] if len(ms) == 1 else None
        elif len(ts) == 1:
            hit = ts[0]
        if hit is None:
            return "Which timer? " + (list_timers() if ts else "None are running.")
        d["timers"] = [t for t in ts if t is not hit]
        _save(d)
    return f"Cancelled timer #{hit['id']} (\"{hit['message']}\")."


def due(clock: Callable[[], float] = time.time) -> Optional[dict]:
    """Oldest timer that has gone off and hasn't been announced yet."""
    now = clock()
    with _lock:
        d = _load()
        ts = d.get("timers", [])
        stale = [t for t in ts if not t.get("done") and now - t["due"] > TIMER_STALE_S]
        for t in stale:
            logger.warning("⏲️ timer #%s dropped unannounced (%.0f min late)", t["id"], (now - t["due"]) / 60)
        if stale:
            d["timers"] = [t for t in ts if t not in stale]
            _save(d)
        live = sorted((t for t in d.get("timers", []) if not t.get("done") and t["due"] <= now),
                      key=lambda x: x["due"])
        return dict(live[0]) if live else None


def claim(tid: int, clock: Callable[[], float] = time.time) -> str:
    """Mark a fired timer announced (removed) and return its cue line."""
    with _lock:
        d = _load()
        ts = d.get("timers", [])
        t = next((x for x in ts if x["id"] == tid), None)
        if t is None:
            return ""
        d["timers"] = [x for x in ts if x is not t]
        _save(d)
    ago = _human(max(0.0, clock() - t["set"]))
    lead = f"[{t['asker']}] " if t.get("asker") else ""
    return (f"{lead}(ROOM CUE #{t['id']}) This is NOT a new line from anyone: it's your own cue. "
            f"A timer set {ago} ago just went off: \"{t['message']}\". Announce it to the room now "
            "in one short line, in character. Don't [HOLD] - they asked for this reminder.")


def cue_note(text: str) -> str:
    import voice_mail
    mn = voice_mail.cue_note(text)
    if mn:
        return mn
    import bets
    bn = bets.cue_note(text)
    if bn:
        return bn
    if ROOM_CUE_RE.search(text or ""):
        return ("REMINDER: this turn is your own timer going off, not someone speaking. Say the "
                "reminder now in one short line, in character. Do not [HOLD].")
    return ""


# ------------------------------------------------------------------ weather
_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
        81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
        95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail"}

_UA = {"User-Agent": "ATLAS-voice-agent/0.3 (+https://github.com/Relic-Studios/atlas)"}


def _get_json(url: str, timeout: float = 6.0):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https hosts)
        return json.loads(r.read().decode("utf-8"))


def check_weather(place: str = "", units: str = "auto", default_place: str = "",
                  fetch: Callable[[str], dict] = _get_json) -> str:
    place = (place or "").strip() or (default_place or "").strip()
    if not place:
        return ("Which place? No default location is set (the owner can set one in the Weather "
                "plugin). Ask where they mean.")
    try:
        g = fetch("https://geocoding-api.open-meteo.com/v1/search?count=1&language=en&format=json&name="
                  + urllib.parse.quote(place[:80]))
        hits = (g or {}).get("results") or []
        if not hits:
            return f"Couldn't find a place called '{place}'. Ask them to say it another way."
        h = hits[0]
        where = ", ".join(x for x in (h.get("name"), h.get("admin1"), h.get("country")) if x)
        imperial = units == "imperial" or (units == "auto" and h.get("country_code") in ("US", "LR", "MM"))
        q = urllib.parse.urlencode({
            "latitude": h["latitude"], "longitude": h["longitude"], "timezone": "auto", "forecast_days": 3,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh"})
        w = fetch("https://api.open-meteo.com/v1/forecast?" + q)
    except Exception as e:  # noqa: BLE001
        return f"The weather service didn't answer ({type(e).__name__}). Say you couldn't get it right now."
    deg = "°F" if imperial else "°C"
    spd = "mph" if imperial else "km/h"
    c = (w or {}).get("current") or {}
    dly = (w or {}).get("daily") or {}
    now = (f"Now in {where}: {_WMO.get(c.get('weather_code'), 'unknown conditions')}, "
           f"{round(c.get('temperature_2m', 0))}{deg} (feels like {round(c.get('apparent_temperature', 0))}{deg}), "
           f"wind {round(c.get('wind_speed_10m', 0))} {spd}, humidity {c.get('relative_humidity_2m', '?')}%.")
    days = []
    for i, label in enumerate(("Today", "Tomorrow", "Day after")):
        try:
            days.append(f"{label}: {_WMO.get(dly['weather_code'][i], '?')}, "
                        f"{round(dly['temperature_2m_min'][i])}-{round(dly['temperature_2m_max'][i])}{deg}, "
                        f"{dly['precipitation_probability_max'][i]}% chance of rain")
        except (KeyError, IndexError, TypeError):
            break
    return now + (" " + "; ".join(days) + "." if days else "") + " (Open-Meteo, live.) Say it briefly."


# ------------------------------------------------------------------ intent prefetch
# Live sim 10-05 (Remote box + qwen3): asked to roll, flip, vote, set a timer or check the
# weather, the model mostly answered with an INVENTED result ("Heads it is", "17!",
# "Timer's set") without calling the tool. Clear requests are therefore detected here
# and the real tool runs first (llm prefetch), so the model only has to say the result.
_N = r"(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|forty-five|sixty|ninety|half an?)"
_WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
            "forty-five": 45, "sixty": 60, "ninety": 90, "half a": 0.5, "half an": 0.5}
_DUR_RE = re.compile(r"\b(?:in|for|after)\s+(" + _N + r")\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|h|m|s)\b", re.I)
_TIMER_RE = re.compile(r"\b(remind|reminder|timer|alarm|wake (?:us|me))\b", re.I)
_TIMER_MSG_RE = re.compile(r"\b(?:to|that|about)\s+(.+?)\s*(?:in|after)\s+" + _N + r"\s*\w+\s*[.!?]*$"
                           r"|\b(?:in|for|after)\s+" + _N + r"\s*\w+\s+(?:to|that|about)\s+(.+?)[.!?]*$", re.I)
_DICE_RE2 = re.compile(r"\broll\b(?:\s+(?:me|us|for\s+\w+))?(?:\s+(?:a|an|the|some|my))?\s+"
                       r"(\d*\s*d\s*\d+(?:\s*[+-]\s*\d+)?|dice|die)\b", re.I)
_COIN_RE = re.compile(r"\b(?:(?:flip|toss)\s+(?:a|the|us\s+a|me\s+a)?\s*coin|(?:with|by|do|use)\s+(?:a\s+)?coin\s+(?:flip|toss)|heads\s+or\s+tails)\b", re.I)
_POLL_RE = re.compile(r"\b(?:quick\s+)?(?:vote|poll)\b", re.I)
_VOTE_RE = re.compile(r"\b(?:put\s+me\s+down\s+for|i\s+vote(?:\s+for)?|my\s+vote\s+(?:is|goes\s+to)|"
                      r"i(?:'ll|\s+will)?\s+(?:pick|choose|go\s+with)|count\s+me\s+(?:in\s+)?for|i'm\s+(?:voting|going)\s+(?:for\s+)?)"
                      r"\s+(.+?)[.!?]*$", re.I)
_WEATHER_RE = re.compile(r"\b(weather|forecast|temperature)\b|\bis it (?:going to )?(?:rain|snow)", re.I)
_PLACE_RE = re.compile(r"\b(?:in|for|at)\s+([A-Z][\w'.-]*(?:[ ,]+[A-Z][\w'.-]*){0,3})")
_PAST_RE = re.compile(r"\b(?:was|were|yesterday|last (?:night|week)|had)\b", re.I)
_STOP = re.compile(r"\s+(?:right now|now|today|tonight|tomorrow|this week(?:end)?)$", re.I)


def _num(s: str) -> float:
    s = s.lower().strip()
    try:
        return float(s)
    except ValueError:
        return float(_WORDNUM.get(s, 0))


def _speech(text: str) -> str:
    return re.sub(r"^\s*\[S\d+\]\s*", "", text or "").strip()


def intent(text: str, poll_open: Optional[bool] = None):
    """(tool, args) for an unmistakable room-tool request in this line, else None."""
    t = _speech(text)
    if not t:
        return None
    m = _DICE_RE2.search(t)
    if m:
        spec = re.sub(r"\s+", "", m[1].lower())
        return ("roll_dice", {"dice": "d6" if spec in ("die", "dice") else spec})
    if _COIN_RE.search(t):
        return ("flip_coin", {})
    if _TIMER_RE.search(t):
        d = _DUR_RE.search(t)
        if d:
            unit = d[2].lower()[0]
            mins = _num(d[1]) * (1 / 60 if unit == "s" else 60 if unit == "h" else 1)
            mm = _TIMER_MSG_RE.search(t)
            msg = (mm[1] or mm[2]) if mm else ""
            msg = re.sub(r"^(?:us|me|everyone|you)\s+(?:to\s+)?", "", (msg or "").strip(" ,.!?"))
            if mins > 0:
                return ("set_timer", {"minutes": round(mins, 2), "message": msg or "time's up"})
    try:
        import call_summary as _cs
        rc = _cs.intent(t)
    except Exception:  # noqa: BLE001
        rc = None
    if rc:
        return rc
    try:
        import voice_mail as _vm
        vm = _vm.intent(t)
    except Exception:  # noqa: BLE001
        vm = None
    if vm:
        return vm
    try:
        import bets as _bets
        bt = _bets.intent(t)
    except Exception:  # noqa: BLE001
        bt = None
    if bt:
        return bt
    try:
        import floor_referee as _fr
        fr = _fr.intent(t)
    except Exception:  # noqa: BLE001
        fr = None
    if fr:
        return fr
    try:
        import game_info as _gi
        gi = _gi.intent(t)
    except Exception:  # noqa: BLE001
        gi = None
    if gi:
        return gi
    if _WEATHER_RE.search(t) and not _PAST_RE.search(t):
        pm = _PLACE_RE.search(t)
        place = _STOP.sub("", pm[1]).strip(" ,.?!") if pm else ""
        return ("check_weather", {"place": place})
    if _POLL_RE.search(t) and re.search(r"\bor\b", t, re.I):
        body = t.split(":", 1)[1] if ":" in t else re.split(r"\b(?:vote|poll)\b", t, flags=re.I)[-1]
        body = re.sub(r"^\s*(?:on|for|between|about)\s+", "", body.strip(" ,.?!"), flags=re.I)
        opts = _options(body)
        if 2 <= len(opts) <= 6 and all(len(o.split()) <= 5 for o in opts):
            q = t.split(":", 1)[0].strip() if ":" in t else ""
            q = re.sub(r"^[A-Za-z]+,\s*", "", q)
            q = re.sub(r"(?i)^(?:ok(?:ay)?\s+)?(?:let'?s\s+)?(?:do\s+a\s+)?(?:quick\s+)?(?:vote|poll)\s*(?:for|on|about)?\s*", "", q).strip()
            q = (q[:1].upper() + q[1:] + "?") if q else "Quick vote"
            return ("start_poll", {"question": q, "options": ", ".join(opts)})
    if poll_open is None:
        with _lock:
            poll_open = bool(_load().get("poll"))
    if poll_open:
        m = _VOTE_RE.search(t)
        if m:
            with _lock:
                p = _load().get("poll") or {}
            choice = _match_option(m[1], p.get("options", []))
            if choice:
                return ("cast_vote", {"choice": choice})
    return None


# ------------------------------------------------------------------ tool schemas
def _tool(name: str, desc: str, props: dict, required=()) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOLS_BY_PLUGIN: Dict[str, List[dict]] = {
    "dice_polls": [
        _tool("roll_dice", "Roll dice fairly (d20, 2d6, 3d8+2). Always use this instead of making up a number.",
              {"dice": {"type": "string", "description": "dice notation, default d20"}}),
        _tool("flip_coin", "Flip a fair coin (to settle something). Never invent the result.",
              {"times": {"type": "integer", "description": "1-10, default 1"}}),
        _tool("pick_one", "Pick one option fairly at random (e.g. which game, who goes first).",
              {"options": {"type": "string", "description": "the options, comma separated"}}, ["options"]),
        _tool("start_poll", "Open a quick room vote. Replaces any open poll.",
              {"question": {"type": "string"},
               "options": {"type": "string", "description": "2-10 options, comma separated"}}, ["options"]),
        _tool("cast_vote", "Record someone's pick in the open poll when they say it. The voter is the speaker.",
              {"choice": {"type": "string", "description": "the option they picked"}}, ["choice"]),
        _tool("poll_results", "Read the open poll's tally; close=true to end it and announce the winner.",
              {"close": {"type": "boolean"}}),
    ],
    "timers": [
        _tool("set_timer", "Set a timer/reminder for the room ('remind us in 10 minutes to start the raid'). "
              "It goes off on its own later and you'll be cued to announce it.",
              {"minutes": {"type": "number", "description": "how long, in minutes (0.5 = 30 seconds)"},
               "message": {"type": "string", "description": "what to remind them of"}}, ["minutes"]),
        _tool("list_timers", "List the timers that are running and how long is left.", {}),
        _tool("cancel_timer", "Cancel a running timer by id, or by words from its message.",
              {"id": {"type": "integer"}, "message": {"type": "string"}}),
    ],
    "weather": [
        _tool("check_weather", "Live weather now and the next 3 days for a place. Empty place = the "
              "owner's default location.",
              {"place": {"type": "string", "description": "city or town, e.g. 'Seattle' or 'Osaka, Japan'"}}),
    ],
    "call_summary": [
        _tool("recap_call", "Get the real transcript of the last N minutes of this call so you can "
              "recap it ('what did we decide?', 'what did I miss?', 'sum up the last 10 minutes'). "
              "Never recap from memory alone.",
              {"minutes": {"type": "number", "description": "how far back, default 10"},
               "focus": {"type": "string", "description": "optional topic to focus on"}}),
    ],
    "voice_mail": [
        _tool("leave_message", "Save a message for someone who isn't here (or not listening) to pass on "
              "the next time you hear their voice: 'tell Riley the raid moved to 9 when she joins'.",
              {"to": {"type": "string", "description": "the recipient's first name"},
               "message": {"type": "string", "description": "what to pass on, in the sender's words"}},
              ["to", "message"]),
        _tool("list_messages", "List messages waiting to be passed on.", {}),
        _tool("cancel_message", "Cancel a waiting message by id, recipient name or words from it.",
              {"id": {"type": "integer"}, "words": {"type": "string"}}),
    ],
    "bets": [
        _tool("record_bet", "Log a bet someone makes ('I bet you five bucks the Lakers win tonight'). "
              "The bettor is the speaker unless they say otherwise. It comes back up on its own when "
              "it can be settled. Only say it's logged after this returns.",
              {"claim": {"type": "string", "description": "what the bettor says will happen"},
               "who": {"type": "string", "description": "the bettor's name, if not the speaker"},
               "against": {"type": "string", "description": "who took the other side, if anyone"},
               "stakes": {"type": "string", "description": "what's on the line, e.g. 'five bucks'"},
               "settle_by": {"type": "string", "description": "when it can be settled: 'tonight', "
                             "'Sunday', 'end of the month', '2026-10-20'"}}, ["claim"]),
        _tool("list_bets", "List open bets (all=true includes settled ones) and the scoreboard.",
              {"all": {"type": "boolean"}}),
        _tool("settle_bet", "Settle a bet once the room says (or search shows) how it went. Give the "
              "winner's name (or the loser's) and ATLAS works out the rest; use outcome only for a push "
              "(tie) or void (called off). Never guess who won. Only say a bet is settled after this returns.",
              {"winner": {"type": "string", "description": "name of the person who won the bet"},
               "loser": {"type": "string", "description": "name of the person who lost, if easier"},
               "id": {"type": "integer"}, "words": {"type": "string", "description": "words from the bet if no id"},
               "outcome": {"type": "string", "enum": ["push", "void", "won", "lost"],
                           "description": "push/void; or won/lost from the bettor's side"},
               "note": {"type": "string", "description": "e.g. the final score"}}),
    ],
    "floor_referee": [
        _tool("floor_stats", "Who has talked how much in the last N minutes (humans only). Use when asked "
              "who's been talking most, who hasn't spoken, or 'am I talking too much'.",
              {"minutes": {"type": "number", "description": "window, default 15"}}),
        _tool("start_topic", "Timebox a topic ('give us 5 minutes on the budget'). Time's-up is called "
              "at the next gap on its own.",
              {"topic": {"type": "string"}, "minutes": {"type": "number"}}, ["topic", "minutes"]),
        _tool("debate_sides", "Get the real lines grouped by person so you can sum up each side fairly. "
              "Use when asked to summarise both sides / everyone's positions.",
              {"topic": {"type": "string", "description": "optional topic words to focus on"},
               "minutes": {"type": "number", "description": "window, default 15"}}),
    ],
    "game_info": [
        _tool("game_info", "Live Steam info for a game: price, whether it's on sale, how many people are "
              "playing right now, release date. Never guess these numbers.",
              {"game": {"type": "string", "description": "the game's name, e.g. 'Helldivers 2'"}}, ["game"]),
    ],
}
NAMES = {t["function"]["name"] for ts in TOOLS_BY_PLUGIN.values() for t in ts}
TOOLS = [t for ts in TOOLS_BY_PLUGIN.values() for t in ts]


def execute(name: str, args: dict, asker: str = "", voter_name: str = "",
            settings: Callable[[str], dict] = lambda pid: {},
            name_of: Optional[Callable] = None, agent_names=()) -> str:
    a = args if isinstance(args, dict) else {}
    if name == "roll_dice":
        return roll_dice(str(a.get("dice") or a.get("spec") or "d20"))
    if name == "flip_coin":
        return flip_coin(a.get("times") or 1)
    if name == "pick_one":
        return pick_one(a.get("options"))
    if name == "start_poll":
        return start_poll(str(a.get("question") or ""), a.get("options"))
    if name == "cast_vote":
        return cast_vote(str(a.get("choice") or ""), voter_name or asker)
    if name == "poll_results":
        return poll_results(bool(a.get("close")))
    if name == "set_timer":
        return set_timer(a.get("minutes"), str(a.get("message") or ""), asker)
    if name == "list_timers":
        return list_timers()
    if name == "cancel_timer":
        return cancel_timer(a.get("id"), str(a.get("message") or ""))
    if name == "check_weather":
        s = settings("weather") or {}
        return check_weather(str(a.get("place") or ""), s.get("units", "auto"), s.get("home", ""))
    if name == "recap_call":
        import call_summary
        s = settings("call_summary") or {}
        return call_summary.recap(a.get("minutes") or s.get("minutes") or call_summary.DEFAULT_MIN,
                                  str(a.get("focus") or ""), name_of=name_of)
    if name in ("leave_message", "list_messages", "cancel_message"):
        import voice_mail
        days = float((settings("voice_mail") or {}).get("days") or voice_mail.DEFAULT_DAYS)
        if name == "list_messages":
            return voice_mail.list_messages(days)
        if name == "cancel_message":
            return voice_mail.cancel(a.get("id"), str(a.get("words") or ""))
        return voice_mail.leave(str(a.get("to") or ""), str(a.get("message") or ""),
                                voter_name or "", agent_names=agent_names, days=days)
    if name in ("record_bet", "list_bets", "settle_bet"):
        import bets
        if name == "list_bets":
            out = bets.list_bets(bool(a.get("all")))
            return out + " " + bets.scoreboard()
        if name == "settle_bet":
            me = voter_name or asker or ""
            w = str(a.get("winner") or "").replace("@speaker", me)
            lo = str(a.get("loser") or "").replace("@speaker", me)
            return bets.settle_bet(a.get("id"), str(a.get("outcome") or ""), str(a.get("note") or ""),
                                   str(a.get("words") or ""), winner=w, loser=lo)
        return bets.record_bet(str(a.get("claim") or ""), str(a.get("who") or "") or voter_name or "",
                               str(a.get("against") or ""), str(a.get("stakes") or ""),
                               str(a.get("settle_by") or a.get("when") or ""))
    if name in ("floor_stats", "start_topic", "debate_sides"):
        import floor_referee as fr
        mins = a.get("minutes")
        if name == "floor_stats":
            return fr.floor_stats(mins or fr.DEFAULT_MIN, name_of=name_of)
        if name == "debate_sides":
            return fr.debate_sides(mins or fr.DEFAULT_MIN, str(a.get("topic") or ""), name_of=name_of)
        return fr.start_topic(str(a.get("topic") or ""), mins if mins is not None else 5, asker)
    if name == "game_info":
        import game_info
        s = settings("game_info") or {}
        return game_info.lookup(str(a.get("game") or a.get("name") or ""), s.get("cc", "us"))
    return f"unknown tool: {name}"
