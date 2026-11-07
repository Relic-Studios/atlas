"""Current date/time for every agent (owner 10-05: "we need a tool that lets us check the
actual date"). Two layers: a one-line NOW note in every turn's context (works even on
small models that never call tools), and a check_date_time tool for other time zones."""
import datetime as _dt
import re as _re

try:
    import zoneinfo as _zi
except Exception:  # noqa: BLE001
    _zi = None


def _fmt(d: _dt.datetime) -> str:
    hour = d.strftime("%I").lstrip("0") or "12"
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B %Y')}, {hour}:{d.strftime('%M %p')}"


def part_of_day(d: _dt.datetime) -> str:
    """Live 11-03: at 8:34 PM a night-owl persona opened with "You're up late too, huh?"; the clock time alone
    didn't stop it, so the note says which part of the day it is."""
    h = d.hour
    if h < 5:
        return "the middle of the night"
    if h < 12:
        return "morning"
    if h < 17:
        return "afternoon"
    if h < 22:
        return "evening"
    return "late night"


def now_note(now=None, zone: bool = True) -> str:
    """zone=False (pack private_setup) leaves the time-zone name out: live 11-02 a guest got
    "Pacific Standard Time ... somewhere on the Pacific coast" out of it."""
    d = (now or _dt.datetime.now()).astimezone()
    where = f"local time, {d.tzname()}" if zone else "the owner's local time"
    return (f"Right now it is {_fmt(d)}, {part_of_day(d)} ({where}). Trust this over anything "
            "in your training or old search results when talking about dates or the time of day.")


def check(timezone: str = "", now=None, zone: bool = True) -> str:
    base = (now or _dt.datetime.now()).astimezone()
    tz = (timezone or "").strip()
    here = f" ({base.tzname()})" if zone else ""
    if tz and _zi is not None:
        try:
            d = base.astimezone(_zi.ZoneInfo(tz.replace(" ", "_")))
            return f"In {tz} it is {_fmt(d)} ({d.tzname()}). Local time is {_fmt(base)}."
        except Exception:  # noqa: BLE001
            return (f"Unknown time zone '{tz}' (use a name like America/New_York or "
                    f"Asia/Tokyo). Local time is {_fmt(base)}{here}.")
    return f"It is {_fmt(base)} local time{here}. ISO date: {base.date().isoformat()}."


TOOL = {"type": "function", "function": {
    "name": "check_date_time",
    "description": ("Check the real current date and time (today's date, weekday, clock time), "
                    "optionally in another time zone. Use it whenever the date or time matters."),
    "parameters": {"type": "object", "properties": {
        "timezone": {"type": "string",
                     "description": "optional IANA zone, e.g. Asia/Tokyo; empty = local"}},
        "required": []}}}


def intent(text: str):
    """An explicit named-clock command; ordinary time questions stay with the LLM.

    The pipeline separately requires direct address and an enabled plugin. A city
    needs model interpretation, so only local time or an explicit IANA zone can be
    executed here without guessing an argument.
    """
    text = text or ""
    match = _re.search(r"\b(?:use|call|run)\s+(?:(?:your|the)\s+)?"
                       r"(?:(?:date\s+(?:and\s+)?time|date|time|clock)\s+tool|check_date_time)\b", text, _re.I)
    if not match:
        return None
    if _re.search(r"\b(?:not|never|don['’]?t|shouldn['’]?t)\s*$", text[:match.start()], _re.I):
        return None
    # Quoted/reported commands are not owner commands.
    if any(left in text[:match.start()] and right in text[match.end():]
           for left, right in (('"', '"'), ('“', '”'))):
        return None
    if _re.search(r"\b(?:said|says|told|quote|quoted|repeat|recite|example)\b", text[:match.start()], _re.I):
        return None
    zone = _re.search(r"\b[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?\b", text)
    if not zone and _re.search(r"\bin\s+[A-Za-z]", text[match.end():], _re.I):
        return None
    return "check_date_time", {"timezone": zone[0] if zone else ""}
