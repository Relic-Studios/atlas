"""Current date/time for every agent (owner 10-05: "we need a tool that lets us check the
actual date"). Two layers: a one-line NOW note in every turn's context (works even on
small models that never call tools), and a check_date_time tool for other time zones."""
import datetime as _dt

try:
    import zoneinfo as _zi
except Exception:  # noqa: BLE001
    _zi = None


def _fmt(d: _dt.datetime) -> str:
    hour = d.strftime("%I").lstrip("0") or "12"
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B %Y')}, {hour}:{d.strftime('%M %p')}"


def now_note(now=None) -> str:
    d = (now or _dt.datetime.now()).astimezone()
    return (f"Right now it is {_fmt(d)} (local time, {d.tzname()}). Trust this over anything "
            "in your training or old search results when talking about dates.")


def check(timezone: str = "", now=None) -> str:
    base = (now or _dt.datetime.now()).astimezone()
    tz = (timezone or "").strip()
    if tz and _zi is not None:
        try:
            d = base.astimezone(_zi.ZoneInfo(tz.replace(" ", "_")))
            return f"In {tz} it is {_fmt(d)} ({d.tzname()}). Local time is {_fmt(base)}."
        except Exception:  # noqa: BLE001
            return (f"Unknown time zone '{tz}' (use a name like America/New_York or "
                    f"Asia/Tokyo). Local time is {_fmt(base)} ({base.tzname()}).")
    return f"It is {_fmt(base)} local time ({base.tzname()}). ISO date: {base.date().isoformat()}."


TOOL = {"type": "function", "function": {
    "name": "check_date_time",
    "description": ("Check the real current date and time (today's date, weekday, clock time), "
                    "optionally in another time zone. Use it whenever the date or time matters."),
    "parameters": {"type": "object", "properties": {
        "timezone": {"type": "string",
                     "description": "optional IANA zone, e.g. Asia/Tokyo; empty = local"}},
        "required": []}}}
