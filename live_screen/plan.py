"""Planning one look and wording its result (docs/designs/live_screen.md, sections 3-4).

Live screen watches one target (the owner's screen or one program), so a plan only decides the kind of look:
`recent` (the newest frame plus three earlier ones) for "what just happened" lines, otherwise `now`; and whether
it counts as asked (the caller decides; a glance never does). One set of eyes (owner, 2026-11-03): by default that
is Model, which gets the frames itself (`seen`); with a separate watcher server configured, the watcher sees
them and the agent gets its report as text (`reported`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Plan:
    kind: str = "now"          # now | recent
    asked: bool = True
    reason: str = ""
    line: str = ""             # what was said (the watcher's question)

    @property
    def glance(self) -> bool:
        return not self.asked

    @property
    def call(self) -> bool:
        return self.reason.startswith("call:")


def plan_look(line: str = "", reason: str = "", asked: bool = True) -> Plan:
    import capability as cap
    reason = reason or ""
    glance = reason.startswith("glance:") or not asked
    kind = "recent" if (not glance and cap.just_happened(line)) else "now"
    return Plan(kind=kind, asked=not glance, reason=reason, line=line or "")


def question_of(plan: Plan) -> str:
    """What the watcher is asked: the line itself, without the speaker label."""
    import re
    q = re.sub(r"^\s*\[S\d+\]\s*", "", plan.line or "").strip()
    if q:
        return q
    if plan.call:
        return "Who is in the call and which channel is it?"
    return "What is on the screen right now?"


def reported(plan: Plan, report: str, speaker: str, watching: str, title: str, frames: list) -> dict:
    """The watcher looked: its report, as text the agent answers from (no images)."""
    import untrusted
    clean, _ = untrusted.sanitize(report or "", limit=900)
    title = _clean_title(title)
    what = f"{watching}" + (f", the window in front is \"{title}\"" if title and title not in watching else "")
    span = "over the last few seconds" if len(frames) > 1 else "just now"
    body = untrusted.wrap(clean, source="screen report", origin=f"what your eyes saw on {watching}, {span}")
    if plan.glance:
        text = ("You glanced at the screen on your own; nobody asked you to. Your eyes report what's there "
                f"({what}):\n{body}\nUse it only if it fits what's being said, the way someone who looked over "
                "would: a quick, natural reaction, not a description of the screen, and don't announce that you "
                "looked. Don't add anything the report doesn't say.")
    else:
        text = (f"You looked at {what}. Your eyes report:\n{body}\nAnswer from this report only; don't add "
                "details it doesn't mention, and if it says it can't tell, say you can't see it clearly." +
                (f" The person who asked is {speaker}." if speaker else ""))
        if plan.call:
            text += _call_hint()
    return {"text": text, "images": [], "live": True, "seen": True, "glance": plan.glance, "frames": frames,
            "title": title}


_WHY = {
    "images": "looking is switched off right now",
    "offline": "your eyes are offline for a moment",
    "eyes_offline": "your eyes didn't answer",
    "starting": "the live view is still starting",
    "not_responding": "the live view didn't answer in time",
    "app_not_open": "the program it watches isn't open",
    "minimized": "it's minimized",
    "black": "it's showing nothing but black",
    "hdr_unsupported": "the screen is in HDR, which the live view can't read yet",
    "not_visible": "the live view can't get a picture of it",
    "stale": "the live view has no fresh picture",
    "off": "the live view is off",
}


def cant_see(plan: Plan, why: str) -> dict:
    words = _WHY.get(why, "the live view isn't working right now")
    if plan.glance:
        text = (f"You tried to glance at the screen but can't see it right now ({words}). Carry on with the "
                "conversation from what you heard; don't say you looked.")
    else:
        text = (f"You can't see the screen right now ({words}). Say plainly that you can't see it at the moment; "
                "don't describe or guess what's on it.")
    return {"text": text, "images": [], "live": True, "glance": plan.glance, "frames": [], "why": why}


def _clean_title(title: str) -> str:
    try:
        import untrusted
        title, _ = untrusted.sanitize(title or "", limit=80)
    except Exception:  # noqa: BLE001
        title = (title or "")[:80]
    return title.replace('"', "'")


def _ages_line(frames: list) -> str:
    parts = []
    for i, f in enumerate(frames, 1):
        age = float(f.get("age_s") or 0.0)
        parts.append(f"Frame {i}, just now" if age < 1.0 else f"Frame {i}, about {round(age)} s before you looked")
    return "; ".join(parts)


def seen(plan: Plan, res: dict, speaker: str, watching: str) -> dict:
    """Model is the eyes (the default): the frames go with the tool result, and it looks at them itself."""
    title = _clean_title(res.get("title") or "")
    what = f"{watching}" + (f", the window in front is \"{title}\"" if title and title not in watching else "")
    who = (f" The person who just spoke to you is {speaker}." if speaker else "") + (
        " If camera tiles are on screen, the one outlined in green is whoever is talking; if no tile is outlined "
        "or there are no tiles, you can't tell who they are on screen.")
    frames = res.get("frames") or []
    if plan.glance:
        text = ("You glanced at the screen on your own; nobody asked you to. Use what you see only if it fits "
                "what's being said, the way someone who looked over would: a quick, natural reaction, not a "
                "description of the screen, and don't announce that you looked. Don't invent anything that isn't "
                "visible: if the person or thing they mentioned isn't in it, don't pretend you saw it." + who +
                f" This is the live view of {what}, right now.")
    else:
        text = (f"These frames are from the live view of {what}, just now. Describe or react to what you actually "
                "see in them; don't invent anything that isn't visible, and don't guess at text too small to read. "
                "If what they asked about isn't in them, say plainly that you can't see it." + who)
        if len(frames) > 1:
            text += (" The frames are in order, oldest first: " + _ages_line(frames) + ". Use them to tell what "
                     "happened, and say if they don't show it clearly.")
        text += (" Describe only what these frames show, not a game or topic from your notes or from earlier in "
                 "the call.")
        if plan.call:
            text += _call_hint()
    return {"text": text, "images": res["images"], "live": True, "seen": True, "glance": plan.glance,
            "frames": frames, "title": title}


def _call_hint() -> str:
    try:
        import screen
        c = screen.discord_call(fresh=True)
    except Exception:  # noqa: BLE001
        c = None
    where = (f" Right now the Discord window's title shows the server \"{c['server']}\", channel \"{c['channel']}\" "
             "(the channel open in Discord; the 'Voice Connected' panel shows the call itself)." if c else "")
    return (" They asked about the call itself. If the Discord window is in these frames, find the answer there: "
            "the server name at the top left, the voice channel with the speaker icon and the people under it, "
            "the 'Voice Connected' panel at the bottom left, and the people's tiles. If Discord isn't in them, "
            "say so and tell them what you do know." + where)
