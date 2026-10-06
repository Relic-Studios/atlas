"""Plugins: every agent ability is a module you can switch on/off and configure.

A plugin is a manifest (what it is, which agent tools it provides, which settings it
takes) plus optional hooks (apply settings, test the connection). The Plugins page
(static/plugins.html) renders a settings window for each plugin straight from its
`settings` schema, so a new plugin needs no UI code:

  field types: text | secret | select | toggle | number | url
  secret values are write-only: the API reports whether one is set (and its last 4
  characters), never the value itself.

State lives in user/plugins.json (enabled flags + non-secret settings); secrets live
in user/<plugin>.<field>.key-style files that the code that needs them already reads
(e.g. user/exa.key for websearch). Both are in user/, which git and the public
export ignore.

Marketplace: plugin_market/catalog.json lists plugins that can be installed later.
Nothing in it runs code yet - an entry is either built in (install = enable) or
"coming soon". Third-party code plugins need signing + review first (SAFETY.md).
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
USER = ROOT / "user"
STATE_PATH = USER / "plugins.json"
MARKET_PATH = ROOT / "plugin_market" / "catalog.json"

_lock = threading.RLock()
_version = 0          # bumps on every change; the pipeline re-filters tools when it moves
_listeners: List[Callable[[], None]] = []


# ------------------------------------------------------------------ hooks
def _secret_path(name: str) -> Path:
    return USER / f"{name}.key"


def _apply_search(settings: dict) -> None:
    try:
        import websearch
        websearch.PREFERRED = settings.get("provider") or "auto"
        websearch.MAX_RESULTS = int(settings.get("max_results") or 4)
    except Exception as e:  # noqa: BLE001
        logger.debug("search plugin apply: %s", e)


def _test_search(settings: dict) -> dict:
    import time
    import websearch
    t = time.time()
    r = websearch.search("weather today") or {}
    ms = round((time.time() - t) * 1000)
    ok = bool(r.get("text")) and "No results" not in (r.get("text") or "")
    via = r.get("backend") or ", ".join(websearch.providers())
    return {"ok": ok, "message": (f"Search works ({via}, {ms} ms)." if ok
                                  else f"No results came back ({ms} ms). Check the key or your connection.")}


def _test_exa_key(value: str) -> dict:
    import websearch
    try:
        res = websearch._search_exa(value, "test", 1)
        return {"ok": bool(res), "message": "Exa key works." if res else "Exa answered with no results."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Exa rejected the key ({getattr(e, 'code', '') or type(e).__name__})."}


def _test_brave_key(value: str) -> dict:
    import websearch
    try:
        res = websearch._search_brave(value, "test", 1)
        return {"ok": bool(res), "message": "Brave key works." if res else "Brave answered with no results."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Brave rejected the key ({getattr(e, 'code', '') or type(e).__name__})."}


def _apply_eyes(settings: dict, enabled: bool) -> None:
    """Plugin off forces the Eyes toggle off. Plugin on only makes looking *available*:
    the live Eyes toggle stays whatever the owner last set (never silently turned on)."""
    try:
        import screen
        if not enabled:
            screen.set_enabled(False)
        with screen._lock:
            screen._state["monitor"] = int(settings.get("monitor") or 1)
    except Exception as e:  # noqa: BLE001
        logger.debug("eyes plugin apply: %s", e)


def _test_eyes(settings: dict) -> dict:
    try:
        import screen
        img = screen._grab()
        return {"ok": True, "message": f"Captured monitor {settings.get('monitor') or 1} ({img.size[0]}x{img.size[1]})."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Couldn't capture the screen: {e}"}


def _monitor_options() -> List[dict]:
    try:
        import mss
        with mss.mss() as s:
            mons = s.monitors[1:]
        return [{"value": str(i + 1), "label": f"Monitor {i + 1} ({m['width']}x{m['height']})"}
                for i, m in enumerate(mons)] or [{"value": "1", "label": "Primary monitor"}]
    except Exception:  # noqa: BLE001
        return [{"value": "1", "label": "Primary monitor"}]


def _test_clock(settings: dict) -> dict:
    import clock
    tz = (settings.get("timezone") or "").strip()
    out = clock.check(tz)
    bad = "unknown" in out.lower() or "not a time zone" in out.lower() or "isn't a time zone" in out.lower()
    return {"ok": not bad, "message": out}


# ------------------------------------------------------------------ catalog
# Shared by the Languages plugin selects ("listen" adds Auto; "primary" is a real language).
_LANG_OPTIONS = [{"value": c, "label": n} for c, n in (("en", "English"), ("es", "Spanish"), ("fr", "French"),
                ("de", "German"), ("it", "Italian"), ("pt", "Portuguese"), ("ru", "Russian"),
                ("zh", "Chinese"), ("ja", "Japanese"), ("ko", "Korean"), ("nl", "Dutch"),
                ("pl", "Polish"), ("tr", "Turkish"), ("ar", "Arabic"), ("hi", "Hindi"))]

def _test_dice(settings: dict) -> dict:
    import room_tools
    return {"ok": True, "message": room_tools.roll_dice("d20").split(" Announce")[0]}


def _test_weather(settings: dict) -> dict:
    import room_tools
    out = room_tools.check_weather(settings.get("home") or "London", settings.get("units", "auto"))
    ok = out.startswith("Now in ")
    return {"ok": ok, "message": out.split(" Today:")[0] if ok else out}


def _test_game(settings: dict) -> dict:
    import game_info
    out = game_info.lookup("Counter-Strike 2", settings.get("cc", "us"))
    ok = "(Steam)" in out
    return {"ok": ok, "message": out.split("\n")[0] if ok else out}


BUILTIN: List[dict] = [
    {
        "id": "web_search", "name": "Web search", "icon": "search", "category": "Knowledge",
        "summary": "Look things up on the web and read pages from the results.",
        "detail": "Tries Exa, then Brave, then a local SearXNG, then a free fallback. "
                  "Search results are treated as untrusted text, never as instructions.",
        "tools": ["web_search", "read_page"], "default": True,
        "settings": [
            {"key": "provider", "label": "Provider", "type": "select", "default": "auto",
             "options": [{"value": "auto", "label": "Best available (recommended)"},
                         {"value": "exa", "label": "Exa only"}, {"value": "brave", "label": "Brave only"},
                         {"value": "free", "label": "Free search only (no key)"}],
             "help": "Auto uses the first provider that has a key and answers."},
            {"key": "exa", "label": "Exa API key", "type": "secret", "file": "exa",
             "help": "Fastest and most accurate. Get one at dashboard.exa.ai.", "link": "https://dashboard.exa.ai/api-keys",
             "test": "exa"},
            {"key": "brave", "label": "Brave Search API key", "type": "secret", "file": "brave",
             "help": "Backup provider with its own index.", "link": "https://brave.com/search/api/",
             "test": "brave"},
            {"key": "max_results", "label": "Results per search", "type": "number", "default": 4,
             "min": 1, "max": 8},
        ],
        "apply": lambda s, on: _apply_search(s), "test": _test_search,
    },
    {
        "id": "eyes", "name": "Eyes (screen)", "icon": "eye", "category": "Senses",
        "summary": "Glance at your screen when someone says \"look!\" or shows something.",
        "detail": "Captures one screenshot per look, only while this is on. The latest capture "
                  "is kept at private/last_screen.jpg so you can see what it saw.",
        "tools": ["look_at_screen"], "default": True,
        "settings": [
            {"key": "monitor", "label": "Which screen", "type": "select", "default": "1",
             "options": _monitor_options},
        ],
        "apply": _apply_eyes, "test": _test_eyes,
    },
    {
        "id": "clock", "name": "Date & time", "icon": "clock", "category": "Knowledge",
        "summary": "Know today's date and the time anywhere.",
        "detail": "The current local time is always in the agent's context; the tool adds other time zones.",
        "tools": ["check_date_time"], "default": True,
        "settings": [
            {"key": "timezone", "label": "Test a time zone", "type": "text", "default": "",
             "placeholder": "e.g. Asia/Tokyo (blank = local)"},
        ],
        "test": _test_clock,
    },
    {
        "id": "languages", "name": "Languages", "icon": "globe", "category": "Senses",
        "summary": "Hear people in their own language and answer in it.",
        "detail": "Whisper detects each speaker's language and writes down what they actually said, "
                  "not an English translation. The agent replies in that language, and its voice "
                  "switches to match (English, Spanish, French, German, Italian, Portuguese, Russian, "
                  "Chinese, Japanese, Korean). Off = everything is treated as your main language.",
        "tools": [], "default": True,
        "settings": [
            {"key": "listen", "label": "Listen for", "type": "select", "default": "auto",
             "options": [{"value": "auto", "label": "Detect automatically (recommended)"}] + _LANG_OPTIONS,
             "help": "Pick one language only if detection keeps guessing wrong in your room."},
            {"key": "primary", "label": "Main language", "type": "select", "default": "en",
             "options": list(_LANG_OPTIONS),
             "help": "Used for short or unclear lines and when this plugin is off."},
        ],
    },
    {
        "id": "dice_polls", "name": "Dice, coins & polls", "icon": "dice", "category": "Games",
        "summary": "Fair dice rolls, coin flips, random picks and quick room votes.",
        "detail": "'Roll a d20', 'settle it with a coin flip', 'let's vote: pizza or tacos'. "
                  "Results come from your PC's secure random source, never from the model. Nothing leaves this PC.",
        "tools": ["roll_dice", "flip_coin", "pick_one", "start_poll", "cast_vote", "poll_results"],
        "default": True, "settings": [], "test": _test_dice,
    },
    {
        "id": "timers", "name": "Timers & reminders", "icon": "timer", "category": "Productivity",
        "summary": "'Remind us in 10 minutes to start the raid.' The agent speaks up when it goes off.",
        "detail": "Timers are shared by all agents and survive a restart. A reminder waits for a gap "
                  "in the talk (at most a few seconds) before it's announced. Nothing leaves this PC.",
        "tools": ["set_timer", "list_timers", "cancel_timer"], "default": True, "settings": [],
    },
    {
        "id": "weather", "name": "Weather", "icon": "cloud", "category": "Knowledge",
        "summary": "Live weather and a 3-day forecast for any place.",
        "detail": "Uses Open-Meteo: free, no account, no key. Only the place name is sent to open-meteo.com.",
        "tools": ["check_weather"], "default": True,
        "settings": [
            {"key": "home", "label": "Default location", "type": "text", "default": "",
             "placeholder": "e.g. Seattle", "help": "Used when someone just asks 'what's the weather?'."},
            {"key": "units", "label": "Units", "type": "select", "default": "auto",
             "options": [{"value": "auto", "label": "Automatic (by country)"},
                         {"value": "metric", "label": "Metric (°C, km/h)"},
                         {"value": "imperial", "label": "Imperial (°F, mph)"}]},
        ],
        "test": _test_weather,
    },
    {
        "id": "call_summary", "name": "Call summary", "icon": "note", "category": "Productivity",
        "summary": "'What did we decide?' A recap of the last few minutes, from what was really said.",
        "detail": "Keeps this session's transcript in memory only (never saved by this plugin, gone "
                  "when ATLAS closes) and gives the agent the real lines to summarise. Nothing leaves this PC "
                  "unless you use a cloud model.",
        "tools": ["recap_call"], "default": True,
        "settings": [
            {"key": "minutes", "label": "Default recap window (minutes)", "type": "number", "default": 10,
             "min": 1, "max": 240, "help": "Used when someone just says 'recap'."},
        ],
    },
    {
        "id": "voice_mail", "name": "Voice mail", "icon": "mail", "category": "Social",
        "summary": "'Tell Riley the raid moved to 9 when she joins.' Messages addressed to a voice, not an account.",
        "detail": "The agent passes a message on the next time it recognises that person's voice, even days "
                  "later. Messages are kept on this PC only and expire. Phone numbers, emails, addresses "
                  "and keys are refused. Recognition is only as good as the voice match, so the agent says "
                  "the name out loud when it delivers.",
        "tools": ["leave_message", "list_messages", "cancel_message"], "default": True,
        "settings": [
            {"key": "days", "label": "Keep undelivered messages (days)", "type": "number", "default": 7,
             "min": 1, "max": 30},
        ],
    },
    {
        "id": "bets", "name": "Bet tracker", "icon": "trophy", "category": "Games",
        "summary": "'I bet you five bucks the Lakers win tonight.' Logs who bet what and calls it when it's due.",
        "detail": "Bets are kept on this PC only. When one can be settled, the agent brings it up at a gap "
                  "in the talk and asks how it went (or checks a public result with search). It never "
                  "decides a winner on its own. Keeps a running scoreboard.",
        "tools": ["record_bet", "list_bets", "settle_bet"], "default": True,
        "settings": [],
    },
    {
        "id": "floor_referee", "name": "Floor referee", "icon": "scale", "category": "Room",
        "summary": "For debates and meetings: talk-time balance, timeboxed topics, fair two-sided summaries.",
        "detail": "Uses only this session's in-memory transcript. 'Who's been talking most?', 'give us 5 "
                  "minutes on the budget', 'sum up both sides'. With the quiet-person hint on, the agent may "
                  "invite someone who's gone quiet into the talk, but only when it's already answering. "
                  "Off by default: it changes how the agent steers a conversation.",
        "tools": ["floor_stats", "start_topic", "debate_sides"], "default": False,
        "settings": [
            {"key": "invite_quiet", "label": "Hint the agent when someone goes quiet", "type": "select",
             "default": "on", "options": [{"value": "on", "label": "On"}, {"value": "off", "label": "Off"}]},
        ],
    },
    {
        "id": "teach", "name": "Teach-me mode", "icon": "book", "category": "Memory",
        "summary": "'Let me teach you how our league scoring works: ...' The agent keeps it and uses it next time.",
        "detail": "Lessons are per agent and stay on this PC (in that agent's memory folder). A lesson only "
                  "comes back when the talk is about its topic. Private details are refused. Wiping an "
                  "agent's memory clears its lessons too.",
        "tools": ["learn_lesson", "recall_lessons", "forget_lesson"], "default": True,
        "settings": [],
    },
    {
        "id": "lore", "name": "Lore keeper", "icon": "scroll", "category": "Memory",
        "summary": "Keeps your group's shared canon - running jokes, nicknames, campaign events - and calls back to it later.",
        "detail": "Say 'add that to the lore' or 'that's canon: ...'. By default each new entry waits here for "
                  "your approval before the agent ever uses it. Approved lore comes back only when the talk "
                  "touches it, at most one callback per turn and not the same one twice in 20 minutes. Lore is "
                  "per agent and stays on this PC; private details are refused; wiping an agent's memory clears it.",
        "tools": ["add_lore", "recall_lore", "forget_lore"], "default": True, "feed": True,
        "settings": [
            {"key": "approval", "label": "New lore", "type": "select", "default": "owner",
             "options": [{"value": "owner", "label": "Waits for my approval"},
                         {"value": "auto", "label": "Used right away"}]},
        ],
    },
    {
        "id": "soul_reflection", "name": "Soul reflection", "icon": "spark", "category": "Memory",
        "summary": "After a call, the agent proposes small changes to how it talks. You approve or reject each one.",
        "detail": "Press 'Reflect on the last call' after a call. The agent reads its own part of the "
                  "transcript and suggests up to three first-person notes ('I'd like to ask more questions "
                  "instead of guessing'). Nothing changes until you approve a note; approved notes are added "
                  "on top of the persona, which is never edited. Notes that deny being an AI, claim a body, "
                  "loosen safety or mention private details are thrown away before you see them. Runs only "
                  "when no call is live. Wiping an agent's memory removes its notes from that window.",
        "tools": [], "default": False, "feed": True,
        "settings": [],
    },
    {
        "id": "highlights", "name": "Highlight reel", "icon": "film", "category": "Creative",
        "summary": "Finds the best moments of a finished call and turns them into captioned vertical clips.",
        "detail": "Press 'Find highlights in the last call'. Moments are ranked from the call transcript: "
                  "laughter and big reactions right after the agent spoke, short punchy replies, being called "
                  "by name. Moments with insults, slurs or private details are never offered. The list is "
                  "private to you; a clip is only made after you confirm below that everyone in your calls "
                  "agreed to appear. Clips are captions on a plain card, saved next to the call recording. "
                  "Runs only when no call is live.",
        "tools": [], "default": False, "feed": True,
        "settings": [
            {"key": "consent", "label": "People in my calls", "type": "select", "default": "not_confirmed",
             "options": [{"value": "not_confirmed", "label": "Not confirmed (list moments only, no clips)"},
                         {"value": "agreed", "label": "Everyone agreed to appear in clips"}]},
            {"key": "count", "label": "Moments to find per call", "type": "number", "default": 6,
             "min": 1, "max": 12},
        ],
    },
    {
        "id": "fact_check", "name": "Quiet fact-check", "icon": "check", "category": "Knowledge",
        "summary": "Checks public claims in the background. The agent only speaks up when asked 'was that true?'",
        "detail": "When someone states a checkable public fact (a number, a date, 'the tallest...'), it's "
                  "looked up in the background with your web search. Results show here, not out loud. "
                  "Personal statements, plans and opinions are ignored; private details are never searched. "
                  "Kept in memory for this session only. Off by default: it uses search credits.",
        "tools": ["check_claim"], "default": False, "feed": True,
        "settings": [
            {"key": "background", "label": "Check claims in the background", "type": "select", "default": "on",
             "options": [{"value": "on", "label": "On (results ready before anyone asks)"},
                         {"value": "off", "label": "Off (only check when asked)"}]},
            {"key": "daily", "label": "Max checks per day", "type": "number", "default": 30, "min": 1, "max": 500},
            {"key": "gap", "label": "Min seconds between background checks", "type": "number",
             "default": 60, "min": 10, "max": 3600},
        ],
    },
    {
        "id": "game_info", "name": "Game info", "icon": "game", "category": "Games",
        "summary": "'Is it on sale?' 'How many people are playing?' Live Steam prices and player counts.",
        "detail": "Uses Steam's public store and player-count endpoints: free, no account, no key. "
                  "Only the game's name is sent to Steam. Player counts are Steam only.",
        "tools": ["game_info"], "default": True,
        "settings": [
            {"key": "cc", "label": "Store region (prices)", "type": "select", "default": "us",
             "options": [{"value": c, "label": l} for c, l in (
                 ("us", "United States ($)"), ("ca", "Canada (CA$)"), ("gb", "United Kingdom (£)"),
                 ("de", "Europe (€)"), ("au", "Australia (A$)"), ("jp", "Japan (¥)"))]},
        ],
        "test": _test_game,
    },
    {
        "id": "notes", "name": "Notes & follow-ups", "icon": "note", "category": "Memory",
        "summary": "A private task board: notes to self, promises, finished searches.",
        "detail": "Notes stay on this PC with the agent's other state.",
        "tools": ["note_to_self", "clear_note"], "default": True, "settings": [],
    },
    {
        "id": "step_back", "name": "Step back", "icon": "pause", "category": "Social",
        "summary": "Lets the agent choose to go quiet for a few minutes when the room is busy.",
        "detail": "Its name still wakes it.", "tools": ["step_back"], "default": True, "settings": [],
    },
    {
        "id": "self_check", "name": "Self-check", "icon": "code", "category": "Reflection",
        "summary": "Log predictions and check them later; read (never change) its own source code.",
        "detail": "Code access is read-only, limited to ATLAS's own code files, and skips keys, "
                  "memories and anything private.",
        "tools": None, "default": True, "settings": [],   # tools filled from self_tools.NAMES
    },
]


def _builtin_tools(p: dict) -> List[str]:
    if p["id"] == "self_check":
        try:
            import self_tools
            return sorted(self_tools.NAMES)
        except Exception:  # noqa: BLE001
            return []
    return list(p.get("tools") or [])


def _by_id() -> Dict[str, dict]:
    return {p["id"]: p for p in BUILTIN}


# ------------------------------------------------------------------ state
def _load() -> dict:
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    USER.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def _entry(state: dict, pid: str) -> dict:
    return state.setdefault(pid, {})


def is_enabled(pid: str, state: Optional[dict] = None) -> bool:
    p = _by_id().get(pid)
    if p is None:
        return False
    e = (state if state is not None else _load()).get(pid, {})
    return bool(e.get("enabled", p.get("default", True)))


def settings_of(pid: str, state: Optional[dict] = None) -> dict:
    p = _by_id().get(pid) or {}
    e = (state if state is not None else _load()).get(pid, {})
    out = {}
    for f in p.get("settings", []):
        if f["type"] == "secret":
            continue
        out[f["key"]] = e.get("settings", {}).get(f["key"], f.get("default"))
    return out


def _secret_info(f: dict) -> dict:
    """Is a key set (from user/, private/ or env)? Never return the value."""
    try:
        import websearch
        v = websearch._key(f["file"])
    except Exception:  # noqa: BLE001
        v = ""
    src = ""
    if v:
        import os
        def _has(d):
            pth = ROOT / d / f"{f['file']}.key"
            try:
                return bool(pth.read_text(encoding="utf-8-sig").strip())
            except OSError:
                return False
        if os.environ.get(f"ATLAS_{f['file'].upper()}_KEY", "").strip():
            src = "environment"
        elif _has("user"):
            src = "saved here"
        else:
            src = "private folder"
    return {"set": bool(v), "last4": v[-4:] if len(v) >= 8 else "", "source": src}


def tool_names_disabled() -> set:
    st = _load()
    out = set()
    for p in BUILTIN:
        if not is_enabled(p["id"], st):
            out.update(_builtin_tools(p))
    return out


def filter_tools(tools: List[dict]) -> List[dict]:
    off = tool_names_disabled()
    return [t for t in (tools or []) if t.get("function", {}).get("name") not in off]


def version() -> int:
    return _version


def on_change(fn: Callable[[], None]) -> None:
    _listeners.append(fn)


def _changed() -> None:
    global _version
    _version += 1
    for fn in list(_listeners):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            logger.warning("plugin listener failed: %s", e)


def apply_all() -> None:
    """Push saved settings into the modules that use them (startup + after edits)."""
    st = _load()
    for p in BUILTIN:
        fn = p.get("apply")
        if fn:
            try:
                fn(settings_of(p["id"], st), is_enabled(p["id"], st))
            except Exception as e:  # noqa: BLE001
                logger.warning("plugin %s apply failed: %s", p["id"], e)


# ------------------------------------------------------------------ API helpers
def _field_view(f: dict) -> dict:
    v = {k: f[k] for k in ("key", "label", "type", "help", "link", "placeholder", "min", "max", "default")
         if k in f}
    if "options" in f:
        v["options"] = f["options"]() if callable(f["options"]) else f["options"]
    if f["type"] == "secret":
        v["secret"] = _secret_info(f)
        v["testable"] = bool(f.get("test"))
    return v


def listing() -> List[dict]:
    st = _load()
    out = []
    for p in BUILTIN:
        out.append({
            "id": p["id"], "name": p["name"], "icon": p["icon"], "category": p["category"],
            "summary": p["summary"], "detail": p.get("detail", ""),
            "tools": _builtin_tools(p), "enabled": is_enabled(p["id"], st),
            "builtin": True, "testable": bool(p.get("test")), "feed": bool(p.get("feed")),
            "settings": [_field_view(f) for f in p.get("settings", [])],
            "values": settings_of(p["id"], st),
        })
    return out


def set_enabled(pid: str, on: bool) -> dict:
    if pid not in _by_id():
        raise KeyError(pid)
    with _lock:
        st = _load()
        _entry(st, pid)["enabled"] = bool(on)
        _save(st)
    logger.info("🧩 plugin %s %s", pid, "ON" if on else "OFF")
    apply_all()
    _changed()
    return next(x for x in listing() if x["id"] == pid)


def save_settings(pid: str, values: dict) -> dict:
    """Validate against the schema; secrets go to their key files, the rest to plugins.json."""
    p = _by_id().get(pid)
    if p is None:
        raise KeyError(pid)
    errors = {}
    with _lock:
        st = _load()
        e = _entry(st, pid).setdefault("settings", {})
        for f in p.get("settings", []):
            k = f["key"]
            if k not in values:
                continue
            v = values[k]
            if f["type"] == "secret":
                v = str(v or "").strip()
                path = _secret_path(f["file"])
                if v == "":                      # explicit clear
                    if path.exists():
                        path.unlink()
                    continue
                if len(v) < 8 or any(c.isspace() for c in v) or len(v) > 400:
                    errors[k] = "That doesn't look like an API key."
                    continue
                USER.mkdir(parents=True, exist_ok=True)
                path.write_text(v, encoding="utf-8")
                continue
            if f["type"] == "number":
                try:
                    n = float(v)
                except (TypeError, ValueError):
                    errors[k] = "Enter a number."
                    continue
                lo, hi = f.get("min"), f.get("max")
                if (lo is not None and n < lo) or (hi is not None and n > hi):
                    errors[k] = f"Between {lo} and {hi}."
                    continue
                v = int(n) if n == int(n) else n
            elif f["type"] == "toggle":
                v = bool(v)
            elif f["type"] == "select":
                opts = f["options"]() if callable(f["options"]) else f["options"]
                if str(v) not in {str(o["value"]) for o in opts}:
                    errors[k] = "Pick one of the options."
                    continue
                v = str(v)
            else:
                v = str(v or "")[:300]
            e[k] = v
        if errors:
            return {"ok": False, "errors": errors}
        _save(st)
    apply_all()
    _changed()
    return {"ok": True, "plugin": next(x for x in listing() if x["id"] == pid)}


_FIELD_TESTS = {"exa": _test_exa_key, "brave": _test_brave_key}


def test(pid: str, field: str = "", value: str = "") -> dict:
    p = _by_id().get(pid)
    if p is None:
        raise KeyError(pid)
    try:
        if field:
            f = next((x for x in p.get("settings", []) if x["key"] == field), None)
            if not f or not f.get("test"):
                return {"ok": False, "message": "Nothing to test for that field."}
            v = (value or "").strip()
            if not v:
                import websearch
                v = websearch._key(f["file"])
            if not v:
                return {"ok": False, "message": "No key entered yet."}
            return _FIELD_TESTS[f["test"]](v)
        fn = p.get("test")
        if not fn:
            return {"ok": True, "message": "Nothing to test; this plugin has no connection."}
        return fn(settings_of(pid))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Test failed: {e}"}


def marketplace() -> List[dict]:
    """Catalog for the Marketplace tab. Built-ins show as installed."""
    try:
        items = json.loads(MARKET_PATH.read_text(encoding="utf-8")).get("plugins", [])
    except (OSError, ValueError):
        items = []
    have = _by_id()
    out = [{"id": p["id"], "name": p["name"], "icon": p["icon"], "category": p["category"],
            "summary": p["summary"], "status": "installed", "author": "ATLAS"} for p in BUILTIN]
    for it in items:
        if it.get("id") in have:
            continue
        out.append({**{k: it.get(k, "") for k in ("id", "name", "icon", "category", "summary", "author")},
                    "status": it.get("status", "coming_soon")})
    return out
