"""Rare, useful tips (docs/designs/live_screen.md, section 7, G4).

A tip candidate exists only when the screen log shows a moment worth one (a new area, an objective that changed,
dying twice within ten minutes) in a game with a known, non-competitive profile, with the Live screen setting
"Tips" on. Then, cheapest first:
  budget     at most TIPS_PER_HOUR, at least TIP_MIN_GAP_S apart (doubled after each ignored tip), not while quiet
  grounding  only facts from the screen log or a web search about that spot, never the model's own game knowledge
  draft      Model (no persona) writes at most one short tip from those sources, or none
  facts      every number and name in the tip must appear in the sources, else it's dropped
  worth      a yes/no from Model: would the player be glad to hear this now, unasked? (WORTH_THRESHOLD)
A tip that survives waits for a natural pause (a loading screen, menu, map or death screen, or TIP_PAUSE_WAIT_S)
and a gap in the talk, and is spoken word for word through a cue turn (as timers and glances are); after
TIP_EXPIRY_S it's dropped. Every tip is written to private/live_screen_tips.jsonl for the owner's ratings
(tools/rate_tips.py), which tune WORTH_THRESHOLD.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path

from live_screen import config as CFG

logger = logging.getLogger(__name__)
LOG_PATH = Path(__file__).resolve().parents[1] / "private" / "live_screen_tips.jsonl"
TIP_CUE_RE = re.compile(r"\(TIP CUE #(\d+)\)")
PAUSE_KINDS = ("loading", "menu", "map", "death", "inventory")

TIP_SYSTEM = (
    "You help a player with rare, useful tips during a voice call. You are not a character and you don't chat. "
    "From what just happened in their game, the recent screen log and any notes from a web search, write at most "
    "one tip of at most 16 words that the player would genuinely want right now: something they may have missed, "
    "or one concrete thing to try. Use only facts from the screen log or the notes, never your own knowledge of the "
    "game. Don't state the obvious, don't describe what they're doing, don't cheer. If nothing is clearly useful, "
    "return an empty tip.")
SPOILER_RULE = {
    "none": "Spoilers: none. Mention nothing the player hasn't already seen in the screen log.",
    "light": "Spoilers: light. You may point the way or suggest what to try, but don't name story events, bosses "
             "or rewards the player hasn't seen yet.",
    "full": "Spoilers: allowed.",
}
TIP_SCHEMA = {"type": "object", "properties": {"tip": {"type": "string", "maxLength": 120},
                                               "uses": {"enum": ["log", "notes", "both", "none"]}},
              "required": ["tip", "uses"]}
_POSITIVE = re.compile(r"\b(?:thanks|thank you|thx|good call|nice|oh (?:yeah|right|true|shit)|true|good (?:tip|idea)|"
                       r"wait really|oh really|didn'?t (?:see|know|notice) that|i missed that|will do)\b", re.I)
_NEGATIVE = re.compile(r"\b(?:shut up|stop|didn'?t ask|nobody asked|i know|i knew|not now|no tips|be quiet|leave me|"
                       r"spoilers?|annoying|let me play)\b", re.I)
_OFF = re.compile(r"\b(?:(?:stop|no more|enough(?:\s+with)?|quit|cut it with)\s+(?:the\s+|your\s+|giving\s+)?tips|"
                  r"no\s+(?:more\s+)?tips(?!\s+for))\b", re.I)
_ON = re.compile(r"\b(?:tips?\s+(?:back\s+)?on|you\s+can\s+give\s+(?:me\s+)?tips(?:\s+again)?|give\s+me\s+tips)\b", re.I)
_QUIET = re.compile(r"\b(?:no\s+tips|quiet|be\s+quiet|shh+)\s+for\s+(?:(\d+)|(a|an|one|five|ten|fifteen|twenty|"
                    r"thirty))\s*(min(?:ute)?s?|hours?)\b", re.I)
_WORDS = {"a": 1, "an": 1, "one": 1, "five": 5, "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30}

_lock = threading.Lock()
_s: dict = {}


def reset() -> None:
    """A new Live screen session: forget everything about tips."""
    with _lock:
        _s.clear()
        _s.update(spoken=[], ignored=0, quiet_until=0.0, off=False, places=set(), objectives={}, deaths=[],
                  was_death=False, ready=None, last=None, searches=0, search_cache={}, next_id=1)


reset()


def setting() -> str:
    try:
        import live_screen
        return (live_screen.settings().get("tips") or "off").lower()
    except Exception:  # noqa: BLE001
        return "off"


def _record(row: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("tip log: %s", e)


# --------------------------------------------------------------------------- triggers
def trigger(look: dict, t: float) -> dict | None:
    """A moment worth a tip in this log look, or None (updates what this session has seen)."""
    kind = look.get("scene_kind")
    with _lock:
        died = kind == "death" and not _s["was_death"]
        _s["was_death"] = kind == "death"
        if died:
            _s["deaths"] = [d for d in _s["deaths"] if t - d < 600] + [t]
            if len(_s["deaths"]) >= 2:
                where = next(iter(sorted(_s["places"])), "") if _s["places"] else ""
                return {"kind": "repeated_death", "topic": (look.get("scene") or where)[:80],
                        "observation": f"died {len(_s['deaths'])} times in the last 10 minutes: "
                                       f"{look.get('scene', '')}"}
        found = None
        for ent in look.get("entities") or []:
            k, name, seen = ent.get("kind"), (ent.get("name") or "").strip(), (ent.get("seen") or "").strip()
            if not name:
                continue
            key = name.lower()
            if k == "place" and key not in _s["places"]:
                first = not _s["places"]
                _s["places"].add(key)
                if not first and found is None:          # the session's first place is just where they started
                    found = {"kind": "new_area", "topic": name, "observation": f"arrived at {name}: {seen}"}
            elif k == "objective":
                prev = _s["objectives"].get(key)
                _s["objectives"][key] = seen
                if prev is not None and prev != seen and found is None:
                    found = {"kind": "objective", "topic": name, "observation": f"objective {name}: {seen}"}
        return found


def _budget_ok(now: float) -> str:
    """'' if a tip may be prepared now, else why not."""
    with _lock:
        last = _s["last"]
        if last and not last.get("reaction") and now - last["t"] > CFG.TIP_REACTION_S:
            last["reaction"] = "none"                        # nobody reacted: treat as ignored
            _s["ignored"] += 1
        if _s["off"]:
            return "tips switched off for this session"
        if now < _s["quiet_until"]:
            return "asked for quiet"
        if _s["ready"] is not None:
            return "a tip is already waiting"
        recent = [t for t in _s["spoken"] if now - t < 3600]
        if len(recent) >= CFG.TIPS_PER_HOUR:
            return "hourly cap"
        gap = CFG.TIP_MIN_GAP_S * (2 ** min(_s["ignored"], 3))
        if _s["spoken"] and now - _s["spoken"][-1] < gap:
            return f"too soon ({round((gap - (now - _s['spoken'][-1])) / 60)} min left)"
    return ""


# --------------------------------------------------------------------------- grounding and checks
def _search_notes(game: str, topic: str) -> str:
    """Web notes about this spot (the Web search plugin), cached per game and topic, capped per session."""
    key = (game.lower(), (topic or "").lower()[:60])
    with _lock:
        if key in _s["search_cache"]:
            return _s["search_cache"][key]
        if _s["searches"] >= CFG.TIP_SEARCH_CAP:
            return ""
        _s["searches"] += 1
    text = ""
    try:
        import plugins
        import websearch
        if plugins.is_enabled("web_search") and websearch.available() and topic:
            r = websearch.search(f"{game} {topic} tips") or {}
            text = (r.get("text") or "")[:1500]
    except Exception as e:  # noqa: BLE001
        logger.debug("tip search failed: %s", e)
    with _lock:
        _s["search_cache"][key] = text
    return text


_CAPS_OK = {"i", "you", "your", "you're", "the", "a", "an", "it", "it's", "that", "this", "there", "try", "check",
            "look", "maybe", "watch", "go", "head", "don't", "heads", "quick", "tip", "hey", "oh", "careful", "also",
            "if", "when", "before", "after", "and", "but", "or", "so", "just", "remember", "on", "in", "at", "to"}


def facts_ok(tip: str, sources: str) -> bool:
    """Every number and capitalized name in the tip appears in the sources (case-insensitive)."""
    src = (sources or "").lower().replace(",", "")
    for n in re.findall(r"\d[\d,.]*", tip or ""):
        if n.rstrip(".,").replace(",", "") not in src:
            return False
    words = re.findall(r"\b[A-Z][\w'\-]*", tip or "")
    for w in words:
        if w.lower() in _CAPS_OK:
            continue
        if w.lower() not in src:
            return False
    return True


def _eyes_model(remote_cfg: dict):
    from live_screen import watcher
    return watcher._client(remote_cfg)


def draft(prof: dict, trig: dict, log_lines: list, notes: str, player: str, remote_cfg: dict) -> str:
    http, base, model = _eyes_model(remote_cfg)
    if http is None:
        return ""
    from live_screen import filters
    import untrusted
    clean_notes, _ = untrusted.sanitize(notes or "", limit=1500)
    user = (f"Game: {prof['name']}. {SPOILER_RULE[prof['spoilers']]}"
            + (f" The player's name is {player}; address them by name." if player else "")
            + f"\nWhat just happened: {trig['observation']}\nScreen log (oldest first):\n"
            + filters.wrap("\n".join(log_lines) or "(empty)")
            + ("\nNotes from a web search:\n" + untrusted.wrap(clean_notes) if clean_notes else
               "\nNo web notes: use only the screen log."))
    body = {"model": model, "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "tip", "schema": TIP_SCHEMA}},
            "messages": [{"role": "system", "content": TIP_SYSTEM}, {"role": "user", "content": user}]}
    try:
        r = http.post(base + "/v1/chat/completions", json=body, timeout=(1.0, 8.0))
        out = json.loads(r.json()["choices"][0]["message"]["content"])
        return (out.get("tip") or "").strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("tip draft failed: %s", e)
        return ""


def worth(prof: dict, trig: dict, tip: str, remote_cfg: dict, log_lines: list | None = None) -> float | None:
    http, base, model = _eyes_model(remote_cfg)
    if http is None:
        return None
    from live_screen import filters
    recent = "\n".join((log_lines or [])[-6:])
    q = (f"A voice assistant is watching someone play {prof['name']} in a call. It could say this tip now, "
         f"unasked: \"{tip}\". What just happened: {trig['observation']}."
         + (f" What it saw earlier (screen log, oldest first):\n{filters.wrap(recent)}\n" if recent else " ")
         + "Would the player most likely be glad to hear it right now (useful, not obvious, not annoying)? "
           "Answer yes or no.")
    body = {"model": model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "Answer yes or no."}, {"role": "user", "content": q}]}
    try:
        r = http.post(base + "/v1/chat/completions", json=body, timeout=(1.0, 4.0))
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception as e:  # noqa: BLE001
        logger.debug("tip worth check failed: %s", e)
        return None
    yes = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() == "yes")
    no = sum(math.exp(x["logprob"]) for x in top if x["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None


# --------------------------------------------------------------------------- the pipeline
def on_look(look: dict, t: float, title: str, process: str, label: str, player: str, remote_cfg: dict) -> dict | None:
    """Called after each screen-log look (scheduler thread). Returns the prepared tip, if one survived."""
    if setting() != "rare" or not isinstance(look, dict):
        return None
    from live_screen import game_profiles, memory
    prof = game_profiles.match(title, process)
    if prof is None or prof["competitive"]:
        return None
    trig = trigger(look, t)
    if trig is None:
        return None
    why = _budget_ok(t)
    row = {"t": t, "game": prof["name"], "kind": trig["kind"], "observation": trig["observation"]}
    if why:
        logger.info("💡 tip moment skipped (%s): %s", why, trig["observation"][:80])
        _record(dict(row, outcome="skipped", why=why))
        return None
    log_lines = [memory.line(e) for e in memory.snapshot(12)]
    notes = _search_notes(prof["name"], trig["topic"])
    tip = draft(prof, trig, log_lines, notes, player, remote_cfg)
    if not tip:
        _record(dict(row, outcome="no tip"))
        return None
    sources = " ".join(log_lines) + " " + notes + " " + trig["observation"] + " " + prof["name"] + " " + (player or "")
    if not facts_ok(tip, sources):
        logger.info("💡 tip dropped (adds names or numbers it can't back): %r", tip)
        _record(dict(row, outcome="failed fact check", tip=tip))
        return None
    p = worth(prof, trig, tip, remote_cfg, log_lines)
    if p is None or p < CFG.WORTH_THRESHOLD:
        logger.info("💡 tip not worth it (p=%s): %r", "n/a" if p is None else f"{p:.2f}", tip)
        _record(dict(row, outcome="not worth it", tip=tip, worth=p))
        return None
    with _lock:
        tid = _s["next_id"]
        _s["next_id"] += 1
        ready = {"id": tid, "tip": tip, "game": prof["name"], "kind": trig["kind"], "t": t,
                 "expires": t + CFG.TIP_EXPIRY_S, "worth": p, "label": label}
        _s["ready"] = ready
    logger.info("💡 tip ready (p=%.2f): %r", p, tip)
    _record(dict(row, outcome="ready", tip=tip, worth=p, id=tid))
    return ready


def ready(now: float) -> bool:
    """A prepared tip may be spoken now: not expired, and at a natural pause (or after waiting long enough)."""
    with _lock:
        tip = _s["ready"]
        if tip is None:
            return False
        if now > tip["expires"]:
            _s["ready"] = None
            logger.info("💡 tip expired unspoken: %r", tip["tip"])
            _record({"t": now, "id": tip["id"], "outcome": "expired", "tip": tip["tip"]})
            return False
        if _s["off"] or now < _s["quiet_until"]:
            return False
    try:
        from live_screen import memory
        cur = memory.current(max_age=30.0) or {}
    except Exception:  # noqa: BLE001
        cur = {}
    return cur.get("scene_kind") in PAUSE_KINDS or now - tip["t"] >= CFG.TIP_PAUSE_WAIT_S


def claim(now: float) -> str:
    """Take the prepared tip for speaking: its cue line (empty if none)."""
    with _lock:
        tip, _s["ready"] = _s["ready"], None
        if tip is None:
            return ""
        _s["spoken"].append(now)
        _s["last"] = {"t": now, "id": tip["id"], "tip": tip["tip"], "reaction": None}
    _record({"t": now, "id": tip["id"], "outcome": "spoken", "tip": tip["tip"], "game": tip["game"]})
    return (f"(TIP CUE #{tip['id']}) This is NOT a new line from anyone: it's your own cue. Watching "
            f"{tip['label']}, you noticed something worth a quick tip. Say exactly this, word for word, as one short "
            f"line: \"{tip['tip']}\" Add nothing to it. If the room is clearly in the middle of something else, "
            "output [HOLD].")


def cue_note(text: str) -> str:
    if not TIP_CUE_RE.search(text or ""):
        return ""
    return ("TIP: this turn is your own cue, not someone speaking. Say the tip from the cue word for word, as one "
            "short line, and nothing else; [HOLD] if the moment has clearly passed.")


def hear(line: str, speaker: str = "") -> str:
    """Spoken tip commands and reactions to the last tip; returns what it did ('' if nothing)."""
    now = time.time()
    text = line or ""
    with _lock:
        m = _QUIET.search(text)
        if m:
            n = int(m.group(1)) if m.group(1) else _WORDS.get((m.group(2) or "").lower(), 10)
            mins = n * 60 if (m.group(3) or "").lower().startswith("hour") else n
            _s["quiet_until"] = now + mins * 60
            return f"quiet for {mins} min"
        if _OFF.search(text):
            _s["off"], _s["ready"] = True, None
            return "tips off for this session"
        if _ON.search(text):
            _s["off"], _s["quiet_until"] = False, 0.0
            return "tips back on"
        last = _s["last"]
        if last and not last.get("reaction") and now - last["t"] <= CFG.TIP_REACTION_S:
            if _NEGATIVE.search(text):
                last["reaction"] = "negative"
                _s["ignored"] += 1
                _s["quiet_until"] = now + CFG.TIP_QUIET_S
            elif _POSITIVE.search(text):
                last["reaction"] = "positive"
                _s["ignored"] = 0
            if last.get("reaction"):
                _record({"t": now, "id": last["id"], "outcome": "reaction", "reaction": last["reaction"],
                         "line": text[:120]})
                return f"reaction {last['reaction']}"
    return ""


def status() -> dict:
    with _lock:
        return {"setting": setting(), "off": _s["off"], "quiet_until": _s["quiet_until"],
                "spoken_last_hour": len([t for t in _s["spoken"] if time.time() - t < 3600]),
                "ignored": _s["ignored"], "waiting": bool(_s["ready"])}
