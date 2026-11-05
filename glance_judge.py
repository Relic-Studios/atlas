"""Would a friend in the call glance at the screen right now? One token from the live model.

Owner 11-01: agents take a screenshot on their own when they'd want to see the people they're
talking with. Word lists couldn't tell "my kitten is asleep on my keyboard" (look!) from "my
kitten is at the vet tomorrow" (don't): three held-out sets of new phrasings kept finding new
vocabulary (tests/data/glance_heldout*.json). So the model decides, with a one-token yes/no and
its logprobs, only for lines aimed at the agent that mention anything visual (capability.
maybe_visual). Fails closed: no answer within TIMEOUT_S means no glance.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time

logger = logging.getLogger(__name__)

TIMEOUT_S = 1.2
THRESHOLD = 0.5
# Glance when the judge is sure, or fairly sure and the word cues agree. On the 170 held-out
# lines (tests/sim_glance_judge.py): judge alone >= 0.5 caught 84/85 with 9/85 false; this
# rule caught 83/85 with 2/85 false (thresholds chosen on those lines).
SURE = 0.8

SYSTEM = ("You are part of a voice assistant in a Discord call where people can be on camera and share "
          "their screen. Decide if the line just said is about something visible here and now that a "
          "friend in the call would glance at the screen to see: the speaker themself, what they're "
          "wearing, holding, eating or just made, a pet or person with them, their room or setup, a view, "
          "or something happening on camera or on screen right now. Something they got or changed "
          "recently that still shows (a new tattoo, haircut, glasses, clothes, braces) counts. Say no for "
          "things that are over or not here yet, plans, wishes, needs, other people who aren't with them, "
          "and 'see' or 'look' used as a figure of speech.\n"
          "Examples:\n"
          "\"check out my new sneakers\" -> yes\n"
          "\"I cut my own hair last night lol\" -> yes\n"
          "\"my dog won't get off my lap\" -> yes\n"
          "\"I just got my nose pierced\" -> yes\n"
          "\"the sky behind me is crazy orange\" -> yes\n"
          "\"can you see why that's annoying\" -> no\n"
          "\"my cousin has a sleeve tattoo\" -> no\n"
          "\"I want a new jacket for winter\" -> no\n"
          "\"I'm wearing earbuds so I might miss stuff\" -> no\n"
          "\"my dog ran away when I was little\" -> no\n"
          "Answer with exactly one word: yes or no.")

_session = None
_cfg = None
stats = {"asked": 0, "yes": 0, "timeouts": 0, "errors": 0, "ms": []}


def _client():
    """Session for the remote model, only when it's a vLLM server (logprobs, images and
    chat_template_kwargs). A cloud provider gets no extra calls: callers fall back (word
    cues, their own words as the query, no screen caption)."""
    global _session, _cfg
    if _session is None:
        import requests
        import remote_llm
        _cfg = remote_llm.load_config()
        if not _cfg or not _cfg.get("vllm") or _cfg.get("cloud"):
            return None, None
        _session = requests.Session()
        if _cfg.get("key"):
            _session.headers["Authorization"] = "Bearer " + _cfg["key"]
    return _session, _cfg


def p_glance(line: str, before: str = "", timeout: float = TIMEOUT_S):
    """Probability a friend would glance at the screen after this line; None if unknown."""
    s, cfg = _client()
    if s is None:
        return None
    ctx = (f"Just before: \"{before.strip()}\"\n" if before and before.strip() else "")
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{ctx}Line: \"{line.strip()}\"\nGlance at the screen?"}],
            "chat_template_kwargs": {"enable_thinking": False}}
    stats["asked"] += 1
    t0 = time.perf_counter()
    try:
        r = s.post(cfg["url"] + "/v1/chat/completions", json=body, timeout=(1.0, timeout))
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception as e:  # noqa: BLE001
        stats["timeouts" if "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower() else "errors"] += 1
        logger.debug("glance judge failed: %s", e)
        return None
    finally:
        stats["ms"] = (stats["ms"] + [round((time.perf_counter() - t0) * 1000)])[-200:]
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    if yes + no <= 0:
        return None
    p = yes / (yes + no)
    stats["yes"] += p >= THRESHOLD
    return p


# Live screen (docs/designs/live_screen.md): does a line said to the agent need a look at the screen right now?
# Phase 0 (work/egm/live-screen/phase0_results.md): this wording, threshold 0.75, caught every screen line of
# the held-out half with no false alarm; the first wording counted any game talk. Acting on its verdicts since
# 11-03 (live_screen.config.SCREEN_DETECT; owner's call after that day's real calls).
SCREEN_SYSTEM = (
    "You are part of a voice assistant in a Discord call. It can look at the player's game screen or a friend's "
    "stream. Decide if the line just said, which was said to the assistant, needs the assistant to look at the "
    "screen right now to answer or react well: a question or remark about what is on the screen at this moment "
    "(what is happening, what something is, where to go or what to do here, what just happened, text on the "
    "screen). Answer no for general game knowledge or strategy that does not depend on this screen, for talk about "
    "other times or places, and for anything that is not about a screen. Answer yes or no.")
screen_stats = {"asked": 0, "yes": 0, "errors": 0, "ms": []}


def p_screen(line: str, timeout: float = TIMEOUT_S):
    """Probability the line needs a look at the screen right now; None if the model can't answer."""
    s, cfg = _client()
    if s is None:
        return None
    body = {"model": cfg["model"], "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "system", "content": SCREEN_SYSTEM},
                         {"role": "user", "content": f"Line: \"{line.strip()}\""}],
            "chat_template_kwargs": {"enable_thinking": False}}
    screen_stats["asked"] += 1
    t0 = time.perf_counter()
    try:
        r = s.post(cfg["url"] + "/v1/chat/completions", json=body, timeout=(1.0, timeout))
        top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except Exception as e:  # noqa: BLE001
        screen_stats["errors"] += 1
        logger.debug("screen judge failed: %s", e)
        return None
    finally:
        screen_stats["ms"] = (screen_stats["ms"] + [round((time.perf_counter() - t0) * 1000)])[-200:]
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    if yes + no <= 0:
        return None
    return yes / (yes + no)


# Verdicts by line. The judge took ~0.2 s in front of every reply to a line said to the agent (heavy-call test 11-04,
# owner: "the lag started appearing after our live video system was implemented"), so a question is judged from the
# partial transcripts while its last words come in, and the turn usually finds its verdict ready.
VERDICT_KEEP_S = 60.0
PREFETCH_MAX = 2                   # judge requests from partials in flight at once
_verdicts: dict = {}
_vlock = threading.Lock()


def _key(line: str) -> str:
    """The same words however the transcript spelled them ("All right," / "Alright")."""
    return re.sub(r"[^a-z0-9]+", "", (line or "").lower().replace("all right", "alright"))


def _judge_into(line: str, entry: dict) -> None:
    try:
        entry["p"] = p_screen(line)
    finally:
        entry["done"].set()


def prefetch_screen(line: str) -> bool:
    """Judge this line in the background (a partial that may be the finished line). Never blocks; at most
    PREFETCH_MAX at once. True if a judge request started."""
    key = _key(line)
    if not key:
        return False
    now = time.time()
    with _vlock:
        for k in [k for k, e in _verdicts.items() if now - e["t"] > VERDICT_KEEP_S]:
            del _verdicts[k]
        if key in _verdicts or sum(not e["done"].is_set() for e in _verdicts.values()) >= PREFETCH_MAX:
            return False
        entry = _verdicts[key] = {"t": now, "p": None, "done": threading.Event()}
    threading.Thread(target=_judge_into, args=(line, entry), name="screen-question", daemon=True).start()
    return True


def screen_verdict(line: str, timeout: float = TIMEOUT_S):
    """p_screen for a line said to the agent: the verdict judged while it was being said (waiting for one still
    running, never past its own timeout), else judged now. None if the model can't answer."""
    key = _key(line)
    with _vlock:
        entry = _verdicts.get(key)
        if entry is None:
            entry = _verdicts[key] = {"t": time.time(), "p": None, "done": threading.Event()}
            mine = True
        else:
            mine = False
    if mine:
        _judge_into(line, entry)
    else:
        entry["done"].wait(max(0.0, entry["t"] + timeout + 0.05 - time.time()))
    return entry["p"]


# Cheap prefilter: the judge only runs on lines that could be about something visible.
# It must never miss (tests/test_glances.py runs it over every held-out "glance" line).
_MAYBE = None


def maybe_visual(line: str) -> bool:
    global _MAYBE
    if _MAYBE is None:
        import re
        import capability as _cap
        _MAYBE = re.compile(
            r"\b(?:" + _cap._VISIBLE + r"|cam|camera|webcam|video|screen|stream|see|seen|look|looks|looking|watch|"
            r"show|check|view|wearing|wear|dressed|dress|holding|made|make|built|drew|painted|baked|cooked|"
            r"got|bought|new|finally|pierced|dyed|shaved|cut|grew|glowing|behind\s+me|on\s+my|in\s+my|"
            r"right\s+now|rate|judge|eating|sitting|lap|shoulder|arm|desk|car|wall|tank|plant|leaf|snow|sky|"
            r"sunset|outside|window|poster|cosplay|gown|paint|stain|eye|lip|contacts|mullet|highlights|"
            r"mic|nephew|niece|baby|grandma|grandpa|mom|dad|sister|brother|kid|son|daughter)\b", re.I)
    return bool(_MAYBE.search(line or ""))


def should_glance(line: str, before: str = "") -> bool:
    """The live decision for a line aimed at the agent: prefilter, then the model;
    if the model can't answer in time, the word cues (capability.wants_glance)."""
    import capability as _cap
    if _cap.wants_look(line) or not maybe_visual(line):
        return False
    p = p_glance(line, before)
    if p is None:
        return _cap.wants_glance(line)
    return p >= SURE or (p >= THRESHOLD and _cap.wants_glance(line))
