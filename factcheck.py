"""Quiet fact-check plugin (owner-approved 10-05).

Hears checkable public claims in the room ("the Great Wall is visible from space",
"Mount Everest is 9,000 metres"), looks them up in the background with the normal
search chain, and keeps the evidence for the owner's panel. The agent says nothing
about it unless someone asks ("Fae, was that true?", "fact-check that").

Design rules:
- Never a verdict from this module. It gathers evidence; the agent judges when asked
  and is told to say plainly when the evidence doesn't settle it.
- In-memory only (this session). Nothing is written to disk.
- Rate limited (min gap + daily cap) so a chatty room can't burn search credits.
- Private details (phone numbers, emails, addresses, keys) are never searched.
- Personal statements, plans, opinions and questions are not claims.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_ITEMS = 50          # claims kept for the panel
RECENT_LINES = 40       # fallback lines for "was that true?" with no queued claim
ASK_WINDOW_S = 10 * 60  # "was that true?" only reaches back this far
DEFAULT_GAP_S = 60.0
DEFAULT_DAILY = 30
WAIT_PENDING_S = 6.0

_lock = threading.Lock()
_items: deque = deque(maxlen=MAX_ITEMS)
_lines: deque = deque(maxlen=RECENT_LINES)
_seen: Dict[str, float] = {}
_day: Dict[str, int] = {"day": -1, "n": 0}
_last_bg = [0.0]
_next_id = [1]
_worker: Dict[str, Optional[threading.Thread]] = {"t": None}
_queue: deque = deque()

# Overridable for tests (no network).
SEARCH: Callable[[str], dict] = None  # type: ignore[assignment]


def _search(q: str) -> dict:
    if SEARCH is not None:
        return SEARCH(q)
    import websearch
    return websearch.search(q, max_results=3)


def _speech(text: str) -> str:
    return re.sub(r"^\s*\[S\d+\]\s*", "", text or "").strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def _private(text: str) -> bool:
    try:
        from hypergraph_memory import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ claim detection
_QUESTION_START = re.compile(
    r"(?i)^(?:who|what|when|where|why|how|which|is|are|was|were|do|does|did|can|could|"
    r"would|will|should|have|has)\b")
_NOT_CLAIM = re.compile(
    r"(?i)\b(?:i|i'm|im|i've|i'll|i'd|me|my|mine|we're|we'll|let's|lets|you're|your)\b|"
    r"\b(?:gonna|going to|tomorrow|tonight|later|next week|in \d+ minutes?|at \d{1,2}(?::\d\d)?\b)|"
    r"\b(?:bet|think|feel|guess|probably|maybe|love|hate|favou?rite|best|worst|sucks|overrated|"
    r"underrated|should)\b")
_SIGNAL = re.compile(
    r"(?i)(?:\d|\b(?:hundred|thousand|million|billion|trillion|percent|half|twice)\b|"
    r"\bthe\s+(?:largest|biggest|smallest|tallest|longest|oldest|first|last|fastest|deepest|"
    r"highest|most|richest|only|hottest|coldest)\b|"
    r"\b(?:invented|discovered|founded|built|born|died|wrote|painted|capital of|population|"
    r"visible from|located in|made of|released in)\b)")
_FACT_VERB = re.compile(r"(?i)\b(?:is|are|was|were|has|have|had|can|invented|discovered|founded|built|"
                        r"born|died|wrote|painted|released|holds|contains|weighs|measures)\b")


def is_claim(text: str) -> bool:
    t = _speech(text)
    words = t.split()
    if not (5 <= len(words) <= 45) or t.endswith("?") or _QUESTION_START.match(t):
        return False
    if _NOT_CLAIM.search(t) or not _SIGNAL.search(t) or not _FACT_VERB.search(t):
        return False
    return not _private(t)


# ------------------------------------------------------------------ ask intent
_ASK_RE = re.compile(
    r"(?i)\b(?:(?:is|was)\s+(?:that|this|it|he|she|they)\s+(?:true|right|correct|accurate|a\s+fact|real)|"
    r"(?:are|were)\s+(?:they|you)\s+right\s+about\s+that|"
    r"fact[\s-]?check\s+(?:that|this|him|her|them|it)|"
    r"did\s+(?:he|she|they)\s+get\s+that\s+right|check\s+(?:that|this)\s+claim|"
    r"is\s+that\s+(?:actually|really)\s+(?:true|right|a\s+thing))\b")


def intent(text: str):
    t = _speech(text)
    if _ASK_RE.search(t):
        return ("check_claim", {})
    return None


# ------------------------------------------------------------------ evidence
def _evidence(res: dict) -> List[dict]:
    import untrusted
    out = []
    for r in (res or {}).get("results") or []:
        # websearch already sanitises; again here so any SEARCH backend is covered.
        title = untrusted.sanitize(str(r.get("title", "")), 160)[0]
        snip = untrusted.sanitize(str(r.get("snippet", "")), 400)[0]
        out.append({"title": title, "snippet": snip,
                    "url": r.get("url", ""), "date": r.get("date") or r.get("age") or ""})
    return out[:3]


def _run(item: dict) -> None:
    try:
        res = _search(item["claim"][:180])
        ev = _evidence(res)
        with _lock:
            item["evidence"] = ev
            item["status"] = "checked" if ev else "no results"
            item["via"] = res.get("backend") or ""
    except Exception as e:  # noqa: BLE001
        with _lock:
            item["status"] = "error"
            item["error"] = str(e)[:120]
        logger.warning("fact-check search failed: %s", e)


def _work() -> None:
    while True:
        with _lock:
            if not _queue:
                _worker["t"] = None
                return
            item = _queue.popleft()
        _run(item)


def _kick() -> None:
    with _lock:
        if _worker["t"] is not None:
            return
        t = threading.Thread(target=_work, name="factcheck", daemon=True)
        _worker["t"] = t
    t.start()


def _budget_ok(daily: int, now: float) -> bool:
    day = int(now // 86400)
    if _day["day"] != day:
        _day["day"], _day["n"] = day, 0
    return _day["n"] < daily


def _new_item(claim: str, speaker: str, now: float, source: str) -> dict:
    item = {"id": _next_id[0], "claim": claim, "speaker": speaker or "someone", "at": now,
            "status": "pending", "evidence": [], "via": "", "source": source}
    _next_id[0] += 1
    _items.appendleft(item)
    _day["n"] += 1
    return item


def observe(text: str, speaker: str = "", gap_s: float = DEFAULT_GAP_S,
            daily: int = DEFAULT_DAILY, background: bool = True, clock=time.time) -> Optional[dict]:
    """Every heard line goes here. Queues a background check for checkable claims."""
    t = _speech(text)
    if not t:
        return None
    now = clock()
    with _lock:
        _lines.append({"text": t, "speaker": speaker or "someone", "at": now})
    if not background or not is_claim(t):
        return None
    key = _norm(t)
    with _lock:
        if key in _seen or now - _last_bg[0] < gap_s or not _budget_ok(daily, now):
            return None
        _seen[key] = now
        _last_bg[0] = now
        item = _new_item(t, speaker, now, "heard")
        _queue.append(item)
    _kick()
    logger.info("🔎 fact-check queued: %s", t[:80])
    return item


def _wait(item: dict, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with _lock:
            if item["status"] != "pending":
                return
        time.sleep(0.1)


def _format(item: dict) -> str:
    import untrusted
    ev = item.get("evidence") or []
    if not ev:
        body = "The search found nothing useful."
    else:
        body = "\n".join(f"- {('[' + e['date'] + '] ') if e.get('date') else ''}{e['title']}: {e['snippet']}"
                         for e in ev)
        body = untrusted.wrap(body, "fact-check search")
    return (f"CLAIM (said by {item['speaker']}): \"{item['claim']}\"\n{body}\n"
            "Judge it yourself from this evidence, briefly and kindly: say plainly if it supports the claim, "
            "contradicts it, or doesn't settle it. Don't overstate, and don't embarrass the person.")


def check(claim: str = "", asker_line: str = "", daily: int = DEFAULT_DAILY, clock=time.time) -> str:
    """Tool body for check_claim. Empty claim = the most recent claim/statement in the room."""
    now = clock()
    claim = _speech(claim)
    ask_key = _norm(_speech(asker_line))
    item = None
    speaker = ""
    with _lock:
        if claim:
            for it in _items:
                if _norm(it["claim"]) == _norm(claim):
                    item = it
                    break
        else:
            for it in _items:
                if now - it["at"] <= ASK_WINDOW_S:
                    item = it
                    break
            if item is None:
                for ln in reversed(_lines):
                    if now - ln["at"] > ASK_WINDOW_S:
                        break
                    if _norm(ln["text"]) == ask_key or ln["text"].endswith("?") or intent(ln["text"]):
                        continue
                    if len(ln["text"].split()) >= 4 and not _private(ln["text"]):
                        claim, speaker = ln["text"], ln["speaker"]
                        break
        if item is None and not claim:
            return "There's no recent claim to check. Ask which statement they mean."
        if item is None:
            if _private(claim):
                return "That contains private details, so it won't be searched."
            if not _budget_ok(daily, now):
                return "The daily fact-check limit is used up. Say you can't check it right now."
            item = _new_item(claim, speaker or "someone", now, "asked")
            pending = True
        else:
            pending = item["status"] == "pending"
    if pending and item.get("source") == "asked":
        _run(item)
    elif pending:
        _wait(item, WAIT_PENDING_S)
    with _lock:
        if item["status"] == "pending":
            return f"Still checking \"{item['claim']}\". Say you're looking into it."
        return _format(item)


def feed() -> List[dict]:
    """Owner panel: most recent claims with evidence."""
    with _lock:
        out = []
        for it in list(_items)[:20]:
            ev = it.get("evidence") or []
            out.append({
                "title": f"{it['speaker']}: {it['claim']}",
                "meta": time.strftime("%H:%M", time.localtime(it["at"])) + " · " + it["status"]
                        + (f" via {it['via']}" if it.get("via") else ""),
                "items": [{"text": f"{e['title']}: {e['snippet'][:200]}", "url": e.get("url", "")} for e in ev],
            })
        return out


def reset() -> None:
    with _lock:
        _items.clear(); _lines.clear(); _seen.clear(); _queue.clear()
        _day.update(day=-1, n=0); _last_bg[0] = 0.0


TOOL = {
    "type": "function",
    "function": {
        "name": "check_claim",
        "description": ("Fact-check a claim someone made in the call, using a web search. Use only when "
                        "someone asks if something is true ('was that true?', 'fact-check that'). "
                        "Leave claim empty to check the most recent claim. Returns evidence; you judge it."),
        "parameters": {"type": "object", "properties": {
            "claim": {"type": "string", "description": "The exact claim, or empty for the latest one."}}},
    },
}
