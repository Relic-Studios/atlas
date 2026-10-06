"""Bet tracker (Bet tracker plugin).

"I bet you five bucks the Lakers win tonight." The agent logs who bet what, against
whom, for what stakes, and when it can be settled. When that time comes (and a call
is on), the agent brings it up at a gap: asks the room how it turned out, or checks
a public result with search, then settles it. A running scoreboard per person.

State: memory_db/bets.json (never exported). Private details are refused, same
filter as memory. Bets nobody settles fall off 30 days after they were due.
"""
from __future__ import annotations

import calendar
import datetime as _dt
import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "memory_db" / "bets.json"
_lock = threading.RLock()

MAX_OPEN = 40
MAX_TEXT = 200
DEFAULT_DAYS = 7          # settle-by when nobody says
STALE_DAYS = 30           # unsettled bets are dropped this long after they were due
BET_CUE_RE = re.compile(r"\(BET CUE #(\d+)\)")
OUTCOMES = {"won": "won", "win": "won", "right": "won", "true": "won",
            "lost": "lost", "lose": "lost", "loss": "lost", "wrong": "lost", "false": "lost",
            "push": "push", "tie": "push", "draw": "push",
            "void": "void", "off": "void", "cancel": "void", "cancelled": "void"}


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


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _clean(s, n=MAX_TEXT) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip(" ,.;:")[:n]


def _private(text: str) -> bool:
    try:
        from hypergraph_memory import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001
        return False


def _human_when(ts: float, now: float) -> str:
    d = (_dt.date.fromtimestamp(ts) - _dt.date.fromtimestamp(now)).days
    if d < -1:
        return f"{-d} days ago"
    if d == -1:
        return "yesterday"
    if d == 0:
        return "today"
    if d == 1:
        return "tomorrow"
    if d < 7:
        return _dt.date.fromtimestamp(ts).strftime("%A")
    return _dt.date.fromtimestamp(ts).strftime("%b %d").replace(" 0", " ")


_DAYS = {d.lower(): i for i, d in enumerate(calendar.day_name)}
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple": 2, "few": 3}


def settle_time(when: str, now: float) -> float:
    """Free text ('tonight', 'Sunday', 'end of the month', 'in 2 weeks', '2026-10-12')
    -> end-of-day epoch. Unknown -> DEFAULT_DAYS from now."""
    w = (when or "").lower().strip()
    today = _dt.date.fromtimestamp(now)

    def eod(d: _dt.date) -> float:
        return _dt.datetime.combine(d, _dt.time(23, 59)).timestamp()

    if not w:
        return eod(today + _dt.timedelta(days=DEFAULT_DAYS))
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", w)
    if m:
        try:
            return eod(_dt.date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            pass
    m = re.search(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})\b", w)
    if m:
        try:
            d = _dt.date(today.year, _MONTHS[m[1]], int(m[2]))
            if d < today:
                d = d.replace(year=today.year + 1)
            return eod(d)
        except ValueError:
            pass
    m = re.search(r"\bin\s+(\d+|" + "|".join(_NUMW) + r")\s+(day|week|month)s?\b", w)
    if m:
        n = int(m[1]) if m[1].isdigit() else _NUMW[m[1]]
        days = n * {"day": 1, "week": 7, "month": 30}[m[2]]
        return eod(today + _dt.timedelta(days=days))
    if re.search(r"\b(tonight|today|later|this (?:evening|afternoon|game|match|round))\b", w):
        return eod(today)
    if "tomorrow" in w:
        return eod(today + _dt.timedelta(days=1))
    if re.search(r"\bend of (?:the )?(?:month|this month)\b|\bthis month\b", w):
        return eod(today.replace(day=calendar.monthrange(today.year, today.month)[1]))
    if re.search(r"\bend of (?:the )?year\b|\bthis year\b", w):
        return eod(_dt.date(today.year, 12, 31))
    if re.search(r"\b(?:this )?weekend\b", w):
        return eod(today + _dt.timedelta(days=6 - today.weekday()))   # this Sunday (today if Sunday)
    if re.search(r"\bnext week\b", w):
        return eod(today + _dt.timedelta(days=7))
    if re.search(r"\bend of (?:the )?week\b|\bthis week\b", w):
        return eod(today + _dt.timedelta(days=6 - today.weekday()))
    for name, i in _DAYS.items():
        if re.search(r"\b" + name + r"\b", w):
            ahead = (i - today.weekday()) % 7 or 7
            return eod(today + _dt.timedelta(days=ahead))
    return eod(today + _dt.timedelta(days=DEFAULT_DAYS))


def _prune(d: dict, now: float) -> list:
    bets = [b for b in d.get("bets", [])
            if b.get("status") != "open" or now - b.get("settle_at", now) < STALE_DAYS * 86400]
    d["bets"] = bets[-300:]
    return d["bets"]


def record_bet(claim: str, who: str = "", against: str = "", stakes: str = "", settle_by: str = "",
               clock: Callable[[], float] = time.time) -> str:
    claim = _clean(claim)
    who, against, stakes = _clean(who, 30), _clean(against, 30), _clean(stakes, 60)
    if not claim:
        return "Couldn't log that: what's the bet? (what someone says will happen)"
    if _private(" ".join((claim, stakes))):
        return "Not logged: it looks like it has a phone number, email, address or key in it."
    now = clock()
    at = settle_time(settle_by, now)
    with _lock:
        d = _load()
        bets = _prune(d, now)
        if sum(1 for b in bets if b.get("status") == "open") >= MAX_OPEN:
            return f"Too many open bets ({MAX_OPEN}). Settle or void some first."
        k = _key(claim)
        dup = next((b for b in bets if b.get("status") == "open" and _key(b["claim"]) == k
                    and _key(b.get("who", "")) == _key(who)), None)
        if dup:
            return f"Already logged as bet #{dup['id']}: {_describe(dup, now)}"
        bid = int(d.get("next_id", 1))
        d["next_id"] = bid + 1
        b = {"id": bid, "claim": claim, "who": who, "against": against, "stakes": stakes,
             "made": now, "settle_at": at, "status": "open", "cued": False}
        bets.append(b)
        _save(d)
    return f"Logged bet #{bid}: {_describe(b, now)}. I'll bring it up when it can be settled."


def _describe(b: dict, now: float) -> str:
    who = b.get("who") or "someone"
    s = f"{who} bets that {b['claim']}"
    if b.get("against"):
        s += f" (against {b['against']})"
    if b.get("stakes"):
        s += f", for {b['stakes']}"
    if b.get("status") == "open":
        s += f"; settles {_human_when(b['settle_at'], now)}"
    else:
        s += f"; {b['status'].upper()}" + (f" ({b['note']})" if b.get("note") else "")
    return s


def list_bets(include_settled: bool = False, clock: Callable[[], float] = time.time) -> str:
    now = clock()
    with _lock:
        d = _load()
        bets = _prune(d, now)
        _save(d)
    show = [b for b in bets if include_settled or b.get("status") == "open"]
    if not show:
        return "No open bets." if not include_settled else "No bets logged."
    return "; ".join(f"#{b['id']} {_describe(b, now)}" for b in show[-12:])


def _find(bets: list, bid=None, words: str = "") -> Optional[dict]:
    try:
        if bid not in (None, ""):
            return next((b for b in bets if b["id"] == int(bid)), None)
    except (TypeError, ValueError):
        pass
    w = (words or "").lower().strip()
    if not w:
        return None
    opens = [b for b in reversed(bets) if b.get("status") == "open"]
    return next((b for b in opens if w in b["claim"].lower() or _key(w) in (_key(b.get("who")),
                                                                              _key(b.get("against"))) ), None)


def _side_outcome(b: dict, winner: str, loser: str) -> Optional[str]:
    """Turn 'who won' into the bettor-side outcome. None if it can't be told."""
    w, l, who, ag = _key(winner), _key(loser), _key(b.get("who")), _key(b.get("against"))
    if w:
        if w == who:
            return "won"
        if ag and w == ag:
            return "lost"
        if not ag and who:
            return "lost"     # someone other than the bettor won a one-sided bet
    if l:
        if l == who:
            return "lost"
        if ag and l == ag:
            return "won"
    return None


def settle_bet(bid=None, outcome: str = "", note: str = "", words: str = "",
               winner: str = "", loser: str = "",
               clock: Callable[[], float] = time.time) -> str:
    """Settle by `winner` / `loser` name (preferred: ATLAS works out the side), or by
    `outcome` from the bettor's side: won / lost / push / void."""
    oc = OUTCOMES.get((outcome or "").lower().strip())
    winner, loser = _clean(winner, 30), _clean(loser, 30)
    if not oc and not (winner or loser):
        return ("Say who won (winner / loser name), or how it went for the person who made the bet: "
                "won, lost, push or void.")
    now = clock()
    with _lock:
        d = _load()
        bets = _prune(d, now)
        b = _find(bets, bid, words)
        if b is None and (winner or loser) and bid in (None, "") and not (words or "").strip():
            names = {_key(winner), _key(loser)} - {""}
            cand = [x for x in bets if x.get("status") == "open"
                    and names & {_key(x.get("who")), _key(x.get("against"))}]
            if not cand:
                cand = [x for x in bets if x.get("status") == "open"]
            if len(cand) > 1:
                due_now = [x for x in cand if x["settle_at"] <= now + 3600]
                cand = due_now if len(due_now) == 1 else cand
            if len(cand) == 1:
                b = cand[0]
            elif len(cand) > 1:
                return ("Which bet? Open: " + "; ".join(f"#{x['id']} {_describe(x, now)}" for x in cand[-6:])
                        + ". Settle it by id.")
        if b is None:
            return "No such open bet."
        if b.get("status") != "open":
            return f"Bet #{b['id']} was already settled ({b['status']})."
        if not oc:
            oc = _side_outcome(b, winner, loser)
            if not oc:
                return (f"Couldn't tell who won bet #{b['id']} ({_describe(b, now)}) from that. Say whether "
                        f"{b.get('who') or 'the bettor'} won or lost.")
        b.update(status=oc, settled=now, note=_clean(note, 120))
        _save(d)
    who, ag = b.get("who") or "the bettor", b.get("against")
    if oc == "won":
        res = f"{who} wins" + (f"; {ag} owes {b['stakes']}" if ag and b.get("stakes") else "")
    elif oc == "lost":
        res = (f"{ag} wins" if ag else f"{who} loses") + (
            f"; {who} owes {b['stakes']}" if b.get("stakes") else "")
    elif oc == "push":
        res = "it's a push, nobody pays"
    else:
        res = "bet called off"
    return f"Settled bet #{b['id']} ({b['claim']}): {res}. {scoreboard(clock=clock)}"


def scoreboard(clock: Callable[[], float] = time.time) -> str:
    with _lock:
        bets = _load().get("bets", [])
    tally: dict = {}

    def add(name, w, l):
        if not name:
            return
        k = _key(name)
        n, a, b2 = tally.get(k, (name, 0, 0))
        tally[k] = (n, a + w, b2 + l)

    for b in bets:
        if b.get("status") == "won":
            add(b.get("who"), 1, 0)
            add(b.get("against"), 0, 1)
        elif b.get("status") == "lost":
            add(b.get("who"), 0, 1)
            add(b.get("against"), 1, 0)
    if not tally:
        return "Scoreboard: no settled bets yet."
    rows = sorted(tally.values(), key=lambda r: (r[1] - r[2], r[1]), reverse=True)
    return "Scoreboard: " + ", ".join(f"{n} {w}-{l}" for n, w, l in rows[:8]) + "."


def due(clock: Callable[[], float] = time.time) -> Optional[dict]:
    """Oldest open bet that is past its settle time and hasn't been brought up yet."""
    now = clock()
    with _lock:
        bets = _prune(_load(), now)
    live = sorted((b for b in bets if b.get("status") == "open" and not b.get("cued")
                   and b["settle_at"] <= now), key=lambda b: b["settle_at"])
    return dict(live[0]) if live else None


def claim(bid: int, clock: Callable[[], float] = time.time) -> str:
    """Mark a due bet as brought up (once) and return the agent's cue line."""
    now = clock()
    with _lock:
        d = _load()
        b = next((x for x in d.get("bets", []) if x["id"] == bid), None)
        if b is None or b.get("status") != "open" or b.get("cued"):
            return ""
        b["cued"] = True
        _save(d)
    return (f"(BET CUE #{b['id']}) This is NOT a new line from anyone: it's your own cue. "
            f"Bet #{b['id']} can be settled now: {_describe(b, now)}, made "
            f"{_human_when(b['made'], now)}. Bring it up in one short line, in character: if it's a "
            "public result (a game score, a release date) you may check it with search; otherwise ask "
            "the room how it turned out. When you know, settle it with settle_bet. Don't [HOLD], and "
            "never invent the result.")


def cue_note(text: str) -> str:
    if BET_CUE_RE.search(text or ""):
        return ("BET: this turn is your own cue that a logged bet is due. Bring it up now in one short "
                "line, in character, to the whole room. Name both people by the names in the cue "
                "(e.g. 'Riley, your bet with Sam...'), never 'you' unless the cue says so. "
                "Never invent who won. Do not [HOLD].")
    return ""


# ------------------------------------------------------------------ intent (prefetch)
# Only unmistakable lines: a bet with real stakes ("I bet you five bucks the Lakers win"),
# or a request to see the bets. "I bet you're tired" has no stakes and is left alone.
_STAKES = (r"(?:\$\s?\d+(?:\.\d\d)?|\d+\s*(?:bucks|dollars|quid|euros|pounds)|"
           r"(?:a|one|two|three|five|ten|twenty|fifty|a\s+hundred)\s+(?:bucks|dollars|quid|euros|pounds)|"
           r"(?:a|an|the)\s+(?:pizza|beer|coffee|drink|round|dinner|lunch|taco|soda|cookie|burger)s?)")
_BET_RE = re.compile(r"(?i)\b(?P<who>I|I'll|I will)\s+(?:bet|wager)\s+"
                     r"(?:(?P<vs>you(?:\s+guys)?|(?-i:[A-Z][a-z]+))\s+)?"
                     r"(?P<stakes>" + _STAKES + r")\s+(?:that\s+)?(?P<claim>[^.?!]{4,160})")
_LIST_RE = re.compile(r"(?i)\b(?:what(?:'s| are| is)?\s+(?:the\s+)?(?:open\s+)?bets|open\s+bets|"
                      r"bet(?:ting)?\s+(?:scoreboard|leaderboard|record|tally)|who(?:'s| is)\s+winning\s+"
                      r"(?:the\s+)?bets|list\s+(?:the\s+|our\s+)?bets)\b")
_WHEN_RE = re.compile(r"(?i)\b(tonight|today|tomorrow|this\s+(?:weekend|week|month|year|game|season)|"
                      r"next\s+week|by\s+(?:the\s+)?end\s+of\s+(?:the\s+)?(?:week|month|year|season)|"
                      r"(?:on|by)\s+(?:" + "|".join(calendar.day_name) + r")|"
                      r"in\s+\w+\s+(?:days?|weeks?|months?))\b")


# "I won the bet", "Riley lost that bet": settle by who won; ATLAS works out the side.
_SETTLE_RE = re.compile(r"(?i)\b(?:(?P<me>I)(?:'ve|\s+have|\s+just)?|(?P<name>(?-i:[A-Z][a-z]+)))\s+"
                        r"(?P<verb>won|lost)\s+(?:the|that|our|my|this|his|her|their)\s+bet\b")
_NOT_NAME = {"I", "We", "You", "They", "He", "She", "It", "So", "And", "But", "Then", "Who"}


def intent(text: str) -> Optional[tuple]:
    t = text or ""
    if _LIST_RE.search(t):
        return ("list_bets", {})
    m = _SETTLE_RE.search(t)
    if m and (m["me"] or m["name"] not in _NOT_NAME):
        side = "winner" if m["verb"].lower().startswith("w") else "loser"
        return ("settle_bet", {side: "@speaker" if m["me"] else m["name"],
                               "note": re.sub(r"\s+", " ", t)[:120]})
    m = _BET_RE.search(t)
    if m:
        claim = m["claim"].strip(" ,.")
        wm = _WHEN_RE.search(claim)
        when = wm[1] if wm else ""
        vs = m["vs"] or ""
        args = {"claim": claim, "stakes": m["stakes"], "settle_by": when}
        if vs and not vs.lower().startswith("you") and vs not in _NOT_NAME:
            args["against"] = vs
        return ("record_bet", args)
    return None
