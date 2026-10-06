"""Game info plugin: "what's the player count on Helldivers 2?" / "is Elden Ring on sale?".

Uses Steam's public, keyless endpoints (store search, app details, current
players). Only the game name leaves this PC. Store text is third-party, so it is
sanitized and wrapped as untrusted data before the model sees it.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from typing import Callable, Optional

_UA = {"User-Agent": "ATLAS-voice-agent/0.3 (+https://github.com/Relic-Studios/atlas)"}
_CACHE: dict = {}
CACHE_S = 120.0


def _get_json(url: str, timeout: float = 6.0):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (fixed https hosts)
        return json.loads(r.read().decode("utf-8", "replace"))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower().replace("™", "").replace("®", "")).strip()


def _pick(items, query: str):
    """Best store hit: exact name, then base game over DLC/soundtracks, then first."""
    q = _norm(query)
    apps = [i for i in items if i.get("type") == "app"]
    for i in apps:
        if _norm(i.get("name")) == q:
            return i
    bad = re.compile(r"\b(soundtrack|dlc|pack|bundle|demo|set|season pass|ost|artbook|skin)\b", re.I)
    clean = [i for i in apps if not bad.search(i.get("name", ""))]
    return (clean or apps or [None])[0]


def _money(cents, cur: str) -> str:
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "CAD": "CA$", "AUD": "A$", "JPY": "¥"}.get(cur, cur + " ")
    if cur == "JPY":
        return f"{sym}{int(cents) // 100:,}"
    return f"{sym}{int(cents) / 100:,.2f}"


def lookup(game: str, cc: str = "us", fetch: Callable = _get_json, clock: Callable[[], float] = time.time) -> str:
    game = (game or "").strip(" .?!\"'")[:80]
    if not game:
        return "Which game? I need a name to look it up."
    cc = (cc or "us").lower()[:2]
    key = (_norm(game), cc)
    hit = _CACHE.get(key)
    if hit and clock() - hit[0] < CACHE_S:
        return hit[1]
    try:
        res = fetch("https://store.steampowered.com/api/storesearch/?"
                    + urllib.parse.urlencode({"term": game, "cc": cc, "l": "en"}))
    except Exception as e:  # noqa: BLE001
        return f"Couldn't reach Steam right now ({type(e).__name__}). Say you couldn't check, don't guess."
    item = _pick((res or {}).get("items") or [], game)
    if not item:
        return f"No game called '{game}' on Steam. Say you couldn't find it; don't guess numbers."
    appid = item["id"]
    name = item.get("name", game).replace("™", "").replace("®", "")
    parts = [f"{name} (Steam)"]

    try:
        d = fetch(f"https://store.steampowered.com/api/appdetails?appids={appid}&cc={cc}&l=en")
        data = ((d or {}).get(str(appid)) or {}).get("data") or {}
    except Exception:  # noqa: BLE001
        data = {}
    if data.get("is_free"):
        parts.append("free to play")
    else:
        po = data.get("price_overview") or {}
        if not po and isinstance(item.get("price"), dict):
            p = item["price"]
            po = {"initial": p.get("initial"), "final": p.get("final"), "currency": p.get("currency", "USD"),
                  "discount_percent": round(100 * (1 - p["final"] / p["initial"])) if p.get("initial") else 0}
        if po.get("final") is not None:
            cur = po.get("currency", "USD")
            disc = int(po.get("discount_percent") or 0)
            if disc > 0:
                parts.append(f"ON SALE: {disc}% off, {_money(po['final'], cur)} "
                             f"(normally {_money(po['initial'], cur)})")
            else:
                parts.append(f"not on sale, {_money(po['final'], cur)}")
        elif data.get("release_date", {}).get("coming_soon"):
            parts.append("not released yet")
    try:
        pc = fetch("https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid="
                   + str(appid))
        n = ((pc or {}).get("response") or {}).get("player_count")
        if isinstance(n, int):
            parts.append(f"{n:,} people playing on Steam right now")
    except Exception:  # noqa: BLE001
        pass
    rdo = data.get("release_date") or {}
    rd = rdo.get("date")
    if rd:
        parts.append(f"release date: {rd}" if rdo.get("coming_soon") else f"released {rd}")
    mc = (data.get("metacritic") or {}).get("score")
    if mc:
        parts.append(f"Metacritic {mc}")
    desc = data.get("short_description") or ""
    out = "; ".join(parts) + "."
    if desc:
        try:
            import untrusted
            clean, _ = untrusted.sanitize(re.sub(r"<[^>]+>", " ", desc), limit=220)
            out += "\n" + untrusted.wrap(clean, "steam store")
        except Exception:  # noqa: BLE001
            pass
    out += ("\nThese are live Steam numbers fetched just now (Steam players only, not other platforms). "
            "Answer briefly in your own words.")
    _CACHE[key] = (clock(), out)
    if len(_CACHE) > 200:
        _CACHE.clear()
    return out


# ------------------------------------------------------------------ intent
_GAME_TAIL = r"(?:for|on|of|in)\s+(.{2,60}?)\s*(?:on\s+steam\s*)?(?:right\s+now|today|currently)?\s*[.!?]*$"
_PLAYERS_RE = re.compile(
    r"\b(?:player\s*count|how\s+many\s+players\s+(?:are\s+)?(?:playing|on|in)|"
    r"how\s+many\s+people\s+(?:are\s+)?(?:still\s+)?play(?:ing)?|"
    r"concurrent\s+players|people\s+playing)\b", re.I)
_SALE_RE = re.compile(r"\b(?:is|are)\s+(.{2,60}?)\s+(?:on\s+sale|discounted|free\s+to\s+play)\b", re.I)
_PRICE_RE = re.compile(r"\bhow\s+much\s+(?:is|does|are)\s+(.{2,60}?)\s*(?:cost|on\s+steam|go\s+for)?\s*[.!?]*$",
                       re.I)
_NOT_GAMES = {"it", "that", "this", "steam", "call", "the call", "chat", "here", "server", "discord", "room",
              "vc", "voice", "lobby", "party", "now"}
_STEAM = re.compile(r"\bsteam\b", re.I)


def _clean_game(s: str) -> str:
    s = re.sub(r"(?i)\s+(?:on\s+steam|right\s+now|today|currently|these\s+days)\b.*$", "", s or "")
    s = re.sub(r"(?i)^(?:the\s+game\s+|the\s+)", "", s.strip(" ,.?!\"'"))
    return s.strip(" ,.?!\"'")


def intent(text: str) -> Optional[tuple]:
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "")
    t = re.sub(r"^[A-Z][a-z]+,\s*", "", t.strip())   # leading vocative ("Fae, ...")
    if _PLAYERS_RE.search(t):
        m = re.search(_GAME_TAIL, t, re.I) or re.search(r"\bplaying\s+(.{2,60}?)\s*[.!?]*$", t, re.I)
        if m:
            g = _clean_game(m[1])
            if g and g.lower() not in _NOT_GAMES:
                return ("game_info", {"game": g})
        return None
    m = _SALE_RE.search(t)
    if m:
        g = _clean_game(m[1])
        if g and g.lower() not in ("it", "that", "this", "everything", "anything", "stuff"):
            return ("game_info", {"game": g})
    m = _PRICE_RE.search(t)
    if m and _STEAM.search(t):
        g = _clean_game(m[1])
        if g and g.lower() not in ("it", "that", "this"):
            return ("game_info", {"game": g})
    return None
