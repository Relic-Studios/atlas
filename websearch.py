"""Web search backend for the agent.

Keyed search APIs only: Exa, then Brave (each used only if a key is present:
ATLAS_EXA_KEY / ATLAS_BRAVE_KEY, or private|user/<name>.key). One HTTPS call per
search to the provider's API.

Owner 10-06: no scraping from the user's PC. The old keyless fallbacks (ddgs, which
raced Brave/DuckDuckGo/Google HTML in parallel, and a local SearXNG, which scrapes
engines the same way) got users rate-limited and flagged by network security.
With no key, web search is simply unavailable: the tool is hidden from the agent
and the Plugins page says a key is needed.
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)
_YEAR = re.compile(r"\b(20[0-9]{2})\b")


def freshen_query(query: str, user_text: str = "", today=None) -> str:
    """Live call 10-05: the model searched 'China fusion research progress 2025' in 2026
    (its training cutoff leaks into queries), so results came back a year stale.
    A RECENT past year (last 3) that nobody in the room said is dropped, so the engine
    returns current results. Older years and years people actually said are kept
    ('World Cup 2022 winner' still works)."""
    import datetime as _dt
    year = (today or _dt.date.today()).year
    said = set(_YEAR.findall(user_text or ""))
    def fix(m):
        y = m.group(1)
        return "" if (year - 3 <= int(y) < year and y not in said) else y
    out = " ".join(_YEAR.sub(fix, query or "").split())
    return out or (query or "")


# ---------------------------------------------------------------- keyed providers
# Owner 10-05: "something PERFECT". Chain = Exa -> Brave (keyed APIs only); the first
# that returns results wins. Keys are never bundled: env var, or a one-line file in
# private/ (dev) or user/ (public installs). No key -> that provider is skipped.
import json as _json
import os as _os
import time as _time
import urllib.parse as _uparse
import urllib.request as _ureq
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
PROVIDER_TIMEOUT_S = 6.0
PREFERRED = "auto"   # plugins.py: auto | exa | brave
MAX_RESULTS = 4


def _key(name: str) -> str:
    env = _os.environ.get(f"ATLAS_{name.upper()}_KEY", "").strip()
    if env:
        return env
    for d in ("user", "private"):   # user/ first: keys saved from the Plugins page win
        f = _HERE / d / f"{name}.key"
        try:
            k = f.read_text(encoding="utf-8-sig").strip()
            if k:
                return k
        except OSError:
            pass
    return ""


_TAGS = re.compile(r"<[^>]+>")


def _search_exa(key: str, query: str, max_results: int) -> List[Dict[str, str]]:
    body = {"query": query, "numResults": max_results, "type": "fast",
            "contents": {"highlights": {"numSentences": 3, "highlightsPerUrl": 1}}}
    req = _ureq.Request("https://api.exa.ai/search", data=_json.dumps(body).encode(),
                        headers={"x-api-key": key, "Content-Type": "application/json"})
    d = _json.load(_ureq.urlopen(req, timeout=PROVIDER_TIMEOUT_S))
    out = []
    for r in d.get("results", [])[:max_results]:
        hl = " ".join(r.get("highlights") or []) or (r.get("text") or "")[:400]
        date = (r.get("publishedDate") or "")[:10]
        out.append({"title": r.get("title") or "", "url": r.get("url") or "",
                    "snippet": (f"[{date}] " if date else "") + " ".join(hl.split())[:500]})
    return out


def _search_brave(key: str, query: str, max_results: int) -> List[Dict[str, str]]:
    url = "https://api.search.brave.com/res/v1/web/search?" + _uparse.urlencode(
        {"q": query, "count": max_results})
    req = _ureq.Request(url, headers={"X-Subscription-Token": key, "Accept": "application/json"})
    d = _json.load(_ureq.urlopen(req, timeout=PROVIDER_TIMEOUT_S))
    out = []
    for r in d.get("web", {}).get("results", [])[:max_results]:
        age = r.get("age") or ""
        out.append({"title": _TAGS.sub("", r.get("title") or ""), "url": r.get("url") or "",
                    "snippet": (f"[{age}] " if age else "") + _TAGS.sub("", r.get("description") or "")})
    return out


def providers() -> List[str]:
    """Which providers would be tried, in order (for status/telemetry)."""
    allowed = {"auto": ("exa", "brave"), "exa": ("exa",), "brave": ("brave",)}.get(PREFERRED, ("exa", "brave"))
    return [n for n in allowed if _key(n)]


def available() -> bool:
    """True when at least one keyed provider is configured (no scraping fallback)."""
    return bool(providers())


NO_KEY_TEXT = ("Web search isn't set up: it needs an Exa or Brave API key "
               "(Plugins > Web search). Tell the person that plainly.")


def _run_chain(query: str, max_results: int):
    tried = []
    for name in providers():
        k = _key(name)
        t = _time.time()
        try:
            fn = _search_exa if name == "exa" else _search_brave
            res = fn(k, query, max_results)
            ms = (_time.time() - t) * 1000
            if res:
                logger.info("web search %r via %s: %d results in %.0f ms", query, name, len(res), ms)
                return res, name
            tried.append(f"{name}:empty")
        except Exception as e:  # noqa: BLE001  network/quota/auth -> next provider
            code = getattr(e, "code", "")
            tried.append(f"{name}:{code or type(e).__name__}")
            logger.warning("web search via %s failed (%s), trying next", name, code or e)
    if tried:
        logger.info("web search %r: no provider answered (%s)", query, ", ".join(tried))
    return [], (tried[-1].split(":")[0] if tried else "none")


def search(query: str, max_results: int = 4) -> Dict:
    """Run a web search and return a normalized result.

    Returns {"ok": bool, "query": str, "results": [...], "text": str}.
    `text` is a compact dump of the top results for LLM context injection.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "query": query, "results": [], "text": ""}

    if not available():
        return {"ok": False, "query": query, "results": [], "text": NO_KEY_TEXT, "backend": "none"}
    try:
        results, backend = _run_chain(query, max_results if max_results != 4 else MAX_RESULTS)
    except Exception as e:  # noqa: BLE001
        logger.warning("web search failed: %s", e)
        return {"ok": False, "query": query, "results": [], "text": f"search error: {e}"}

    if not results:
        return {"ok": True, "query": query, "results": [], "text": "No results found."}

    # Outside text: neutralize prompt injection before any model sees it.
    import untrusted
    flagged = 0
    for r in results:
        r["title"], f1 = untrusted.sanitize(r.get("title", ""), 160)
        r["snippet"], f2 = untrusted.sanitize(r.get("snippet", ""), 500)
        r["url"] = "".join(ch for ch in str(r.get("url", "")) if ch.isprintable() and ch not in " <>[]")[:200]
        flagged += f1 + f2
    if flagged:
        logger.warning("web search %r: neutralized %d injection/control fragment(s)", query, flagged)
    import datetime as _dt
    lines = [f"Search for '{query}' ({backend}), run just now on {_dt.date.today().isoformat()}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"({i}) {r['title']}\n    {r['url']}\n    {r['snippet']}")
    text = untrusted.wrap("\n".join(lines), "web search")

    return {"ok": True, "query": query, "results": results, "text": text, "backend": backend}


PAGE_CHARS = 1800


def read_page(url: str, max_chars: int = PAGE_CHARS) -> Dict:
    """Fetch one page and return its main text (read-only, bounded).

    Returns {"ok", "url", "title", "text"}. Only http(s); never follows to
    local/private hosts.
    """
    import ipaddress, socket, urllib.parse
    url = (url or "").strip()
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return {"ok": False, "url": url, "title": "", "text": "not a web URL"}
    try:
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return {"ok": False, "url": url, "title": "", "text": "refusing local address"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "url": url, "title": "", "text": f"lookup failed: {e}"}
    try:
        import requests, trafilatura
        r = requests.get(url, timeout=(3.0, 5.0), headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/128 Safari/537.36"})
        raw = r.text if r.ok and "html" in r.headers.get("content-type", "html") else ""
        if not raw:
            return {"ok": False, "url": url, "title": "", "text": "page didn't load"}
        text = trafilatura.extract(raw, include_comments=False, include_tables=False) or ""
        meta = trafilatura.extract_metadata(raw)
        title = (getattr(meta, "title", "") or "") if meta else ""
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "url": url, "title": "", "text": f"fetch failed: {e}"}
    text = " ".join(text.split())
    if not text:
        return {"ok": False, "url": url, "title": title, "text": "no readable text on that page"}
    import untrusted
    title, f1 = untrusted.sanitize(title, 160)
    text, f2 = untrusted.sanitize(text, max_chars)
    if f1 + f2:
        logger.warning("read_page %s: neutralized %d injection/control fragment(s)", url, f1 + f2)
    return {"ok": True, "url": url, "title": title, "text": text}


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(search("current weather in Tokyo", 3), indent=2, ensure_ascii=False))
