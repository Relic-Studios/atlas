"""Web search backend for the agent.

Provider chain (first with results wins): Exa -> Brave (each only if a key is
present: ATLAS_EXA_KEY / ATLAS_BRAVE_KEY, or private|user/<name>.key) -> then the
two keyless backends below, picked automatically:
- SearXNG (self-hosted metasearch, zero per-call fee, nothing leaves the box) —
  preferred when reachable, since it keeps queries off third-party services.
- DuckDuckGo via `ddgs` (already installed, no API key) — fallback so the agent
  can search on first boot without standing up Docker.

Both return a uniform list of {title, url, snippet} plus a short combined text
suitable for injecting into the LLM context.
"""
from __future__ import annotations

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)
for _noisy in ("ddgs", "primp", "ddgs.ddgs", "ddgs.base"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# SearXNG endpoint candidates (docker default; also common host-mapped port).
SEARXNG_URLS = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8888",
]


_SEARX_CACHE = {"t": 0.0, "base": None}


def _searxng_available() -> str | None:
    """Probe SearXNG at most once a minute (a dead probe used to cost ~4.5s per search)."""
    import time
    import urllib.request
    now = time.monotonic()
    if now - _SEARX_CACHE["t"] < 60.0:
        return _SEARX_CACHE["base"]
    found = None
    for base in SEARXNG_URLS:
        try:
            urllib.request.urlopen(f"{base}/config", timeout=0.3).close()
            found = base
            break
        except Exception:
            continue
    _SEARX_CACHE.update(t=now, base=found)
    return found


def _search_searxng(base: str, query: str, max_results: int = 5) -> List[Dict[str, str]]:
    import urllib.parse
    import urllib.request
    import json
    url = f"{base}/search?q={urllib.parse.quote(query)}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "atlas-agent/1.0"})
    with urllib.request.urlopen(req, timeout=10.0) as r:
        data = json.loads(r.read().decode("utf-8"))
    out = []
    for res in data.get("results", [])[:max_results]:
        out.append({
            "title": res.get("title", ""),
            "url": res.get("url", ""),
            "snippet": (res.get("content") or "")[:500],
        })
    return out


def _ddgs_backend(query: str, backend: str, max_results: int) -> List[Dict[str, str]]:
    from ddgs import DDGS
    rows = DDGS(timeout=6).text(query, max_results=max_results, backend=backend)
    return [{
        "title": r.get("title", ""),
        "url": r.get("href", r.get("url", "")),
        "snippet": (r.get("body", "") or "")[:400],
    } for r in rows or []]


# Backends race in parallel; the first non-empty answer wins. Sequential
# fallback cost up to ~15s live (brave empty -> auto multi-engine).
RACE_BACKENDS = ("brave", "duckduckgo", "google", "auto")
RACE_TIMEOUT_S = 9.0


def _search_ddgs(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    import concurrent.futures as cf
    pool = cf.ThreadPoolExecutor(max_workers=len(RACE_BACKENDS), thread_name_prefix="search")
    futs = {pool.submit(_ddgs_backend, query, be, max_results): be for be in RACE_BACKENDS}
    last_err = None
    try:
        for fut in cf.as_completed(futs, timeout=RACE_TIMEOUT_S):
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if out:
                logger.info("web search: %s won the race", futs[fut])
                return out
    except cf.TimeoutError:
        last_err = last_err or TimeoutError(f"no backend answered in {RACE_TIMEOUT_S}s")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if last_err:
        raise last_err
    return []


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
# Owner 10-05: "something PERFECT". Chain = Exa -> Brave -> SearXNG -> ddgs; the first
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


def _key(name: str) -> str:
    env = _os.environ.get(f"ATLAS_{name.upper()}_KEY", "").strip()
    if env:
        return env
    for d in ("private", "user"):
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
    order = [n for n in ("exa", "brave") if _key(n)]
    if _searxng_available():
        order.append("searxng")
    order.append("ddgs")
    return order


def _run_chain(query: str, max_results: int):
    tried = []
    for name in ("exa", "brave"):
        k = _key(name)
        if not k:
            continue
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
            logger.warning("web search via %s failed (%s), falling back", name, code or e)
    base = _searxng_available()
    if base:
        try:
            res = _search_searxng(base, query, max_results)
            if res:
                return res, "searxng"
        except Exception as e:  # noqa: BLE001
            tried.append(f"searxng:{type(e).__name__}")
    if tried:
        logger.info("web search %r: keyed providers unavailable (%s), using ddgs", query, ", ".join(tried))
    return _search_ddgs(query, max_results), "ddgs"


def search(query: str, max_results: int = 4) -> Dict:
    """Run a web search and return a normalized result.

    Returns {"ok": bool, "query": str, "results": [...], "text": str}.
    `text` is a compact dump of the top results for LLM context injection.
    """
    query = (query or "").strip()
    if not query:
        return {"ok": False, "query": query, "results": [], "text": ""}

    try:
        results, backend = _run_chain(query, max_results)
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
