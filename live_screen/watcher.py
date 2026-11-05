"""The agents' one set of eyes for the screen log (docs/designs/live_screen.md, section 6; owner, 2026-11-03).

By default the eyes are Model itself: a log look sends one frame to Model's server with a plain "report only
what is visible" instruction (no persona), temperature 0, and gets JSON per SCHEMA. With ATLAS_WATCHER_PORT set,
the same requests go to a separate watcher server on that port (same host and API key), and `answer` also serves
screen turns. Never raises: a failed, slow or cancelled look is just skipped."""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from live_screen import config as CFG

logger = logging.getLogger(__name__)

SYSTEM = ("You describe a single screenshot of a computer screen (usually a game) for a log. You are not a "
          "character and you don't chat: report only what is visible; when unsure, set unclear to true. Never guess "
          "names or numbers you cannot read. Keep every field short.")
SCHEMA = {
    "type": "object",
    "properties": {
        "scene_kind": {"enum": ["gameplay", "loading", "menu", "map", "inventory", "death", "cutscene", "dialogue",
                                "desktop", "other"]},
        "scene": {"type": "string", "maxLength": 160},
        "events": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 100}},
        "entities": {"type": "array", "maxItems": 5, "items": {
            "type": "object",
            "properties": {"kind": {"enum": ["item", "person", "place", "objective"]},
                           "name": {"type": "string", "maxLength": 50},
                           "seen": {"type": "string", "maxLength": 80}},
            "required": ["kind", "name", "seen"]}},
        "text": {"type": "array", "maxItems": 2, "items": {"type": "string", "maxLength": 160}},
        "unclear": {"type": "boolean"},
    },
    "required": ["scene_kind", "scene", "events", "entities", "text", "unclear"],
}
# A separate watcher answering screen turns (only with ATLAS_WATCHER_PORT): the agent answers from its report.
ANSWER_SYSTEM = ("You are the eyes of a voice assistant in a call. You can see the frames from the user's screen; "
                 "the assistant cannot. Report what is visible that answers the question in one to three short, "
                 "plain sentences of fact: what is there, and any readable text or numbers exactly as shown. Never "
                 "guess or invent; if the frames don't show it or it's unclear, say so. With several frames (oldest "
                 "first), say what changed between them.")

stats = {"looks": 0, "ok": 0, "errors": 0, "cancelled": 0, "ms": [], "answers": 0, "answer_errors": 0,
         "answer_ms": []}
_session = {"http": None, "base": None, "model": None, "checked": 0.0}
_lock = threading.Lock()


def _endpoint(remote_cfg: dict):
    """(base url, key, model or None): Model's own server, or the separate watcher's port on the same host."""
    cfg = remote_cfg or {}
    url = (cfg.get("url") or "").rstrip("/")
    if not url:
        return None, None, None
    if not CFG.SEPARATE_WATCHER:
        return url, cfg.get("key"), cfg.get("model")
    p = urlsplit(url)
    return (urlunsplit((p.scheme, f"{p.hostname}:{CFG.WATCHER_PORT}", p.path, "", "")).rstrip("/"),
            cfg.get("key"), None)


def _client(remote_cfg: dict):
    """(session, base url, model id) or (None, None, None); a separate watcher is re-checked every 30 s."""
    with _lock:
        if _session["model"] and _session["http"] is not None:
            return _session["http"], _session["base"], _session["model"]
        if time.time() - _session["checked"] < 30.0:
            return None, None, None
        _session["checked"] = time.time()
        base, key, model = _endpoint(remote_cfg)
        if not base:
            return None, None, None
        import requests
        http = requests.Session()
        if key:
            http.headers["Authorization"] = "Bearer " + key
        if model is None:
            try:
                model = http.get(base + "/v1/models", timeout=(1.0, 2.0)).json()["data"][0]["id"]
            except Exception as e:  # noqa: BLE001
                logger.debug("watcher not reachable: %s", e)
                return None, None, None
            logger.info("👁️ separate watcher found: %s at %s", model, base)
        _session.update(http=http, base=base, model=model)
        return http, base, model


def reset() -> None:
    with _lock:
        _session.update(http=None, base=None, model=None, checked=0.0)


def _image(jpeg: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")}}


def look(jpeg: bytes, title: str, remote_cfg: dict, cancel: threading.Event | None = None) -> dict | None:
    """One stateless log look: the frame plus the window title; JSON per SCHEMA, or None. Streamed so it can be
    cancelled (closing the request makes the server drop it) the moment someone stops talking."""
    http, base, model = _client(remote_cfg)
    if http is None:
        return None
    body = {"model": model, "temperature": 0, "max_tokens": CFG.WATCHER_MAX_TOKENS, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "look", "schema": SCHEMA}},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [{"type": "text", "text": f"Window in front: {title or 'unknown'}."},
                                                      _image(jpeg)]}]}
    stats["looks"] += 1
    t0 = time.perf_counter()
    resp = None
    try:
        resp = http.post(base + "/v1/chat/completions", json=body, stream=True, timeout=(1.0, CFG.WATCHER_TIMEOUT_S))
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        parts = []
        for raw in resp.iter_lines():
            if cancel is not None and cancel.is_set():
                stats["cancelled"] += 1
                return None
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw or "")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                continue
            parts.append(delta.get("content") or "")
        out = json.loads("".join(parts))
        stats["ok"] += 1
        return out if isinstance(out, dict) else None
    except Exception as e:  # noqa: BLE001
        stats["errors"] += 1
        logger.debug("log look failed: %s", e)
        if "Connection" in type(e).__name__:
            reset()
        return None
    finally:
        if resp is not None:
            resp.close()
        stats["ms"] = (stats["ms"] + [round((time.perf_counter() - t0) * 1000)])[-200:]


def answer(jpegs: list, question: str, title: str, ages: list | None, remote_cfg: dict,
           glance: bool = False) -> str | None:
    """Separate watcher only: a look on the reply path; a short factual report, or None if it can't answer in time
    (then the agent says it can't see right now)."""
    http, base, model = _client(remote_cfg)
    if http is None or not jpegs:
        return None
    ages = list(ages or [0.0] * len(jpegs))
    when = ("" if len(jpegs) == 1 else " The frames are oldest first: " + "; ".join(
        "just now" if a < 1.0 else f"about {round(a)} s before" for a in ages) + ".")
    ask = (f"Someone in the call just said: \"{question.strip()}\". Describe briefly what on the screen fits what "
           "they said, if anything." if glance else f"Question from the call: \"{question.strip()}\"")
    content = [{"type": "text", "text": f"Window in front: {title or 'unknown'}.{when} {ask}"}] + [_image(j) for j in jpegs]
    body = {"model": model, "temperature": 0, "max_tokens": CFG.ANSWER_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": content}]}
    stats["answers"] += 1
    t0 = time.perf_counter()
    try:
        r = http.post(base + "/v1/chat/completions", json=body, timeout=(1.0, CFG.ANSWER_TIMEOUT_S))
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"]["content"] or "").strip() or None
    except Exception as e:  # noqa: BLE001
        stats["answer_errors"] += 1
        logger.warning("👁️ watcher couldn't answer: %s", str(e)[:160])
        if "Connection" in type(e).__name__:
            reset()
        return None
    finally:
        stats["answer_ms"] = (stats["answer_ms"] + [round((time.perf_counter() - t0) * 1000)])[-200:]
