"""Optional remote LLM (vLLM, OpenAI-compatible) behind the existing Ollama code path.

Self-contained so it is trivial to strip for a public release:
  delete this file + private/remote_llm.json; the hooks in server.py/llm_module.py
  are guarded by `import remote_llm` failing and fall back to local Ollama.

How it works: LLM keeps talking "Ollama" (payload + NDJSON stream). When remote is
active, LLM.ollama_session is swapped for RemoteSession, which translates the
Ollama /api/chat payload to OpenAI /v1/chat/completions, streams SSE back and
re-emits it as Ollama NDJSON. Tool calls, images, cancellation and the
header parser all keep working unchanged. If the remote host is unreachable,
the request transparently falls back to the local Ollama session.

Config: private/remote_llm.json (dev) or the cloud block of user/settings.json
(public, written by first-run setup):
  {"url": "http://<host>:<port>", "model": "<model id>", "key_file": "private/remote_llm.key",
   "label": "<shown in the dashboard>"}
"""
import itertools
from collections import deque
import json
import re
import logging
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("ATLAS_REMOTE_LLM_CONFIG", ROOT / "private" / "remote_llm.json"))
FAIL_COOLDOWN_S = 30.0  # after a connection failure, go straight to local for this long
# Live 11-02: one reply's first token took 42 s with nothing else going on; the 600 s read limit callers pass would
# let a stuck request hold a turn for minutes and then answer something long gone. Each wait for the next bytes from
# the server is capped; a slow server only loses that turn (it is not marked down: see post()).
READ_TIMEOUT_S = float(os.environ.get("ATLAS_LLM_READ_TIMEOUT_S", "12"))


def is_read_timeout(e: BaseException) -> bool:
    """The connection worked but no bytes came in time (requests raises ReadTimeout before the response, and wraps
    urllib3's ReadTimeoutError in a ConnectionError while a stream is being read)."""
    return isinstance(e, requests.ReadTimeout) or "read timed out" in str(e).lower()
WATCH_INTERVAL_S = float(os.environ.get("ATLAS_REMOTE_WATCH_S", "10"))
DOWN_AFTER_FAILS = 2      # consecutive failed pings before we warm the local model
UP_STABLE_S = 60.0        # remote healthy this long before local VRAM is freed again


def load_config():
    """Return the remote config dict, or None if not configured (public build)."""
    # A choice saved in Settings has the same precedence in every build.
    # An invalid selected cloud stays unavailable; never switch to a private host.
    import user_settings
    if user_settings.selected_llm_mode() == "cloud":
        return user_settings.cloud_config()
    selected_cloud = user_settings.cloud_config()
    if selected_cloud is not None:
        return selected_cloud
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Public build: the user's own cloud provider from first-run setup (or none).
        try:
            import user_settings
            return user_settings.cloud_config()
        except Exception:  # noqa: BLE001
            return None
    key = cfg.get("key") or ""
    kf = cfg.get("key_file")
    if not key and kf:
        try:
            key = (ROOT / kf if not os.path.isabs(kf) else Path(kf)).read_text(encoding="utf-8").strip()
        except OSError:
            key = ""
    if not cfg.get("url") or not cfg.get("model"):
        return None
    return {"url": cfg["url"].rstrip("/"), "model": cfg["model"], "key": key,
            "label": cfg.get("label") or cfg["url"], "vllm": bool(cfg.get("vllm", True)),
            "allow_local_fallback": bool(cfg.get("allow_local_fallback", True)),
            "context": int(cfg.get("context") or 8192),
            "vision": cfg.get("vision") if isinstance(cfg.get("vision"), bool) else None}


def remote_selected() -> bool:
    """One startup selector: explicit environment, Settings, then legacy profile."""
    mode = os.environ.get("ATLAS_LLM")
    if mode in ("local", "remote"):
        return mode == "remote"
    import user_settings
    selected = user_settings.selected_llm_mode()
    if selected is not None:
        return selected == "cloud"
    try:
        return (ROOT / "private" / "llm_mode.txt").read_text(encoding="utf-8").strip() == "remote"
    except OSError:
        return False


def restore_at_startup(llm):
    """Restore the selected remote transport before prewarm or latency probes."""
    if not remote_selected():
        return
    cfg = load_config()
    if cfg is None:
        llm._remote_startup_error = "Remote mode selected but no remote configuration is available"
        raise requests.ConnectionError(llm._remote_startup_error)
    activate(llm, cfg)
    logger.info("Remote startup transport selected: %s (%s)", cfg["label"], cfg["model"])


# ---------------------------------------------------------------------------
# Ollama payload  ->  OpenAI request
# ---------------------------------------------------------------------------
def _image_part(b64: str) -> dict:
    mime = "image/png" if b64.startswith("iVBOR") else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def translate_messages(messages):
    """Ollama chat messages -> OpenAI messages (tool call ids, images, arg JSON)."""
    out, pending_ids, n = [], [], 0
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        images = m.get("images") or []
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                cid = tc.get("id") or f"call_{n}"
                n += 1
                pending_ids.append(cid)
                calls.append({"id": cid, "type": "function",
                              "function": {"name": fn.get("name", ""),
                                           "arguments": args if isinstance(args, str) else json.dumps(args)}})
            out.append({"role": "assistant", "content": content, "tool_calls": calls})
            continue
        if role == "tool":
            cid = m.get("tool_call_id") or (pending_ids.pop(0) if pending_ids else f"call_{n}")
            out.append({"role": "tool", "tool_call_id": cid, "content": content})
            if images:
                # Tool messages are text-only in the OpenAI schema: show the image
                # right after, as the tool's attachment.
                out.append({"role": "user", "content": [
                    {"type": "text", "text": "(image returned by the tool above)"}] +
                    [_image_part(b) for b in images]})
            continue
        if images:
            out.append({"role": role, "content": [{"type": "text", "text": content}] +
                        [_image_part(b) for b in images]})
        else:
            out.append({"role": role, "content": content})
    return out


def translate_payload(payload: dict, model: str, vllm: bool = True) -> dict:
    """vllm=False: strict OpenAI-compatible body for cloud providers (no
    chat_template_kwargs / top_k / stream_options, which some reject)."""
    opts = payload.get("options") or {}
    body = {"model": model, "messages": translate_messages(payload.get("messages") or []),
            "stream": bool(payload.get("stream", True))}
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                     ("num_predict", "max_tokens"), ("stop", "stop")):
        if opts.get(src) is not None:
            body[dst] = opts[src]
    if body.get("max_tokens", 1) is not None and body.get("max_tokens", 1) < 0:
        body.pop("max_tokens")
    if payload.get("tools"):
        body["tools"] = payload["tools"]
    fmt = payload.get("format")
    if not vllm:
        body.pop("top_k", None)
        if fmt == "json" or isinstance(fmt, dict):     # a schema is only asked for, never enforced, by cloud providers
            body["response_format"] = {"type": "json_object"}
        return body
    think = payload.get("think")
    ctk = dict(payload.get("chat_template_kwargs") or {})
    if think is False:
        ctk["enable_thinking"] = False
    if ctk:
        body["chat_template_kwargs"] = ctk
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):                        # a JSON schema: vLLM's guided decoding (experience memory, S2)
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "record", "schema": fmt}}
    if body["stream"]:
        body["stream_options"] = {"include_usage": False}
    return body


def provider_request_body(body: dict, endpoint: str, *, think=None) -> dict:
    """Apply a provider's documented wire option, never a model/prompt heuristic.

    DeepSeek enables thinking by default. Atlas's explicit think=False must
    disable it through its Chat Completions field, including tool rounds.
    Other hosts retain the strict OpenAI-compatible body unchanged.
    """
    if think is False and urlsplit(endpoint).hostname == "api.deepseek.com":
        return {**body, "thinking": {"type": "disabled"}}
    return body


# ---------------------------------------------------------------------------
# OpenAI SSE stream  ->  Ollama NDJSON response
# ---------------------------------------------------------------------------
class NdjsonResponse:
    """Looks enough like requests.Response for llm_module's Ollama readers."""

    def __init__(self, resp: requests.Response, model: str, lines=None, entry=None):
        self._resp = resp
        self._model = model
        self.status_code = resp.status_code
        # lines: an already-started line iterator (think_listen reads ahead to the first visible
        # text and hands the rest over here); entry: a TURN_LOG entry that gets reply_cap.
        self._lines = lines
        self._entry = entry

    def raise_for_status(self):
        self._resp.raise_for_status()

    def close(self):
        self._resp.close()

    def _line(self, obj) -> bytes:
        obj.setdefault("model", self._model)
        return (json.dumps(obj) + "\n").encode("utf-8")

    def iter_content(self, chunk_size=None):
        calls = {}  # index -> {"name":..., "arguments": str}
        done_reason = "stop"
        lines = self._lines if self._lines is not None else self._resp.iter_lines(decode_unicode=False)
        for raw in lines:
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("error"):
                yield self._line({"error": str(ev["error"])})
                return
            for ch in ev.get("choices") or []:
                delta = ch.get("delta")
                # chat chunks carry delta.content; /v1/completions chunks (think_listen) carry text
                c = delta.get("content") if delta is not None else ch.get("text")
                delta = delta or {}
                if c:
                    yield self._line({"message": {"role": "assistant", "content": c}, "done": False})
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if ch.get("finish_reason"):
                    done_reason = ch["finish_reason"]
                    if done_reason == "length" and self._entry is not None:
                        self._entry["reply_cap"] = True
                    if ch.get("stop_reason") == "<tool_call>":
                        logger.info("🧠💭 think reply stopped at a tool call; the call is dropped")
        if calls:
            out = []
            for _, slot in sorted(calls.items()):
                try:
                    args = json.loads(slot["arguments"] or "{}")
                except ValueError:
                    args = {}
                out.append({"function": {"name": slot["name"], "arguments": args}})
            yield self._line({"message": {"role": "assistant", "content": "", "tool_calls": out}, "done": False})
        yield self._line({"message": {"role": "assistant", "content": ""}, "done": True,
                          "done_reason": done_reason})


class _SyntheticNdjson:
    """Pre-decided Ollama NDJSON (used for greedy [HOLD] / greedy tool calls)."""

    status_code = 200

    def __init__(self, model, content="", tool_calls=None):
        self._model, self._content, self._calls = model, content, tool_calls or []

    def raise_for_status(self):
        pass

    def close(self):
        pass

    def iter_content(self, chunk_size=None):
        if self._content:
            yield (json.dumps({"model": self._model, "message": {"role": "assistant", "content": self._content},
                               "done": False}) + "\n").encode("utf-8")
        if self._calls:
            yield (json.dumps({"model": self._model, "message": {"role": "assistant", "content": "",
                               "tool_calls": self._calls}, "done": False}) + "\n").encode("utf-8")
        yield (json.dumps({"model": self._model, "message": {"role": "assistant", "content": ""},
                           "done": True, "done_reason": "stop"}) + "\n").encode("utf-8")


class _PrefixedNdjson(NdjsonResponse):
    """Stage-2 stream: emit the greedy header first, then the sampled body."""

    # Stage 1 already chose SPEAK over any tool call, so this round is speech: the
    # caller may voice it as it streams instead of buffering it whole.
    speak_body = True

    def __init__(self, resp, model, prefix, lines=None, entry=None):
        super().__init__(resp, model, lines=lines, entry=entry)
        self._prefix = prefix

    def iter_content(self, chunk_size=None):
        yield self._line({"message": {"role": "assistant", "content": self._prefix}, "done": False})
        yield from super().iter_content(chunk_size)


# Two-stage sampling: the speak/hold header (and any tool decision) is taken at
# temperature 0; only the spoken body is sampled. Live remote at 0.6 butted into
# quiet-room muttering ~7/15 vs ~1/15 greedy (tests/sim_passivity.py 09-30).
TWO_STAGE = os.environ.get("ATLAS_TWO_STAGE", "1") != "0"
# Today's path in one request (RemoteSession._one_request); ATLAS_ONE_REQUEST=0 restores two requests.
ONE_REQUEST = os.environ.get("ATLAS_ONE_REQUEST", "1") != "0"
# A sampled header token after the first within this many nats of the top choice counts as the top
# choice: a near tie (e.g. "to=S1" vs "to=S2" at 0.50/0.50) flips with the server's prefix-cache noise
# (~1 logit) anyway, and falling back for it cost ~0.3 s on 3 of 20 bench lines (11-03). The first token
# (SPEAK vs HOLD) must be the exact top choice.
ONE_REQUEST_TIE = float(os.environ.get("ATLAS_ONE_REQUEST_TIE", "0.2"))
_PROTOCOL_MARK = "output only [HOLD]"
# Calibrated hold: the remote quant's greedy header is ~always [SPEAK] (live 09-30:
# 1345 SPEAK / 1 HOLD), but its P([HOLD]) still separates side chatter (~1-2%) from
# direct address (~0.2%). Hold when P(HOLD)/(P(HOLD)+P(SPEAK)) >= this. 0 disables.
HOLD_THRESH = float(os.environ.get("ATLAS_HOLD_THRESH", "0"))
LAST_HOLD_SHARE = None   # last measured share (telemetry / calibration)
# Think while listening (think_listen.py): telemetry of the last think attempt, the last header
# decision time, and one entry per _two_stage call (the sims clear TURN_LOG before each turn).
LAST_THINK: dict = {}
LAST_HEADER_MS = None
TURN_LOG: deque = deque(maxlen=256)     # bounded: live calls never clear it


def hold_share(resp: dict):
    """P([HOLD]) / (P([HOLD]) + P([SPEAK])) from the first header token, or None."""
    import math
    try:
        first = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    ph = sum(math.exp(t["logprob"]) for t in first if t["token"].lstrip().startswith("[H"))
    ps = sum(math.exp(t["logprob"]) for t in first if t["token"].lstrip().startswith("[S"))
    return ph / (ph + ps) if ph + ps > 0 else None


class _Interrupted(Exception):
    """A cancelled turn (RemoteSession.interrupt) seen inside a read."""


def _reply_lines(rest, lines, markers, budget=None, entry=None):
    """/v1/completions SSE lines of a reply that runs on in a longer stream: `rest` (text already read)
    first, then `lines`, cut at the first of `markers` (the reply request's own stop strings) and after
    `budget` tokens (the reply request's max_tokens; entry["reply_cap"] is set, as for that request)."""
    def emit(text):
        return ("data: " + json.dumps({"choices": [{"text": text}]})).encode()

    def cut(text):
        hits = [text.find(m) for m in markers if m in text]
        return (text[:min(hits)], True) if hits else (text, False)
    if rest:
        text, stop = cut(rest)
        if text:
            yield emit(text)
        if stop:
            return
    used = 0
    for raw in lines:
        piece = _completion_text(raw)
        if piece is None:
            yield raw
            return
        text, stop = cut(piece)
        if stop:
            if text:
                yield emit(text)
            return
        yield raw
        if budget is not None and piece:
            try:
                ch = (json.loads(raw.decode("utf-8", "replace").strip()[5:]).get("choices") or [{}])[0]
                used += len((ch.get("logprobs") or {}).get("tokens") or []) or 1
            except (ValueError, AttributeError):
                used += 1
            if used >= budget:
                if entry is not None:
                    entry["reply_cap"] = True
                return


def _completion_text(raw):
    """Text of one /v1/completions SSE line ("" for non-data lines), or None at the end of the stream."""
    line = raw.decode("utf-8", "replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
    if not line.startswith("data:"):
        return ""
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        ev = json.loads(data)
    except ValueError:
        return ""
    return "".join((ch.get("text") or "") for ch in ev.get("choices") or [])


def _is_participation(body) -> bool:
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return False
    c = msgs[0].get("content")
    return isinstance(c, str) and _PROTOCOL_MARK in c


# "search X" / "google X" / "look up X" said TO the agent: a clear command, so
# the greedy decision must be the tool call, not a "[SPEAK] On it." header.
_EXPLICIT_SEARCH = re.compile(
    r"\b(?:(?:can|could|would|will)\s+you\s+(?:go\s+)?research|(?:please|go)\s+research|\w+,\s*research\b|research\s+(?:this|that|it)\b|search(?:\s+up)?|google|look\s+(?:it\s+)?up|look\s+up|lookup|check\s+online|"
    r"find\s+out\s+online|search\s+the\s+(?:web|internet))\b", re.I)


def _last_user_text(body) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else ""
    return ""


def _offers_tool(body, name) -> bool:
    return any(((t.get("function") or {}).get("name") == name) for t in (body.get("tools") or []))


class _JsonResponse:
    """Non-streaming result in Ollama shape (prewarm/measure use stream=True, but be safe)."""

    def __init__(self, resp, model):
        self._resp, self._model, self.status_code = resp, model, resp.status_code

    def raise_for_status(self):
        self._resp.raise_for_status()

    def close(self):
        self._resp.close()

    def json(self):
        d = self._resp.json()
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        return {"model": self._model, "message": {"role": "assistant", "content": msg.get("content") or ""},
                "done": True}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class RemoteSession:
    """Drop-in for LLM.ollama_session. Falls back to `local` when remote is down."""

    def __init__(self, cfg: dict, local: requests.Session):
        self.cfg = cfg
        self.local = local
        self._http = requests.Session()
        if cfg.get("key"):
            self._http.headers["Authorization"] = "Bearer " + cfg["key"]
        if cfg.get("cloud"):
            self._http.headers["X-Title"] = "ATLAS"
        self._lock = threading.Lock()
        self.down_until = 0.0
        # Per-turn calibrated-HOLD threshold for the next two-stage call only (floor.undirected_hold_thresh:
        # lines that ask the agent nothing); None = the global ATLAS_HOLD_THRESH.
        self.next_hold_thresh = None
        # Think while listening: (thought or None, line text, think threshold) for the next request
        # only; set by the pipeline next to next_hold_thresh, read and cleared by _two_stage.
        self.next_thought = None
        # A running thought finish (think_listen "finish" setting) and a counter interrupt() bumps.
        self._finish_lock = threading.Lock()
        self._finish_resp = None
        self._finish_gen = 0
        self.last_error = ""
        self.fallbacks = 0
        self.requests = 0
        self.read_timeouts = 0      # turns dropped because the server sent nothing for READ_TIMEOUT_S
        # Watchdog state (see tick()). local_warm: local model is loaded as a
        # hot standby because the remote looked down.
        self.fails = 0
        self.up_since = time.time()
        self.local_warm = False
        self._stop = threading.Event()
        self._thread = None

    def interrupt(self):
        """Cancel hook (LLM.cancel_generation): stop a running thought finish so a cancelled turn
        doesn't wait for it."""
        with self._finish_lock:
            self._finish_gen += 1
            resp, self._finish_resp = self._finish_resp, None
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

    # health / connection check (llm_module calls session.get(base + "/"))
    def ping(self, timeout=3.0) -> bool:
        try:
            r = self._http.get(self.cfg["url"] + "/v1/models", timeout=timeout)
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            self.last_error = str(e)[:200]
            return False

    def get(self, url, **kw):
        if self.ping(timeout=min(kw.get("timeout", 5.0), 5.0)):
            r = requests.Response()
            r.status_code = 200
            return r
        if not self.cfg.get("allow_local_fallback", True):
            raise requests.ConnectionError("Remote LLM unavailable: " + self.last_error)
        return self.local.get(url, **kw)

    def _remote_ok(self) -> bool:
        return time.time() >= self.down_until

    def post(self, url, json=None, stream=False, timeout=None, **kw):
        request_fallback = kw.pop("allow_local_fallback", True)
        allow_fallback = self.cfg.get("allow_local_fallback", True) and request_fallback
        payload = json or {}
        if self.next_thought is not None and url.endswith("/api/chat") and not self._remote_ok():
            self.next_thought = None             # a handover only belongs to a two-stage request
        if url.endswith("/api/chat") and self._remote_ok():
            body = translate_payload(payload, self.cfg["model"], vllm=self.cfg.get("vllm", True))
            body = provider_request_body(body, self.cfg["url"], think=payload.get("think"))
            body["stream"] = bool(stream)
            if not stream:
                body.pop("stream_options", None)
            read_t = min(timeout[1] if isinstance(timeout, tuple) else 600.0, READ_TIMEOUT_S)
            try:
                self.requests += 1
                if (stream and TWO_STAGE and self.cfg.get("vllm", True) and (body.get("temperature") or 0) > 0
                        and _is_participation(body)):
                    staged = self._two_stage(body, (4.0, read_t))
                elif self.next_thought is not None and _is_participation(body):
                    self.next_thought = None
                    staged = None
                else:
                    staged = None
                if staged is not None:
                    return staged
                resp = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=body,
                                       stream=stream, timeout=(4.0, read_t))
                if resp.status_code >= 500:
                    raise requests.ConnectionError(f"remote HTTP {resp.status_code}: {resp.text[:200]}")
                return NdjsonResponse(resp, self.cfg["model"]) if stream else _JsonResponse(resp, self.cfg["model"])
            except (requests.ConnectionError, requests.Timeout) as e:
                if is_read_timeout(e):
                    # The server answered the connection but sent nothing for read_t seconds: this turn is lost,
                    # the server isn't down (marking it down would mute every turn for FAIL_COOLDOWN_S).
                    self.read_timeouts += 1
                    logger.warning("🌐⏱️ Remote LLM sent nothing for %g s: dropping this turn", read_t)
                    raise
                self.down_until = time.time() + FAIL_COOLDOWN_S
                self.last_error = str(e)[:200]
                if not allow_fallback:
                    logger.warning("Remote LLM unavailable; local fallback is disabled: %s", self.last_error)
                    raise
                self.fallbacks += 1
                logger.warning("🌐🤖 Remote LLM unreachable (%s) — falling back to local for %.0fs",
                               self.last_error, FAIL_COOLDOWN_S)
        if url.endswith("/api/chat") and not allow_fallback:
            raise requests.ConnectionError("Remote LLM unavailable; local fallback is disabled")
        if not allow_fallback and not (
                url.endswith("/api/generate") and payload.get("keep_alive") == 0
                and not any(payload.get(key) for key in ("prompt", "messages", "images"))):
            raise requests.ConnectionError("Remote-only transport prohibits local model generation")
        # Non-chat endpoints (e.g. /api/generate unload) and fallback -> local Ollama.
        return self.local.post(url, json=payload, stream=stream, timeout=timeout, **kw)

    def _two_stage(self, body, timeout):
        """Greedy header/tool decision, then a sampled body continued from the header.
        Returns None to fall back to the ordinary single sampled stream."""
        handover, self.next_thought = self.next_thought, None
        t0 = time.perf_counter()
        entry = {"ms": None, "path": "today", "reason": None, "decision": None, "p_hold": None,
                 "reply_cap": False}
        TURN_LOG.append(entry)

        def _done(decision, ph=None):
            global LAST_HEADER_MS
            entry["ms"] = round((time.perf_counter() - t0) * 1000)
            entry["decision"] = decision
            if ph is not None:
                entry["p_hold"] = ph
            LAST_HEADER_MS = entry["ms"]

        read_t = timeout[1] if isinstance(timeout, tuple) else 600.0
        s1 = dict(body, stream=False, temperature=0.0, max_tokens=16, stop=["]"])
        _lu = _last_user_text(body)
        try:
            from backchannels import mention_only as _mention_only
        except Exception:  # noqa: BLE001
            def _mention_only(_t):
                return False
        try:
            from tasks import CUE_RE as _CUE_RE
        except Exception:  # noqa: BLE001
            _CUE_RE = None
        # The agent's own TASK CUE says "your background search ... finished": not a
        # spoken command (live: all 34 cue turns re-searched and said "looking it up"
        # right before delivering the result they already had).
        _cue = bool(_CUE_RE is not None and _CUE_RE.search(_lu))
        if (_offers_tool(body, "web_search") and _EXPLICIT_SEARCH.search(_lu) and not _mention_only(_lu)
                and not _cue):
            s1 = dict(body, stream=False, temperature=0.0, max_tokens=256,
                      tool_choice={"type": "function", "function": {"name": "web_search"}})
        thresh = self.next_hold_thresh if self.next_hold_thresh is not None else HOLD_THRESH
        self.next_hold_thresh = None
        if handover is not None:
            # Think while listening: the thought cut at the decision point decides; on any
            # fallback today's request below runs unchanged with today's threshold.
            staged = self._two_stage_think(body, handover, "tool_choice" in s1, read_t, entry, _done)
            if staged is not None:
                return staged
        if ONE_REQUEST and "tool_choice" not in s1:
            one = self._one_request(body, thresh, read_t, entry, _done)
            if one is not None:
                return one
        if thresh > 0:
            s1["logprobs"], s1["top_logprobs"] = True, 10
        s1.pop("stream_options", None)
        s1.pop("top_p", None)
        r = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=s1, timeout=(4.0, read_t))
        if r.status_code >= 500:
            raise requests.ConnectionError(f"remote HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            _done(None)
            return None
        s1r = r.json()
        ch1 = (s1r.get("choices") or [{}])[0]
        msg = ch1.get("message") or {}
        calls = msg.get("tool_calls") or []
        head = (msg.get("content") or "").strip()
        # A tool call needs more than 16 tokens: vLLM reports a cut one as either
        # "length" or "tool_calls" with arguments '{}' (live 10-01: every search
        # ran with an empty query). Redo any truncated decision with room to finish.
        if s1.get("max_tokens", 0) <= 16 and (calls or (ch1.get("finish_reason") == "length"
                                                        and not head.startswith("["))):
            # 16 tokens fits a [SPEAK] header but NOT a tool call: the arguments
            # were cut mid-JSON and would parse as {} (live 10-01: every search
            # ran with an empty query -> "came back empty"). Redo the greedy
            # decision with room to finish the call.
            full = dict(s1, max_tokens=256)
            full.pop("stop", None)
            r = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=full,
                                timeout=(4.0, read_t))
            if r.status_code >= 400:
                _done(None)
                return None
            msg = ((r.json().get("choices") or [{}])[0].get("message") or {})
            calls = msg.get("tool_calls") or []
            head = (msg.get("content") or "").strip()
            if not calls and not head.startswith("[SPEAK to="):
                _done(None)
                return None
        if calls:
            out = []
            for tc in calls:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    _done(None)
                    return None          # broken call: let the ordinary stream decide
                out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            _done("tool")
            return _SyntheticNdjson(self.cfg["model"], tool_calls=out)
        if head.startswith("[HOLD"):
            _done("hold")
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        ph = hold_share(s1r)
        global LAST_HOLD_SHARE
        LAST_HOLD_SHARE = ph
        if thresh > 0 and ph is not None and ph >= thresh:
            logger.info("🌐🤖 Calibrated HOLD (p_hold share %.3f >= %.3f)", ph, thresh)
            _done("hold", ph)
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        if head.startswith("[SPEAK to=") and "]" in head:
            head = head[:head.index("]")]
        if not head.startswith("[SPEAK to=") or len(head) > 40:
            _done(None, ph)
            return None
        header = head + "]"
        s2 = dict(body)
        s2["messages"] = list(body["messages"]) + [{"role": "assistant", "content": header}]
        s2["continue_final_message"] = True
        s2["add_generation_prompt"] = False
        _done("speak", ph)
        resp = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=s2, stream=True,
                               timeout=(4.0, read_t))
        if resp.status_code >= 500:
            raise requests.ConnectionError(f"remote HTTP {resp.status_code}: {resp.text[:200]}")
        return _PrefixedNdjson(resp, self.cfg["model"], header, entry=entry)

    def _one_request(self, body, thresh, read_t, entry, done):
        """The decision and the reply in one request (11-03: first words ~0.18 s sooner than the two
        requests). The reply is sampled exactly as before and the header is read off the same stream
        with its logprobs: it counts only if every header token was the model's top choice (or within
        ONE_REQUEST_TIE of it), i.e. the header today's greedy request would have produced, and P(HOLD)
        comes from the same first-token
        distribution. Returns a response (SPEAK or HOLD), or None for today's two requests (a header
        token that wasn't the top choice, a tool call, anything unexpected)."""
        s = dict(body, stream=True, logprobs=True, top_logprobs=10)
        resp = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=s, stream=True,
                               timeout=(4.0, read_t))
        if resp.status_code >= 500:
            raise requests.ConnectionError(f"remote HTTP {resp.status_code}: {resp.text[:200]}")

        def _drop(why):
            entry["one"] = why
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            return None
        if resp.status_code >= 400:
            return _drop("http")
        lines = resp.iter_lines(decode_unicode=False)
        head, greedy, first_top, ended, closed = "", True, None, False, False
        try:
            for raw in lines:
                line = raw.decode("utf-8", "replace").strip() if raw else ""
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    ended = True
                    break
                ch = (json.loads(data).get("choices") or [{}])[0]
                delta = ch.get("delta") or {}
                if delta.get("tool_calls"):
                    return _drop("tool")
                lps = (ch.get("logprobs") or {}).get("content") or []
                if (delta.get("content") or "") and not lps and not closed:
                    greedy = False      # header text we can't check
                for lp in lps:
                    if closed:          # reply tokens sharing the chunk (speculative decoding): sampled
                        break
                    closed = "]" in (lp.get("token") or "")
                    top = lp.get("top_logprobs") or []
                    best = max((t["logprob"] for t in top), default=None)
                    gap = None if best is None else best - float(lp.get("logprob", -1e9))
                    if first_top is None:
                        first_top = top
                        # SPEAK vs HOLD itself: only the exact top choice (a near tie must not flip
                        # a line aimed at the agent to HOLD)
                        if gap is None or gap > 1e-6:
                            greedy = False
                    elif gap is None or gap > ONE_REQUEST_TIE:
                        greedy = False
                head += delta.get("content") or ""
                stripped = head.lstrip()
                if "]" in head or len(stripped) > 48 or (stripped and not stripped.startswith("[")):
                    break
                if ch.get("finish_reason"):
                    ended = True
                    break
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("one-request header read failed: %s", e)
            return _drop("read")
        top_first = max(first_top or [], key=lambda t: t["logprob"], default={}).get("token", "")
        if not top_first.lstrip().startswith("["):
            return _drop("tool")      # the greedy decision isn't a header (e.g. a tool call): today's path decides
        ph = hold_share({"choices": [{"logprobs": {"content": [{"top_logprobs": first_top or []}]}}]})
        global LAST_HOLD_SHARE
        LAST_HOLD_SHARE = ph
        stripped = head.lstrip()
        if thresh > 0 and ph is not None and ph >= thresh:
            logger.info("🌐🤖 Calibrated HOLD (p_hold share %.3f >= %.3f)", ph, thresh)
            done("hold", ph)
            entry["one"] = "ok"
            _drop("ok")
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        if not greedy or first_top is None:
            return _drop("not_greedy")
        if stripped.startswith("[HOLD"):
            done("hold", ph)
            _drop("ok")
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        if not stripped.startswith("[SPEAK to=") or "]" not in stripped or stripped.index("]") > 40:
            return _drop("header")
        cut = stripped.index("]")
        header, rest = stripped[:cut + 1], stripped[cut + 1:]
        done("speak", ph)
        entry["one"] = "ok"
        first = [("data: " + json.dumps({"choices": [{"delta": {"content": rest}}]})).encode()] if rest else []
        more = iter(()) if ended else lines
        return _PrefixedNdjson(resp, self.cfg["model"], header, lines=itertools.chain(first, more), entry=entry)

    def _fast_finish_start(self, TL, base, model, text, finish, params, read_t, finish_gen):
        """Start the thought's finish and the reply after it as one stream (think_listen.FAST_FINISH),
        registered for interrupt(). None if the turn was already cancelled."""
        with self._finish_lock:
            if finish_gen != self._finish_gen:
                return None
        # The thought is sampled as the thinker and the measured finish were (TL.SAMPLING); one stream can't
        # switch settings, so the reply after it gets top_p 0.95 instead of the turn's 0.9.
        sampling = dict(TL.SAMPLING)
        resp = TL.complete(self._http, base, model, text, stream=True, timeout=(4.0, read_t),
                           max_tokens=finish + int(params.get("max_tokens") or 256), logprobs=1,
                           skip_special_tokens=False, **sampling)
        with self._finish_lock:
            if finish_gen == self._finish_gen:
                self._finish_resp = resp
                return resp
        self._fast_finish_drop(resp)
        return None

    def _fast_finish_drop(self, resp):
        if resp is None:
            return
        with self._finish_lock:
            if self._finish_resp is resp:
                self._finish_resp = None
        try:
            resp.close()
        except Exception:  # noqa: BLE001
            pass

    def _fast_finish_read(self, TL, resp, header, finish, finish_gen):
        """Read the finishing thought off the stream started by _fast_finish_start. Returns
        ("reply", rest, lines, cont, n): the model's own header after </think> matched `header`, with every
        header token its top choice (within ONE_REQUEST_TIE), and the reply goes on in `lines`;
        ("slow", cont, n): finished (or cut at `finish` tokens) but the reply needs its own request;
        ("interrupted",): the turn was cancelled. cont: the thought's continuation; n: its tokens."""
        lines = resp.iter_lines(decode_unicode=False)
        full, toks, n, k = "", [], 0, None

        def slow(why):
            self._fast_finish_drop(resp)
            with self._finish_lock:
                if finish_gen != self._finish_gen:
                    return ("interrupted",)
            LAST_THINK["fast_why"] = why
            return ("slow", (full[:k] if k is not None else full).rstrip("\n"), n)
        try:
            for raw in lines:
                line = raw.decode("utf-8", "replace").strip() if raw else ""
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ch = (json.loads(data).get("choices") or [{}])[0]
                piece = ch.get("text") or ""
                lp = ch.get("logprobs") or {}
                offs, tls, tops = lp.get("text_offset") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or []
                for j, off in enumerate(offs):
                    top = tops[j] if j < len(tops) and tops[j] else {}
                    toks.append((off, tls[j] if j < len(tls) else None, max(top.values()) if top else None))
                full += piece
                if k is None:
                    n += len(offs) or (1 if piece else 0)
                    pos = full.find("</think>")
                    if pos < 0:
                        if n >= finish:
                            return slow("cap")
                        if ch.get("finish_reason"):
                            return slow("ended")
                        continue
                    k = pos
                after = full[k + len("</think>"):]
                hs = k + len("</think>") + len(after) - len(after.lstrip())
                got = full[hs:hs + len(header)]
                if not header.startswith(got):
                    return slow("header")
                if len(got) < len(header):
                    if ch.get("finish_reason"):
                        return slow("ended")
                    continue
                he = hs + len(header)
                for j, (off, tl, best) in enumerate(toks):
                    end = toks[j + 1][0] if j + 1 < len(toks) else len(full)
                    if off < he and end > hs and (tl is None or best is None or best - tl > ONE_REQUEST_TIE):
                        return slow("not_greedy")
                n_rest = sum(1 for off, _, _ in toks if off >= he)
                with self._finish_lock:
                    if finish_gen != self._finish_gen:
                        raise _Interrupted()
                    if self._finish_resp is resp:
                        self._finish_resp = None          # it is the reply now; its owner closes it
                return ("reply", full[hs + len(header):], lines, full[:k].rstrip("\n"), n, n_rest)
            return slow("ended")
        except Exception as e:  # noqa: BLE001
            with self._finish_lock:
                interrupted = finish_gen != self._finish_gen
            self._fast_finish_drop(resp)
            if interrupted:
                return ("interrupted",)
            if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                raise
            logger.warning("think fast finish failed: %s", e)
            return ("slow", (full[:k] if k is not None else full).rstrip("\n"), n)

    def _two_stage_think(self, body, handover, forced, read_t, entry, done):
        """Think while listening (think_listen.py): decide from the thought cut at the decision point.
        Returns a response, or None for today's path (the reason is recorded). Connection errors and
        timeouts propagate so post() fails over as it does for today's request."""
        import think_listen as TL
        thought, line_text, thr = handover
        with self._finish_lock:
            finish_gen = self._finish_gen
        LAST_THINK.clear()
        LAST_THINK.update(path="fallback", reason=None, tokens=getattr(thought, "tokens", 0),
                          age_s=round(getattr(thought, "age_s", 0.0), 2),
                          closed=getattr(thought, "closed", False), repeated=getattr(thought, "repeated", False),
                          reopened=getattr(thought, "reopened", 0), p_hold=None, decision=None,
                          header_ms=None, corrupt=TL.corrupt(getattr(thought, "text", "") or ""))

        def fallback(reason):
            entry.update(path="fallback", reason=reason)
            LAST_THINK.update(path="fallback", reason=reason)
            if reason not in ("no_thought",):
                logger.info("🧠💭 think fallback: %s", reason)
            return None

        if thought is None or not (thought.text or "").strip():
            return fallback("no_thought")
        msgs = body.get("messages") or []
        if any(isinstance(m.get("content"), list)
               and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]) for m in msgs):
            return fallback("image")
        if not msgs or msgs[-1].get("role") != "user":
            return fallback("prerun_tool")
        last = re.sub(r"^/no_think\s+", "", _last_user_text(body))
        if not line_text or not last.startswith(line_text):
            return fallback("mismatch")
        if forced:
            return fallback("forced_tool")
        base, model = self.cfg["url"], self.cfg["model"]
        params = dict(TL.REPLY)
        for k in ("temperature", "top_p"):
            if body.get(k) is not None:
                params[k] = body[k]
        finish = TL.finish_tokens()
        finishing = finish > 0 and not getattr(thought, "closed", False)
        fast = None
        try:
            rendered = TL.render(self._http, base, model, TL.strip_no_think(msgs), body.get("tools"))
            prompt = rendered + thought.text + TL.CUT_NOTE + TL.CLOSE
            if finishing and TL.FAST_FINISH:
                # The finish (and the reply after it) starts now, alongside the header request.
                try:
                    fast = self._fast_finish_start(TL, base, model, rendered + thought.text, finish, params,
                                                   read_t, finish_gen)
                except (requests.ConnectionError, requests.Timeout):
                    raise
                except Exception as e:  # noqa: BLE001 - the sequential finish below still runs
                    logger.warning("think fast finish start failed: %s", e)
            ch = TL.complete(self._http, base, model, prompt, timeout=(4.0, read_t), **TL.HEADER)
        except (requests.ConnectionError, requests.Timeout):
            self._fast_finish_drop(fast)
            raise
        except Exception as e:  # noqa: BLE001
            self._fast_finish_drop(fast)
            logger.warning("think header failed: %s", e)
            return fallback("http_error")
        text = (ch.get("text") or "").strip()
        ph = TL.hold_share_completion(ch)
        global LAST_HOLD_SHARE
        LAST_HOLD_SHARE = ph
        entry["p_hold"] = ph
        LAST_THINK["p_hold"] = ph

        def decided(decision):
            entry["path"] = "think"
            done(decision, ph)
            LAST_THINK.update(path="think", decision=decision, header_ms=entry["ms"])

        if not (text.startswith("[SPEAK") and not (ph is not None and ph >= thr)):
            self._fast_finish_drop(fast)       # tool, hold or a bad header: no finish needed
            fast = None
        if text.startswith("<tool_call>"):
            try:
                ch2 = TL.complete(self._http, base, model, prompt, timeout=(4.0, read_t), **TL.TOOL_FINISH)
            except (requests.ConnectionError, requests.Timeout):
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("think tool finish failed: %s", e)
                return fallback("http_error")
            calls = TL.read_tool_calls(ch2.get("text") or "", body.get("tools"))
            if not calls:
                return fallback("unreadable_tool")
            decided("tool")
            return _SyntheticNdjson(model, tool_calls=calls)
        if text.startswith("[HOLD") or (ph is not None and ph >= thr):
            decided("hold")
            return _SyntheticNdjson(model, content="[HOLD]")
        header = TL.header_from_text(text)
        if header is None:
            self._fast_finish_drop(fast)
            return fallback("bad_header")
        reply_prompt = prompt + header
        if fast is not None:
            t_fin = time.perf_counter()
            got = self._fast_finish_read(TL, fast, header, finish, finish_gen)
            LAST_THINK["fast"] = got[0]
            if got[0] == "reply":
                _, rest, lines, cont, n, n_rest = got
                LAST_THINK["finish_tokens"] = n
                LAST_THINK["finish_ms"] = round((time.perf_counter() - t_fin) * 1000)
                read, first = [], None
                body_lines = _reply_lines(rest, lines, TL.REPLY_MARKERS,
                                          budget=int(params.get("max_tokens") or 256) - n_rest, entry=entry)
                try:
                    for raw in body_lines:
                        read.append(raw)
                        piece = _completion_text(raw)
                        if piece is None:
                            break
                        if piece.strip():
                            first = piece
                            break
                except Exception as e:  # noqa: BLE001
                    logger.warning("think reply stream failed: %s", e)
                if first is not None and not first.lstrip().startswith("<"):
                    decided("speak")
                    return _PrefixedNdjson(fast, model, header, lines=itertools.chain(read, body_lines), entry=entry)
                self._fast_finish_drop(fast)
                reply_prompt = rendered + thought.text + cont + TL.CLOSE + header      # the reply on its own
            elif got[0] == "interrupted":
                LAST_THINK["finish_interrupted"] = True
            else:
                _, cont, n = got
                reply_prompt = rendered + thought.text + cont + TL.CLOSE + header
                LAST_THINK["finish_tokens"] = n
                LAST_THINK["finish_ms"] = round((time.perf_counter() - t_fin) * 1000)
        elif finishing:
            # The decision is already made (SPEAK); let the thought run to its end, on the whole line,
            # before the first word (reply_check 2026-11-02: invented details 16 -> 8 of 50).
            # Streamed so interrupt() (a cancelled turn) can stop it at once; then the reply goes out from
            # the cut thought and the turn's owner closes it.
            t_fin = time.perf_counter()
            resp, cont, n = None, "", 0
            try:
                with self._finish_lock:
                    live = finish_gen == self._finish_gen
                if live:
                    resp = TL.complete(self._http, base, model, rendered + thought.text, stream=True,
                                       timeout=(4.0, read_t), max_tokens=finish, **TL.SAMPLING, **TL.THINK_REQUEST)
                    with self._finish_lock:
                        live = finish_gen == self._finish_gen
                        if live:
                            self._finish_resp = resp
                if live:
                    cont, n = TL.read_stream(resp)
                    cont = cont.rstrip("\n")          # CLOSE brings its own newline
                with self._finish_lock:
                    self._finish_resp = None
                    live = live and finish_gen == self._finish_gen
                if live:
                    reply_prompt = rendered + thought.text + cont + TL.CLOSE + header
                    LAST_THINK["finish_tokens"] = n
                    LAST_THINK["finish_ms"] = round((time.perf_counter() - t_fin) * 1000)
                else:
                    LAST_THINK["finish_interrupted"] = True
            except (requests.ConnectionError, requests.Timeout):
                with self._finish_lock:
                    interrupted = finish_gen != self._finish_gen
                    self._finish_resp = None
                if not interrupted:
                    raise
                LAST_THINK["finish_interrupted"] = True
            except Exception as e:  # noqa: BLE001 - speak from the cut thought
                with self._finish_lock:
                    interrupted = finish_gen != self._finish_gen
                    self._finish_resp = None
                if interrupted:
                    LAST_THINK["finish_interrupted"] = True
                else:
                    logger.warning("think finish failed: %s", e)
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:  # noqa: BLE001
                        pass
        if LAST_THINK.get("finish_interrupted"):
            # Cancelled while the thought finished (someone kept talking): no reply request, which would
            # only delay the next turn (review 11-03); the turn's owner discards this.
            decided("cancelled")
            return _SyntheticNdjson(model, content="[HOLD]")
        try:
            resp = TL.complete(self._http, base, model, reply_prompt, stream=True, timeout=(4.0, read_t),
                               **params)
        except (requests.ConnectionError, requests.Timeout):
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("think reply failed: %s", e)
            return fallback("http_error")
        lines = resp.iter_lines(decode_unicode=False)
        read, first = [], None
        try:
            for raw in lines:
                read.append(raw)
                piece = _completion_text(raw)
                if piece is None:
                    break
                if piece.strip():
                    first = piece
                    break
        except Exception as e:  # noqa: BLE001
            logger.warning("think reply stream failed: %s", e)
        if first is None or first.lstrip().startswith("<"):
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            return fallback("empty_reply" if first is None else "markup_reply")
        decided("speak")
        return _PrefixedNdjson(resp, model, header, lines=itertools.chain(read, lines), entry=entry)

    # ------------------------------------------------------------------
    # Watchdog: keep a hot local standby only while the remote is down.
    # ------------------------------------------------------------------
    def tick(self, ok: bool, now: float = None):
        """Pure state step. Returns "warm_local", "unload_local" or None."""
        now = time.time() if now is None else now
        if not ok:
            self.fails += 1
            self.up_since = 0.0
            if not self.cfg.get("allow_local_fallback", True):
                self.down_until = max(self.down_until, now + WATCH_INTERVAL_S * 2)
                return None
            if self.fails >= DOWN_AFTER_FAILS or not self._remote_ok():
                self.down_until = max(self.down_until, now + WATCH_INTERVAL_S * 2)
                if not self.local_warm:
                    self.local_warm = True
                    return "warm_local"
            return None
        # healthy ping
        if self.fails or self.down_until > now:
            logger.info("🌐🤖 Remote LLM reachable again — routing turns back to it")
        self.fails = 0
        self.down_until = 0.0
        if not self.up_since:
            self.up_since = now
        if self.local_warm and now - self.up_since >= UP_STABLE_S:
            self.local_warm = False
            return "unload_local"
        return None

    def start_watchdog(self, warm_local, unload_local):
        if self._thread is not None:
            return
        def loop():
            while not self._stop.wait(WATCH_INTERVAL_S):
                try:
                    action = self.tick(self.ping(timeout=3.0))
                    if action == "warm_local":
                        logger.warning("🌐🤖 Remote LLM down — warming local model as standby")
                        warm_local()
                    elif action == "unload_local":
                        logger.info("🌐🤖 Remote stable — unloading local standby to free VRAM")
                        unload_local()
                except Exception as e:  # noqa: BLE001 — watchdog must never die
                    logger.warning("remote watchdog error: %s", e)
        self._thread = threading.Thread(target=loop, name="RemoteWatchdog", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        try:
            self._http.close()
        except Exception:
            pass

    def status(self) -> dict:
        return {"label": self.cfg["label"], "url": self.cfg["url"], "model": self.cfg["model"],
                "allow_local_fallback": bool(self.cfg.get("allow_local_fallback",True)),
                "healthy": self._remote_ok(), "last_error": self.last_error,
                "requests": self.requests, "fallbacks": self.fallbacks}


# ---------------------------------------------------------------------------
# Switching helpers used by server.py
# ---------------------------------------------------------------------------
def unload_local(ollama_url: str, model: str) -> bool:
    """Free the local model's VRAM (Ollama keep_alive=0)."""
    try:
        r = requests.post(ollama_url.rstrip("/") + "/api/generate",
                          json={"model": model, "keep_alive": 0}, timeout=15)
        return r.ok
    except requests.RequestException:
        return False


def make_session(cfg, local=None):
    if cfg.get("provider") in ("codex", "claude_code"):
        from native_providers import NativeSession
        return NativeSession(cfg, local)
    return RemoteSession(cfg, local)


def activate(llm, cfg: dict) -> "RemoteSession":
    """Point an LLM instance at the remote host. Returns the RemoteSession."""
    llm._ensure_local_session()
    rs = make_session(cfg, llm._local_session)
    llm.ollama_session = rs
    llm.remote = rs
    llm._remote_startup_error = None
    llm._client_initialized = False

    if getattr(rs,"skip_startup_inference",False):
        rs.ping(timeout=5)

    def _warm():
        # prewarm() goes through llm.ollama_session, which is the RemoteSession;
        # with down_until set it routes to local Ollama and loads the model.
        try:
            llm.prewarm()
        except Exception as e:  # noqa: BLE001
            logger.warning("local standby warm failed: %s", e)

    rs.start_watchdog(_warm, lambda: unload_local(llm.effective_ollama_url, llm.model))
    return rs


def deactivate(llm):
    llm._remote_startup_error = None
    llm._client_initialized = False
    rs = getattr(llm, "remote", None)
    if rs is not None:
        llm.ollama_session = llm._local_session
        llm.remote = None
        rs.close()
