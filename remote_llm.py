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
import json
import re
import logging
import os
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("ATLAS_REMOTE_LLM_CONFIG", ROOT / "private" / "remote_llm.json"))
FAIL_COOLDOWN_S = 30.0  # after a connection failure, go straight to local for this long
WATCH_INTERVAL_S = float(os.environ.get("ATLAS_REMOTE_WATCH_S", "10"))
DOWN_AFTER_FAILS = 2      # consecutive failed pings before we warm the local model
UP_STABLE_S = 60.0        # remote healthy this long before local VRAM is freed again


def load_config():
    """Return the remote config dict, or None if not configured (public build)."""
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
            "label": cfg.get("label") or cfg["url"], "vllm": bool(cfg.get("vllm", True))}


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
    if not vllm:
        body.pop("top_k", None)
        if payload.get("format") == "json":
            body["response_format"] = {"type": "json_object"}
        return body
    think = payload.get("think")
    ctk = dict(payload.get("chat_template_kwargs") or {})
    if think is False:
        ctk["enable_thinking"] = False
    if ctk:
        body["chat_template_kwargs"] = ctk
    if payload.get("format") == "json":
        body["response_format"] = {"type": "json_object"}
    if body["stream"]:
        body["stream_options"] = {"include_usage": False}
    return body


# ---------------------------------------------------------------------------
# OpenAI SSE stream  ->  Ollama NDJSON response
# ---------------------------------------------------------------------------
class NdjsonResponse:
    """Looks enough like requests.Response for llm_module's Ollama readers."""

    def __init__(self, resp: requests.Response, model: str):
        self._resp = resp
        self._model = model
        self.status_code = resp.status_code

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
        for raw in self._resp.iter_lines(decode_unicode=False):
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
                delta = ch.get("delta") or {}
                c = delta.get("content")
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

    def __init__(self, resp, model, prefix):
        super().__init__(resp, model)
        self._prefix = prefix

    def iter_content(self, chunk_size=None):
        yield self._line({"message": {"role": "assistant", "content": self._prefix}, "done": False})
        yield from super().iter_content(chunk_size)


# Two-stage sampling: the speak/hold header (and any tool decision) is taken at
# temperature 0; only the spoken body is sampled. Live remote at 0.6 butted into
# quiet-room muttering ~7/15 vs ~1/15 greedy (tests/sim_passivity.py 09-30).
TWO_STAGE = os.environ.get("ATLAS_TWO_STAGE", "1") != "0"
_PROTOCOL_MARK = "output only [HOLD]"
# Calibrated hold: the remote quant's greedy header is ~always [SPEAK] (live 09-30:
# 1345 SPEAK / 1 HOLD), but its P([HOLD]) still separates side chatter (~1-2%) from
# direct address (~0.2%). Hold when P(HOLD)/(P(HOLD)+P(SPEAK)) >= this. 0 disables.
HOLD_THRESH = float(os.environ.get("ATLAS_HOLD_THRESH", "0"))
LAST_HOLD_SHARE = None   # last measured share (telemetry / calibration)


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
        self.last_error = ""
        self.fallbacks = 0
        self.requests = 0
        # Watchdog state (see tick()). local_warm: local model is loaded as a
        # hot standby because the remote looked down.
        self.fails = 0
        self.up_since = time.time()
        self.local_warm = False
        self._stop = threading.Event()
        self._thread = None

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
        return self.local.get(url, **kw)

    def _remote_ok(self) -> bool:
        return time.time() >= self.down_until

    def post(self, url, json=None, stream=False, timeout=None, **kw):
        payload = json or {}
        if url.endswith("/api/chat") and self._remote_ok():
            body = translate_payload(payload, self.cfg["model"], vllm=self.cfg.get("vllm", True))
            body["stream"] = bool(stream)
            if not stream:
                body.pop("stream_options", None)
            try:
                self.requests += 1
                if (stream and TWO_STAGE and self.cfg.get("vllm", True) and (body.get("temperature") or 0) > 0
                        and _is_participation(body)):
                    staged = self._two_stage(body, timeout)
                    if staged is not None:
                        return staged
                resp = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=body,
                                       stream=stream, timeout=(4.0, timeout[1] if isinstance(timeout, tuple) else 600.0))
                if resp.status_code >= 500:
                    raise requests.ConnectionError(f"remote HTTP {resp.status_code}: {resp.text[:200]}")
                return NdjsonResponse(resp, self.cfg["model"]) if stream else _JsonResponse(resp, self.cfg["model"])
            except (requests.ConnectionError, requests.Timeout) as e:
                self.down_until = time.time() + FAIL_COOLDOWN_S
                self.last_error = str(e)[:200]
                self.fallbacks += 1
                logger.warning("🌐🤖 Remote LLM unreachable (%s) — falling back to local for %.0fs",
                               self.last_error, FAIL_COOLDOWN_S)
        # Non-chat endpoints (e.g. /api/generate unload) and fallback -> local Ollama.
        return self.local.post(url, json=payload, stream=stream, timeout=timeout, **kw)

    def _two_stage(self, body, timeout):
        """Greedy header/tool decision, then a sampled body continued from the header.
        Returns None to fall back to the ordinary single sampled stream."""
        read_t = timeout[1] if isinstance(timeout, tuple) else 600.0
        s1 = dict(body, stream=False, temperature=0.0, max_tokens=16, stop=["]"])
        _lu = _last_user_text(body)
        try:
            from backchannels import mention_only as _mention_only
        except Exception:  # noqa: BLE001
            def _mention_only(_t):
                return False
        if _offers_tool(body, "web_search") and _EXPLICIT_SEARCH.search(_lu) and not _mention_only(_lu):
            s1 = dict(body, stream=False, temperature=0.0, max_tokens=256,
                      tool_choice={"type": "function", "function": {"name": "web_search"}})
        if HOLD_THRESH > 0:
            s1["logprobs"], s1["top_logprobs"] = True, 10
        s1.pop("stream_options", None)
        s1.pop("top_p", None)
        r = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=s1, timeout=(4.0, read_t))
        if r.status_code >= 500:
            raise requests.ConnectionError(f"remote HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
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
                return None
            msg = ((r.json().get("choices") or [{}])[0].get("message") or {})
            calls = msg.get("tool_calls") or []
            head = (msg.get("content") or "").strip()
            if not calls and not head.startswith("[SPEAK to="):
                return None
        if calls:
            out = []
            for tc in calls:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    return None          # broken call: let the ordinary stream decide
                out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            return _SyntheticNdjson(self.cfg["model"], tool_calls=out)
        if head.startswith("[HOLD"):
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        ph = hold_share(s1r)
        global LAST_HOLD_SHARE
        LAST_HOLD_SHARE = ph
        if HOLD_THRESH > 0 and ph is not None and ph >= HOLD_THRESH:
            logger.info("🌐🤖 Calibrated HOLD (p_hold share %.3f >= %.3f)", ph, HOLD_THRESH)
            return _SyntheticNdjson(self.cfg["model"], content="[HOLD]")
        if head.startswith("[SPEAK to=") and "]" in head:
            head = head[:head.index("]")]
        if not head.startswith("[SPEAK to=") or len(head) > 40:
            return None
        header = head + "]"
        s2 = dict(body)
        s2["messages"] = list(body["messages"]) + [{"role": "assistant", "content": header}]
        s2["continue_final_message"] = True
        s2["add_generation_prompt"] = False
        resp = self._http.post(self.cfg["url"] + "/v1/chat/completions", json=s2, stream=True,
                               timeout=(4.0, read_t))
        if resp.status_code >= 500:
            raise requests.ConnectionError(f"remote HTTP {resp.status_code}: {resp.text[:200]}")
        return _PrefixedNdjson(resp, self.cfg["model"], header)

    # ------------------------------------------------------------------
    # Watchdog: keep a hot local standby only while the remote is down.
    # ------------------------------------------------------------------
    def tick(self, ok: bool, now: float = None):
        """Pure state step. Returns "warm_local", "unload_local" or None."""
        now = time.time() if now is None else now
        if not ok:
            self.fails += 1
            self.up_since = 0.0
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


def activate(llm, cfg: dict) -> "RemoteSession":
    """Point an LLM instance at the remote host. Returns the RemoteSession."""
    llm._ensure_local_session()
    rs = RemoteSession(cfg, llm._local_session)
    llm.ollama_session = rs
    llm.remote = rs

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
    rs = getattr(llm, "remote", None)
    if rs is not None:
        llm.ollama_session = llm._local_session
        llm.remote = None
        rs.close()
