"""Think while listening (owner 2026-11-02). Design: work/egm/think-listen/design_draft.md.

While a person talks, the model writes a private thought about the growing partial line; at the
decision point the thought is cut where it is and the speak/hold header is read right after it
(remote_llm._two_stage_think); if it speaks, the reply follows the thought in the same prompt text.

Off by default: live use needs the `think_listen` plugin (Phase 2). `active()` gates live entry
points; the sims and unit tests drive the Thinker and the handover directly. This module imports
nothing from the pipeline at module level (remote_llm imports it)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Optional

logger = logging.getLogger(__name__)

EFFORT = "low"                   # audit_low: the decision gain of xhigh at a quarter of the length
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
START_WORDS = 3
RESTART_WORDS = 2                # a thinking request restarts on 2 new words ...
RESTART_MIN_TOKENS = 16          # ... once it has produced this many tokens
REOPEN_WORDS = 3                 # a closed (or handed-over) thought reopens on 3 new words
CAP = int(os.environ.get("ATLAS_THINK_CAP", "400"))           # thinking tokens per utterance
CUT_WAIT_S = 0.05
# Calibrated on the calibration half of the saved calls (gate p1_fresh, 2026-11-02), fresh thinking.
HOLD_DIRECTED = float(os.environ.get("ATLAS_THINK_HOLD_DIRECTED", "0.01"))
HOLD_OTHER = float(os.environ.get("ATLAS_THINK_HOLD_OTHER", "0.005"))
THINK_TIMEOUT = (3.0, 10.0)
RENDER_TIMEOUT = 5.0
THINK_REQUEST = {"stop": ["</think>"], "skip_special_tokens": False, "logprobs": 0}
HEADER = {"max_tokens": 16, "temperature": 0.0, "stop": ["]"], "logprobs": 10, "skip_special_tokens": False}
TOOL_FINISH = {"max_tokens": 256, "temperature": 0.0, "stop": ["<|im_end|>"], "skip_special_tokens": False}
REPLY = {"max_tokens": 256, "stop": ["<|im_end|>", "<tool_call>", "<think>", "</think>"],
         "skip_special_tokens": False}
CLOSE = "\n</think>\n\n"
# Finish the thought, faster (owner 11-03: keep finish, just quicker): the finish starts at the cut,
# alongside the speak/hold header request instead of after it, and the reply follows the finished
# thought in the same stream when the model's own header after </think> matches the decision.
# ATLAS_THINK_FAST_FINISH=0 restores finish-then-reply as separate requests.
FAST_FINISH = os.environ.get("ATLAS_THINK_FAST_FINISH", "1") == "1"
REPLY_MARKERS = ("<|im_end|>", "<tool_call>", "<think>", "</think>")   # REPLY's stop strings
# Gate run 1 (2026-11-02): thoughts on half-heard lines concluded "the message is incomplete, HOLD" and the
# decision followed them on finished lines. PARTIAL_MARK tells the thinker the line is still being spoken;
# CUT_NOTE (fixed text, no generation) says it has finished, right before </think>.
PARTIAL_MARK = os.environ.get("ATLAS_THINK_MARK", "")
CUT_NOTE = os.environ.get("ATLAS_THINK_CUT_NOTE", "")
# FRESH: a thinking request that sees new words starts the thought from empty (the thought at the cut
# is always about the latest line), instead of continuing a thought written on fewer words.
FRESH = os.environ.get("ATLAS_THINK_FRESH", "1") == "1"   # gate p1 failed with reopen; p1_fresh nearly passed
# Live only: pause between thinking requests so other users of the model server keep their tail
# latency (loadcheck 2026-11-02: p95 x1.64 with back-to-back thinking). Tuned with `loadcheck --pause-ms`.
PAUSE_S = float(os.environ.get("ATLAS_THINK_PAUSE_MS", "0")) / 1000.0
# FINISH: once the immediate decision is SPEAK, let the thought run on (up to this many tokens) before the
# first spoken word, so the reply follows a complete thought (reply_check 2026-11-02: invented details
# 16 -> 8 of 50, median +0.87 s before the first word). The plugin setting "finish" turns it on;
# ATLAS_THINK_FINISH_TOKENS overrides (sims).
FINISH_DEFAULT_TOKENS = 320
FINISH_TOKENS = os.environ.get("ATLAS_THINK_FINISH_TOKENS")


def finish_tokens() -> int:
    if FINISH_TOKENS is not None:
        try:
            return max(0, int(FINISH_TOKENS))
        except ValueError:
            return 0
    return FINISH_DEFAULT_TOKENS if _plugin_settings().get("finish") == "on" else 0

_WORD = re.compile(r"[a-z0-9']+")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")   # the audit's ranges, plus Hangul


# ----------------------------------------------------------------------------- switch
_flag = None
_settings = {}
_flag_lock = threading.Lock()


def _load_flag():
    """Cached `think_listen` plugin flag and settings, refreshed when the Plugins page saves."""
    global _flag, _settings
    with _flag_lock:
        if _flag is None:
            try:
                import plugins

                def _refresh():
                    global _flag, _settings
                    try:
                        _flag = bool(plugins.is_enabled("think_listen"))
                        _settings = dict(plugins.settings_of("think_listen") or {})
                    except Exception:  # noqa: BLE001
                        _flag, _settings = False, {}
                plugins.on_change(_refresh)
                _flag = bool(plugins.is_enabled("think_listen"))
                _settings = dict(plugins.settings_of("think_listen") or {})
            except Exception:  # noqa: BLE001
                _flag, _settings = False, {}


def _plugin_on() -> bool:
    _load_flag()
    return bool(_flag)


def _plugin_settings() -> dict:
    _load_flag()
    return _settings


def active(rs) -> bool:
    """Think mode may run for this remote session (live entry points only)."""
    if os.environ.get("ATLAS_THINK", "") == "0" or rs is None:
        return False
    try:
        import llm_module
        import remote_llm
        cfg = getattr(rs, "cfg", None) or {}
        if not (remote_llm.TWO_STAGE and cfg.get("vllm", True) and not cfg.get("cloud")):
            return False
        if (llm_module.PARTICIPATION_SAMPLING.get("temperature") or 0) <= 0:
            return False
        if not rs._remote_ok():
            return False
    except Exception:  # noqa: BLE001
        return False
    return _plugin_on()


# ----------------------------------------------------------------------------- text helpers
def words(text: str) -> list:
    return _WORD.findall((text or "").lower())


def new_words(a: str, b: str) -> int:
    """Words of `b` not in `a` (the tokenizer of the pipeline's `_adds_content`)."""
    seen = set(words(a))
    return sum(1 for w in words(b) if w not in seen)


def strip_label(text: str) -> str:
    from floor import strip_label as _strip
    return _strip(text or "")


def same_utterance(a: str, b: str) -> bool:
    from speech_pipeline_manager import SpeechPipelineManager
    return SpeechPipelineManager._same_utterance(a, b)


def trim_line(text: str) -> str:
    from speech_pipeline_manager import SpeechPipelineManager
    return SpeechPipelineManager._trim_transcript(text)


def repeat_at(text: str, n: int = 5, times: int = 3) -> Optional[int]:
    """Character index where the first 5-word sequence to reach a third occurrence starts, or None."""
    spans = [(m.start(), m.group(0).lower()) for m in re.finditer(r"\w+", text or "")]
    seen = Counter()
    for i in range(len(spans) - n + 1):
        key = tuple(w for _, w in spans[i:i + n])
        seen[key] += 1
        if seen[key] >= times:
            return spans[i][0]
    return None


def corrupt(text: str) -> bool:
    return bool(_CJK.search(text or "")) or any(ord(c) < 32 and c not in "\n\t\r" for c in text or "")


def strip_no_think(messages: list) -> list:
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            m["content"] = re.sub(r"^/no_think\s+", "", m["content"])
            break
    return out


def header_from_text(text: str) -> Optional[str]:
    """Today's header rule: vLLM drops the stop `]`, so cut at `]` if present, require
    `[SPEAK to=` and at most 40 characters, and add the bracket back."""
    head = (text or "").strip()
    if "]" in head:
        head = head[:head.index("]")]
    if not head.startswith("[SPEAK to=") or len(head) > 40:
        return None
    return head + "]"


# ----------------------------------------------------------------------------- server helpers
_render_cache: "OrderedDict[str, str]" = OrderedDict()
_render_lock = threading.Lock()


def render(http, base: str, model: str, messages: list, tools: Optional[list], effort: Optional[str] = EFFORT,
           timeout: float = RENDER_TIMEOUT) -> str:
    """The model's own chat template with thinking on, ending in `<think>\\n` (applied by the server)."""
    ctk: dict = {"enable_thinking": True}
    if effort:
        ctk["reasoning_effort"] = effort
    key = hashlib.sha1(json.dumps([base, model, messages, tools, ctk], sort_keys=True, default=str)
                       .encode("utf-8")).hexdigest()
    with _render_lock:
        if key in _render_cache:
            _render_cache.move_to_end(key)
            return _render_cache[key]
    body = {"model": model, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": ctk}
    if tools:
        body["tools"] = tools
    r = http.post(base + "/tokenize", json=body, timeout=timeout)
    r.raise_for_status()
    d = http.post(base + "/detokenize", json={"model": model, "tokens": r.json()["tokens"]}, timeout=timeout)
    d.raise_for_status()
    text = d.json()["prompt"]
    if not text.rstrip().endswith("<think>"):
        text += "<think>\n"
    with _render_lock:
        _render_cache[key] = text
        while len(_render_cache) > 64:
            _render_cache.popitem(last=False)
    return text


def complete(http, base: str, model: str, prompt: str, stream: bool = False, timeout=(4.0, 120.0), **params):
    """POST /v1/completions. Non-streaming: the first choice dict. Streaming: the open Response."""
    body = {"model": model, "prompt": prompt, "stream": stream, **params}
    r = http.post(base + "/v1/completions", json=body, stream=stream, timeout=timeout)
    if r.status_code >= 400:
        text = (getattr(r, "text", "") or "")[:200]
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"/v1/completions HTTP {r.status_code}: {text}")
    if stream:
        return r
    return r.json()["choices"][0]


def read_stream(resp) -> tuple:
    """A streamed completion's text and token count, read to the end."""
    parts, n = [], 0
    for raw in resp.iter_lines():
        txt = (raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw or "")).strip()
        if not txt.startswith("data:"):
            continue
        data = txt[5:].strip()
        if data == "[DONE]":
            break
        try:
            ch = (json.loads(data).get("choices") or [{}])[0]
        except ValueError:
            continue
        text = ch.get("text") or ""
        parts.append(text)
        n += len(((ch.get("logprobs") or {}).get("tokens")) or ([text] if text else []))
    return "".join(parts), n


def hold_share_completion(choice: dict) -> Optional[float]:
    """P(HOLD) from a completion's first-token top_logprobs ([H... vs [S...)."""
    top = (choice.get("logprobs") or {}).get("top_logprobs") or []
    first = (top[0] if top else None) or {}
    ph = sum(math.exp(v) for k, v in first.items() if k.lstrip().startswith("[H"))
    ps = sum(math.exp(v) for k, v in first.items() if k.lstrip().startswith("[S"))
    return ph / (ph + ps) if ph + ps > 0 else None


_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def read_tool_calls(text: str, tools: Optional[list]) -> Optional[list]:
    """The template's tool-call text -> [{"function": {"name", "arguments"}}], or None if anything is off."""
    text = (text or "").strip()
    if text.endswith("<|im_end|>"):
        text = text[:-len("<|im_end|>")].rstrip()
    if not text.startswith("<tool_call>"):
        return None
    schemas = {(t.get("function") or {}).get("name"): ((t.get("function") or {}).get("parameters") or {})
               for t in tools or []}
    calls, pos = [], 0
    for m in _CALL.finditer(text):
        if text[pos:m.start()].strip():
            return None
        pos = m.end()
        name = m[1].strip()
        if name not in schemas:
            return None
        props = schemas[name].get("properties") or {}
        body = m[2]
        if _PARAM.sub("", body).strip():
            return None
        args = {}
        for p in _PARAM.finditer(body):
            key, raw = p[1].strip(), p[2]
            typ = (props.get(key) or {}).get("type", "string")
            try:
                if typ == "integer":
                    val = int(raw.strip())
                elif typ == "number":
                    val = float(raw.strip())
                elif typ == "boolean":
                    low = raw.strip().lower()
                    if low not in ("true", "false"):
                        return None
                    val = low == "true"
                elif typ in ("object", "array"):
                    val = json.loads(raw)
                else:
                    val = raw
            except ValueError:
                return None
            args[key] = val
        if any(k not in args for k in schemas[name].get("required") or []):
            return None
        calls.append({"function": {"name": name, "arguments": args}})
    if not calls or text[pos:].strip():
        return None
    return calls


def thinking_messages(llm, mgr, line: str, history: list, inputs: tuple, room: str = "", clean: bool = True,
                      memory: str = "") -> tuple:
    """(messages, tools) for a thinking request: system prompt + decision protocol (+ room and memory:
    the live thinker and the sims' "full" variant), the cleaned history and the labeled partial line,
    without /no_think.
    clean=False: the history is already cleaned (it came from _llm_context)."""
    from conversation_log import _norm
    partial = _norm(strip_label(line))
    hist = []
    for m in history or []:
        if m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
            continue
        if m.get("role") == "user":
            h = _norm(strip_label(m["content"]))
            if h and (partial == h or partial.startswith(h + " ")):
                continue                     # this utterance, already committed
        hist.append(dict(m))
    if clean:
        try:
            import capability as _cap
            hist = _cap.scrub_refusals(hist, inputs[0], inputs[1])
        except Exception:  # noqa: BLE001
            pass
        hist = mgr._clean_history(hist)
    messages = llm.build_messages(line, hist, True, True, memory or "", room, log=False)
    try:
        import remote_llm
        messages = remote_llm.translate_messages(messages)
    except Exception:  # noqa: BLE001
        pass
    return strip_no_think(messages), getattr(llm, "tools", None)


# ----------------------------------------------------------------------------- thinker
@dataclass
class Thought:
    text: str = ""
    tokens: int = 0
    closed: bool = False
    repeated: bool = False
    reopened: int = 0
    started_at: float = 0.0
    last_token_at: float = 0.0
    label: Optional[str] = None
    age_s: float = 0.0


class Thinker:
    """One utterance's private thought. `step` runs one thinking request; `cut` hands it over once."""

    def __init__(self, cfg_fn: Callable[[], dict], messages_fn: Callable[[str, Optional[str]], tuple],
                 enabled_fn: Callable[[], bool] = lambda: True, clock: Callable[[], float] = time.monotonic,
                 skip_fn: Optional[Callable[[str], bool]] = None, http=None):
        self.cfg_fn, self.messages_fn, self.enabled_fn, self.clock = cfg_fn, messages_fn, enabled_fn, clock
        self.skip_fn = skip_fn
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._resp = None
        self._epoch = 0                     # bumped by cut() and by a new utterance: stops a step
        self._http = http
        self._http_key = None if http is None else ("injected",)
        self.event = threading.Event()      # set by hear(); the live worker waits on it
        self._clear()

    # -- state
    def _clear(self):
        self._epoch += 1
        self.partial = ""
        self.label = None
        self.label_fn = None
        self.label_done = False
        self.skipped = False
        self.failed = False
        self.ended = False
        self.thought = Thought()
        self.used = 0
        self.text_at_close = None           # partial line when the thought closed or repeated
        self.text_at_cut = None             # partial line when a cut handed the thought over
        self.consumed = False
        self.reopen_pending = False
        self._req_partial = ""              # partial line the last thinking request saw (FRESH)

    def reset(self):
        with self._lock:
            self._clear()

    def hear(self, text: str, label_fn: Optional[Callable[[], Optional[str]]] = None) -> None:
        text = strip_label(text or "").strip()
        if not text:
            return
        with self._lock:
            if not self.partial or self.ended or not same_utterance(self.partial, text):
                self._clear()
                self.label_fn = label_fn
            self.partial = text
        self.event.set()

    def end(self) -> None:
        with self._lock:
            self.ended = True

    def snapshot(self) -> dict:
        with self._lock:
            return {"partial": self.partial, "used": self.used, "tokens": self.thought.tokens,
                    "closed": self.thought.closed, "consumed": self.consumed, "failed": self.failed}

    # -- rules
    def want_step(self) -> int:
        """Tokens the next thinking request may produce now; 0 = idle (the Thinking rules)."""
        try:
            if not self.enabled_fn():
                return 0
        except Exception:  # noqa: BLE001
            return 0
        with self._lock:
            if not self.partial or self.ended or self.skipped or self.failed:
                return 0
            if len(words(self.partial)) < START_WORDS:
                return 0
            left = CAP - self.used
            if left <= 0:
                return 0
            if self.consumed:
                if new_words(self.text_at_cut or "", self.partial) < REOPEN_WORDS:
                    return 0
                self.consumed = False
                self.thought.closed = self.thought.repeated = False
                self.reopen_pending = bool(self.thought.text)
            elif self.thought.closed or self.thought.repeated:
                if new_words(self.text_at_close or "", self.partial) < REOPEN_WORDS:
                    return 0
                self.thought.closed = self.thought.repeated = False
                self.reopen_pending = True
            partial = self.partial
        if self.skip_fn is not None:
            try:
                if self.skip_fn(partial):
                    return 0
            except Exception:  # noqa: BLE001
                pass
        return left

    def _session(self, cfg):
        if self._http_key == ("injected",):
            return self._http
        import requests
        key = (cfg.get("url"), cfg.get("key"))
        if self._http is None or self._http_key != key:
            self._http = requests.Session()
            if cfg.get("key"):
                self._http.headers["Authorization"] = "Bearer " + cfg["key"]
            self._http_key = key
        return self._http

    def _line(self) -> str:
        body = trim_line(self.partial) + PARTIAL_MARK
        return f"[{self.label}] {body}" if self.label else body

    def _fail(self, why: str):
        with self._lock:
            first = not self.failed
            self.failed = True
        if first:
            logger.warning("think_listen: thinking stopped for this utterance (%s)", why[:160])

    def step(self, max_tokens: int, should_stop: Callable[[], bool] = lambda: False) -> int:
        """One streamed thinking request; returns the tokens added."""
        if max_tokens <= 0:
            return 0
        with self._lock:
            fn = None if self.label_done else self.label_fn
            self.label_done = True
        if fn is not None:
            try:
                lab = fn()
            except Exception:  # noqa: BLE001
                lab = None
            with self._lock:
                if lab == "self":
                    self.skipped = True
                    return 0
                self.label = lab if lab and re.fullmatch(r"S\d+", str(lab)) else None
        with self._lock:
            ep = self._epoch
            line, label = self._line(), self.label
            if FRESH and self.thought.text and new_words(self._req_partial, self.partial) > 0:
                self.thought = Thought(reopened=self.thought.reopened + 1)
                self.reopen_pending = False
            self._req_partial = self.partial
            if self.reopen_pending and self.thought.text:
                self.thought.text = self.thought.text.rstrip("\n") + "\n\n"
                self.thought.reopened += 1
            self.reopen_pending = False
            if not self.thought.started_at:
                self.thought.started_at = self.clock()
                self.thought.label = label
            thought = self.thought.text
        try:
            cfg = self.cfg_fn() or {}
            base, model = cfg.get("url", ""), cfg.get("model", "")
            http = self._session(cfg)
            messages, tools = self.messages_fn(line, label)
            prompt = render(http, base, model, strip_no_think(messages), tools)
            resp = complete(http, base, model, prompt + thought, stream=True, timeout=THINK_TIMEOUT,
                            max_tokens=int(max_tokens), **SAMPLING, **THINK_REQUEST)
        except Exception as e:  # noqa: BLE001
            self._fail(f"{type(e).__name__}: {e}")
            return 0
        with self._lock:
            if self._epoch != ep:
                try:
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass
                return 0
            self._resp = resp
            self._idle.clear()
        added = 0
        try:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                txt = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
                txt = txt.strip()
                if not txt.startswith("data:"):
                    continue
                data = txt[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                ch = (ev.get("choices") or [{}])[0]
                text = ch.get("text") or ""
                toks = ((ch.get("logprobs") or {}).get("tokens")) or ([text] if text else [])
                stop = False
                with self._lock:
                    if self._epoch != ep:
                        break
                    if text:
                        self.thought.text += text
                        self.thought.tokens += len(toks)
                        self.thought.last_token_at = self.clock()
                        self.used += len(toks)
                        added += len(toks)
                        idx = repeat_at(self.thought.text)
                        if idx is not None:
                            self.thought.text = self.thought.text[:idx].rstrip()
                            self.thought.repeated = True
                            self.text_at_close = self.partial
                            stop = True
                    if ch.get("finish_reason") == "stop" and ch.get("stop_reason") == "</think>":
                        self.thought.closed = True
                        self.text_at_close = self.partial
                        stop = True
                    if self.used >= CAP:
                        stop = True
                if stop:
                    break
                try:
                    if not self.enabled_fn() or should_stop():
                        break
                except Exception:  # noqa: BLE001
                    break
        except Exception:  # noqa: BLE001 - urllib3 raises when a cut closes the stream from another thread
            pass
        finally:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass
            with self._lock:
                self._resp = None
            self._idle.set()
        return added

    def cut(self, line: str) -> Optional[Thought]:
        """End thinking and hand the thought over once (None if not this utterance or nothing to give)."""
        with self._lock:
            self._epoch += 1
            resp = self._resp
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001 - urllib3 raises when closed from another thread
                pass
        self._idle.wait(CUT_WAIT_S)
        with self._lock:
            if not self.partial or self.consumed or not self.thought.text.strip():
                return None
            if not same_utterance(strip_label(line or ""), self.partial):
                return None
            self.consumed = True
            self.text_at_cut = self.partial
            out = replace(self.thought)
            out.age_s = max(0.0, self.clock() - (self.thought.started_at or self.clock()))
            return out
