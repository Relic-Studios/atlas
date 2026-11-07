"""Native account sessions implementing Atlas's existing Ollama response contract."""
from __future__ import annotations

import json
import subprocess
import threading
import uuid
import weakref

import requests

from provider_errors import ProviderError

_sessions = weakref.WeakSet()
_sessions_lock = threading.Lock()


def invalidate_provider(provider):
    with _sessions_lock:
        sessions = [s for s in _sessions if s.cfg.get("provider") == provider]
    for session in sessions:
        session.invalidate_account()


class NativeResponse:
    def __init__(self, session, payload):
        self.session, self.payload = session, payload
        self.request_id = payload.get("_atlas_request_id") or uuid.uuid4().hex
        self.cancelled = threading.Event()
        self._hook = None
        self._lock = threading.Lock()
        self._consumed = False
        self.status_code = 200

    def set_cancel_hook(self, hook):
        with self._lock:
            self._hook = hook
            cancelled = self.cancelled.is_set()
        if cancelled:
            hook()

    def raise_for_status(self):
        if self.cancelled.is_set():
            raise ProviderError(self.session.cfg["provider"], "cancelled", "The request was cancelled.")

    def _events(self):
        if self._consumed:
            raise ProviderError(self.session.cfg["provider"], "protocol", "The provider response was already consumed.")
        self._consumed = True
        try:
            if self.cancelled.is_set():
                return
            provider = self.session.cfg["provider"]
            if provider == "codex":
                from codex_native import generate
            elif provider == "claude_code":
                from claude_native import generate
            else:
                raise ProviderError(provider, "configuration", "Unknown native account transport.")
            self.session.requests += 1
            total = 0
            done = False
            for event in generate(self.session.cfg, self.payload, self):
                if self.cancelled.is_set():
                    return
                total += len(json.dumps(event,ensure_ascii=False).encode("utf-8"))
                if total > 8 * 1024 * 1024:
                    raise ProviderError(provider, "protocol", "The provider response exceeded Atlas's size limit.")
                done = bool(event.get("done"))
                yield {"model": self.session.cfg["model"], **event}
            if not done and not self.cancelled.is_set():
                raise ProviderError(provider, "protocol", "The provider ended without completing the response.")
            if self.cancelled.is_set():
                return
            self.session.last_error = ""
            self.session.last_provider_error = None
            self.session._remember_outcome()
        except ProviderError as exc:
            if not self.cancelled.is_set():
                self.session.last_error = str(exc)
                self.session.last_provider_error = exc.public()
                if exc.code == "usage_limit":
                    self.session.usage_paused = True
                    self.session._remember_outcome(exc.public())
                raise
        except (OSError,ValueError,TypeError,AttributeError,subprocess.SubprocessError) as exc:
            if not self.cancelled.is_set():
                error=ProviderError(self.session.cfg["provider"],"protocol","The account connector returned an invalid response or stopped unexpectedly. Verify the account and retry explicitly.")
                self.session.last_error=str(error)
                self.session.last_provider_error=error.public()
                raise error from exc
        finally:
            self.close()

    def iter_content(self, chunk_size=None):
        for event in self._events():
            yield (json.dumps(event, ensure_ascii=False)+"\n").encode("utf-8")

    def json(self):
        parts, calls = [], []
        for event in self._events():
            message = event.get("message", {})
            parts.append(message.get("content", ""))
            calls.extend(message.get("tool_calls", []))
        return {"model": self.session.cfg["model"], "done": True,
                "message": {"role":"assistant", "content":"".join(parts), **({"tool_calls":calls} if calls else {})}}

    def close(self):
        with self._lock:
            if self.cancelled.is_set():
                return
            self.cancelled.set()
            hook, self._hook = self._hook, None
        if hook:
            try:
                hook()
            except Exception:
                pass
        self.session._retire(self)


class NativeSession:
    _request_scoped_interrupt = True
    skip_startup_inference = True

    def __init__(self, cfg, local=None):
        self.cfg = dict(cfg)
        self.local = local
        self._lock = threading.Lock()
        self._responses = set()
        self.last_error = ""
        self.last_provider_error = None
        self.isolation_verified = False
        self.cli_version = self.resolved_model = None
        self.requests = self.fallbacks = self.read_timeouts = self.fails = 0
        self.local_warm = False
        self.next_thought = self.next_hold_thresh = None
        self._closed = False
        self.account_invalidated = False
        evidence = self.cfg.get("verification") or {}
        self.usage_paused = evidence.get("status") == "limited" and evidence.get("limit_active",True)
        if self.usage_paused:
            self.last_provider_error = evidence.get("provider_error")
            self.last_error = (self.last_provider_error or {}).get("message", "The selected account is usage limited. Retry explicitly from Settings.")
        with _sessions_lock:
            _sessions.add(self)

    def post(self, url, json=None, stream=False, timeout=None, **kwargs):
        if not url.endswith("/api/chat"):
            raise ProviderError(self.cfg["provider"], "protocol", "The account connector only accepts Atlas chat requests.")
        with self._lock:
            if self._closed:
                raise ProviderError(self.cfg["provider"], "cancelled", "The provider connection was closed.")
            if self.account_invalidated:
                raise ProviderError(self.cfg["provider"],"account_changed","The account connection changed. Verify and save this brain again in Settings.")
            if self.usage_paused:
                info=self.last_provider_error or {}
                raise ProviderError(self.cfg["provider"],"usage_limit",self.last_error,reset_at=info.get("reset_at"),retry_after=info.get("retry_after"))
            response = NativeResponse(self, json or {})
            self._responses.add(response)
        return response

    def _retire(self, response):
        with self._lock:
            self._responses.discard(response)

    def _remember_outcome(self,error=None):
        try:
            import user_settings
            user_settings.remember_account_outcome(self.cfg,error)
        except (OSError,ValueError):
            pass

    def interrupt(self, request_id=None):
        with self._lock:
            responses = [r for r in self._responses if request_id is None or r.request_id == request_id]
        for response in responses:
            response.close()

    def ping(self, timeout=20):
        from provider_accounts import account_status
        try:
            status = account_status(self.cfg["provider"], self.cfg.get("executable"))
            if not status["connected"]:
                raise ProviderError(self.cfg["provider"], "sign_in_required", status["message"])
            if self.cfg.get("account_binding") and status.get("account_binding")!=self.cfg["account_binding"]:
                self.invalidate_account()
                return False
            if self.account_invalidated:
                return False
            if not self.usage_paused:
                self.last_error = ""
            return True
        except ProviderError as exc:
            self.last_error, self.last_provider_error = str(exc), exc.public()
            return False

    def get(self, url, **kwargs):
        if not self.ping(timeout=kwargs.get("timeout",20)):
            raise ProviderError(self.cfg["provider"], "sign_in_required", self.last_error)
        response = requests.Response()
        response.status_code = 200
        return response

    def status(self):
        return {"label":self.cfg["label"],"model":self.cfg["model"],"cloud":True,"vllm":False,"url":"","healthy":not bool(self.last_error),
                "requests":self.requests,"fallbacks":0,"last_error":self.last_error,
                "provider_error":self.last_provider_error,"usage_paused":self.usage_paused,"allow_local_fallback":False,"local_warm":False}

    def clear_usage_limit(self):
        with self._lock:
            self.usage_paused=False
            self.last_error=""
            self.last_provider_error=None

    def invalidate_account(self):
        error=ProviderError(self.cfg["provider"],"account_changed","The account connection changed. Verify and save this brain again in Settings.")
        with self._lock:
            self.account_invalidated=True
            self.last_error=str(error)
            self.last_provider_error=error.public()
        self.interrupt()

    def start_watchdog(self, *_):
        pass

    def close(self):
        with self._lock:
            self._closed = True
        self.interrupt()
