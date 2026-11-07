"""Official CLI-owned account verification and cancellable browser sign-in."""
from __future__ import annotations

import json
import hashlib
import queue
import re
import subprocess
import threading
import time
import uuid
from urllib.parse import urlsplit

from codex_rpc import CodexRPC, child_environment, executable
from provider_errors import ProviderError

PROVIDERS = {"codex", "claude_code"}
AUTH_HOSTS = {"codex": {"auth.openai.com"}, "claude_code": {
    "claude.ai", "auth.claude.com", "auth.anthropic.com", "console.anthropic.com", "platform.claude.com"}}
_attempts = {}
_lock = threading.RLock()
LOGIN_LIFETIME = 600


def auth_url(provider, url):
    try:
        p = urlsplit(url)
        if (p.scheme == "https" and p.hostname in AUTH_HOSTS[provider] and not p.username and not p.password
                and p.port in (None, 443) and p.path in ("/oauth/authorize", "/authorize", "/codex/device")):
            return url
    except (ValueError, KeyError):
        pass
    raise ProviderError(provider, "protocol", "The CLI returned an unexpected authorization address.")


def _check_provider(provider):
    if provider not in PROVIDERS:
        raise ProviderError("account", "configuration", "Choose a supported account connector.")


def account_binding(provider, info):
    fields = ("type", "email", "planType", "subscriptionType", "accountId", "chatgptAccountId", "orgId", "organizationId", "workspaceId")
    identity = {key: info[key] for key in fields if info.get(key) is not None}
    if not any(identity.get(k) for k in ("email", "accountId", "chatgptAccountId")):
        raise ProviderError(provider, "incompatible_cli", "The CLI did not provide an account identity. Update it before verifying this connection.")
    return hashlib.sha256((provider+json.dumps(identity,sort_keys=True)).encode("utf-8")).hexdigest()


def bounded_output(args, env, timeout=20, maximum=1024*1024):
    """Read official account status without unbounded captured output or stderr."""
    proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    chunks=[]
    invalid=threading.Event()
    def read():
        size=0
        try:
            while True:
                raw=proc.stdout.read(4096)
                if not raw:break
                size+=len(raw)
                if size>maximum:
                    invalid.set();proc.terminate();break
                chunks.append(raw)
        except (OSError,ValueError):
            invalid.set()
    reader=threading.Thread(target=read,daemon=True);reader.start()
    try:
        proc.wait(timeout=timeout)
        reader.join(timeout=2)
        if invalid.is_set() or reader.is_alive():
            raise ProviderError("claude_code","protocol","Claude Code returned an invalid or oversized account status.")
        return b"".join(chunks).decode("utf-8",errors="replace")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait(timeout=2)
        proc.stdout.close();reader.join(timeout=2)


def _remaining(provider, deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:
        raise ProviderError(provider,"timeout","The provider account discovery timed out. Retry explicitly.")
    return remaining


def account_status(provider, configured=None):
    _check_provider(provider)
    if provider == "codex":
        deadline=time.monotonic()+20
        with CodexRPC(configured) as rpc:
            result = rpc.call("account/read", {"refreshToken": False},timeout=_remaining(provider,deadline))
            account = result.get("account") or {}
            connected = account.get("type") == "chatgpt"
            limits = rpc.call("account/rateLimits/read",timeout=_remaining(provider,deadline)) if connected else {}
            return {"provider": provider, "connected": connected, "auth_mode": "subscription" if connected else "signed_out",
                    "account_binding": account_binding(provider, account) if connected else None,
                    "plan": account.get("planType") if connected else None, "rate_limits": limits.get("rateLimits"),
                    "message": "Codex account verified." if connected else "Connect your ChatGPT account through Codex. API-key login is a separate billing path."}
    try:
        output = bounded_output([executable("claude", configured), "auth", "status"],child_environment("claude"))
        info = json.loads(output)
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(provider, "timeout", "Claude Code did not answer its account check in time.") from exc
    except (ValueError, OSError) as exc:
        raise ProviderError(provider, "protocol", "Claude Code could not report its account status. Update the official CLI and retry.") from exc
    connected = (info.get("loggedIn") is True and info.get("authMethod")=="claude.ai"
                 and info.get("apiProvider") in (None,"firstParty"))
    return {"provider": provider, "connected": connected, "auth_mode": "subscription" if connected else "signed_out",
            "account_binding": account_binding(provider, info) if connected else None,
            "plan": info.get("subscriptionType") if connected else None,
            "message": "Claude Code subscription verified." if connected else "Connect your Claude subscription through Claude Code. Atlas will not use an API key in account mode."}


def account_models(provider, configured=None):
    _check_provider(provider)
    if provider == "claude_code":
        # Supported CLI aliases; this is not an entitlement/capability test or inference.
        return [{"id": name, "label": label, "vision": None, "tools": None, "source": "Claude Code model aliases"}
                for name, label in (("sonnet", "Sonnet"), ("opus", "Opus"), ("haiku", "Haiku"))]
    models, cursor = [], None
    deadline=time.monotonic()+20
    with CodexRPC(configured) as rpc:
        for _ in range(20):
            page = rpc.call("model/list", {"cursor": cursor, "limit": 100, "includeHidden": False},timeout=_remaining(provider,deadline))
            for model in page.get("data", []):
                modalities = model.get("inputModalities")
                models.append({"id": model.get("model") or model["id"], "label": model.get("displayName") or model["id"],
                    "vision": ("image" in modalities) if isinstance(modalities, list) else None,
                    "tools": None, "source": "Codex model/list", "default": bool(model.get("isDefault"))})
            cursor = page.get("nextCursor")
            if not cursor:
                break
        else:
            raise ProviderError(provider, "protocol", "The model catalog exceeded Atlas's discovery limit.")
    return models


class LoginAttempt:
    def __init__(self, provider, configured):
        self.id, self.provider = uuid.uuid4().hex, provider
        self.configured = configured
        self.created = time.monotonic()
        self.state = "starting"
        self.url = None
        self.account = None
        self.error = None
        self.proc = None
        self.rpc = None
        self.login_id = None
        self.cancelled = threading.Event()
        self._lock = threading.RLock()

    def snapshot(self):
        with self._lock:
            return {"id": self.id, "provider": self.provider, "state": self.state,
                    "authorization_url": self.url, "account": self.account, "error": self.error}

    def cancel(self, expired=False):
        with self._lock:
            if self.state in ("connected", "error", "cancelled", "expired"):
                return
            self.cancelled.set()
            self.state = "expired" if expired else "cancelled"
            self.url = None
            rpc, proc, login_id = self.rpc, self.proc, self.login_id
        if rpc:
            try:
                if login_id:
                    rpc.call("account/login/cancel", {"loginId": login_id}, timeout=2)
            except Exception:
                pass
            rpc.close()
        if proc and proc.poll() is None:
            proc.terminate()

    def _set_url(self, url):
        with self._lock:
            if not self.cancelled.is_set():
                self.url = auth_url(self.provider, url)
                self.state = "waiting_for_browser"

    def _run_codex(self):
        with CodexRPC(self.configured,allow_login=True) as rpc:
            with self._lock:
                self.rpc = rpc
                if self.cancelled.is_set():
                    return
            result = rpc.call("account/login/start", {"type": "chatgpt"})
            with self._lock:
                self.login_id = result.get("loginId")
            self._set_url(result["authUrl"])
            while not self.cancelled.is_set():
                if time.monotonic() - self.created > LOGIN_LIFETIME:
                    self.cancel(expired=True)
                    return
                event = rpc.next_event(timeout=1)
                if not event:
                    continue
                if "id" in event:
                    rpc.reject(event)
                    raise ProviderError(self.provider, "protocol", "The account connection requested an unsupported operation.")
                params = event.get("params", {})
                if event.get("method") == "account/login/completed" and params.get("loginId") == self.login_id:
                    if not params.get("success"):
                        raise ProviderError(self.provider, "sign_in_required", "Codex sign-in did not complete. Connect again to retry.")
                    return

    def _run_claude(self):
        with self._lock:
            if self.cancelled.is_set():
                return
            proc = subprocess.Popen([executable("claude", self.configured), "auth", "login", "--claudeai"],
                env=child_environment("claude"), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self.proc = proc
        output = queue.Queue(maxsize=256)

        def read():
            total=0
            try:
                for line in iter(lambda:proc.stdout.readline(1024*1024+1), b""):
                    total+=len(line)
                    if total>1024*1024:
                        proc.terminate()
                        break
                    output.put(line, timeout=1)
            except (OSError, queue.Full):
                proc.terminate()
            finally:
                try:
                    output.put(None, timeout=1)
                except queue.Full:
                    pass
        threading.Thread(target=read, daemon=True).start()
        size = 0
        try:
            while not self.cancelled.is_set():
                if time.monotonic() - self.created > LOGIN_LIFETIME:
                    self.cancel(expired=True)
                    return
                try:
                    line = output.get(timeout=1)
                except queue.Empty:
                    continue
                if line is None:
                    break
                size += len(line)
                if size > 1024 * 1024:
                    raise ProviderError(self.provider, "protocol", "Claude Code sign-in exceeded its output limit.")
                for url in re.findall(r'https://[^\s\x1b<>"\']+', line.decode("utf-8", errors="replace")):
                    try:
                        self._set_url(url)
                    except ProviderError:
                        continue  # ignore help/documentation links, never expose them as authorization
            if not self.cancelled.is_set() and proc.wait(timeout=2) != 0:
                raise ProviderError(self.provider, "sign_in_required", "Claude Code sign-in did not complete. Connect again to retry.")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
            proc.stdout.close()

    def run(self):
        try:
            existing = account_status(self.provider, self.configured)
            if existing["connected"]:
                with self._lock:
                    if not self.cancelled.is_set():
                        self.account, self.state = existing, "connected"
                return
            if self.cancelled.is_set():
                return
            from native_providers import invalidate_provider
            invalidate_provider(self.provider)
            (self._run_codex if self.provider == "codex" else self._run_claude)()
            if self.cancelled.is_set():
                return
            account = account_status(self.provider, self.configured)
            if not account["connected"]:
                raise ProviderError(self.provider, "sign_in_required", account["message"])
            with self._lock:
                if not self.cancelled.is_set():
                    self.account, self.state, self.url = account, "connected", None
        except Exception as exc:
            error = exc if isinstance(exc, ProviderError) else ProviderError(self.provider, "connection_failed", "The account connection failed. Update the official CLI and retry Connect.")
            with self._lock:
                if not self.cancelled.is_set():
                    self.state, self.error, self.url = "error", error.public(), None


def start_login(provider, configured=None):
    _check_provider(provider)
    executable("claude" if provider == "claude_code" else "codex", configured)
    with _lock:
        for key, old in list(_attempts.items()):
            if time.monotonic() - old.created > LOGIN_LIFETIME:
                old.cancel(expired=True)
                del _attempts[key]
            elif old.provider == provider and old.state in ("starting", "waiting_for_browser"):
                return old.snapshot()
        if len(_attempts) >= 16:
            terminal = [key for key, a in _attempts.items() if a.state in ("connected", "error", "cancelled", "expired")]
            if terminal:
                del _attempts[terminal[0]]
            else:
                raise ProviderError(provider, "busy", "An account connection is already running. Finish or cancel it first.")
        attempt = LoginAttempt(provider, configured)
        _attempts[attempt.id] = attempt
    threading.Thread(target=attempt.run, daemon=True).start()
    return attempt.snapshot()


def login_state(attempt_id, cancel=False):
    with _lock:
        attempt = _attempts.get(attempt_id)
    if not attempt:
        raise ProviderError("account", "not_found", "This sign-in attempt has expired. Connect again.")
    if cancel or time.monotonic() - attempt.created > LOGIN_LIFETIME:
        attempt.cancel(expired=not cancel)
    return attempt.snapshot()


def close_logins():
    with _lock:
        attempts = list(_attempts.values())
    for attempt in attempts:
        attempt.cancel()
