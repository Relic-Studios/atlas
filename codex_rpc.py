"""Bounded stdio client for the installed official Codex app-server.

Never reads auth.json, invokes a shell, or writes the owner's CLI configuration.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from pathlib import Path

from provider_errors import ProviderError, failure

DISABLED_FEATURES = (
    "shell_tool", "multi_agent", "multi_agent_v2", "goals", "hooks", "plugins", "apps",
    "memories", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "computer_use", "in_app_browser", "code_mode", "code_mode_host", "artifact",
    "image_generation", "view_image", "remote_plugin", "skill_search", "skill_mcp_dependency_install",
    "tool_suggest", "request_permissions_tool", "shell_snapshot",
)
ALLOWED_RPC = frozenset({"initialize", "account/read", "account/login/start", "account/login/cancel",
    "account/rateLimits/read", "model/list", "config/read", "experimentalFeature/list", "skills/list",
    "thread/start", "thread/inject_items", "thread/unsubscribe", "turn/start", "turn/interrupt"})
LOGIN_RPC = frozenset({"account/login/start", "account/login/cancel"})


def executable(provider: str, configured: str | None = None) -> str:
    if configured:
        p = Path(configured).expanduser()
        if p.is_file() and p.suffix.lower() not in (".cmd", ".bat", ".ps1"):
            return str(p.resolve())
        raise ProviderError(provider, "missing_cli", f"Choose the installed official {provider} executable.")
    candidates = [shutil.which(provider + ".exe" if os.name == "nt" else provider)]
    if provider == "codex":
        candidates.append(str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/OpenAI/Codex/bin/codex.exe"))
    else:
        candidates.append(str(Path.home() / ".local/bin/claude.exe"))
    for item in candidates:
        if item and Path(item).is_file() and Path(item).suffix.lower() not in (".cmd", ".bat", ".ps1"):
            return str(Path(item).resolve())
    raise ProviderError(provider, "missing_cli", f"Install the official {provider} CLI, then retry Connect.")


def child_environment(provider: str) -> dict:
    # Keep platform/keychain/certificate access, not inherited agent configuration or credentials.
    allowed={"SYSTEMROOT","WINDIR","COMSPEC","PATH","PATHEXT","TEMP","TMP","USERPROFILE","HOME",
        "HOMEDRIVE","HOMEPATH","APPDATA","LOCALAPPDATA","PROGRAMDATA","PROGRAMFILES","PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES","COMMONPROGRAMFILES(X86)","LANG","LC_ALL","LC_CTYPE","TERM",
        "HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","NO_PROXY","SSL_CERT_FILE","SSL_CERT_DIR","NODE_EXTRA_CA_CERTS",
        "REQUESTS_CA_BUNDLE","CLAUDE_CONFIG_DIR","CODEX_HOME","XDG_RUNTIME_DIR","XDG_CONFIG_HOME","XDG_CACHE_HOME",
        "DBUS_SESSION_BUS_ADDRESS","DISPLAY","WAYLAND_DISPLAY"}
    env = {key:value for key,value in os.environ.items() if key.upper() in allowed}
    # Account mode cannot be overridden by an inherited API key or alternate billing endpoint.
    drop = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID") if provider == "codex" else (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")
    for key in drop:
        env.pop(key, None)
    env.pop("CLAUDECODE", None)
    return env


def config_overrides() -> list[str]:
    """Only names/paths are read from config; credentials never leave their CLI owner."""
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        config = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
    except FileNotFoundError:
        config = {}
    except (OSError, ValueError) as exc:
        raise ProviderError("codex", "configuration", "Codex configuration could not be inspected safely.") from exc
    selected_profile = config.get("profiles", {}).get(config.get("profile"), {})
    if any(layer.get(key) for layer in (config, selected_profile)
           for key in ("openai_base_url", "chatgpt_base_url")):
        raise ProviderError("codex", "configuration", "Atlas account inference requires the official Codex service endpoints. Remove custom endpoint overrides in Codex before connecting.")
    values = {"model_provider": "openai", "approval_policy": "never", "sandbox_mode": "read-only",
              "web_search": "disabled", "project_doc_max_bytes": 0,
              "skills.bundled.enabled": False, "skills.include_instructions": False}
    values.update({"features." + key: False for key in DISABLED_FEATURES})
    for name in config.get("mcp_servers", {}):
        if "." in name:
            raise ProviderError("codex", "configuration", "An inherited Codex MCP identifier cannot be isolated by this CLI version.")
        values['mcp_servers.' + name + '.enabled'] = False
    for name in config.get("plugins", {}):
        if "." in name:
            raise ProviderError("codex", "configuration", "An inherited Codex plugin identifier cannot be isolated by this CLI version.")
        values['plugins.' + name + '.enabled'] = False
    args = []
    for key, value in values.items():
        args += ["-c", key + "=" + json.dumps(value, ensure_ascii=False)]
    return args


class CodexRPC:
    def __init__(self, configured=None, *, timeout=20.0, extra_overrides=(), cancel_event=None, allow_login=False):
        self.timeout = timeout
        self.cancel_event = cancel_event
        self.allowed_rpc = ALLOWED_RPC if allow_login else ALLOWED_RPC - LOGIN_RPC
        self._max_bytes=(1 if allow_login else 8)*1024*1024
        self._login_output=0 if allow_login else None
        self._temp = tempfile.TemporaryDirectory(prefix="atlas-codex-")
        self.cwd = self._temp.name
        self._write_lock = threading.Lock()
        self._lock = threading.Lock()
        self._next_id = 0
        self._pending = {}
        self.events = queue.Queue(maxsize=256 if allow_login else 1024)
        self._event_bytes = 0
        self._closed = threading.Event()
        self.error = None
        self.proc = None
        try:
            command = [executable("codex", configured), "app-server", "--listen", "stdio://"]
            command += config_overrides() + list(extra_overrides)
            self.proc = subprocess.Popen(command, cwd=self.cwd, env=child_environment("codex"),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._reader = threading.Thread(target=self._read, daemon=True)
            self._reader.start()
            self.call("initialize", {"clientInfo": {"name": "atlas", "title": "ATLAS", "version": "0.7.0"},
                                     "capabilities": {"experimentalApi": True}})
            self._send({"method": "initialized", "params": {}})
        except BaseException:
            self.close()
            raise

    def _send(self, value):
        if self._closed.is_set():
            raise ProviderError("codex", "cancelled", "The Codex request has ended.")
        raw = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        if len(raw) > 8 * 1024 * 1024:
            raise ProviderError("codex", "request_too_large", "The Codex request exceeds Atlas's transport limit.")
        with self._write_lock:
            try:
                self.proc.stdin.write(raw)
                self.proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise ProviderError("codex", "process_closed", "The Codex connection closed.") from exc

    def _read(self):
        try:
            while not self._closed.is_set():
                raw = self.proc.stdout.readline(self._max_bytes + 1)
                if not raw:
                    break
                if self._login_output is not None:
                    self._login_output+=len(raw)
                if len(raw) > self._max_bytes or self._login_output is not None and self._login_output>self._max_bytes:
                    raise ValueError("oversized event")
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError("invalid event")
                if "id" in event and "method" not in event:
                    with self._lock:
                        waiter = self._pending.get(event["id"])
                    if waiter:
                        waiter.put_nowait(event)
                elif "method" in event:
                    with self._lock:
                        self._event_bytes += len(raw)
                        if self._event_bytes > self._max_bytes:
                            raise queue.Full()
                    event["_atlas_wire_bytes"] = len(raw)
                    self.events.put_nowait(event)
        except (OSError, ValueError, queue.Full) as exc:
            self.error = ProviderError("codex", "protocol", "The Codex protocol stream was invalid or exceeded its limit.")
        finally:
            if not self.error:
                self.error = ProviderError("codex", "process_closed", "The Codex connection ended before the request completed.")
            with self._lock:
                for waiter in self._pending.values():
                    try:
                        waiter.put_nowait({"_error": self.error})
                    except queue.Full:
                        pass

    def call(self, method, params=None, timeout=None):
        if method not in self.allowed_rpc:
            raise ProviderError("codex", "protocol", "Atlas refused an unsupported Codex operation.")
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            waiter = queue.Queue(maxsize=1)
            self._pending[rid] = waiter
        try:
            self._send({"id": rid, "method": method, "params": params or {}})
            deadline = time.monotonic() + (timeout or self.timeout)
            while True:
                if method != "turn/interrupt" and self.cancel_event is not None and self.cancel_event.is_set():
                    raise ProviderError("codex", "cancelled", "The Codex request was cancelled.")
                try:
                    result = waiter.get(timeout=min(.1, max(.001, deadline-time.monotonic())))
                    break
                except queue.Empty as exc:
                    if time.monotonic() >= deadline:
                        raise ProviderError("codex", "timeout", "Codex did not answer in time.") from exc
            if "_error" in result:
                raise result["_error"]
            if result.get("error"):
                raise failure("codex", result["error"])
            return result.get("result", {})
        finally:
            with self._lock:
                self._pending.pop(rid, None)

    def next_event(self, timeout=1):
        try:
            event = self.events.get(timeout=timeout)
            with self._lock:
                self._event_bytes -= event.pop("_atlas_wire_bytes", 0)
            return event
        except queue.Empty:
            if self.error:
                raise self.error
            return None

    def reject(self, event):
        if "id" in event:
            self._send({"id": event["id"], "error": {"code": -32601, "message": "Atlas does not permit this operation"}})

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            for stream in (self.proc.stdin, self.proc.stdout):
                if stream:
                    stream.close()
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
