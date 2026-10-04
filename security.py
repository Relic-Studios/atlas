"""Local-server hardening (ASGI middleware), v0.2.0.

Threat: ATLAS is an HTTP + WebSocket server on the user's machine. Without
checks, ANY web page the user visits can talk to it:
  * cross-site requests: a page POSTs to http://localhost:8000/api/setup/llm
    (text/plain bodies skip CORS preflight, and the old CORS was "*");
  * cross-site WebSockets: browsers apply no CORS to ws://, so a page could
    open /ws or /telemetry and read live transcripts and logs;
  * DNS rebinding: evil.example re-resolves to 127.0.0.1, making requests
    "same-origin" for the attacker while the Host header says evil.example.
The loopback "owner" check alone stops none of these: the browser IS local.

Rules:
  1. Host header must be a name this server is really reachable as
     (loopback names; in the dev/LAN build also this machine's own IPs and
     hostname; plus ATLAS_ALLOWED_HOSTS). Blocks DNS rebinding.
  2. A browser Origin, when present, must be one of those hosts. Applies to
     every WebSocket and to every non-GET/HEAD/OPTIONS request. Non-browser
     clients (curl, the installer, tests) send no Origin and are unaffected.
  3. Security headers on every HTTP response.
"""
from __future__ import annotations

import os
import socket
from urllib.parse import urlsplit

LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"content-security-policy", b"frame-ancestors 'none'; base-uri 'self'; form-action 'self'"),
    (b"permissions-policy", b"camera=(), geolocation=(), payment=()"),
]


def _strip_port(host: str) -> str:
    host = (host or "").strip().lower()
    if host.startswith("["):                       # [::1]:8000
        end = host.find("]")
        return host[: end + 1] if end > 0 else host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


def local_names(lan: bool) -> set[str]:
    names = set(LOOPBACK_NAMES)
    extra = os.environ.get("ATLAS_ALLOWED_HOSTS", "")
    names |= {h.strip().lower() for h in extra.split(",") if h.strip()}
    if lan:
        try:
            hn = socket.gethostname()
            names.add(hn.lower())
            for info in socket.getaddrinfo(hn, None):
                names.add(str(info[4][0]).lower())
        except OSError:
            pass
    return names


class LocalGuard:
    """Pure ASGI middleware so it covers HTTP and WebSocket scopes alike."""

    def __init__(self, app, lan: bool = False, names: set[str] | None = None):
        self.app = app
        self.names = names if names is not None else local_names(lan)

    # -- decisions (pure; unit-tested) ------------------------------------
    def host_ok(self, host: str) -> bool:
        if not host:                                # HTTP/1.0 clients; only local tools do this
            return True
        return _strip_port(host) in self.names

    def origin_ok(self, origin: str | None) -> bool:
        if origin is None:                          # not a browser
            return True
        o = origin.strip().lower()
        if o in ("", "null"):                       # sandboxed iframes / file:// pages
            return False
        parts = urlsplit(o)
        if parts.scheme not in ("http", "https"):
            return False
        return _strip_port(parts.netloc) in self.names

    # -- ASGI --------------------------------------------------------------
    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        hdrs = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        origin = hdrs.get("origin")
        bad = None
        if not self.host_ok(hdrs.get("host", "")):
            bad = "host"
        elif kind == "websocket" and not self.origin_ok(origin):
            bad = "origin"
        elif kind == "http" and scope.get("method", "GET") not in SAFE_METHODS and not self.origin_ok(origin):
            bad = "origin"
        elif kind == "http" and origin is not None and not self.origin_ok(origin) \
                and str(scope.get("path", "")).startswith("/api/"):
            bad = "origin"                          # cross-site reads of API data too
        if bad:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            body = b'{"error":"blocked: request did not come from the ATLAS app"}'
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode())] + SECURITY_HEADERS})
            await send({"type": "http.response.body", "body": body})
            return
        if kind == "websocket":
            return await self.app(scope, receive, send)

        async def send_wrapped(msg):
            if msg.get("type") == "http.response.start":
                have = {k.lower() for k, _ in msg.get("headers") or []}
                msg = dict(msg)
                msg["headers"] = list(msg.get("headers") or []) + [h for h in SECURITY_HEADERS if h[0] not in have]
            await send(msg)
        return await self.app(scope, receive, send_wrapped)


# -- secrets at rest -----------------------------------------------------------
# Windows: DPAPI (CurrentUser) via ctypes, no extra dependency. The settings file
# then holds "dpapi:<base64>", useless if copied to another account or machine.
# Elsewhere: plaintext in a 0600 file (user_settings.save chmods it).
def protect(secret: str) -> str:
    if not secret or os.name != "nt" or secret.startswith("dpapi:"):
        return secret
    try:
        import base64
        blob = _dpapi(secret.encode("utf-8"), encrypt=True)
        return "dpapi:" + base64.b64encode(blob).decode("ascii")
    except Exception:  # noqa: BLE001  never lose the user's key over this
        return secret


def unprotect(stored: str) -> str:
    if not stored or not stored.startswith("dpapi:"):
        return stored or ""
    try:
        import base64
        return _dpapi(base64.b64decode(stored[6:]), encrypt=False).decode("utf-8")
    except Exception:  # noqa: BLE001  e.g. settings copied from another account
        return ""


def _dpapi(data: bytes, encrypt: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    inp = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = BLOB()
    crypt = ctypes.windll.crypt32
    fn = crypt.CryptProtectData if encrypt else crypt.CryptUnprotectData
    ok = fn(ctypes.byref(inp), None, None, None, None, 0x1, ctypes.byref(out))  # UI_FORBIDDEN
    if not ok:
        raise OSError("DPAPI failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)
