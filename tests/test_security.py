"""v0.2.0 hardening: cross-site pages, cross-site WebSockets, DNS rebinding, headers, key at rest."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import security as S


async def _get(request):
    return JSONResponse({"ok": True})


async def _ws(ws):
    await ws.accept()
    await ws.send_text("transcript")
    await ws.close()


def make(names=None):
    app = Starlette(routes=[Route("/api/x", _get, methods=["GET", "POST"]), Route("/static/a", _get),
                            WebSocketRoute("/ws", _ws)])
    app.add_middleware(S.LocalGuard, names=names or {"localhost", "127.0.0.1", "testserver"})
    return TestClient(app)


class HostAndOrigin(unittest.TestCase):
    def setUp(self):
        self.c = make()

    def test_same_origin_ui_allowed(self):
        r = self.c.post("/api/x", headers={"origin": "http://localhost:8000", "host": "localhost:8000"})
        self.assertEqual(r.status_code, 200)

    def test_cross_site_post_blocked(self):
        r = self.c.post("/api/x", headers={"origin": "https://evil.example"}, content="{}")
        self.assertEqual(r.status_code, 403)

    def test_null_origin_post_blocked(self):  # sandboxed iframe / file:// page
        self.assertEqual(self.c.post("/api/x", headers={"origin": "null"}).status_code, 403)

    def test_cross_site_api_read_blocked(self):
        self.assertEqual(self.c.get("/api/x", headers={"origin": "https://evil.example"}).status_code, 403)

    def test_non_browser_tool_allowed(self):  # curl / installer / tests: no Origin header
        self.assertEqual(self.c.post("/api/x").status_code, 200)

    def test_dns_rebinding_blocked(self):
        r = self.c.get("/static/a", headers={"host": "evil.example:8000"})
        self.assertEqual(r.status_code, 403)

    def test_rebinding_post_with_matching_origin_blocked(self):
        r = self.c.post("/api/x", headers={"host": "evil.example:8000", "origin": "http://evil.example:8000"})
        self.assertEqual(r.status_code, 403)

    def test_cross_site_websocket_blocked(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.c.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_text()

    def test_same_origin_websocket_allowed(self):
        with self.c.websocket_connect("/ws", headers={"origin": "http://localhost:8000"}) as ws:
            self.assertEqual(ws.receive_text(), "transcript")

    def test_ipv6_loopback(self):
        g = S.LocalGuard(None, names=S.LOOPBACK_NAMES)
        self.assertTrue(g.host_ok("[::1]:8000"))
        self.assertTrue(g.origin_ok("http://[::1]:8000"))
        self.assertFalse(g.origin_ok("http://localhost.evil.example"))

    def test_security_headers(self):
        h = self.c.get("/static/a").headers
        self.assertEqual(h["x-frame-options"], "DENY")
        self.assertEqual(h["x-content-type-options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", h["content-security-policy"])


class ServerWiring(unittest.TestCase):
    def test_no_wildcard_cors_and_guard_installed(self):
        src = Path(__file__).resolve().parents[1].joinpath("server.py").read_text(encoding="utf-8")
        self.assertNotIn("CORSMiddleware", src)
        self.assertIn("app.add_middleware(_security.LocalGuard", src)

    def test_electron_window_locked_down(self):
        js = Path(__file__).resolve().parents[1].joinpath("desktop", "main.js").read_text(encoding="utf-8")
        for need in ("sandbox: true", "setWindowOpenHandler", "contextIsolation: true", "nodeIntegration: false"):
            self.assertIn(need, js)


class KeyAtRest(unittest.TestCase):
    def test_roundtrip_and_file_never_holds_plaintext_on_windows(self):
        import user_settings as US
        old = US.PATH
        with tempfile.TemporaryDirectory() as d:
            US.PATH = Path(d) / "settings.json"
            try:
                US.update({"llm": {"cloud": {"provider": "openai", "base_url": "https://x/v1",
                                             "model": "m", "api_key": "sk-test-SECRET-123456"}}})
                self.assertEqual(US.load()["llm"]["cloud"]["api_key"], "sk-test-SECRET-123456")
                self.assertEqual(US.cloud_config()["key"], "sk-test-SECRET-123456")
                raw = US.PATH.read_text(encoding="utf-8")
                if os.name == "nt":
                    self.assertNotIn("SECRET", raw)
                    self.assertIn("dpapi:", raw)
                US.update({"audio": {"input": 1}})          # re-save must not double-encrypt
                self.assertEqual(US.load()["llm"]["cloud"]["api_key"], "sk-test-SECRET-123456")
                self.assertNotIn("SECRET", json.dumps(US.public_view()))
            finally:
                US.PATH = old

    def test_unreadable_blob_fails_closed(self):
        self.assertEqual(S.unprotect("dpapi:AAAA"), "")
        self.assertEqual(S.unprotect("plain"), "plain")


if __name__ == "__main__":
    unittest.main()
