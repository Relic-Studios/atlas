"""Plugins page backend (owner 10-05): every ability is a plugin you can switch off,
with a settings window for keys and choices. Secrets are write-only."""
import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plugins as P


def _tools(*names):
    return [{"type": "function", "function": {"name": n}} for n in names]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._old = (P.USER, P.STATE_PATH, P.ROOT)
        P.USER = self.tmp / "user"
        P.STATE_PATH = P.USER / "plugins.json"
        P.ROOT = self.tmp             # secret "source" lookups stay inside the temp dir
        import websearch
        self.ws = websearch
        self._ws = (websearch._HERE, websearch.PREFERRED, websearch.MAX_RESULTS)
        websearch._HERE = self.tmp    # websearch._key reads tmp/user + tmp/private only
        self._env = {k: v for k, v in __import__("os").environ.items() if k.startswith("ATLAS_") and k.endswith("_KEY")}
        for k in self._env:
            del __import__("os").environ[k]

    def tearDown(self):
        P.USER, P.STATE_PATH, P.ROOT = self._old
        self.ws._HERE, self.ws.PREFERRED, self.ws.MAX_RESULTS = self._ws
        __import__("os").environ.update(self._env)


class Registry(Base):
    def test_every_agent_tool_belongs_to_a_plugin(self):
        try:
            import speech_pipeline_manager as S
        except ModuleNotFoundError as e:   # light CI deps (no transformers/torch)
            self.skipTest(f"pipeline import needs {e.name}")
        owned = set()
        for p in P.BUILTIN:
            owned.update(P._builtin_tools(p))
        names = {t["function"]["name"] for t in S.AGENT_TOOLS}
        self.assertFalse(names - owned, f"tools with no plugin switch: {names - owned}")

    def test_defaults(self):
        # Every plugin starts at its declared default; opt-in ones (floor referee) start off.
        dflt = {p["id"]: p.get("default", True) for p in P.BUILTIN}
        self.assertTrue(all(p["enabled"] == dflt[p["id"]] for p in P.listing()))
        off_tools = {t for p in P.BUILTIN if not p.get("default", True) for t in p["tools"]}
        self.assertEqual(P.tool_names_disabled(), off_tools)
        self.assertEqual(off_tools, {"floor_stats", "start_topic", "debate_sides"})

    def test_switch_off_removes_tools(self):
        P.set_enabled("web_search", False)
        kept = [t["function"]["name"] for t in P.filter_tools(_tools("web_search", "read_page", "check_date_time"))]
        self.assertEqual(kept, ["check_date_time"])
        self.assertFalse(P.is_enabled("web_search"))
        P.set_enabled("web_search", True)
        self.assertEqual(len(P.filter_tools(_tools("web_search", "read_page"))), 2)

    def test_state_persists(self):
        P.set_enabled("clock", False)
        self.assertFalse(json.loads(P.STATE_PATH.read_text())["clock"]["enabled"])

    def test_listener_fires(self):
        hits = []
        P.on_change(lambda: hits.append(1))
        try:
            P.set_enabled("notes", False)
        finally:
            P._listeners.pop()
        self.assertTrue(hits)

    def test_unknown_plugin(self):
        with self.assertRaises(KeyError):
            P.set_enabled("nope", True)


class Settings(Base):
    def test_secret_is_write_only(self):
        r = P.save_settings("web_search", {"exa": "exa-test-key-123456789"})
        self.assertTrue(r["ok"])
        self.assertEqual((P.USER / "exa.key").read_text(), "exa-test-key-123456789")
        dump = json.dumps(P.listing())
        self.assertNotIn("exa-test-key-123456789", dump)
        f = next(x for x in r["plugin"]["settings"] if x["key"] == "exa")
        self.assertEqual(f["secret"], {"set": True, "last4": "6789", "source": "saved here"})
        self.assertNotIn("exa", json.loads(P.STATE_PATH.read_text())["web_search"]["settings"])

    def test_saved_key_beats_private_key(self):
        (self.tmp / "private").mkdir()
        (self.tmp / "private" / "exa.key").write_text("old-private-key-1111")
        P.save_settings("web_search", {"exa": "new-user-key-22222222"})
        self.assertEqual(self.ws._key("exa"), "new-user-key-22222222")

    def test_secret_clear_and_validation(self):
        P.save_settings("web_search", {"exa": "exa-test-key-123456789"})
        P.save_settings("web_search", {"exa": ""})
        self.assertFalse((P.USER / "exa.key").exists())
        r = P.save_settings("web_search", {"exa": "has space key"})
        self.assertFalse(r["ok"])
        self.assertIn("exa", r["errors"])

    def test_select_and_number_validation(self):
        bad = P.save_settings("web_search", {"provider": "google", "max_results": 50})
        self.assertFalse(bad["ok"])
        self.assertEqual(set(bad["errors"]), {"provider", "max_results"})
        ok = P.save_settings("web_search", {"provider": "brave", "max_results": "6"})
        self.assertTrue(ok["ok"])
        self.assertEqual(P.settings_of("web_search"), {"provider": "brave", "max_results": 6})
        self.assertEqual((self.ws.PREFERRED, self.ws.MAX_RESULTS), ("brave", 6))  # applied live

    def test_free_provider_skips_keys(self):
        P.USER.mkdir(parents=True, exist_ok=True)
        (P.USER / "exa.key").write_text("exa-test-key-123456789")
        called = []
        self.ws.PREFERRED = "free"
        old = (self.ws._search_exa, self.ws._search_ddgs, self.ws._searxng_available)
        self.ws._search_exa = lambda *a: called.append("exa") or [{"title": "x", "href": "u", "body": "b"}]
        self.ws._search_ddgs = lambda q, n=5: [{"title": "free", "href": "u", "body": "b"}]
        self.ws._searxng_available = lambda: None
        try:
            self.ws._run_chain("q", 2)
        finally:
            self.ws._search_exa, self.ws._search_ddgs, self.ws._searxng_available = old
        self.assertEqual(called, [])

    def test_key_test_without_key(self):
        self.assertFalse(P.test("web_search", "exa")["ok"])
        self.assertIn("Nothing to test", P.test("web_search", "provider")["message"])

    def test_clock_test(self):
        P.save_settings("clock", {"timezone": "Asia/Tokyo"})
        self.assertTrue(P.test("clock")["ok"])


class Eyes(Base):
    def test_plugin_off_forces_eyes_off_and_on_never_turns_it_on(self):
        import screen
        before = screen.status()["enabled"]
        try:
            screen.set_enabled(False)
            P.set_enabled("eyes", True)
            self.assertFalse(screen.status()["enabled"])   # owner's toggle respected
            screen.set_enabled(True)
            P.set_enabled("eyes", False)
            self.assertFalse(screen.status()["enabled"])
        finally:
            screen.set_enabled(before)


class Abilities(unittest.TestCase):
    def test_off_abilities_not_claimed(self):
        import capability as C
        on = C.abilities_note(False, True, "x")
        off = C.abilities_note(False, False, "x", {"clock", "self_check", "notes"})
        self.assertIn("source code", on)
        self.assertNotIn("source code", off)
        self.assertNotIn("search the web", off)
        self.assertNotIn("leave notes", off)


class Market(Base):
    def test_builtins_installed_plus_catalog(self):
        m = P.marketplace()
        ids = {p["id"] for p in P.BUILTIN}
        self.assertTrue(all(x["status"] == "installed" for x in m if x["id"] in ids))
        self.assertTrue(any(x["status"] == "coming_soon" for x in m))


class Api(Base):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import plugins_api
        app = FastAPI()
        app.include_router(plugins_api.router)
        self.local = TestClient(app, client=("127.0.0.1", 5000))
        self.remote = TestClient(app, client=("192.0.2.9", 5000))

    def test_remote_denied(self):
        self.assertEqual(self.remote.get("/api/plugins").status_code, 403)
        self.assertEqual(self.remote.post("/api/plugins/eyes/enabled", json={"enabled": False}).status_code, 403)

    def test_toggle_and_settings(self):
        self.assertEqual(len(self.local.get("/api/plugins").json()["plugins"]), len(P.BUILTIN))
        r = self.local.post("/api/plugins/notes/enabled", json={"enabled": False})
        self.assertFalse(r.json()["plugin"]["enabled"])
        self.assertEqual(self.local.post("/api/plugins/notes/enabled", json={"enabled": "no"}).status_code, 400)
        self.assertEqual(self.local.post("/api/plugins/nope/enabled", json={"enabled": True}).status_code, 404)
        bad = self.local.post("/api/plugins/web_search/settings", json={"values": {"provider": "x"}})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("provider", bad.json()["errors"])
        self.assertEqual(self.local.get("/api/plugins/marketplace").status_code, 200)


if __name__ == "__main__":
    unittest.main()
