"""First-run setup: settings store, cloud config, key redaction, setup API routes (no models)."""
import json, os, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.environ["ATLAS_SETTINGS"] = os.path.join(self.d, "settings.json")
        import importlib, user_settings
        self.US = importlib.reload(user_settings)

    def tearDown(self):
        os.environ.pop("ATLAS_SETTINGS", None)

    def test_empty_defaults(self):
        self.assertEqual(self.US.llm_mode(), "local")
        self.assertIsNone(self.US.cloud_config())
        self.assertEqual(self.US.audio_devices(), {"input": None, "output": None})

    def test_cloud_config_strips_v1_and_key_redacted(self):
        self.US.update({"llm": {"mode": "cloud", "cloud": {"provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1/", "model": "m", "api_key": "sk-abcdefgh12345"}}})
        c = self.US.cloud_config()
        self.assertEqual(c["url"], "https://openrouter.ai/api")
        self.assertEqual(c["key"], "sk-abcdefgh12345")
        self.assertFalse(c["vllm"])
        v = self.US.public_view()
        self.assertEqual(v["llm"]["cloud"]["api_key"], "")
        self.assertTrue(v["llm"]["cloud"]["has_key"])
        self.assertNotIn("sk-abcdefgh12345", json.dumps(v))

    def test_deep_merge_keeps_siblings(self):
        self.US.update({"llm": {"mode": "cloud", "cloud": {"model": "x", "base_url": "http://h/v1"}}})
        self.US.update({"llm": {"mode": "local"}})
        self.assertEqual(self.US.load()["llm"]["cloud"]["model"], "x")

    def test_recommend_by_vram(self):
        self.assertEqual(self.US.recommend_local_model(24)["model"], "qwen3:14b")
        self.assertEqual(self.US.recommend_local_model(16)["model"], "qwen3:14b")
        self.assertEqual(self.US.recommend_local_model(12)["model"], "qwen3:8b")
        self.assertTrue(self.US.recommend_local_model(12)["fits"])
        low = self.US.recommend_local_model(8)
        self.assertEqual(low["model"], "qwen3:8b"); self.assertFalse(low["fits"])
        self.assertFalse(self.US.recommend_local_model(0)["fits"])
        self.assertGreaterEqual(self.US.MIN_CONTEXT, 4096)

    def test_no_bundled_keys(self):
        for k, p in self.US.PROVIDERS.items():
            self.assertNotIn("key", {kk for kk in p if kk not in ("keys",)}, k)


class TranslateTests(unittest.TestCase):
    def test_cloud_body_is_strict_openai(self):
        import remote_llm as R
        b = R.translate_payload({"messages": [{"role": "user", "content": "hi"}], "think": False,
                                 "options": {"temperature": 0.6, "top_k": 20}}, "m", vllm=False)
        self.assertNotIn("chat_template_kwargs", b)
        self.assertNotIn("top_k", b)
        self.assertEqual(b["model"], "m")


class SetupApiTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.environ["ATLAS_SETTINGS"] = os.path.join(self.d, "settings.json")
        import importlib, user_settings, setup_api
        importlib.reload(user_settings)
        self.SA = importlib.reload(setup_api)
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(self.SA.router)
        self.c = TestClient(app)

    def tearDown(self):
        os.environ.pop("ATLAS_SETTINGS", None)

    def test_status_shape(self):
        j = self.c.get("/api/setup/status").json()
        for k in ("setup_needed", "gpu", "ollama", "recommended", "providers", "devices", "pull"):
            self.assertIn(k, j)

    def test_local_saves_and_remote_client_denied(self):
        # TestClient's host is "testclient" -> not loopback -> denied
        r = self.c.post("/api/setup/llm", json={"mode": "local", "model": "qwen3:8b"})
        self.assertEqual(r.status_code, 403)
        self.SA._owner = lambda req: True
        r = self.c.post("/api/setup/llm", json={"mode": "local", "model": "qwen3:8b"})
        self.assertEqual(r.status_code, 200)
        import user_settings as US
        self.assertEqual(US.local_llm()["model"], "qwen3:8b")

    def test_cloud_bad_endpoint_not_saved(self):
        self.SA._owner = lambda req: True
        r = self.c.post("/api/setup/llm", json={"mode": "cloud", "provider": "custom",
                                               "base_url": "http://127.0.0.1:9/v1", "model": "m", "api_key": "k"})
        self.assertEqual(r.status_code, 400)
        import user_settings as US
        self.assertIsNone(US.cloud_config())

    def test_complete_flag(self):
        self.SA._owner = lambda req: True
        self.c.post("/api/setup/complete", json={"agent": "juniper"})
        import user_settings as US
        self.assertTrue(US.load()["setup_complete"])
        self.assertEqual(US.load()["agent"], "juniper")


if __name__ == "__main__":
    unittest.main()
