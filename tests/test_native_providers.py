"""Account ownership, native transport lifecycle and subscription setup contracts."""
import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from codex_rpc import child_environment
from codex_native import history_items, valid_arguments
from native_providers import NativeSession
from provider_accounts import LoginAttempt, account_status, auth_url
from provider_errors import ProviderError, failure


CFG={"provider":"codex","model":"fixture-model","label":"Codex account","url":""}


class AccountContracts(unittest.TestCase):
    def test_account_child_cannot_inherit_api_billing(self):
        with patch.dict(os.environ,{"OPENAI_API_KEY":"fixture-secret","OPENAI_BASE_URL":"https://fixture.invalid",
                                   "ANTHROPIC_API_KEY":"fixture-secret","ANTHROPIC_AUTH_TOKEN":"fixture-token"}):
            self.assertNotIn("OPENAI_API_KEY",child_environment("codex"))
            self.assertNotIn("OPENAI_BASE_URL",child_environment("codex"))
            self.assertNotIn("ANTHROPIC_API_KEY",child_environment("claude"))
            self.assertNotIn("ANTHROPIC_AUTH_TOKEN",child_environment("claude"))
            self.assertEqual(os.environ["ANTHROPIC_API_KEY"],"fixture-secret")

    def test_authorization_link_cannot_escape_official_host(self):
        self.assertEqual(auth_url("claude_code","https://claude.ai/oauth/authorize?state=fixture"),"https://claude.ai/oauth/authorize?state=fixture")
        for value in ("http://claude.ai/oauth/authorize","https://claude.ai.evil.test/oauth/authorize",
                      "https://claude.ai@evil.test/oauth/authorize","https://claude.ai:8443/oauth/authorize",
                      "https://claude.ai/help"):
            with self.subTest(value=value),self.assertRaises(ProviderError):auth_url("claude_code",value)

    @patch("provider_accounts.executable",return_value="fixture-cli")
    @patch("provider_accounts.bounded_output")
    def test_claude_api_auth_does_not_count_as_subscription(self,run,_):
        run.return_value=json.dumps({"loggedIn":True,"authMethod":"api_key"})
        self.assertFalse(account_status("claude_code")["connected"])

    @patch("provider_accounts.executable",return_value="fixture-cli")
    @patch("provider_accounts.bounded_output")
    def test_claude_verified_identity_is_hashed_not_exposed(self,run,_):
        run.return_value=json.dumps({"loggedIn":True,"authMethod":"claude.ai","email":"fixture@example.invalid","subscriptionType":"max"})
        result=account_status("claude_code")
        self.assertTrue(result["connected"])
        self.assertEqual(len(result["account_binding"]),64)
        self.assertNotIn("fixture@example.invalid",json.dumps(result))

    def test_login_cancellation_wins_over_late_completion(self):
        attempt=LoginAttempt("claude_code",None)
        with patch("provider_accounts.account_status",return_value={"connected":False}),patch.object(attempt,"_run_claude",side_effect=attempt.cancel):
            attempt.run()
        self.assertEqual(attempt.snapshot()["state"],"cancelled")
        self.assertIsNone(attempt.snapshot()["authorization_url"])

    def test_limit_message_never_exposes_untrusted_diagnostics(self):
        error=failure("codex","usage_limit secret=fixture-secret https://private.invalid",reset_at=123)
        self.assertEqual(error.code,"usage_limit")
        self.assertEqual(error.public()["reset_at"],123)
        self.assertNotIn("fixture-secret",json.dumps(error.public()))
        self.assertNotIn("private.invalid",json.dumps(error.public()))


class NativeLifecycle(unittest.TestCase):
    def test_cancelled_startup_cannot_clear_a_persisted_limit(self):
        session=NativeSession(CFG)
        def cancelled(cfg,payload,response):
            response.close()
            return
            yield
        with patch("codex_native.generate",side_effect=cancelled),patch.object(session,"_remember_outcome") as remember:
            self.assertEqual(list(session.post("/api/chat",json={}).iter_content()),[])
            remember.assert_not_called()
        session.close()

    def test_malformed_native_output_has_safe_error(self):
        session=NativeSession(CFG)
        def malformed(*_):
            raise ValueError("secret fixture-private-provider-output")
            yield
        with patch("codex_native.generate",side_effect=malformed):
            with self.assertRaises(ProviderError) as caught:session.post("/api/chat",json={}).json()
        self.assertNotIn("fixture-private",str(caught.exception))
        self.assertEqual(session.status()["provider_error"]["code"],"protocol")
        session.close()

    def test_usage_limit_latches_without_local_fallback_or_retry(self):
        local=Mock()
        session=NativeSession(CFG,local)
        def limited(*_):
            raise failure("codex","usage limit",reset_at=123)
            yield
        with patch("codex_native.generate",side_effect=limited) as gen:
            with self.assertRaises(ProviderError):session.post("/api/chat",json={}).json()
            self.assertTrue(session.status()["usage_paused"])
            with self.assertRaises(ProviderError):session.post("/api/chat",json={}).json()
            self.assertEqual(gen.call_count,1)
        local.post.assert_not_called()
        session.clear_usage_limit()
        with patch("codex_native.generate",return_value=iter([{"message":{"content":"hello"},"done":True}])):
            self.assertEqual(session.post("/api/chat",json={}).json()["message"]["content"],"hello")
        session.close()

    def test_voice_cancellation_does_not_cancel_draft_or_replacement_session(self):
        from llm_module import LLM
        llm=LLM.__new__(LLM)
        llm._requests_lock=threading.Lock();llm._active_requests={}
        owner=NativeSession(CFG);replacement=NativeSession(CFG)
        llm.ollama_session=owner
        voice=owner.post("/api/chat",json={"_atlas_request_id":"voice"})
        draft=owner.post("/api/chat",json={"_atlas_request_id":"draft"})
        for rid,response in (("voice",voice),("draft",draft)):
            llm._register_request(rid,"ollama",None)
            llm._register_request(rid,"ollama",response)
        llm.ollama_session=replacement
        self.assertTrue(llm.cancel_generation("voice"))
        self.assertTrue(voice.cancelled.is_set())
        self.assertFalse(draft.cancelled.is_set())
        self.assertFalse(replacement._closed)
        owner.close();replacement.close()

    def test_collected_and_streamed_text_have_same_content_without_duplicates(self):
        events=[{"message":{"content":"A "},"done":False},{"message":{"content":"reply."},"done":False},{"message":{"content":""},"done":True}]
        session=NativeSession(CFG)
        with patch("codex_native.generate",side_effect=lambda *_:iter(events)):
            collected=session.post("/api/chat",json={}).json()["message"]["content"]
            streamed="".join(json.loads(raw)["message"]["content"] for raw in session.post("/api/chat",json={}).iter_content())
        self.assertEqual(collected,streamed)
        self.assertEqual(collected,"A reply.")
        session.close()

    def test_missing_terminal_is_not_success(self):
        session=NativeSession(CFG)
        with patch("codex_native.generate",return_value=iter([{"message":{"content":"partial"},"done":False}])):
            with self.assertRaises(ProviderError):session.post("/api/chat",json={}).json()
        session.close()


class NativeHistory(unittest.TestCase):
    def test_tool_result_images_retain_call_id_and_real_attachment(self):
        from claude_native import cli_input
        _,items=history_items([
            {"role":"assistant","tool_calls":[{"id":"screen-call","function":{"name":"look_at_screen","arguments":{}}}]},
            {"role":"tool","tool_call_id":"screen-call","content":"Current screen","images":["aGVsbG8="]}])
        self.assertEqual(items[-1]["call_id"],"screen-call")
        self.assertEqual(items[-1]["output"][-1]["type"],"input_image")
        blocks=cli_input(items)["message"]["content"]
        self.assertNotIn("aGVsbG8=",blocks[0]["text"])
        self.assertEqual(blocks[-1]["type"],"image")
        self.assertEqual(blocks[-1]["source"]["data"],"aGVsbG8=")

    def test_claude_images_are_attachments_not_base64_in_prompt_text(self):
        from claude_native import cli_input
        data=cli_input([{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,aGVsbG8="}]}])
        blocks=data["message"]["content"]
        self.assertNotIn("aGVsbG8=",blocks[0]["text"])
        self.assertEqual(blocks[-1]["type"],"image")
        self.assertEqual(blocks[-1]["source"]["data"],"aGVsbG8=")

    def test_tool_history_keeps_real_roles_and_ids(self):
        system,items=history_items([{"role":"system","content":"persona"},
            {"role":"user","content":"/no_think actual request"},
            {"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"name":"clock","arguments":{}}}]},
            {"role":"tool","tool_call_id":"c1","content":"real result"},
            {"role":"assistant","content":"/no_think is quoted text"}])
        self.assertEqual(system,"persona")
        self.assertEqual(items[0]["role"],"user")
        self.assertEqual(items[0]["content"][0]["text"],"actual request")
        self.assertEqual(items[1]["call_id"],items[2]["call_id"])
        self.assertEqual(items[-1]["content"][0]["text"],"/no_think is quoted text")

    def test_orphan_tool_result_is_rejected(self):
        with self.assertRaises(ProviderError):history_items([{"role":"tool","content":"orphan"}])

    def test_tool_arguments_are_bounded_and_typed(self):
        schema={"type":"object","properties":{"count":{"type":"integer","minimum":1,"maximum":3}},"required":["count"],"additionalProperties":False}
        self.assertTrue(valid_arguments({"count":2},schema))
        for bad in ({"count":True},{"count":4},{},{"count":2,"other":1}):self.assertFalse(valid_arguments(bad,schema))


class NativeSetup(unittest.TestCase):
    def test_local_browser_origin_cannot_be_forged_by_external_site(self):
        import setup_api as api
        request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"),headers={"origin":"https://evil.example"})
        self.assertEqual(asyncio.run(api.setup_provider_connect(request)).status_code,403)

    def test_all_account_routes_reject_lan_clients(self):
        import setup_api as api
        request=SimpleNamespace(client=SimpleNamespace(host="192.0.2.2"))
        calls=[(api.setup_provider_status,()),(api.setup_provider_models,()),(api.setup_provider_connect,()),
               (api.setup_provider_connection,("fixture",)),(api.setup_provider_disconnect,("fixture",))]
        for fn,args in calls:
            with self.subTest(route=fn.__name__):
                self.assertEqual(asyncio.run(fn(request,*args)).status_code,403)

    def test_subscription_setup_never_calls_compatible_v1_route(self):
        import setup_api as api,user_settings as us
        cloud={"provider":"claude_code","protocol":"claude_code","auth_mode":"subscription","model":"fixture-native",
               "account_binding":"fixture-binding","verification":{"status":"limited","isolation_verified":True}}
        async def body():return {"mode":"cloud","provider":"claude_code","model":"sonnet","base_url":"","api_key":"fixture-old-key","context":16384}
        request=SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"),json=body)
        with tempfile.TemporaryDirectory() as folder,patch.object(us,"PATH",Path(folder)/"settings.json"),patch.object(api,"test_cloud") as compatible,patch.object(api,"test_account",return_value={"ok":True,"verification":"limited","cloud":cloud}):
            us.save({"llm":{"mode":"cloud","cloud":{"provider":"openrouter","api_key":"fixture-old-key","base_url":"https://fixture.invalid","model":"old"}}})
            result=asyncio.run(api.setup_llm(request))
            self.assertTrue(result["ok"])
            compatible.assert_not_called()
            self.assertNotIn("api_key",us.load()["llm"]["cloud"])
            config=us.cloud_config()
            self.assertEqual(config["url"],"")
            self.assertEqual(config["context"],16384)
            self.assertFalse(config["allow_local_fallback"])
            self.assertIsNone(api.configured_brain_error())


class EffectiveCodexIsolation(unittest.TestCase):
    def valid_config(self):
        return {"model_provider":"openai", "approval_policy":"never", "sandbox_mode":"read-only",
                "web_search":"disabled", "project_doc_max_bytes":0, "features":{"view_image":False},
                "skills":{"include_instructions":False,"bundled":{"enabled":False}}}

    def test_effective_config_rejects_redirected_or_unverified_capabilities(self):
        from codex_native import verify_config
        verify_config(self.valid_config())
        bad = {"model_provider":"other", "approval_policy":"on-request", "sandbox_mode":"workspace-write",
               "openai_base_url":"https://fixture.invalid", "chatgpt_base_url":"https://fixture.invalid",
               "features":{"view_image":True}}
        for field, value in bad.items():
            with self.subTest(field=field):
                cfg=self.valid_config(); cfg[field]=value
                with self.assertRaises(ProviderError): verify_config(cfg)
        for field in ("model_provider","approval_policy","sandbox_mode","features"):
            cfg=self.valid_config(); del cfg[field]
            with self.subTest(missing=field), self.assertRaises(ProviderError): verify_config(cfg)

    def test_root_and_selected_profile_endpoint_overrides_fail_before_spawn(self):
        from codex_rpc import config_overrides
        configs = ['openai_base_url = "https://fixture.invalid"',
                   'chatgpt_base_url = "https://fixture.invalid"',
                   'profile = "custom"\n[profiles.custom]\nchatgpt_base_url = "https://fixture.invalid"']
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,{"CODEX_HOME":folder}):
            path=Path(folder)/"config.toml"
            for text in configs:
                path.write_text(text,encoding="utf-8")
                with self.subTest(config=text), self.assertRaises(ProviderError): config_overrides()
            path.write_text('',encoding="utf-8")
            self.assertIn('features.view_image=false',config_overrides())

    def test_falsey_non_text_history_is_rejected(self):
        for value in ([],{},0,False):
            with self.subTest(content=value), self.assertRaises(ProviderError):
                history_items([{"role":"user","content":value}])


if __name__=="__main__":unittest.main()
