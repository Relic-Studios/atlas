"""First-run setup: settings store, cloud config, key redaction, setup API routes (no models)."""
import json, os, sys, tempfile, unittest
from unittest.mock import patch
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
        self.assertGreaterEqual(self.US.MIN_CONTEXT, 8192)

    def test_vision_suggestions_are_separate_and_have_headroom(self):
        self.assertEqual(self.US.recommend_local_model(24, vision=True)['model'], 'qwen3-vl:8b-instruct')
        self.assertTrue(self.US.recommend_local_model(16, vision=True)['fits'])
        self.assertFalse(self.US.recommend_local_model(12, vision=True)['fits'])
        self.assertTrue(all(m['vision'] and m['tools'] for m in self.US.VISION_MODELS))
        self.assertTrue(all('recommended' not in m['label'].lower() for m in self.US.LOCAL_MODELS + self.US.VISION_MODELS))

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


class ProbeRoutingTests(unittest.TestCase):
    def test_models_must_answer_the_addressed_probe_with_spoken_text(self):
        import setup_api as SA
        from unittest.mock import Mock
        for cloud in (False, True):
            for answer, expected in (("[HOLD]", False), ("[SPEAK to=S1]", False),
                                     ("[SPEAK to=S1] Blue.", True)):
                with self.subTest(cloud=cloud, answer=answer):
                    replies = []
                    for text in (answer, "[HOLD]"):
                        response = Mock(status_code=200, ok=True)
                        response.json.return_value = {"choices": [{"message": {"content": text}}]} if cloud else {"message": {"content": text}}
                        replies.append(response)
                    with patch.object(SA.requests, "post", side_effect=replies):
                        result = SA.test_cloud("http://127.0.0.1:11434/v1", "fixture", "test-only") if cloud else SA.test_local("http://127.0.0.1:11434", "fixture")
                    self.assertEqual(result["ok"], expected, result)

    def test_cloud_probe_uses_shared_provider_format_without_qwen_fields(self):
        import setup_api as SA
        from unittest.mock import Mock
        for endpoint, disabled in (('https://api.deepseek.com/v1', True), ('https://api.openai.com/v1', False)):
            with self.subTest(endpoint=endpoint):
                replies = []
                for _, want in SA.PROBE_CASES:
                    response = Mock(status_code=200, ok=True)
                    response.json.return_value = {'choices':[{'message':{'content':
                        '[SPEAK to=S1] A short reply.' if want == 'SPEAK' else '[HOLD]'}}]}
                    replies.append(response)
                with patch.object(SA.requests, 'post', side_effect=replies) as post:
                    result = SA.test_cloud(endpoint, 'any-model', 'fixture-only-key')
                self.assertTrue(result['ok'], result)
                self.assertEqual(post.call_count, len(SA.PROBE_CASES))
                for call in post.call_args_list:
                    body = call.kwargs['json']
                    self.assertNotIn('chat_template_kwargs', body)
                    self.assertEqual(body.get('thinking'), {'type':'disabled'} if disabled else None)


class SetupApiTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.environ["ATLAS_SETTINGS"] = os.path.join(self.d, "settings.json")
        import importlib, user_settings, setup_api, agent_registry
        self._settings_path = user_settings.PATH
        importlib.reload(user_settings)
        self.SA = importlib.reload(setup_api)
        root = Path(self.d)
        self.registry = patch.multiple(agent_registry, ROOT=root, DEV_PACK=root / 'dev_pack/pack.json',
            BUNDLED_PACK=root / 'starter_pack/pack.json', USER_AGENTS=root / 'personas/agents.json',
            USER_VOICES=root / 'voices/user/voices.json')
        self.registry.start()
        self.bundle()
        self.probes = [patch.object(self.SA, 'gpu_info', return_value={'total_gb':16, 'name':'Fixture GPU'}),
            patch.object(self.SA, 'ollama_info', return_value={'running':False, 'installed':False, 'models':[]})]
        for probe in self.probes: probe.start()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(self.SA.router)
        self.app = app
        self.c = TestClient(app)

    def tearDown(self):
        for probe in self.probes: probe.stop()
        self.registry.stop()
        os.environ.pop("ATLAS_SETTINGS", None)
        import user_settings, agent_registry
        user_settings.PATH = self._settings_path
        agent_registry.reload()

    def bundle(self):
        import wave
        root = Path(self.d)
        folder = root / 'starter_pack'; folder.mkdir(exist_ok=True)
        with wave.open(str(folder / 'juniper-natural.wav'), 'wb') as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(24000)
            audio.writeframes(b'\x01\x00' * 2400)
        (folder / 'juniper-natural.txt').write_text('Hello there.')
        (folder / 'pack.json').write_text(json.dumps({'default':'juniper', 'voices':{'juniper-natural':{
            'wav':'starter_pack/juniper-natural.wav', 'text':'starter_pack/juniper-natural.txt'}},
            'agents':[{'id':'juniper', 'name':'Juniper', 'voice':'juniper-natural', 'prompt':'You are Juniper.'}]}))

    def test_status_shape(self):
        j = self.c.get("/api/setup/status").json()
        for k in ("setup_needed", "gpu", "ollama", "recommended", "providers", "devices", "pull"):
            self.assertIn(k, j)
        self.assertEqual(j['ready_agent'], {'id':'juniper', 'name':'Juniper', 'voice':'juniper-natural', 'bundled':True})
        self.assertEqual(j['agent_choices'], [j['ready_agent']])
        self.assertEqual(j['agents'], ['juniper'])
        self.assertIsNone(j['bundle_error'])
        self.assertFalse(j['brain_ready'])
        self.assertTrue(j['brain_error'])

    def test_local_saves_and_remote_client_denied(self):
        # TestClient's host is "testclient" -> not loopback -> denied
        r = self.c.post("/api/setup/llm", json={"mode": "local", "model": "qwen3:8b"})
        self.assertEqual(r.status_code, 403)
        self.SA._owner = lambda req: True
        # Hermetic: no live Ollama in CI (this passed locally only because Ollama was up).
        real = self.SA.test_local
        try:
            self.SA.test_local = lambda url, model, timeout=120.0: {"ok": False, "error": "format"}
            r = self.c.post("/api/setup/llm", json={"mode": "local", "model": "qwen3:4b"})
            self.assertEqual(r.status_code, 400)      # failed probe -> not saved
            self.SA.test_local = lambda url, model, timeout=120.0: {"ok": True}
            r = self.c.post("/api/setup/llm", json={"mode": "local", "model": "qwen3:8b"})
            self.assertEqual(r.status_code, 200)
        finally:
            self.SA.test_local = real
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
        import user_settings as US
        US.update({'llm':{'mode':'local', 'local':{'model':'configured-fixture'}}})
        self.assertEqual(self.c.post("/api/setup/complete", json={"agent": "juniper"}).status_code, 200)
        self.assertTrue(US.load()["setup_complete"])
        self.assertEqual(US.load()["agent"], "juniper")

    def test_completion_default_and_stale_explicit_selection(self):
        self.SA._owner = lambda req: True
        import user_settings as US
        US.update({'llm':{'mode':'local', 'local':{'model':'configured-fixture'}}})
        for invalid in ('missing-agent', '', None):
            before = US.load()
            r = self.c.post('/api/setup/complete', json={'agent':invalid})
            self.assertEqual(r.status_code, 400)
            self.assertIsInstance(r.json()['error'], str)
            self.assertEqual(US.load(), before)
        r = self.c.post('/api/setup/complete', json={})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(US.load()['agent'], 'juniper')

    def test_missing_bundle_is_actionable_and_cannot_complete(self):
        import agent_registry as R, user_settings as US
        self.SA._owner = lambda req: True
        US.update({'llm':{'mode':'local', 'local':{'model':'configured-fixture'}}})
        R.BUNDLED_PACK.unlink()
        status = self.c.get('/api/setup/status').json()
        self.assertIsNone(status['ready_agent'])
        self.assertTrue(status['bundle_error'])
        before = US.load()
        self.assertEqual(self.c.post('/api/setup/complete', json={}).status_code, 400)
        self.assertEqual(US.load(), before)

    def test_completion_requires_configured_brain_and_required_cloud_key(self):
        import user_settings as US
        self.SA._owner = lambda req: True
        incomplete = [{}, {'mode':'local', 'local':{'model':'  '}},
            {'mode':'cloud', 'cloud':{'provider':'custom', 'base_url':'', 'model':'fixture'}},
            {'mode':'cloud', 'cloud':{'provider':'custom', 'base_url':'http://fixture', 'model':''}},
            {'mode':'cloud', 'cloud':{'provider':'openai', 'base_url':'http://fixture', 'model':'fixture'}}]
        for llm in incomplete:
            with self.subTest(llm=llm):
                US.save({'llm':llm})
                before = US.load()
                status = self.c.get('/api/setup/status').json()
                self.assertFalse(status['brain_ready'])
                self.assertTrue(status['brain_error'])
                response = self.c.post('/api/setup/complete', json={'agent':'juniper'})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(US.load(), before)
        # Keyless custom local API endpoints are valid; required-provider keys
        # are configuration fields, not a request to perform paid inference.
        for provider, key in (('custom',''), ('openai','fixture-only-key')):
            US.save({'llm':{'mode':'cloud', 'cloud':{'provider':provider, 'base_url':'http://fixture',
                'model':'fixture', 'api_key':key}}})
            self.assertTrue(self.c.get('/api/setup/status').json()['brain_ready'])
            self.assertEqual(self.c.post('/api/setup/complete', json={'agent':'juniper'}).status_code, 200)

    def test_cloud_vision_is_explicit_boolean_and_cleared_for_other_model(self):
        self.SA._owner = lambda req: True
        import user_settings as US
        body = {'mode':'cloud', 'provider':'custom', 'base_url':'http://fixture/v1', 'model':'vision-model'}
        with patch.object(self.SA, 'test_cloud', return_value={'ok':True}):
            self.assertEqual(self.c.post('/api/setup/llm', json={**body, 'vision':True}).status_code, 200)
            self.assertTrue(US.load()['llm']['cloud']['vision'])
            self.assertEqual(self.c.post('/api/setup/llm', json={**body, 'model':'text-model'}).status_code, 200)
            self.assertFalse(US.load()['llm']['cloud']['vision'])

    def test_deepseek_preset_saves_official_defaults_as_strict_cloud(self):
        self.SA._owner = lambda req: True
        import user_settings as US
        with patch.object(self.SA, 'test_cloud', return_value={'ok':True}) as probe:
            result = self.c.post('/api/setup/llm', json={'mode':'cloud', 'provider':'deepseek',
                'api_key':'fixture-only-key'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(probe.call_args.args[:2], ('https://api.deepseek.com/v1', 'deepseek-flash'))
        cloud = US.load()['llm']['cloud']
        self.assertFalse(cloud['vllm'])
        self.assertEqual(cloud['provider'], 'deepseek')
        self.assertEqual(US.cloud_config()['url'], 'https://api.deepseek.com')

    def test_blank_key_is_reused_only_for_the_same_provider_endpoint(self):
        self.SA._owner = lambda req: True
        import user_settings as US
        cases = [
            ("custom", "http://127.0.0.1:11434/v1/", "saved-test-credential"),
            ("custom", "http://127.0.0.1:11435/v1", ""),
            ("openai", "http://127.0.0.1:11434/v1", ""),
        ]
        for provider, endpoint, expected in cases:
            with self.subTest(provider=provider, endpoint=endpoint):
                US.update({"llm": {"cloud": {"provider": "custom", "base_url": "http://127.0.0.1:11434/v1",
                                             "model": "old-model", "api_key": "saved-test-credential"}}})
                with patch.object(self.SA, "test_cloud", return_value={"ok": True}) as probe:
                    r = self.c.post("/api/setup/llm", json={"mode": "cloud", "provider": provider,
                        "base_url": endpoint, "model": "new-model", "api_key": ""})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(probe.call_args.args[2], expected)
                self.assertEqual(US.load()["llm"]["cloud"]["api_key"], expected)
                self.assertNotIn("saved-test-credential", r.text)

    def test_slow_model_probe_keeps_async_routes_responsive(self):
        import asyncio
        import threading
        import httpx
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        self.SA._owner = lambda req: True

        @self.app.get("/probe-responsiveness")
        async def ping():
            return {"ok": True}

        def slow_probe(*args):
            started.set()
            release.wait(5)
            finished.set()
            return {"ok": True}

        async def check():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as client:
                for mode in ("local", "cloud"):
                    started.clear(); release.clear(); finished.clear()
                    body = {"mode": mode, "model": "test-model", "provider": "custom",
                            "base_url": "http://127.0.0.1:11434/v1", "api_key": "test-only"}
                    pending = asyncio.create_task(client.post("/api/setup/llm", json=body))
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        self.assertFalse(finished.is_set(), "probe blocked the event loop")
                        response = await asyncio.wait_for(client.get("/probe-responsiveness"), timeout=1)
                        self.assertEqual(response.status_code, 200)
                    finally:
                        release.set()
                        response = await pending
                    self.assertEqual(response.status_code, 200)

        with patch.object(self.SA, "test_local", slow_probe), patch.object(self.SA, "test_cloud", slow_probe):
            asyncio.run(check())


class RestartCleanupTests(unittest.TestCase):
    def test_only_owned_descendants_are_stopped(self):
        import subprocess, time, psutil, setup_api
        flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        executable = getattr(sys, '_base_executable', None) or sys.executable
        code = ('import subprocess,sys,time; '
                'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                'print(p.pid,flush=True); time.sleep(60)')
        parent = subprocess.Popen([executable, '-u', '-c', code], stdout=subprocess.PIPE,
                                  text=True, creationflags=flags)
        unrelated = subprocess.Popen([executable, '-c', 'import time; time.sleep(60)'], creationflags=flags)
        child = None
        try:
            child = psutil.Process(int(parent.stdout.readline().strip()))
            started = time.monotonic()
            self.assertTrue(setup_api._stop_restart_children(psutil.Process(parent.pid)))
            self.assertLess(time.monotonic() - started, 5)
            self.assertFalse(child.is_running() and child.status() != psutil.STATUS_ZOMBIE)
            self.assertIsNone(parent.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            if child and child.is_running():
                child.kill()
            for proc in (parent, unrelated):
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
            parent.stdout.close()

    def test_survivors_get_forced_kill_with_shared_bounded_waits(self):
        from unittest.mock import Mock
        import psutil, setup_api
        child, parent = Mock(), Mock()
        parent.children.side_effect = [[child], []]
        with patch.object(psutil, 'wait_procs', side_effect=[([], [child]), ([child], [])]) as wait:
            self.assertTrue(setup_api._stop_restart_children(parent))
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual([c.kwargs['timeout'] for c in wait.call_args_list], [2.0, 1.0])
        parent.terminate.assert_not_called()

    def test_late_child_refuses_automatic_restart(self):
        from unittest.mock import Mock
        import psutil, setup_api
        child, parent = Mock(), Mock()
        child.pid, child.status.return_value = 12345, 'running'
        parent.children.side_effect = [[], [child]]
        with patch.object(psutil, 'wait_procs', return_value=([], [])):
            self.assertFalse(setup_api._stop_restart_children(parent))

    def test_exit_75_requires_successful_cleanup(self):
        import setup_api
        from types import SimpleNamespace
        from unittest.mock import Mock
        for success in (True, False):
            close = Mock()
            with self.subTest(cleanup=success), patch.object(setup_api.time, 'sleep'), \
                    patch.dict(sys.modules, {'call_recorder':SimpleNamespace(REC=SimpleNamespace(close=close))}), \
                    patch.object(setup_api, '_stop_restart_children', return_value=success), \
                    patch.object(setup_api.os, '_exit') as exit_mock:
                setup_api.exit_for_restart()
                exit_mock.assert_called_once_with(75 if success else 1)
                close.assert_called_once_with(timeout=2.0)

    def test_recorder_close_failure_refuses_restart(self):
        import setup_api
        from types import SimpleNamespace
        from unittest.mock import Mock
        for error in (TimeoutError('drain timeout'), OSError('recording write failed')):
            with self.subTest(error=type(error).__name__), patch.object(setup_api.time, 'sleep'), \
                    patch.dict(sys.modules, {'call_recorder':SimpleNamespace(REC=SimpleNamespace(close=Mock(side_effect=error)))}), \
                    patch.object(setup_api, '_stop_restart_children') as stop, \
                    patch.object(setup_api.os, '_exit') as exit_mock:
                setup_api.exit_for_restart()
                exit_mock.assert_called_once_with(1)
                stop.assert_not_called()


class SetupWizardTests(unittest.TestCase):
    def test_executed_wizard_uses_live_ready_agent_and_optional_customization(self):
        """Execute shipped JS against fake DOM/API; no browser, backend or models."""
        import shutil, subprocess
        node = shutil.which('node')
        if not node:
            self.skipTest('Node is needed to execute wizard JavaScript')
        page = Path(__file__).resolve().parents[1] / 'static/setup.html'
        harness = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
let source = fs.readFileSync(process.argv[1], 'utf8').split('<script>')[1].split('</script>')[0];
source = source.slice(0, source.lastIndexOf('load().then'));
const nodes = new Map(), requests = [];
const get = id => { if (!nodes.has(id)) nodes.set(id, {value:'', checked:false, disabled:false, options:[],
  innerHTML:'', textContent:'', className:'', style:{}, classList:{toggle(name,value){this[name]=value;},add(){},remove(){}},
  addEventListener(){},removeAttribute(name){delete this[name];}}); return nodes.get(id); };
const juniper = {id:'juniper',name:'Juniper',voice:'juniper-natural',bundled:true};
const moon = {id:'moon',name:'Moon',voice:'moon-voice',bundled:false};
const status = {gpu:{}, ollama:{running:false,installed:false,models:[]}, devices:{inputs:[],outputs:[]},
  settings:{audio:{},llm:{cloud:{provider:'custom',vision:true}}}, recommended:{}, local_models:[],
  providers:{custom:{label:'Custom',model:'m',base_url:'http://fixture'},
             other:{label:'Other',model:'other-model',base_url:'http://other'}},
  agent_choices:[juniper],ready_agent:juniper,voices:['juniper-natural'],pull:{status:'idle'},brain_ready:true};
const context = {document:{getElementById:get,querySelectorAll:()=>[]}, console, URLSearchParams,
  localStorage:{getItem:()=>null,setItem(){},removeItem(){}}, setTimeout, clearTimeout,
  fetch:async(path, opts={})=>{requests.push({path,body:opts.body && JSON.parse(opts.body)});
    return {ok:true,json:async()=>path==='/api/setup/status'?status:{ok:true,ms:7}};},
  assert, status, juniper, moon, requests};
vm.runInNewContext(source + `
(async()=>{
  await load();
  assert.equal($('cVision').checked,true);
  $('prov').value='other'; provChanged();
  assert.equal($('cVision').checked,false);
  provChanged({provider:'custom',vision:true});
  assert.equal($('cVision').checked,false);
  $('prov').value='custom'; provChanged({provider:'custom',vision:true});
  assert.equal($('cVision').checked,true);
  mode='cloud'; restartAndWait=async()=>{};
  await saveBrain();
  assert.equal(requests.find(r=>r.path==='/api/setup/llm').body.vision,true);
  W.agent='stale'; W.agentChosen=true; await fillSummary();
  assert.equal(W.agent,'juniper'); assert.equal($('finishBtn').disabled,false);
  assert.ok($('summary').innerHTML.includes('juniper-natural'));
  await saveAudio(); await fillSummary();
  assert.equal(W.step,5);
  assert.equal(requests.some(r=>['/api/voices/import','/api/agents/draft','/api/agents'].includes(r.path)),false);
  await saveAudio(true); assert.equal(W.step,3);
  status.agent_choices=[juniper,moon]; status.ready_agent=moon;
  W.agent='juniper'; W.agentChosen=false; await fillSummary(); assert.equal(W.agent,'moon');
  W.agent='juniper'; W.agentChosen=true; await fillSummary(); assert.equal(W.agent,'juniper');
  status.brain_ready=false;status.brain_error='Choose and save a brain.';
  await fillSummary();assert.equal(W.agent,'juniper');assert.equal($('finishBtn').disabled,true);
  assert.equal($('summaryBrainBack').classList.hide,false);
  assert.equal($('doneMsg').textContent,status.brain_error);
  const beforeBrainCompletes=requests.filter(r=>r.path==='/api/setup/complete').length;
  await finish();assert.equal(requests.filter(r=>r.path==='/api/setup/complete').length,beforeBrainCompletes);
  status.brain_ready=true;
  status.agent_choices=[moon]; await fillSummary(); assert.equal(W.agent,'moon');
  status.agent_choices=[];status.ready_agent=null;status.bundle_error='Repair the bundle or create an agent.';
  await saveAudio(); assert.equal(W.step,3);
  assert.equal($('voiceMsg').textContent,status.bundle_error);
  await fillSummary(); assert.equal($('finishBtn').disabled,true);
  const completes=requests.filter(r=>r.path==='/api/setup/complete').length;
  await finish();
  assert.equal(requests.filter(r=>r.path==='/api/setup/complete').length,completes);
})()`, context).then(()=>console.log('wizard flow passed')).catch(e=>{console.error(e);process.exitCode=1;});
'''
        result = subprocess.run([node, '-e', harness, str(page)], capture_output=True, text=True, timeout=15,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('wizard flow passed', result.stdout)


if __name__ == "__main__":
    unittest.main()
