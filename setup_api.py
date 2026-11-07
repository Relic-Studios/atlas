"""First-run setup API (public build). Everything the wizard needs, nothing hidden.

No API keys ship with ATLAS. The user either runs a local model through Ollama
(we detect it, recommend a size for their GPU and pull it with a progress bar)
or links their own cloud provider key. Voice import and persona creation reuse
the creator endpoints in server.py (/api/voices/import, /api/agents/*).

Owner-only (this machine) for anything that writes or spends: the server binds
0.0.0.0, so a LAN client must never be able to change the model or read keys.
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
import threading
import time

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import user_settings as US

log = logging.getLogger(__name__)
router = APIRouter()

def _stop_restart_children(parent=None, term_timeout=2.0, kill_timeout=1.0) -> bool:
    """Release this backend's descendants before Electron replaces it.

    RealtimeSTT's Windows worker can outlive os._exit, retaining Whisper on the
    GPU. Use captured process objects and bounded waits, never executable names.
    """
    import psutil
    try:
        parent = parent or psutil.Process()
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                log.warning("restart: could not terminate child %s: %s", child.pid, exc)
        _, alive = psutil.wait_procs(children, timeout=term_timeout)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as exc:
                log.warning("restart: could not kill child %s: %s", child.pid, exc)
        _, alive = psutil.wait_procs(alive, timeout=kill_timeout)
        remaining = parent.children(recursive=True)
        def running(child):
            try:
                return child.is_running() and child.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                return False
        left = [p.pid for p in alive + remaining if running(p)]
        if left:
            log.error("restart refused: backend children still alive: %s", sorted(set(left)))
        return not left
    except psutil.Error:
        log.exception("restart refused: could not verify backend child cleanup")
        return False


def exit_for_restart():
    """Deliver the response, then restart only after owned workers are gone."""
    time.sleep(0.6)
    try:
        from call_recorder import REC
        REC.close(timeout=2.0)
        code = 75 if _stop_restart_children() else 1
    except Exception:
        log.exception("restart refused: recorder close or child cleanup failed")
        code = 1
    os._exit(code)


_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _owner(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", "") or ""
    if not (host in _LOOPBACK or host.startswith("127.")):
        return False
    origin=(getattr(request,"headers",{}) or {}).get("origin")
    if origin:
        from urllib.parse import urlsplit
        try:
            parsed=urlsplit(origin)
            port=parsed.port or (443 if parsed.scheme=="https" else 80)
            return (parsed.scheme in ("http","https") and parsed.hostname in _LOOPBACK
                    and port==int(os.environ.get("ATLAS_PORT","8000")))
        except ValueError:
            return False
    return True


def _deny():
    return JSONResponse({"error": "setup can only be changed from this computer"}, status_code=403)


# ---------------------------------------------------------------------------
# probes (pure-ish, unit-testable)
# ---------------------------------------------------------------------------
def gpu_info() -> dict:
    """Name + total/free VRAM of the first NVIDIA GPU, or {} if none."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        name, total, free = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "total_gb": round(int(total) / 1024, 1), "free_gb": round(int(free) / 1024, 1)}
    except Exception:  # noqa: BLE001
        return {}


def ollama_info(url: str) -> dict:
    """Is Ollama installed / running, and which models does it have?"""
    info = {"installed": bool(shutil.which("ollama")), "running": False, "models": [], "version": None}
    try:
        r = requests.get(url.rstrip("/") + "/api/version", timeout=2)
        info["running"] = r.ok
        info["version"] = r.json().get("version") if r.ok else None
        if r.ok:
            info["installed"] = True
            t = requests.get(url.rstrip("/") + "/api/tags", timeout=3).json()
            info["models"] = sorted(m.get("name", "") for m in t.get("models", []))
    except Exception:  # noqa: BLE001
        pass
    return info


def test_cloud(base_url: str, model: str, api_key: str, timeout: float = 20.0) -> dict:
    """One tiny real chat call. Returns {"ok": bool, "error"?, "reply"?, "ms"}."""
    url = (base_url or "").strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    if not model:
        return {"ok": False, "error": "pick a model"}
    h = {"Content-Type": "application/json", "X-Title": "ATLAS"}
    if api_key:
        h["Authorization"] = "Bearer " + api_key.strip()
    # Same routing probe as local models: a key that works on a model that can't emit
    # ATLAS's [SPEAK]/[HOLD] header is still a broken setup (10-03: reasoning models
    # spent an 8-token probe thinking and returned an empty reply that was marked "ok").
    import re as _re
    from remote_llm import provider_request_body
    hdr = _re.compile(r"^\s*\[(SPEAK to=[^\]]+|HOLD)\]")
    got, ms = [], None
    for line, want in PROBE_CASES:
        body = {"model": model, "max_tokens": 60, "temperature": 0,
                "messages": [{"role": "system", "content": PROBE_SYSTEM},
                             {"role": "user", "content": line}]}
        body = provider_request_body(body, url, think=False)
        t0 = time.time()
        try:
            r = requests.post(url + "/chat/completions", headers=h, timeout=timeout, json=body)
        except requests.RequestException as e:
            return {"ok": False, "error": f"could not reach {url}: {type(e).__name__}"}
        if ms is None:
            ms = int((time.time() - t0) * 1000)
        bad_http = _cloud_status(r, ms)
        if bad_http is not None:
            return bad_http
        try:
            text = r.json()["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            return {"ok": False, "error": "unexpected response (is this an OpenAI-compatible endpoint?)", "ms": ms}
        m = hdr.match(text)
        got.append((want, (m.group(1).split()[0] if m else None), text.strip()[:120]))
    bad = [g for g in got if g[1] is None]
    if bad:
        return {"ok": False, "ms": ms, "probe": got,
                "error": (f"the key works, but {model} doesn't follow the reply format ATLAS needs "
                          f"(it answered: \"{bad[0][2] or '(nothing)'}\"). Try a larger or non-reasoning model.")}
    if got[0][1] != "SPEAK" or not hdr.sub("", got[0][2], count=1).strip():
        return {"ok": False, "ms": ms, "probe": got,
                "error": f"the key works, but {model} gave no spoken reply when directly addressed. Try another model."}
    return {"ok": True, "ms": ms, "probe": got, "reply": got[0][2][:60],
            "routing_correct": sum(1 for w, g, _ in got if w == g)}


def _cloud_status(r, ms):
    """None if the HTTP response is usable, else a user-facing error dict."""
    if r.status_code in (401, 403):
        return {"ok": False, "error": "the provider rejected the API key", "ms": ms}
    if r.status_code == 404:
        return {"ok": False, "error": "model or endpoint not found (check the model name)", "ms": ms}
    if not r.ok:
        msg = ""
        try:
            msg = (r.json().get("error") or {}).get("message", "") if isinstance(r.json().get("error"), dict) \
                else str(r.json().get("error", ""))
        except ValueError:
            msg = r.text[:160]
        return {"ok": False, "error": f"HTTP {r.status_code}: {msg}"[:240], "ms": ms}
    return None


# ---------------------------------------------------------------------------
# Ollama pull with progress (one at a time, background thread)
# ---------------------------------------------------------------------------
_PULL = {"model": None, "status": "idle", "completed": 0, "total": 0, "error": None}
_PULL_LOCK = threading.Lock()



PROBE_SYSTEM = ("You are Sam, a voice agent in a group call. Begin EVERY reply with a control header: "
                "[SPEAK to=S1] followed by your short spoken reply, or [HOLD] alone if the line is not for you.")
PROBE_CASES = [("[S1] Sam, what's your favourite colour?", "SPEAK"),
               ("[S2] S3, did you feed the dog?", "HOLD")]


def test_local(url: str, model: str, timeout: float = 120.0) -> dict:
    """Can this local model drive ATLAS? One routing probe per case (10-03 bench: thinking-only
    builds like qwen3:4b emit no header ~78% of the time; they must be rejected, not saved)."""
    import re as _re
    hdr = _re.compile(r"^\s*\[(SPEAK to=[^\]]+|HOLD)\]")
    got = []
    for line, want in PROBE_CASES:
        try:
            r = requests.post(url.rstrip("/") + "/api/chat", timeout=timeout, json={
                "model": model, "stream": False, "think": False, "keep_alive": "10m",
                "options": {"temperature": 0, "num_ctx": US.MIN_CONTEXT, "num_predict": 40},
                "messages": [{"role": "system", "content": PROBE_SYSTEM},
                             {"role": "user", "content": "/no_think " + line}]})
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"could not reach Ollama: {e}"}
        if r.status_code != 200:
            return {"ok": False, "error": f"Ollama said {r.status_code}: {r.text[:200]}"}
        text = ((r.json().get("message") or {}).get("content") or "")
        m = hdr.match(text)
        got.append((want, (m.group(1).split()[0] if m else None), text[:120]))
    bad = [g for g in got if g[1] is None]
    if bad:
        return {"ok": False, "error": (f"{model} doesn't follow the reply format ATLAS needs "
                                       f"(it answered: \"{bad[0][2]}\"). Pick a recommended model "
                                       "or use a cloud provider."), "probe": got}
    if got[0][1] != "SPEAK" or not hdr.sub("", got[0][2], count=1).strip():
        return {"ok": False, "probe": got,
                "error": f"{model} gave no spoken reply when directly addressed. Pick another model or a cloud provider."}
    return {"ok": True, "probe": got, "routing_correct": sum(1 for w, g, _ in got if w == g)}


def _pull_worker(url: str, model: str):
    try:
        with requests.post(url.rstrip("/") + "/api/pull", json={"model": model, "stream": True},
                           stream=True, timeout=(5, 600)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                with _PULL_LOCK:
                    if d.get("error"):
                        _PULL.update(status="error", error=d["error"])
                        return
                    _PULL["status"] = d.get("status", _PULL["status"])
                    if d.get("total"):
                        _PULL["total"] = d["total"]
                        _PULL["completed"] = d.get("completed", 0)
        with _PULL_LOCK:
            _PULL.update(status="success")
    except Exception as e:  # noqa: BLE001
        with _PULL_LOCK:
            _PULL.update(status="error", error=f"{type(e).__name__}: {e}"[:200])


def pull_state() -> dict:
    with _PULL_LOCK:
        d = dict(_PULL)
    d["pct"] = round(100 * d["completed"] / d["total"], 1) if d["total"] else 0.0
    return d


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
def configured_brain_error() -> str | None:
    """Configuration readiness only; setup's existing save route owns model probes."""
    llm = US.load().get("llm") or {}
    if not isinstance(llm, dict):
        return "Choose and save a brain before completing setup."
    nonblank = lambda value: isinstance(value, str) and bool(value.strip())
    mode = llm.get("mode") or "local"
    if mode == "local":
        local = llm.get("local") or {}
        if isinstance(local, dict) and nonblank(local.get("model")):
            return None
    elif mode == "cloud":
        cloud = llm.get("cloud") or {}
        if isinstance(cloud, dict):
            provider = cloud.get("provider") or "custom"
            if provider in ("codex","claude_code"):
                return None if US.cloud_config() else "Connect, verify and save your subscription brain before completing setup."
            if (isinstance(provider, str)
                    and nonblank(cloud.get("base_url")) and nonblank(cloud.get("model"))):
                preset = US.PROVIDERS.get(provider, US.PROVIDERS["custom"])
                if not preset.get("keys") or nonblank(cloud.get("api_key")):
                    return None
                return "Save the API key for your chosen brain before completing setup."
    return "Choose and save a brain before completing setup."


@router.get("/api/setup/status")
def setup_status():   # plain def: FastAPI runs it off the event loop (nvidia-smi, audio devices, Ollama probe)
    loc = US.local_llm()
    gpu = gpu_info()
    try:
        import audio_io
        devices = audio_io.list_audio_devices()
    except Exception:  # noqa: BLE001
        devices = {"inputs": [], "outputs": []}
    try:
        import agent_registry as registry
        registry.reload()
        agent_list = registry.agent_ids()
        voice_list = list(registry.voices())
        choices, ready, bundle_error = registry.ready_agents(), registry.ready_agent(), registry.bundle_error()
    except Exception:  # noqa: BLE001
        agent_list = []
        voice_list = []
        choices, ready, bundle_error = [], None, "Agent configuration could not be read. Repair ATLAS, or create an agent."
    brain_error = configured_brain_error()
    return {
        "setup_needed": US.setup_needed(),
        "public": US.is_public(),
        "settings": US.public_view(),
        "gpu": gpu,
        "ollama": ollama_info(loc["url"]),
        "recommended": US.recommend_local_model(gpu.get("total_gb", 0)),
        "recommended_vision": US.recommend_local_model(gpu.get("total_gb", 0), vision=True),
        "local_models": [dict(m, vision=False, tools=True) for m in US.LOCAL_MODELS] + US.VISION_MODELS,
        "providers": {k: {kk: v.get(kk) for kk in ("label", "base_url", "model", "keys", "protocol", "auth_mode", "install_url")} for k, v in US.PROVIDERS.items()},
        "devices": devices,
        "voices": voice_list,
        "agents": agent_list,
        "agent_choices": choices,
        "ready_agent": ready,
        "bundle_error": bundle_error,
        "brain_ready": brain_error is None,
        "brain_error": brain_error,
        "pull": pull_state(),
    }


@router.post("/api/setup/ollama/pull")
async def setup_pull(request: Request):
    if not _owner(request):
        return _deny()
    model = ((await request.json()) or {}).get("model", "").strip()
    if not model:
        return JSONResponse({"error": "no model"}, status_code=400)
    with _PULL_LOCK:
        if _PULL["status"] not in ("idle", "success", "error"):
            return JSONResponse({"error": "a download is already running", "pull": dict(_PULL)}, status_code=409)
        _PULL.update(model=model, status="starting", completed=0, total=0, error=None)
    threading.Thread(target=_pull_worker, args=(US.local_llm()["url"], model),
                     name="OllamaPull", daemon=True).start()
    return {"pull": pull_state()}


@router.get("/api/setup/ollama/pull")
async def setup_pull_state():
    return {"pull": pull_state()}


@router.post("/api/setup/ollama/install")
async def setup_ollama_install(request: Request):
    """Install Ollama with winget (the user clicked the button). Returns at once;
    the UI polls /api/setup/status until Ollama answers."""
    if not _owner(request):
        return _deny()
    if os.name != "nt":
        # Linux/macOS: the official installer needs sudo, which a web button
        # can't ask for. Hand the user the one-liner instead.
        return JSONResponse({"error": "Run this in a terminal, then come back: "
                             "curl -fsSL https://ollama.com/install.sh | sh"}, status_code=501)
    w = shutil.which("winget")
    if not w:
        return JSONResponse({"error": "winget not available — download Ollama from https://ollama.com/download"},
                            status_code=501)
    subprocess.Popen([w, "install", "-e", "--id", "Ollama.Ollama", "--accept-source-agreements",
                      "--accept-package-agreements"], creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    return {"started": True}


def _account_error(exc):
    from provider_errors import ProviderError
    if isinstance(exc, ProviderError):
        status={"not_found":404,"busy":409,"sign_in_required":401,"usage_limit":429}.get(exc.code,400)
        return JSONResponse({"error": str(exc), "provider_error": exc.public()}, status_code=status,headers={"Cache-Control":"no-store"})
    return JSONResponse({"error": "The account operation failed. Update the official CLI and retry."}, status_code=400,headers={"Cache-Control":"no-store"})


def _account_response(value):
    return JSONResponse(value,headers={"Cache-Control":"no-store"})


def test_account(provider, model, configured=None, vision=None):
    """Same native session as voice; a verified limit is not a successful reply probe."""
    import re
    from provider_accounts import account_status, account_models
    from provider_errors import ProviderError
    from native_providers import NativeSession
    from codex_native import instruction_fingerprint
    account = account_status(provider, configured)
    if not account["connected"]:
        raise ProviderError(provider,"sign_in_required",account["message"])
    catalog = account_models(provider,configured)
    if not model:
        model = next((m["id"] for m in catalog if m.get("default")), catalog[0]["id"] if catalog else "")
    if not isinstance(model,str) or not model.strip():
        raise ProviderError(provider,"model_access","Choose an account model before continuing.")
    metadata = next((m for m in catalog if m["id"]==model),{})
    if provider=="codex" and not metadata:
        raise ProviderError(provider,"model_access","Choose a model listed for this Codex account.")
    cfg={"provider":provider,"protocol":US.PROVIDERS[provider]["protocol"],"auth_mode":"subscription",
         "model":model,"executable":configured or None,"account_binding":account["account_binding"],
         "label":US.PROVIDERS[provider]["label"],"url":"","allow_local_fallback":False}
    if provider=="codex":cfg["instruction_fingerprint"]=instruction_fingerprint()
    probe=[]
    start=time.monotonic()
    session=NativeSession(cfg)
    state="tested"
    provider_error=None
    try:
        for line,want in PROBE_CASES:
            try:
                reply=session.post("/api/chat",json={"messages":[{"role":"system","content":PROBE_SYSTEM},
                    {"role":"user","content":line}],"options":{"temperature":0,"num_predict":60}}).json()["message"]["content"]
            except ProviderError as exc:
                if exc.code!="usage_limit" or not session.isolation_verified:
                    raise
                state,provider_error="limited",exc.public()
                break
            match=re.match(r"^\s*\[(SPEAK to=[^\]]+|HOLD)\]",reply)
            got=match.group(1).split()[0] if match else None
            probe.append((want,got,reply.strip()[:120]))
            if not match or want=="SPEAK" and (got!="SPEAK" or not reply[match.end():].strip()):
                raise ProviderError(provider,"reply_format","The account connected, but this model did not follow Atlas's required reply format. Choose another model.")
        fresh=account_status(provider,configured)
        if not fresh["connected"] or fresh["account_binding"]!=cfg["account_binding"]:
            raise ProviderError(provider,"account_changed","The account changed during setup. Verify it and retry.")
        actual=session.resolved_model or model
        if provider=="claude_code" and model in ("sonnet","opus","haiku"):
            cfg["model"]=actual
        cfg.pop("label",None);cfg.pop("url",None);cfg.pop("allow_local_fallback",None)
        known=metadata.get("vision")
        cfg["vision"]=known if isinstance(known,bool) else vision if isinstance(vision,bool) else None
        cfg["verification"]={"status":state,"verified_at":int(time.time()),"isolation_verified":session.isolation_verified,
            "cli_version":session.cli_version,"provider_error":provider_error,"participation_tested":state=="tested"}
        return {"ok":True,"ms":int((time.monotonic()-start)*1000),"probe":probe,"verification":state,
                "provider_error":provider_error,"routing_correct":sum(w==g for w,g,_ in probe),"cloud":cfg}
    finally:
        session.close()


@router.post("/api/setup/provider/status")
async def setup_provider_status(request: Request):
    if not _owner(request):
        return _deny()
    b = await request.json()
    try:
        from provider_accounts import account_status
        return _account_response(await asyncio.to_thread(account_status, b.get("provider"), b.get("executable")))
    except Exception as exc:
        return _account_error(exc)


@router.post("/api/setup/provider/models")
async def setup_provider_models(request: Request):
    if not _owner(request):
        return _deny()
    b = await request.json()
    try:
        from provider_accounts import account_models
        return _account_response({"models": await asyncio.to_thread(account_models, b.get("provider"), b.get("executable"))})
    except Exception as exc:
        return _account_error(exc)


@router.post("/api/setup/provider/connect")
async def setup_provider_connect(request: Request):
    if not _owner(request):
        return _deny()
    b = await request.json()
    try:
        from provider_accounts import start_login
        return _account_response(await asyncio.to_thread(start_login, b.get("provider"), b.get("executable")))
    except Exception as exc:
        return _account_error(exc)


@router.get("/api/setup/provider/connect/{attempt_id}")
async def setup_provider_connection(request: Request, attempt_id: str):
    if not _owner(request):
        return _deny()
    try:
        from provider_accounts import login_state
        return _account_response(login_state(attempt_id))
    except Exception as exc:
        return _account_error(exc)


@router.delete("/api/setup/provider/connect/{attempt_id}")
async def setup_provider_disconnect(request: Request, attempt_id: str):
    if not _owner(request):
        return _deny()
    try:
        from provider_accounts import login_state
        return _account_response(await asyncio.to_thread(login_state, attempt_id, True))
    except Exception as exc:
        return _account_error(exc)


@router.post("/api/setup/llm")
async def setup_llm(request: Request):
    """{mode:"local", model, url?} or {mode:"cloud", provider, base_url, model, api_key}.
    Cloud is tested with one real request before it is saved."""
    if not _owner(request):
        return _deny()
    b = (await request.json()) or {}
    mode = b.get("mode")
    if mode == "local":
        model = (b.get("model") or "").strip()
        if not model:
            return JSONResponse({"error": "pick a model"}, status_code=400)
        url = b.get("url") or US.local_llm()["url"]
        res = await asyncio.to_thread(test_local, url, model) if not b.get("skip_probe") else {"ok": True}
        if not res["ok"]:
            return JSONResponse(res, status_code=400)
        US.update({"llm": {"mode": "local", "local": {"model": model, "url": url}}})
        return {**res, "settings": US.public_view(), "restart": True}
    if mode == "cloud":
        prov = b.get("provider") or "custom"
        if prov in ("codex","claude_code"):
            try:
                res=await asyncio.to_thread(test_account,prov,b.get("model"),b.get("executable"),b.get("vision"))
                cloud=res.pop("cloud")
                try:
                    context=int(b.get("context") or 0)
                except (TypeError,ValueError):
                    context=0
                if context>0:
                    cloud["context"]=max(US.MIN_CONTEXT,context)
                US.replace_cloud(cloud)
                return {**res,"settings":US.public_view(),"restart":True}
            except Exception as exc:
                return _account_error(exc)
        preset = US.PROVIDERS.get(prov, US.PROVIDERS["custom"])
        base = (b.get("base_url") or preset["base_url"]).strip()
        model = (b.get("model") or preset["model"]).strip()
        key = b.get("api_key")
        if key in (None, ""):  # keep the stored key when the user didn't retype it
            saved = ((US.load().get("llm") or {}).get("cloud") or {})
            # A saved credential belongs to one provider endpoint. A blank field on a different
            # provider/server must never forward the previous server's key to that destination.
            same_endpoint = (saved.get("base_url") or "").strip().rstrip("/") == base.rstrip("/")
            key = saved.get("api_key", "") if saved.get("provider") == prov and same_endpoint else ""
        res = await asyncio.to_thread(test_cloud, base, model, key)
        if not res["ok"]:
            return JSONResponse(res, status_code=400)
        cloud = {"provider": prov, "base_url": base, "model": model, "api_key": key,
                 "vllm": prov == "custom" and bool(b.get("vllm")), "vision": b.get("vision") is True}
        try:   # optional: the model's context window (memory budgets follow it, docs/designs/experience_memory.md D4)
            ctx = int(b.get("context") or 0)
        except (TypeError, ValueError):
            ctx = 0
        if 2048 <= ctx <= 1_048_576:
            cloud["context"] = ctx
        US.replace_cloud(cloud)
        return {**res, "settings": US.public_view(), "restart": True}
    return JSONResponse({"error": "mode must be local or cloud"}, status_code=400)


@router.post("/api/setup/audio")
async def setup_audio(request: Request):
    if not _owner(request):
        return _deny()
    b = (await request.json()) or {}
    US.update({"audio": {"input": b.get("input"), "output": b.get("output")}})
    return {"ok": True, "settings": US.public_view()}


@router.post("/api/setup/complete")
async def setup_complete(request: Request):
    if not _owner(request):
        return _deny()
    b = (await request.json()) or {}
    brain_error = configured_brain_error()
    if brain_error:
        return JSONResponse({"error": brain_error}, status_code=400)
    import agent_registry as registry
    registry.reload()
    selected = b.get("agent") if "agent" in b else registry.default_agent()
    ready = next((a for a in registry.ready_agents() if a["id"] == selected), None)
    if ready is None:
        return JSONResponse({"error": "Choose an available agent with a usable voice, or repair ATLAS to restore Juniper."}, status_code=400)
    patch = {"setup_complete": True, "agent": ready["id"]}
    US.update(patch)
    return {"ok": True, "restart": True}


@router.post("/api/setup/reset")
async def setup_reset(request: Request):
    """Re-run the wizard (keeps voices/agents; clears the done flag)."""
    if not _owner(request):
        return _deny()
    US.update({"setup_complete": False})
    return {"ok": True}
