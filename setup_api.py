"""First-run setup API (public build). Everything the wizard needs, nothing hidden.

No API keys ship with ATLAS. The user either runs a local model through Ollama
(we detect it, recommend a size for their GPU and pull it with a progress bar)
or links their own cloud provider key. Voice import and persona creation reuse
the creator endpoints in server.py (/api/voices/import, /api/agents/*).

Owner-only (this machine) for anything that writes or spends: the server binds
0.0.0.0, so a LAN client must never be able to change the model or read keys.
"""
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

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _owner(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in _LOOPBACK or host.startswith("127.")


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
    hdr = _re.compile(r"^\s*\[(SPEAK to=[^\]]+|HOLD)\]")
    got, ms = [], None
    for line, want in PROBE_CASES:
        body = {"model": model, "max_tokens": 60, "temperature": 0,
                "messages": [{"role": "system", "content": PROBE_SYSTEM},
                             {"role": "user", "content": line}]}
        t0 = time.time()
        try:
            r = requests.post(url + "/chat/completions", headers=h, timeout=timeout,
                              json=dict(body, chat_template_kwargs={"enable_thinking": False}))
            if r.status_code in (400, 422):  # providers that reject unknown fields
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
@router.get("/api/setup/status")
async def setup_status():
    loc = US.local_llm()
    gpu = gpu_info()
    try:
        import audio_io
        devices = audio_io.list_audio_devices()
    except Exception:  # noqa: BLE001
        devices = {"inputs": [], "outputs": []}
    try:
        import voices
        voice_list = voices.voice_names()
    except Exception:  # noqa: BLE001
        voice_list = []
    try:
        import agents as _ag
        agent_list = [a.get("id") for a in _ag.load_registry()]
    except Exception:  # noqa: BLE001
        agent_list = []
    return {
        "setup_needed": US.setup_needed(),
        "public": US.is_public(),
        "settings": US.public_view(),
        "gpu": gpu,
        "ollama": ollama_info(loc["url"]),
        "recommended": US.recommend_local_model(gpu.get("total_gb", 0)),
        "local_models": US.LOCAL_MODELS,
        "providers": {k: {kk: v[kk] for kk in ("label", "base_url", "model", "keys")} for k, v in US.PROVIDERS.items()},
        "devices": devices,
        "voices": voice_list,
        "agents": agent_list,
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
        res = test_local(url, model) if not b.get("skip_probe") else {"ok": True}
        if not res["ok"]:
            return JSONResponse(res, status_code=400)
        US.update({"llm": {"mode": "local", "local": {"model": model, "url": url}}})
        return {**res, "settings": US.public_view(), "restart": True}
    if mode == "cloud":
        prov = b.get("provider") or "custom"
        preset = US.PROVIDERS.get(prov, US.PROVIDERS["custom"])
        base = (b.get("base_url") or preset["base_url"]).strip()
        model = (b.get("model") or preset["model"]).strip()
        key = b.get("api_key")
        if key in (None, ""):  # keep the stored key when the user didn't retype it
            key = ((US.load().get("llm") or {}).get("cloud") or {}).get("api_key", "")
        res = test_cloud(base, model, key)
        if not res["ok"]:
            return JSONResponse(res, status_code=400)
        US.update({"llm": {"mode": "cloud", "cloud": {"provider": prov, "base_url": base, "model": model,
                                                       "api_key": key, "vllm": prov == "custom" and bool(b.get("vllm"))}}})
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
    patch = {"setup_complete": True}
    if b.get("agent"):
        patch["agent"] = b["agent"]
    US.update(patch)
    return {"ok": True, "restart": True}


@router.post("/api/setup/reset")
async def setup_reset(request: Request):
    """Re-run the wizard (keeps voices/agents; clears the done flag)."""
    if not _owner(request):
        return _deny()
    US.update({"setup_complete": False})
    return {"ok": True}
