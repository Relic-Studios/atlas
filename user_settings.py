"""User settings for ATLAS (first-run setup writes these; nothing is bundled).

One JSON file next to the app: user/settings.json. No API keys ship with ATLAS:
a cloud key only exists here if the user typed it into setup, and it never
leaves this machine except in requests to the provider the user picked.

    {
      "setup_complete": true,
      "llm": {
        "mode": "local" | "cloud",
        "local": {"url": "http://127.0.0.1:11434", "model": "qwen3:8b"},
        "cloud": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                  "model": "...", "api_key": "...", "vllm": false}
      },
      "audio": {"input": "Microphone (…)", "output": "Speakers (…)"},
      "agent": "my-agent"
    }

The dev build (dev_pack/ present) keeps working with no settings file at all.
"""
import json
import os
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = Path(os.environ.get("ATLAS_SETTINGS", ROOT / "user" / "settings.json"))
_LOCK = threading.RLock()

# Cloud presets: OpenAI-compatible chat endpoints. Users bring their own key.
PROVIDERS = {
    "claude_code": {"label": "Claude Code · subscription account", "base_url": "", "model": "sonnet", "keys": "",
                    "protocol": "claude_code", "auth_mode": "subscription", "install_url": "https://code.claude.com/docs/en/setup"},
    "codex": {"label": "Codex · ChatGPT account", "base_url": "", "model": "", "keys": "",
              "protocol": "codex_app_server", "auth_mode": "subscription", "install_url": "https://developers.openai.com/codex/cli"},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "model": "qwen/qwen3-235b-a22b-2507", "keys": "https://openrouter.ai/keys"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1",
               "model": "gpt-4.1-mini", "keys": "https://platform.openai.com/api-keys"},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                 "model": "deepseek-flash", "keys": "https://platform.deepseek.com"},
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1",
             "model": "llama-3.3-70b-versatile", "keys": "https://console.groq.com/keys"},
    "together": {"label": "Together AI", "base_url": "https://api.together.xyz/v1",
                 "model": "Qwen/Qwen2.5-72B-Instruct-Turbo", "keys": "https://api.together.ai/settings/api-keys"},
    "custom": {"label": "Custom (OpenAI-compatible, e.g. your own vLLM / LM Studio)",
               "base_url": "http://127.0.0.1:8001/v1", "model": "", "keys": ""},
}

# Local model ladder by free VRAM (Ollama tags). Picked automatically, user can override.
# Measured 10-03 (tests/bench_min_model.py + bench_quality.py, ctx 4096, the full
# ~3.1k-token prompt). min_vram_gb = the model's own measured VRAM + margin.
#   qwen3:14b  8.7 GB  0 broken / 0 butt-in / 0 missed / 12/12 requests / 29/30 clean
#   qwen3:8b   4.9 GB  0 broken / 0 butt-in / 3/12 missed / 10/12 requests / 18/21 clean
#   qwen3:4b-instruct  6/12 requests -> below the bar
#   qwen3:4b (thinking-only build) 31/40 broken headers -> unusable
# Below 8B a local model can't carry the prompt; the wizard steers to cloud.
LOCAL_MODELS = [
    {"min_vram_gb": 9.5, "model": "qwen3:14b", "label": "Qwen3 14B (~9 GB)"},
    {"min_vram_gb": 5.5, "model": "qwen3:8b", "label": "Qwen3 8B (minimum, ~5 GB)"},
]
# Official Ollama listings declare both image input and tool calling. Memory
# estimates reserve space for image/context processing in addition to weights;
# these are sizing suggestions, not a promise about a machine's free VRAM.
VISION_MODELS = [
    {"min_vram_gb": 10.0, "model": "qwen3-vl:8b-instruct", "label": "Qwen3-VL 8B Instruct (~6.2 GB download)",
     "vision": True, "tools": True, "source": "https://ollama.com/library/qwen3-vl:8b-instruct"},
    {"min_vram_gb": 12.0, "model": "qwen3.5:9b", "label": "Qwen3.5 9B (~6.6–7.6 GB download)",
     "vision": True, "tools": True, "source": "https://ollama.com/library/qwen3.5:9b"},
    {"min_vram_gb": 10.0, "model": "ministral-3:8b", "label": "Ministral 3 8B (~6 GB download)",
     "vision": True, "tools": True, "source": "https://ollama.com/library/ministral-3:8b"},
]
MIN_CONTEXT = 8192  # persona + participation + enabled tool schemas exceeded 5.5k in real calls
# Whisper + TTS already need ~6 GB of the GPU; leave that headroom.
SPEECH_VRAM_GB = 6.0


def is_public() -> bool:
    """Public build = the owner's dev pack is absent."""
    return not (ROOT / "dev_pack").is_dir()


def _cloud_block(d: dict):
    c = (d.get("llm") or {}).get("cloud") if isinstance(d.get("llm"), dict) else None
    return c if isinstance(c, dict) else None


def load() -> dict:
    """Settings with the cloud key decrypted (it is stored DPAPI-protected on Windows)."""
    try:
        d = json.loads(PATH.read_text(encoding="utf-8"))
        d = d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}
    c = _cloud_block(d)
    if c and c.get("api_key"):
        import security
        c["api_key"] = security.unprotect(c["api_key"])
    return d


def save(data: dict) -> None:
    import copy
    import security
    data = copy.deepcopy(data)
    c = _cloud_block(data)
    if c and c.get("api_key"):
        c["api_key"] = security.protect(c["api_key"])
    with _LOCK:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if os.name != "nt":
            os.chmod(tmp, 0o600)            # the key is plaintext off Windows: owner-only file
        os.replace(tmp, PATH)


def update(patch: dict) -> dict:
    """Deep-merge `patch` into the stored settings and save."""
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a
    with _LOCK:
        d = load()
    d = merge(d, patch or {})
    save(d)
    return d


def setup_needed() -> bool:
    return is_public() and not load().get("setup_complete")


def replace_cloud(cloud: dict) -> None:
    """Replace credential ownership atomically; never merge credentials across providers."""
    with _LOCK:
        data = load()
        block = data.setdefault("llm", {})
        block["mode"] = "cloud"
        block["cloud"] = dict(cloud)
        save(data)


def remember_account_outcome(cfg: dict, error: dict | None = None) -> None:
    """Persist a limit only for the still-selected account/model, without claiming probe success."""
    with _LOCK:
        data=load()
        cloud=(data.get("llm") or {}).get("cloud") or {}
        if any(cloud.get(key)!=cfg.get(key) for key in ("provider","model","account_binding")):
            return
        if cloud.get("provider") not in ("codex","claude_code") or not cloud.get("account_binding"):
            return
        evidence=cloud.setdefault("verification",{})
        if error is not None:
            evidence.update(status="limited",limit_active=True,provider_error=dict(error))
        else:
            evidence.update(limit_active=False,provider_error=None)
        save(data)


def selected_llm_mode() -> str | None:
    """An explicit Settings choice, shared by dev and packaged launches."""
    mode = (load().get("llm") or {}).get("mode")
    return mode if mode in ("local", "cloud") else None


def llm_mode() -> str:
    return selected_llm_mode() or "local"


def startup_local_config() -> dict | None:
    """Use Settings in either build; retain an unconfigured legacy dev profile."""
    if selected_llm_mode() is None and not is_public():
        return None
    local = local_llm()
    return {"url": local["url"], "model": local["model"] or "qwen3:8b"}


def local_llm() -> dict:
    loc = (load().get("llm") or {}).get("local") or {}
    return {"url": loc.get("url") or "http://127.0.0.1:11434", "model": loc.get("model") or ""}


def remember_llm_backend(mode: str, cfg: dict | None = None) -> None:
    """Persist a switch in either build, migrating an explicitly chosen legacy host."""
    patch = {"mode": "cloud" if mode == "remote" else "local"}
    if mode == "remote" and cloud_config() is None:
        if not cfg or not cfg.get("url") or not cfg.get("model"):
            raise ValueError("The selected remote model is not configured")
        patch["cloud"] = {"provider": "custom", "base_url": cfg["url"],
                          "model": cfg["model"], "api_key": cfg.get("key") or "",
                          "vllm": bool(cfg.get("vllm", True)),
                          "context": int(cfg.get("context") or MIN_CONTEXT),
                          "allow_local_fallback": bool(cfg.get("allow_local_fallback", True))}
        if isinstance(cfg.get("vision"), bool):
            patch["cloud"]["vision"] = cfg["vision"]
    update({"llm": patch})


def cloud_config():
    """Cloud block in remote_llm's format, or None. base_url may end in /v1."""
    c = (load().get("llm") or {}).get("cloud") or {}
    prov = c.get("provider") or "custom"
    if prov in ("codex", "claude_code"):
        evidence = c.get("verification") or {}
        if (not c.get("model") or c.get("auth_mode") != "subscription"
                or not c.get("account_binding") or evidence.get("status") not in ("tested", "limited")
                or evidence.get("isolation_verified") is not True):
            return None
        return {"provider":prov,"protocol":PROVIDERS[prov]["protocol"],"auth_mode":"subscription",
                "url":"","model":c["model"],"key":"","executable":c.get("executable"),
                "account_binding":c["account_binding"],"instruction_fingerprint":c.get("instruction_fingerprint"),
                "verification":evidence,"label":f"{PROVIDERS[prov]['label']} · {c['model']}",
                "cloud":True,"vllm":False,"allow_local_fallback":False,"context":int(c.get("context") or MIN_CONTEXT),
                "vision":c.get("vision") if isinstance(c.get("vision"),bool) else None}
    url = (c.get("base_url") or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if not url or not c.get("model"):
        return None
    vllm = prov == "custom" and c.get("vllm") is True
    return {"url": url, "model": c["model"], "key": c.get("api_key") or "",
            "label": f"{PROVIDERS.get(prov, {}).get('label', 'Cloud')} · {c['model']}",
            "vllm": vllm, "cloud": not vllm, "context": int(c.get("context") or 8192),
            "allow_local_fallback": bool(c.get("allow_local_fallback", True)),
            "vision": c.get("vision") if isinstance(c.get("vision"), bool) else None}


def audio_devices() -> dict:
    a = load().get("audio") or {}
    return {"input": a.get("input"), "output": a.get("output")}


def public_view() -> dict:
    """Settings safe to send to the UI: the API key is never echoed back."""
    d = json.loads(json.dumps(load()))
    cloud = (d.get("llm") or {}).get("cloud")
    if isinstance(cloud, dict) and cloud.get("api_key"):
        k = cloud["api_key"]
        cloud["api_key"] = ""
        cloud["has_key"] = True
        cloud["key_hint"] = ("…" + k[-4:]) if len(k) > 8 else "set"
    return d


def recommend_local_model(vram_gb: float, vision: bool = False) -> dict:
    avail = max(0.0, (vram_gb or 0.0) - SPEECH_VRAM_GB)
    choices = VISION_MODELS if vision else LOCAL_MODELS
    for m in choices:
        if avail >= m["min_vram_gb"]:
            return dict(m, fits=True)
    # Too little VRAM for even the minimum: offer it, but say cloud is the better fit.
    return dict(choices[-1], fits=False,
                note="This GPU is below the local minimum; a cloud model will work much better.")
