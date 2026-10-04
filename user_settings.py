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
_LOCK = threading.Lock()

# Cloud presets: OpenAI-compatible chat endpoints. Users bring their own key.
PROVIDERS = {
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "model": "qwen/qwen3-235b-a22b-2507", "keys": "https://openrouter.ai/keys"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1",
               "model": "gpt-4.1-mini", "keys": "https://platform.openai.com/api-keys"},
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
    {"min_vram_gb": 9.5, "model": "qwen3:14b", "label": "Qwen3 14B (recommended, ~9 GB)"},
    {"min_vram_gb": 5.5, "model": "qwen3:8b", "label": "Qwen3 8B (minimum, ~5 GB)"},
]
MIN_CONTEXT = 4096  # the prompt alone is ~3.1k tokens
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


def llm_mode() -> str:
    return (load().get("llm") or {}).get("mode") or "local"


def local_llm() -> dict:
    loc = (load().get("llm") or {}).get("local") or {}
    return {"url": loc.get("url") or "http://127.0.0.1:11434", "model": loc.get("model") or ""}


def cloud_config():
    """Cloud block in remote_llm's format, or None. base_url may end in /v1."""
    c = (load().get("llm") or {}).get("cloud") or {}
    url = (c.get("base_url") or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    if not url or not c.get("model"):
        return None
    prov = c.get("provider") or "custom"
    return {"url": url, "model": c["model"], "key": c.get("api_key") or "",
            "label": f"{PROVIDERS.get(prov, {}).get('label', 'Cloud')} · {c['model']}",
            "vllm": bool(c.get("vllm", False)), "cloud": True}


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


def recommend_local_model(vram_gb: float) -> dict:
    avail = max(0.0, (vram_gb or 0.0) - SPEECH_VRAM_GB)
    for m in LOCAL_MODELS:
        if avail >= m["min_vram_gb"]:
            return dict(m, fits=True)
    # Too little VRAM for even the minimum: offer it, but say cloud is the better fit.
    return dict(LOCAL_MODELS[-1], fits=False,
                note="This GPU is below the local minimum; a cloud model will work much better.")
