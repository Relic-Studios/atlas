import sys as _sys
# Piped stdout on Windows defaults to cp1252; RealtimeTTS prints emoji per sentence,
# which raised UnicodeEncodeError inside synthesis -> zero audio. Force UTF-8.
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
# server.py
from queue import Queue, Empty
import logging
from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)


_REC_NAMES: dict = {}


def _rec_name(speaker, mgr):
    """Write a {"k": "name"} event the first time a speaker's name is known (or changes),
    so offline tools (Highlight reel) can show real names instead of S-labels."""
    name = mgr.people.name_of(speaker) if speaker else None
    if name and _REC_NAMES.get(speaker) != name:
        _REC_NAMES[speaker] = name
        _rec_event("name", spk=speaker, name=name)


def _rec_event(kind, **fields):
    """Best-effort call recording (call_recorder.py); never raises."""
    try:
        from call_recorder import REC
        REC.event(kind, **fields)
    except Exception:  # noqa: BLE001
        pass
    # Demo runs (demo_call.py) also log what was said, so the clipper can find
    # the agent's lines by text and time.
    try:
        import demo_call
        demo_call.note(kind, **fields)
    except Exception:  # noqa: BLE001
        pass
if __name__ == "__main__":
    logger.info("🖥️👋 Welcome to local real-time voice chat")

from upsample_overlap import UpsampleOverlap
from datetime import datetime
from colors import Colors
import uvicorn
import asyncio
import struct
import json
import re
import time
import threading # Keep threading for SpeechPipelineManager internals and AbortWorker
import sys
import os # Added for environment variable access

from typing import Any, Dict, Optional, Callable # Added for type hints in docstrings
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

USE_SSL = False
TTS_START_ENGINE = "qwen"
TTS_ORPHEUS_MODEL = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf"

LLM_START_PROVIDER = "ollama"
LLM_START_MODEL = "local-model:latest"  # same Model weights + Qwen3.8 mmproj (native vision)
try:  # public build: the model the user picked in first-run setup
    import user_settings as _us
    if _us.is_public():
        LLM_START_MODEL = _us.local_llm()["model"] or "qwen3:8b"
        os.environ.setdefault("OLLAMA_BASE_URL", _us.local_llm()["url"])
except Exception:  # noqa: BLE001
    pass
NO_THINK = True
DIRECT_STREAM = TTS_START_ENGINE=="orpheus"

# Discord-call bridge (B2 in / VAIO3 out + AEC + diarization). Off by default so
# the browser mic/speakers are the audio path. Flip to True to run inside a
# Discord call via Voicemeeter.
ENABLE_CALL_BRIDGE = True

if __name__ == "__main__":
    logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Starting engine: {Colors.apply(TTS_START_ENGINE).blue}")
    logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Direct streaming: {Colors.apply('ON' if DIRECT_STREAM else 'OFF').blue}")

# Define the maximum allowed size for the incoming audio queue
try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    if __name__ == "__main__":
        logger.info(f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Audio queue size limit set to: {Colors.apply(str(MAX_AUDIO_QUEUE_SIZE)).blue}")
except ValueError:
    if __name__ == "__main__":
        logger.warning("🖥️⚠️ Invalid MAX_AUDIO_QUEUE_SIZE env var. Using default: 50")
    MAX_AUDIO_QUEUE_SIZE = 50


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#from handlerequests import LanguageProcessor
#from audio_out import AudioOutProcessor
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager
from colors import Colors

LANGUAGE = "en"
# TTS_FINAL_TIMEOUT = 0.5 # unsure if 1.0 is needed for stability
TTS_FINAL_TIMEOUT = 1.0 # unsure if 1.0 is needed for stability

# --------------------------------------------------------------------
# Custom no-cache StaticFiles
# --------------------------------------------------------------------
class NoCacheStaticFiles(StaticFiles):
    """
    Serves static files without allowing client-side caching.

    Overrides the default Starlette StaticFiles to add 'Cache-Control' headers
    that prevent browsers from caching static assets. Useful for development.
    """
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        """
        Gets the response for a requested path, adding no-cache headers.

        Args:
            path: The path to the static file requested.
            scope: The ASGI scope dictionary for the request.

        Returns:
            A Starlette Response object with cache-control headers modified.
        """
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        # These might not be strictly necessary with no-store, but belt and suspenders
        if "etag" in response.headers:
             response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
             response.headers.__delitem__("last-modified")
        return response

# --------------------------------------------------------------------
# Lifespan management
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, initializing and shutting down resources.

    Initializes global components like SpeechPipelineManager, Upsampler, and
    AudioInputProcessor and stores them in `app.state`. Handles cleanup on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("🖥️▶️ Server starting up")
    # Post-call report for the previous call (tools/call_report.py), off the hot path.
    def _prev_call_report():
        try:
            import sys as _s, glob as _g, os as _o
            _s.path.insert(0, _o.path.join(_o.path.dirname(__file__), "tools"))
            import call_report as _cr
            files = sorted(_g.glob(_o.path.join(_o.path.dirname(__file__), "recordings", "*", "*.jsonl")),
                           key=_o.path.getmtime, reverse=True)
            for f in files[:5]:
                rows = _cr.load(f)
                if _cr.is_real_call(rows):
                    out = f[:-len(".jsonl")] + ".report.md"
                    if not _o.path.exists(out):
                        open(out, "w", encoding="utf-8").write(_cr.render(f, *_cr.analyse(rows)))
                        logger.info("🖥️📋 call report written: %s", out)
                    break
        except Exception as e:  # noqa: BLE001
            logger.warning("call report skipped: %s", e)
    import threading as _th
    _th.Thread(target=_prev_call_report, name="CallReport", daemon=True).start()
    # Initialize global components, not connection-specific state
    from echo_guard import EchoGuard
    app.state.EchoGuard = EchoGuard()
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
        no_think=NO_THINK,
        orpheus_model=TTS_ORPHEUS_MODEL,
    )

    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        is_orpheus=TTS_START_ENGINE=="orpheus",
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000, # seconds
    )
    app.state.Aborting = False # Keep this? Its usage isn't clear in the provided snippet. Minimizing changes.

    # Speaker diarization is available in BOTH modes. The bridge builds its own
    # diarizer when enabled; in browser mode we still run one so speaker identity
    # ([S1], echo/self filtering) is consistent, reaches the client as speakerId,
    # and feeds [speaker] history tags.
    app.state.Diarizer = None
    if not ENABLE_CALL_BRIDGE:
        try:
            from diarize import Diarizer
            app.state.Diarizer = Diarizer(self_reference_audio="reference_audio.wav", device="cuda")
        except Exception as e:  # noqa: BLE001
            logger.error("🖥️💥 Diarizer failed to load: %s", e)
            app.state.Diarizer = None

    # Discord-call bridge: B2 -> AEC -> STT, and TTS -> VAIO3. Disabled by default
    # so the browser is the audio source. When enabled, voice I/O moves to the
    # virtual buses so the agent never transcribes its own output.
    app.state.CallBridge = None
    if ENABLE_CALL_BRIDGE:
        try:
            from bridge import CallBridge
            from voices import voice_wav, DEFAULT_VOICE
            app.state.CallBridge = CallBridge(
                enable_aec=True,
                self_reference_audio=voice_wav(DEFAULT_VOICE),
            )
            app.state.CallBridge.start(app.state.AudioInputProcessor.transcriber)
            _sync_bridge_self_reference(app)
            _spm = app.state.SpeechPipelineManager
            _spm.people.voice_lookup = lambda label: (
                app.state.CallBridge.diarizer.centroid(label)
                if app.state.CallBridge and app.state.CallBridge.diarizer else None)
            _spm.on_backchannel = lambda pcm: (app.state.CallBridge.play_tts(pcm)
                                               if app.state.CallBridge else None)
            _spm.backchannels.preload(_spm.audio.current_voice)
        except Exception as e:  # noqa: BLE001
            logger.error("🖥️💥 CallBridge failed to start: %s", e)
            app.state.CallBridge = None

    # Restore the last chosen LLM backend (remote only if configured + reachable;
    # otherwise stay local). Background so a slow remote never delays boot.
    try:
        import user_settings as _us2
        if os.environ.get("ATLAS_LLM") == "remote" or (_us2.is_public() and _us2.llm_mode() == "cloud") or (
                os.path.exists(_LLM_MODE_FILE) and open(_LLM_MODE_FILE, encoding="utf-8").read().strip() == "remote"):
            async def _restore_remote():
                res = await asyncio.to_thread(_switch_llm_backend, "remote")
                if res.get("error"):
                    logger.warning("🌐🤖 Remote LLM not restored: %s (staying local)", res["error"])
            asyncio.create_task(_restore_remote())
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM backend restore skipped: %s", e)

    # Headless agent session: in bridge mode the agent must hear and answer the
    # call whether or not a UI is connected. UIs subscribe to its broadcast feed.
    app.state.headless = None
    if ENABLE_CALL_BRIDGE and app.state.CallBridge is not None:
        app.state.headless = _start_headless_session(app)

    yield

    if getattr(app.state, "headless", None):
        for t in app.state.headless["tasks"]:
            t.cancel()

    logger.info("🖥️⏹️ Server shutting down")
    if app.state.CallBridge is not None:
        app.state.CallBridge.stop()
    app.state.AudioInputProcessor.shutdown()

# --------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
from setup_api import router as _setup_router  # noqa: E402  first-run setup (public build)
app.include_router(_setup_router)
from memory_api import router as _memory_router  # noqa: E402  per-agent memory panel (owner only)
app.include_router(_memory_router)
from plugins_api import router as _plugins_router  # noqa: E402  plugins page (owner only)
app.include_router(_plugins_router)


@app.post("/api/setup/restart")
async def setup_restart(request: Request):
    """Apply a new model/audio choice: exit with code 75; the desktop shell
    relaunches the backend on that code (see desktop/main.js)."""
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    def _bye():
        time.sleep(0.6)
        os._exit(75)
    threading.Thread(target=_bye, daemon=True).start()
    return {"ok": True}

# Local-server hardening (security.py): Host allowlist (DNS rebinding), browser
# Origin check on WebSockets and state-changing requests (cross-site pages),
# security headers. Replaces the old wildcard CORS, which let any web
# page drive the API. Same-origin UI needs no CORS at all.
import security as _security  # noqa: E402
import user_settings as _us_sec  # noqa: E402
_LAN = (os.environ.get("ATLAS_HOST", "").strip() or ("127.0.0.1" if _us_sec.is_public() else "0.0.0.0")) not in ("127.0.0.1", "localhost", "::1")
app.add_middleware(_security.LocalGuard, lan=_LAN)

# Mount static files with no cache
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

# --------------------------------------------------------------------
# Telemetry (read-only; separate from /ws so a dashboard never aborts a gen)
# --------------------------------------------------------------------
import telemetry as _telemetry
TELEMETRY = _telemetry.install(app)

@app.get("/api/telemetry")
async def telemetry_snapshot():
    return await asyncio.to_thread(TELEMETRY.snapshot)

@app.websocket("/telemetry")
async def telemetry_ws(ws: WebSocket):
    """Pushes `fast` frames (~15 Hz: meters/spectrum/state), `snap` frames
    (~1 Hz: GPU/model/pipeline/room) and incremental events + log lines."""
    await ws.accept()
    ev_seq = max((e["seq"] for e in TELEMETRY.events_since(0)), default=0) - 60
    ln_seq = max((l["seq"] for l in TELEMETRY.lines_since(0)), default=0) - 80
    last_snap = 0.0
    try:
        while True:
            now = time.time()
            frames = [TELEMETRY.fast()]
            if now - last_snap >= 1.0:
                last_snap = now
                snap = await asyncio.to_thread(TELEMETRY.snapshot)
                snap["type"] = "snap"
                frames.append(snap)
            evs = TELEMETRY.events_since(ev_seq)
            if evs:
                ev_seq = evs[-1]["seq"]
                frames.append({"type": "events", "items": evs})
            lns = TELEMETRY.lines_since(ln_seq)
            if lns:
                ln_seq = lns[-1]["seq"]
                frames.append({"type": "lines", "items": lns[-120:]})
            for f in frames:
                await ws.send_text(json.dumps(f, default=str))
            await asyncio.sleep(1 / 15)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("telemetry ws closed: %s", e)

# --- Screen tool owner switch (screen.py). Owner-only: HTTP from the ATLAS window. ---
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _is_owner(request) -> bool:
    """Owner = a client on this machine. The server binds 0.0.0.0 (LAN/ZeroTier
    reachable), so owner controls must refuse remote clients."""
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in _LOOPBACK or host.startswith("127.")


@app.get("/api/screen")
async def screen_status():
    import screen as _scr
    return _scr.status()

@app.post("/api/screen")
async def screen_set(request: Request):
    import screen as _scr
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    body = await request.json()
    _scr.set_enabled(bool(body.get("enabled")))
    return _scr.status()

@app.get("/api/screen/last")
async def screen_last(request: Request):
    import screen as _scr
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    if not _scr.LAST_PATH.exists():
        return JSONResponse({"error": "no capture yet"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(str(_scr.LAST_PATH), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

# --- LLM backend selector (optional remote_llm.py; absent in public builds) ---
_LLM_MODE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "private", "llm_mode.txt")


def _llm_backend_status() -> dict:
    llm = getattr(getattr(app.state, "SpeechPipelineManager", None), "llm", None)
    try:
        import remote_llm as _rl
        cfg = _rl.load_config()
    except Exception:  # noqa: BLE001
        cfg = None
    rs = getattr(llm, "remote", None)
    return {"mode": "remote" if rs is not None else "local",
            "available": cfg is not None,
            "local_model": getattr(llm, "model", None),
            "remote_label": cfg["label"] if cfg else None,
            "remote": rs.status() if rs is not None else None,
            "busy": bool(getattr(app.state, "llm_switching", False))}


def _switch_llm_backend(mode: str) -> dict:
    """Blocking (run in a thread). remote: ping -> swap session -> unload local VRAM.
    local: swap back -> reload + warm the local model."""
    spm = app.state.SpeechPipelineManager
    llm = spm.llm
    app.state.llm_switching = True
    try:
        import remote_llm as _rl
        if mode == "remote":
            cfg = _rl.load_config()
            if cfg is None:
                return {"error": "no cloud provider configured (set one up in Settings)"}
            probe = _rl.RemoteSession(cfg, None)
            if not probe.ping(timeout=4.0):
                return {"error": "remote unreachable: " + probe.last_error}
            probe.close()
            if getattr(llm, "remote", None) is None:
                spm.abort_generation(wait_for_completion=True, timeout=5.0, reason="llm_backend:remote")
                _rl.activate(llm, cfg)
            freed = _rl.unload_local(llm.effective_ollama_url, llm.model)
            logger.info("🌐🤖 LLM backend -> REMOTE %s (local unloaded=%s)", cfg["label"], freed)
        else:
            if getattr(llm, "remote", None) is not None:
                spm.abort_generation(wait_for_completion=True, timeout=5.0, reason="llm_backend:local")
                _rl.deactivate(llm)
            logger.info("🖥️🤖 LLM backend -> LOCAL %s (warming)", llm.model)
            try:
                llm.prewarm()
            except Exception as e:  # noqa: BLE001
                logger.warning("local prewarm failed: %s", e)
        try:
            import user_settings as _us3
            if _us3.is_public():
                _us3.update({"llm": {"mode": "cloud" if mode == "remote" else "local"}})
        except Exception:  # noqa: BLE001
            pass
        try:
            os.makedirs(os.path.dirname(_LLM_MODE_FILE), exist_ok=True)
            with open(_LLM_MODE_FILE, "w", encoding="utf-8") as fh:
                fh.write(mode)
        except OSError:
            pass
        return {}
    except ImportError:
        return {"error": "remote_llm.py not installed"}
    finally:
        app.state.llm_switching = False


@app.get("/api/llm_backend")
async def llm_backend_get():
    return _llm_backend_status()


@app.post("/api/llm_backend")
async def llm_backend_set(request: Request):
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    mode = (await request.json()).get("mode")
    if mode not in ("local", "remote"):
        return JSONResponse({"error": "mode must be local|remote"}, status_code=400)
    if getattr(app.state, "llm_switching", False):
        return JSONResponse({"error": "switch already in progress"}, status_code=409)
    res = await asyncio.to_thread(_switch_llm_backend, mode)
    out = _llm_backend_status()
    if res.get("error"):
        out["error"] = res["error"]
        return JSONResponse(out, status_code=503)
    return out


# --- Agent creation (agents.py): template + LLM interpretation + registry ---
@app.get("/api/agents/template")
async def agents_template():
    import agents as A
    from voices import voice_names, PERSONA_VOICES
    used = set(PERSONA_VOICES.values())
    return {"fields": A.FIELDS, "voices": voice_names(), "used_voices": sorted(used),
            "palette": A.PALETTE, "example": A.assemble_prompt("Name", {
                "identity": "(who they are)", "who": "(personality)", "talk": "(speech style)",
                "bait": "(bait reaction)"})}


@app.post("/api/agents/draft")
async def agents_draft(request: Request):
    import agents as A
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    body = await request.json()
    name = (body.get("name") or "").strip()
    desc = (body.get("description") or "").strip()
    if len(name) < 2 or len(desc) < 10:
        return JSONResponse({"error": "Give the agent a name and a short description."}, status_code=400)
    mgr = app.state.SpeechPipelineManager
    # Through the pipeline's LLM wrapper, so it follows the active backend (local/remote).
    draft = await asyncio.to_thread(
        A.build_draft, name, desc, body.get("voice") or (__import__("voices").DEFAULT_VOICE or ""),
        mgr.llm_model, lambda n, d, m: A.draft_fields_llm(mgr.llm, n, d))
    return draft


@app.post("/api/agents/heart")
async def agents_heart(request: Request):
    """heart.md upload -> an editable draft (nothing saved until the user hits Create)."""
    import agents as A
    import heart_md
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    body = await request.json()
    h = heart_md.parse(body.get("text") or "")
    name = (h.get("name") or body.get("name") or "").strip()
    if not h.get("fields") and not h.get("prompt"):
        return JSONResponse({"error": "That file has no character in it."}, status_code=400)
    fields = {k: (h.get("fields") or {}).get(k, "") for k in A.TEXT_FIELDS}
    from voices import voice_names
    voice = h.get("voice") if h.get("voice") in voice_names() else (body.get("voice") or "")
    prompt = (h.get("prompt") or "").strip() or A.assemble_prompt(name or "Agent", fields)
    return {"id": A.slugify(name), "name": name, "voice": voice, "fields": fields,
            "role": A.normalize_role(h.get("role") or "") or "custom · agent",
            "interests": A.parse_interests(h.get("interests")),
            "talkativeness": A.clamp_talk(h.get("talkativeness")),
            "accent": A.accent_for(name), "prompt": prompt,
            "prompt_locked": bool(h.get("prompt")), "source": "heart"}


@app.get("/api/agents/{aid}/heart.md")
async def agents_heart_export(aid: str, request: Request):
    """Export a user-made agent as heart.md. Dev-pack agents are never exported."""
    import agent_registry as _r
    import heart_md
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    a = _r.agent(aid)
    if not a or a.get("dev") or not a.get("custom"):
        return JSONResponse({"error": "not exportable"}, status_code=404)
    text = heart_md.dump(a.get("name") or aid, a.get("fields") or {}, a.get("interests") or (),
                         a.get("role") or "", a.get("talkativeness"), a.get("voice") or "",
                         None if a.get("fields") else _r.prompt_text(aid))
    return Response(text, media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="{aid}.heart.md"'})


@app.post("/api/voices/import")
async def voices_import(request: Request, label: str = "", filename: str = "", transcript: str = ""):
    """Raw audio body -> a new cloneable voice under voices/user/ (owner only)."""
    import voice_import as VI
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    try:
        declared = int(request.headers.get("content-length") or 0)
    except ValueError:
        declared = 0
    if declared > VI.MAX_UPLOAD:                     # refuse before buffering it
        return JSONResponse({"error": "That file is too large to import."}, status_code=413)
    data = await request.body()
    from voices import voice_names
    try:
        res = await asyncio.to_thread(VI.import_voice, data, filename, label, transcript, set(voice_names()))
    except VI.VoiceImportError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    app.state.SpeechPipelineManager.refresh_registry()
    return {**res, "voices": voice_names()}


@app.post("/api/agents")
async def agents_create(request: Request):
    import agents as A
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    from voices import voice_names
    from speech_pipeline_manager import PERSONAS
    body = await request.json()
    if isinstance(body.get("fields"), dict) and not (body.get("prompt") or "").strip():
        body["prompt"] = A.assemble_prompt(body.get("name", ""), body["fields"])
    err = A.validate(body, voice_names(), set(PERSONAS))
    if err:
        return JSONResponse({"error": err}, status_code=400)
    rec = await asyncio.to_thread(A.save_agent, body)
    app.state.SpeechPipelineManager.register_agent(rec, body["prompt"])
    return {"agent": rec, **app.state.SpeechPipelineManager.personas_info()}


@app.post("/api/agents/preview")
async def agents_preview(request: Request):
    import agents as A
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    body = await request.json()
    return {"prompt": A.assemble_prompt(body.get("name", ""), body.get("fields") or {}),
            "id": A.slugify(body.get("name", "")), "accent": A.accent_for(body.get("name", ""))}


@app.delete("/api/agents/{aid}")
async def agents_delete(aid: str, request: Request):
    import agents as A
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    mgr = app.state.SpeechPipelineManager
    if not mgr.remove_agent(aid):
        return JSONResponse({"error": "Can't delete a built-in or the active agent."}, status_code=400)
    if getattr(mgr, "hgmem", None) is not None:
        mgr.hgmem.drop_agent(aid)
    await asyncio.to_thread(A.delete_agent, aid)
    return mgr.personas_info()


# --- Owner controls (owner_controls.py): mute, quiet mode, talkativeness, muted speakers ---
def _owner_status() -> dict:
    import owner_controls as _owner
    out = _owner.snapshot()
    mgr = getattr(app.state, "SpeechPipelineManager", None)
    try:
        out["active"] = mgr.agent.id
        out["active_talkativeness"] = round(mgr.agent.profile.talkativeness, 2)
    except Exception:  # noqa: BLE001
        pass
    return out


@app.get("/api/owner")
async def owner_get():
    return _owner_status()


@app.post("/api/owner")
async def owner_set(request: Request):
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    import owner_controls as _owner
    body = await request.json()
    mgr = getattr(app.state, "SpeechPipelineManager", None)
    if "mute" in body:
        on = body["mute"] if body["mute"] != "toggle" else not _owner.STATE["mute"]
        _owner.set_mute(on)
        logger.info(f"🖥️🔇 owner mute {'ON' if on else 'off'}")
        if on and mgr is not None:
            br = getattr(app.state, "CallBridge", None)
            try:
                if br is not None:
                    br.flush_tts()
                if mgr.running_generation is not None:
                    mgr.abort_generation(wait_for_completion=False, reason="owner mute")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"owner mute cut failed: {e}")
    if "quiet" in body:
        on = body["quiet"] if body["quiet"] != "toggle" else not _owner.quiet_active()
        _owner.set_quiet(bool(on), float(body.get("quiet_min", _owner.QUIET_DEFAULT_MIN)))
        logger.info(f"🖥️🤫 owner quiet mode {'ON' if on else 'off'}")
    b = body.get("background")
    if isinstance(b, dict):
        aid = str(b.get("agent") or (mgr.agent.id if mgr is not None else ""))
        on = b.get("on", "toggle")
        on = (not _owner.background_on(aid)) if on == "toggle" else bool(on)
        _owner.set_background(aid, on)
        logger.info(f"🖥️🌙 background mode {'ON' if on else 'off'} for {aid}")
    for key, on in (("mute_speaker", True), ("unmute_speaker", False)):
        if body.get(key):
            _owner.set_speaker_muted(str(body[key]), on)
            logger.info(f"🖥️🙊 owner {'muted' if on else 'unmuted'} speaker {body[key]}")
    t = body.get("talkativeness")
    if isinstance(t, dict) and t.get("agent") and t.get("value") is not None:
        _owner.set_talkativeness(str(t["agent"]), float(t["value"]))
        try:
            mgr.agents.refresh_profile(str(t["agent"]))
        except Exception:  # noqa: BLE001
            pass
    return _owner_status()


@app.get("/api/personas")
async def personas_list():
    return app.state.SpeechPipelineManager.personas_info()


@app.get("/favicon.ico")
async def favicon():
    """
    Serves the favicon.ico file.

    Returns:
        A FileResponse containing the favicon.
    """
    return FileResponse("static/favicon.ico")

@app.get("/")
async def get_index() -> HTMLResponse:
    """
    Serves the main index.html page.

    Reads the content of static/index.html and returns it as an HTML response.

    Returns:
        An HTMLResponse containing the content of index.html.
    """
    page = "static/index.html" if os.path.exists("static/index.html") else "static/atlas.html"
    with open(page, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------
def parse_json_message(text: str) -> dict:
    """
    Safely parses a JSON string into a dictionary.

    Logs a warning if the JSON is invalid and returns an empty dictionary.

    Args:
        text: The JSON string to parse.

    Returns:
        A dictionary representing the parsed JSON, or an empty dictionary on error.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️⚠️ Ignoring client message with invalid JSON")
        return {}

def format_timestamp_ns(timestamp_ns: int) -> str:
    """
    Formats a nanosecond timestamp into a human-readable HH:MM:SS.fff string.

    Args:
        timestamp_ns: The timestamp in nanoseconds since the epoch.

    Returns:
        A string formatted as hours:minutes:seconds.milliseconds.
    """
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# WebSocket data processing
# --------------------------------------------------------------------

def _audio_devices_info(app: FastAPI) -> dict:
    """Current device list + resolved in/out indices for the UI selector."""
    from audio_io import list_audio_devices, resolve_device
    info = {"devices": list_audio_devices(), "current": {"input": None, "output": None}}
    bridge = app.state.CallBridge
    if bridge is not None:
        try:
            info["current"]["input"], _ = resolve_device("input", bridge.input_device)
            info["current"]["output"], _ = resolve_device("output", bridge.output_device)
        except Exception:  # noqa: BLE001
            pass
    try:  # demo/sim calls run on a virtual device: say so instead of naming real hardware
        import demo_call
        if demo_call.enabled():
            info["virtual"] = "Virtual call (demo)"
    except Exception:  # noqa: BLE001
        pass
    return {"type": "devices", **info}


def _sync_bridge_self_reference(app: FastAPI) -> None:
    """Point the bridge diarizer's self-profile at the current active voice."""
    bridge = app.state.CallBridge
    if bridge is None:
        return
    from voices import voice_wav
    voice = app.state.SpeechPipelineManager.personas_info()["current"]["voice"]
    try:
        bridge.update_self_reference(voice_wav(voice))
        logger.info(f"🖥️🎙️ diarizer self-reference -> {voice}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🖥️⚠️ self-reference update failed: {e}")


async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Receives messages via WebSocket, processes audio and text messages.

    Handles binary audio chunks, extracting metadata (timestamp, flags) and
    putting the audio PCM data with metadata into the `incoming_chunks` queue.
    Applies back-pressure if the queue is full.
    Parses text messages (assumed JSON) and triggers actions based on message type
    (e.g., updates client TTS state via `callbacks`, clears history, sets speed).

    Args:
        ws: The WebSocket connection instance.
        app: The FastAPI application instance (for accessing global state if needed).
        incoming_chunks: An asyncio queue to put processed audio metadata dictionaries into.
        callbacks: The TranscriptionCallbacks instance for this connection to manage state.
    """
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                if ENABLE_CALL_BRIDGE:
                    continue  # bridge feeds B2 directly; ignore browser mic PCM
                raw = msg["bytes"]

                # Ensure we have at least an 8‑byte header: 4 bytes timestamp_ms + 4 bytes flags
                if len(raw) < 8:
                    logger.warning("🖥️⚠️ Received packet too short for 8‑byte header.")
                    continue

                # Unpack big‑endian uint32 timestamp (ms) and uint32 flags
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # Build metadata using fixed fields
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                }

                # Record server receive time
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(server_ns)

                # The rest of the payload is raw PCM bytes
                metadata["pcm"] = raw[8:]

                # Check queue size before putting data
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # Now put only the metadata dict (containing PCM audio) into the processing queue.
                    await incoming_chunks.put(metadata)
                else:
                    # Queue is full, drop the chunk and log a warning
                    logger.warning(
                        f"🖥️⚠️ Audio queue full ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk. Possible lag."
                    )

            elif "text" in msg and msg["text"]:
                # Text-based message: parse JSON
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                logger.info(Colors.apply(f"🖥️📥 ←←Client: {data}").orange)


                if msg_type == "tts_start":
                    logger.info("🖥️ℹ️ Received tts_start from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    logger.info("🖥️ℹ️ Received tts_stop from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = False
                # Add to the handleJSONMessage function in server.py
                elif msg_type == "clear_history":
                    logger.info("🖥️ℹ️ Received clear_history from client.")
                    app.state.SpeechPipelineManager.reset()
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    app.state.turn_speed = int(speed_value)
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        logger.info(f"🖥️⚙️ Updated turn detection settings to factor: {speed_factor:.2f}")
                elif msg_type == "set_persona":
                    persona = data.get("persona", "")
                    ok = app.state.SpeechPipelineManager.set_persona(persona)
                    logger.info(f"🖥️🎭 Persona switch request '{persona}' -> {'ok' if ok else 'unknown'}")
                    _sync_bridge_self_reference(app)
                    callbacks.message_queue.put_nowait({
                        "type": "personas",
                        **app.state.SpeechPipelineManager.personas_info(),
                    })
                elif msg_type == "set_voice":
                    voice = data.get("voice", "")
                    ok = app.state.SpeechPipelineManager.set_voice(voice)
                    logger.info(f"🖥️🔊 Voice switch request '{voice}' -> {'ok' if ok else 'failed'}")
                    _sync_bridge_self_reference(app)
                    callbacks.message_queue.put_nowait({
                        "type": "personas",
                        **app.state.SpeechPipelineManager.personas_info(),
                    })
                elif msg_type == "list_personas":
                    callbacks.message_queue.put_nowait({
                        "type": "personas",
                        **app.state.SpeechPipelineManager.personas_info(),
                    })
                elif msg_type == "list_devices":
                    callbacks.message_queue.put_nowait(_audio_devices_info(app))
                elif msg_type == "set_audio_devices":
                    inp, out = data.get("input"), data.get("output")
                    bridge = app.state.CallBridge
                    if bridge is not None:
                        try:
                            bridge.reconfigure(input_device=inp, output_device=out)
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"🖥️⚠️ audio reconfigure failed: {e}")
                    else:
                        logger.warning("🖥️⚠️ set_audio_devices ignored: no CallBridge (browser mode)")
                    callbacks.message_queue.put_nowait(_audio_devices_info(app))


    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in process_incoming_data: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in process_incoming_data: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in process_incoming_data: {repr(e)}")

async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue, quiet: bool = False) -> None:
    """
    Continuously sends text messages from a queue to the client via WebSocket.

    Waits for messages on the `message_queue`, formats them as JSON, and sends
    them to the connected WebSocket client. Logs non-TTS messages.

    Args:
        ws: The WebSocket connection instance.
        message_queue: An asyncio queue yielding dictionaries to be sent as JSON.
    """
    try:
        while True:
            await asyncio.sleep(0.001) # Yield control
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type != "tts_chunk" and not quiet:
                logger.info(Colors.apply(f"🖥️📤 →→Client: {data}").orange)
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_text_messages: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_text_messages: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_text_messages: {repr(e)}")

async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    """
    Resets the microphone interruption flag after a delay (async version).

    Waits for 1 second, then checks if the AudioInputProcessor is still marked
    as interrupted. If so, resets the flag on both the processor and the
    connection-specific callbacks instance.

    Args:
        app: The FastAPI application instance (to access AudioInputProcessor).
        callbacks: The TranscriptionCallbacks instance for the connection.
    """
    await asyncio.sleep(1)
    # Check the AudioInputProcessor's own interrupted state
    if app.state.AudioInputProcessor.interrupted:
        logger.info(f"{Colors.apply('🖥️🎙️ ▶️ Microphone continued (async reset)').cyan}")
        app.state.AudioInputProcessor.interrupted = False
        # Reset connection-specific interruption time via callbacks
        callbacks.interruption_time = 0
        logger.info(Colors.apply("🖥️🎙️ interruption flag reset after TTS chunk (async)").cyan)

async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Continuously sends TTS audio chunks from the SpeechPipelineManager to the client.

    Monitors the state of the current speech generation (if any) and the client
    connection (via `callbacks`). Retrieves audio chunks from the active generation's
    queue, upsamples/encodes them, and puts them onto the outgoing `message_queue`
    for the client. Handles the end-of-generation logic and state resets.

    Args:
        app: The FastAPI application instance (to access global components).
        message_queue: An asyncio queue to put outgoing TTS chunk messages onto.
        callbacks: The TranscriptionCallbacks instance managing this connection's state.
    """
    try:
        logger.info("🖥️🔊 Starting TTS chunk sender")
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None
        _sender_err_t = 0.0

        while True:
          try:
            await asyncio.sleep(0.001) # Yield control

            # Use connection-specific interruption_time via callbacks
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time and time.time() - callbacks.interruption_time > 2.0:
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0 # Reset via callbacks
                logger.info(Colors.apply("🖥️🎙️ interruption flag reset after 2 seconds").cyan)

            _g0 = app.state.SpeechPipelineManager.running_generation
            is_tts_finished = bool(app.state.SpeechPipelineManager.is_valid_gen() and _g0 is not None and _g0.audio_quick_finished)

            def log_status():
                nonlocal prev_status
                last_quick_answer_chunk_decayed = (
                    last_quick_answer_chunk
                    and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                    and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                )

                curr_status = (
                    # Access connection-specific state via callbacks
                    int(callbacks.tts_to_client),
                    int(callbacks.tts_client_playing),
                    int(callbacks.tts_chunk_sent),
                    1, # Placeholder?
                    int(callbacks.is_hot), # from callbacks
                    int(callbacks.synthesis_started), # from callbacks
                    int(app.state.SpeechPipelineManager.running_generation is not None), # Global manager state
                    int(app.state.SpeechPipelineManager.is_valid_gen()), # Global manager state
                    int(is_tts_finished), # Calculated local variable
                    int(app.state.AudioInputProcessor.interrupted) # Input processor state
                )

                if curr_status != prev_status:
                    status = Colors.apply("🖥️🚦 State ").red
                    logger.info(
                        f"{status} ToClient {curr_status[0]}, "
                        f"ttsClientON {curr_status[1]}, " # Renamed slightly for clarity
                        f"ChunkSent {curr_status[2]}, "
                        f"hot {curr_status[4]}, synth {curr_status[5]}"
                        f" gen {curr_status[6]}"
                        f" valid {curr_status[7]}"
                        f" tts_q_fin {curr_status[8]}"
                        f" mic_inter {curr_status[9]}"
                    )
                    prev_status = curr_status

            # Use connection-specific state via callbacks
            if not callbacks.tts_to_client:
                await asyncio.sleep(0.001)
                log_status()
                continue

            _g = app.state.SpeechPipelineManager.running_generation
            if not _g:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if _g.abortion_started:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not _g.audio_quick_finished:
                _g.tts_quick_allowed_event.set()

            # A silent response has no first audio chunk to wait for.
            gen = _g
            if gen.response_held and gen.llm_finished and callbacks.user_history_committed:
                logger.info("Model HOLD retired without audio or assistant history")
                try:
                    app.state.SpeechPipelineManager.threads.retire_active()
                except Exception:  # noqa: BLE001
                    pass
                app.state.SpeechPipelineManager.running_generation = None
                callbacks.reset_state()
                continue

            if not _g.quick_answer_first_chunk_ready:
                await asyncio.sleep(0.001)
                log_status()
                continue

            chunk = None
            try:
                chunk = _g.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                final_expected = _g.quick_answer_provided
                audio_final_finished = _g.audio_final_finished

                if not final_expected or audio_final_finished:
                    logger.info("🖥️🏁 Sending of TTS chunks and 'user request/assistant answer' cycle finished.")
                    if app.state.CallBridge is not None:
                        try:
                            app.state.CallBridge.end_reply()
                            _ps = app.state.CallBridge.playout_stats()
                            if _ps.get("last_reply_underruns"):
                                logger.warning("🖥️🔇 Playout ran dry %d time(s) last reply (total %d, %.0f ms silent, prebuffer now %d ms)",
                                               _ps["last_reply_underruns"], _ps["underruns"], _ps["silence_ms"], _ps["prebuffer_ms"])
                        except Exception:  # noqa: BLE001
                            pass
                    callbacks.send_final_assistant_answer() # Callbacks method
                    try:
                        # Use the snapshot: re-reading running_generation here could
                        # return None after a barge-in and break the line below.
                        app.state.SpeechPipelineManager.threads.answered(
                            getattr(getattr(_g, "decision", None), "target", None))
                    except Exception:  # noqa: BLE001
                        pass

                    assistant_answer = _g.quick_answer + _g.final_answer                    
                    app.state.SpeechPipelineManager.running_generation = None

                    callbacks.tts_chunk_sent = False # Reset via callbacks
                    callbacks.reset_state() # Reset connection state via callbacks

                await asyncio.sleep(0.001)
                log_status()
                continue

            # Route synthesized audio to VAIO3 (into the call) in addition to the
            # browser monitor. AEC uses this same audio as its reference.
            if app.state.CallBridge is not None:
                try:
                    app.state.CallBridge.play_tts(chunk)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"🖥️⚠️ CallBridge.play_tts error: {e}")

            if not ENABLE_CALL_BRIDGE:
                # Bridge mode routes TTS to VAIO3 only (play_tts above). Sending
                # tts_chunk to the client as well plays a SECOND copy on the local
                # speakers — double audio.
                base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
                message_queue.put_nowait({
                    "type": "tts_chunk",
                    "content": base64_chunk
                })
            last_chunk_sent = time.time()

            # Use connection-specific state via callbacks
            if not callbacks.tts_chunk_sent:
                # Use the async helper function instead of a thread
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))

            callbacks.tts_chunk_sent = True # Set via callbacks
          except (asyncio.CancelledError, WebSocketDisconnect):
            raise
          except Exception as e:  # noqa: BLE001
            # 10-04 live call: a barge-in cleared running_generation mid-step, the
            # AttributeError ended this task, and every later reply was synthesized
            # but never played (Fae silent for 12+ min). Log loudly, keep sending.
            _now = time.time()
            if _now - _sender_err_t > 5.0:
                _sender_err_t = _now
                logger.exception(f"🖥️💥 send_tts_chunks step failed (sender kept alive): {e!r}")
            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        pass # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_tts_chunks: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_tts_chunks: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_tts_chunks: {repr(e)}")


# --------------------------------------------------------------------
# Callback class to handle transcription events
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    """
    Manages state and callbacks for a single WebSocket connection's transcription lifecycle.

    This class holds connection-specific state flags (like TTS status, user interruption)
    and implements callback methods triggered by the `AudioInputProcessor` and
    `SpeechPipelineManager`. It sends messages back to the client via the provided
    `message_queue` and manages interaction logic like interruptions and final answer delivery.
    It also includes a threaded worker to handle abort checks based on partial transcription.
    """
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        """
        Initializes the TranscriptionCallbacks instance for a WebSocket connection.

        Args:
            app: The FastAPI application instance (to access global components).
            message_queue: An asyncio queue for sending messages back to the client.
        """
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        self.abort_text = ""
        self.last_abort_text = ""

        # Initialize connection-specific state flags here
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0

        # These were already effectively instance variables or reset logic existed
        self.silence_active: bool = True
        self.is_hot: bool = False
        self.user_finished_turn: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer: str = ""
        self.is_processing_potential: bool = False
        self.is_processing_final: bool = False
        self.last_inferred_transcription: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = "" # Added for clarity

        self.reset_state() # Call reset to ensure consistency

        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(target=self._abort_worker, name="AbortWorker", daemon=True)
        self.abort_worker_thread.start()


    def reset_state(self):
        """Resets connection-specific state flags and variables to their initial values."""
        # Reset all connection-specific state flags
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        # Don't reset tts_client_playing here, it reflects client state reports
        self.interruption_time = 0.0

        # Reset other state variables
        self.silence_active = True
        self.is_hot = False
        self.user_finished_turn = False
        self.user_history_committed = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        self.is_processing_final = False
        self.last_inferred_transcription = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""

        # Keep the abort call related to the audio processor/pipeline manager
        self.app.state.AudioInputProcessor.abort_generation()


    def _abort_worker(self):
        """Background thread worker to check for abort conditions based on partial text."""
        while True:
            was_set = self.abort_request_event.wait(timeout=0.1) # Check every 100ms
            if was_set:
                self.abort_request_event.clear()
                # Only trigger abort check if the text actually changed
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    logger.debug(f"🖥️🧠 Abort check triggered by partial: '{self.abort_text}'")
                    mgr = self.app.state.SpeechPipelineManager
                    spk = getattr(self, "live_speaker", None)
                    act = getattr(mgr, "threads", None) and mgr.threads.active
                    if act is not None and spk and spk != act.speaker and self.abort_text:
                        try:
                            if not mgr.threads.should_preempt(
                                    mgr.threads.note(f"[{spk}] {self.abort_text}", mgr.floor.active_partner())):
                                continue
                        except Exception:  # noqa: BLE001
                            pass
                    if self._backchannel_over_reply(mgr, self.abort_text):
                        continue
                    mgr.check_abort(self.abort_text, False, "on_partial")

    def _yield_decision(self, txt: str) -> tuple[bool, str]:
        """Does speech heard while the agent talks take the floor? (interrupts.py)"""
        mgr = self.app.state.SpeechPipelineManager
        names = tuple(getattr(getattr(mgr, "dynamics", None), "agent_names", ()) or ())
        try:
            from interrupts import should_yield
            m = re.match(r'^\s*\[(S\d+)\]', txt or "")
            speaker = m[1] if m else self._live_speaker()
            if speaker == "self":
                return False, "own echo"
            partner = None
            try:
                partner = mgr.floor.active_partner()
            except Exception:  # noqa: BLE001
                pass
            bridge = getattr(self.app.state, "CallBridge", None)
            left = bridge.pending_seconds() if bridge is not None and hasattr(bridge, "pending_seconds") else None
            talk = 0.35
            try:
                talk = float(mgr._runtime().profile.talkativeness)
            except Exception:  # noqa: BLE001
                pass
            return should_yield(txt, names, speaker=speaker, partner=partner,
                                agent_left_s=left, talkativeness=talk)
        except Exception as e:  # noqa: BLE001 - fall back to the old rule
            logger.warning(f"yield decision failed, using real_interruption: {e}")
            from floor import real_interruption
            return bool(real_interruption(txt, names)), "fallback"

    def _record_cutoff(self, by: str | None = None) -> None:
        """Remember what the agent didn't get to say (for the resume note/cue)."""
        self._cut_said = None
        try:
            mgr = self.app.state.SpeechPipelineManager
            gen = mgr.running_generation
            bridge = getattr(self.app.state, "CallBridge", None)
            if gen is None or bridge is None or not hasattr(bridge, "reply_progress"):
                return
            text = (getattr(gen, "quick_answer", "") or "") + (getattr(gen, "final_answer", "") or "")
            played, pushed = bridge.reply_progress()
            from interrupts import make_cutoff
            cut = make_cutoff(text, played, pushed, bool(getattr(gen, "audio_final_finished", False)),
                              target=getattr(getattr(gen, "decision", None), "target", "") or "", by=by)
            mgr.cutoff = cut
            if cut is not None:
                self._cut_said = cut.said
                logger.info(f"🖥️✂️ cut off after {played:.1f}s; unsaid: '{cut.unsaid[:60]}'")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"cutoff record failed: {e}")

    def _backchannel_over_reply(self, mgr, txt: str) -> bool:
        """Live 09-29 17:47: a listener's "Yeah." / "Oh." aborted Max mid-reply
        (similarity 0.00 -> abort). Once a reply is committed (text decided or
        audio playing), only a REAL interruption may cut it; backchannels can't.
        A partial that grows into real words re-checks on the next change."""
        gen = getattr(mgr, "running_generation", None)
        bridge = getattr(self.app.state, "CallBridge", None)
        speaking = False
        try:
            speaking = bool(bridge is not None and bridge.is_speaking())
        except Exception:  # noqa: BLE001
            pass
        committed = speaking or (gen is not None and bool(getattr(gen, "quick_answer", "")))
        if not committed or not txt:
            return False
        yes, why = self._yield_decision(txt)
        if yes:
            if speaking:
                self._record_cutoff(why)
            return False
        logger.info(f"🖥️👂 talking over ({why}): '{txt[:40]}'")
        return True

    def on_partial(self, txt: str):
        """
        Callback invoked when a partial transcription result is available.

        Updates internal state, sends the partial result to the client,
        and signals the abort worker thread to check for potential interruptions.

        Args:
            txt: The partial transcription text.
        """
        self.final_assistant_answer_sent = False # New user speech invalidates previous final answer sending state
        self.final_transcription = "" # Clear final transcription as this is partial
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt # Update text used for abort check
        self.abort_request_event.set() # Signal the abort worker
        self._check_barge_in(txt)
        # Memory recall prefetch: embed + search while they're still talking, so the
        # turn's recall is ready (~0 ms) when the sentence lands. Non-blocking.
        try:
            _m = self.app.state.SpeechPipelineManager
            if getattr(_m, "hgmem", None) is not None:
                _m.hgmem.prefetch(getattr(_m, "current_persona", ""), txt)
        except Exception:  # noqa: BLE001 - prefetch must never touch the turn
            pass

    def safe_abort_running_syntheses(self, reason: str):
        """Placeholder for safely aborting syntheses (currently does nothing)."""
        # TODO: Implement actual abort logic if needed, potentially interacting with SpeechPipelineManager
        pass

    def on_tts_allowed_to_synthesize(self):
        """Callback invoked when the system determines TTS synthesis can proceed."""
        # Access global manager state
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen and not gen.abortion_started and not gen.tts_quick_allowed_event.is_set():
            # Fires every monitor tick; log/set only on the transition (was ~200 lines/s).
            logger.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED').blue}")
            gen.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        """
        Callback invoked when a potentially complete sentence is detected by the STT.

        Triggers the preparation of a speech generation based on this potential sentence.

        Args:
            txt: The potential sentence text.
        """
        logger.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        # Access global manager state
        mgr = self.app.state.SpeechPipelineManager
        # Deterministic hard-veto: an explicit silence command ("stop talking",
        # "shut up", "be quiet", "hold on") must never reach the LLM — Model
        # otherwise acknowledges it aloud ("okay I'll be quiet"), defeating the
        # silence. The patterns are specific imperatives, so no false-positive risk;
        # the LLM still decides every ambiguous/contextual turn.
        # pre_llm_veto also holds turns vocatively addressed to ANOTHER named
        # person ("Can you pass the salt, Jordan?"). Shared with the sims so the
        # simulated gate is the production gate.
        # Label the in-flight utterance with the LIVE speaker (read-only ECAPA on
        # the last 4s). Without this every Discord turn reached the model as
        # unlabeled "user", so the conversation lock (floor.py) never engaged.
        speaker = self._live_speaker()
        if speaker == "self":
            logger.info(f"🖥️🙉 skip (own voice): '{txt[:40]}'")
            return
        if self._text_echo(txt):
            logger.info(f"🖥️🙉 skip (own words echoed): '{txt[:40]}'")
            return
        if speaker and txt and not txt.lstrip().startswith("["):
            txt = f"[{speaker}] {txt}"
        self.live_speaker = speaker
        from conversation_dynamics import pre_llm_veto
        veto = mgr.floor.turn_gate(txt, speaker, tuple(mgr.dynamics.agent_names)) if txt else None
        if veto:
            logger.info(f"🖥️🤫 HOLD ({veto}): '{txt[:60]}'")
            # An earlier, shorter partial may already have started a generation
            # ("Can you pass the salt" before ", Jordan?"). Retire it.
            if mgr.running_generation is not None:
                mgr.abort_generation(wait_for_completion=False, reason=f"pre_llm_veto: {veto}")
            return
        # Fast-VAD guards (floor.py, pure text): laughter/filler never costs an
        # LLM call; a clause cut mid-breath waits for more speech. A held
        # fragment is remembered so on_before_final can still answer the turn --
        # a guard may delay a turn, never silently drop it.
        from floor import filler_only, incomplete_fragment, status_only
        if not txt.strip() or filler_only(txt):
            logger.info(f"🖥️😂 skip LLM (filler): '{txt[:40]}'")
            return
        try:
            _names = tuple(getattr(getattr(mgr, "threads", None), "names", ()) or ())
        except Exception:  # noqa: BLE001
            _names = ()
        if status_only(txt, _names):
            logger.info(f"🖥️⏸️ skip LLM (status update): '{txt[:40]}'")
            return
        if incomplete_fragment(txt):
            logger.info(f"🖥️⏳ wait (fragment): '{txt[-40:]}'")
            self.held_fragment = txt
            return
        self.held_fragment = ""
        bridge = getattr(self.app.state, "CallBridge", None)
        if bridge is not None and bridge.is_speaking():
            yes, why = self._yield_decision(txt)
            if not yes:
                logger.info(f"🖥️🗣️ keep talking ({why}); not starting a turn for '{txt[:50]}'")
                return
            self._record_cutoff(why)
        if self._monologue_hold(txt, speaker):
            return
        # Thread arbiter (threads.py): don't chase every new voice. A line from a
        # different speaker only preempts the in-flight thread if it clearly
        # outranks it; when a turn fires, answer the best open thread.
        try:
            thr = mgr.threads.note(txt, mgr.floor.active_partner())
            running = mgr.running_generation
            if (running is not None and not running.abortion_started
                    and not mgr.threads.should_preempt(thr)):
                logger.info(f"🖥️🧵 keep thread {mgr.threads.active.speaker if mgr.threads.active else '?'}; "
                            f"not preempted by {thr.speaker}: '{txt[:50]}'")
                return
            chosen = mgr.threads.choose(thr)
            if chosen is not thr:
                logger.info(f"🖥️🧵 answering {chosen.speaker}'s open thread instead of {thr.speaker}")
            mgr.threads.start(chosen)
            txt = chosen.text
        except Exception as e:  # noqa: BLE001 - arbiter must never block a turn
            logger.warning(f"thread arbiter failed, answering latest: {e}")
        # Model decides SPEAK/HOLD inside the normal generation.
        mgr.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """
        Callback invoked when a potential *final* transcription is detected (hot state).

        Logs the potential final transcription.

        Args:
            txt: The potential final transcription text.
        """
        logger.info(f"{Colors.apply('🖥️🧠 HOT: ').magenta}{txt}")

    def on_potential_abort(self):
        """Callback invoked if the STT detects a potential need to abort based on user speech."""
        # Placeholder: Currently logs nothing, could trigger abort logic.
        pass

    def _agent_audible(self, audio) -> bool | None:
        """Bridge mode only: was the agent playing during this utterance?"""
        bridge = getattr(self.app.state, "CallBridge", None)
        fn = getattr(bridge, "agent_audible_during", None) if bridge is not None else None
        if fn is None or audio is None:
            return None
        try:
            return fn(len(audio) / 16000.0)
        except Exception:  # noqa: BLE001
            return None

    def _text_echo(self, txt: str) -> bool:
        """Backstop for echo the voiceprint missed: heard words == our own words."""
        guard = getattr(self.app.state, "EchoGuard", None)
        if guard is None or not txt:
            return False
        try:
            # Echo needs the agent's audio in the air. A person quoting the agent
            # after it finished ("wait, you said the village ever had?") is NOT echo.
            bridge = getattr(self.app.state, "CallBridge", None)
            if bridge is not None:
                est_s = max(1.0, len(txt.split()) / 2.5)  # ~2.5 words/s speech
                aud = bridge.agent_audible_during(est_s)
                if aud is False:
                    return False
            gen = self.app.state.SpeechPipelineManager.running_generation
            in_flight = ((gen.quick_answer or "") + " " + (gen.final_answer or "")) if gen is not None else ""
            return guard.is_echo(txt, in_flight=in_flight.strip())
        except Exception as e:  # noqa: BLE001 - never block a turn
            logger.debug(f"text echo check failed: {e}")
            return False

    def _live_speaker(self) -> str | None:
        """Who is talking in the utterance still in progress (None = unknown)."""
        diarizer = getattr(self.app.state, "Diarizer", None)
        if diarizer is None:
            bridge = getattr(self.app.state, "CallBridge", None)
            diarizer = getattr(bridge, "diarizer", None) if bridge is not None else None
        if diarizer is None or not hasattr(diarizer, "identify"):
            return None
        try:
            audio = self.app.state.AudioInputProcessor.transcriber.live_frames()
            if audio is None:
                return None
            aud = self._agent_audible(audio)
            kw = {} if aud is None else {"agent_audible": aud}
            return diarizer.identify(audio, 16000, **kw)
        except Exception as e:  # noqa: BLE001 - labeling must never block a turn
            logger.debug(f"live speaker id failed: {e}")
            return None

    def _adopt_final_speaker(self, speaker: str) -> None:
        """Generation started from an unlabeled partial; the finished turn now has a
        label. Point the in-flight decision at the real speaker."""
        if not speaker or speaker == "self" or not re.fullmatch(r"S\d+", speaker):
            return
        try:
            gen = self.app.state.SpeechPipelineManager.running_generation
            dec = getattr(gen, "decision", None) if gen is not None else None
            if dec is None or getattr(dec, "expected_target", "") not in ("", "user"):
                return
            dec.expected_target = speaker
            if getattr(dec, "action", "") == "SPEAK" and dec.target != speaker:
                logger.info(f"🖥️🎯 target {dec.target} -> {speaker} (final speaker label)")
                dec.target = speaker
        except Exception as e:  # noqa: BLE001 - labeling must never block a turn
            logger.debug(f"adopt final speaker failed: {e}")

    def _pace_turn(self, speaker, text, audio, mgr) -> None:
        """Pacing accounting + per-turn record (owner 10-06: talk less, match the
        room's pacing; the record doubles as turn-taking training data)."""
        import pacing as _pacing
        feats = _pacing.audio_features(audio) if audio is not None else {}
        rec = mgr.floor.pacing_turn(speaker, text, feats)
        if not rec:
            return
        ov = self._agent_audible(audio)
        if ov is not None:
            rec["agent_audible"] = ov
        _rec_event("pace", **rec)

    def _tag_speaker(self, audio) -> str | None:
        """Diarize one turn using the app-level Diarizer (browser or bridge)."""
        diarizer = getattr(self.app.state, "Diarizer", None)
        if diarizer is None:
            bridge = getattr(self.app.state, "CallBridge", None)
            if bridge is not None:
                diarizer = getattr(bridge, "diarizer", None)
        if diarizer is None:
            return None
        try:
            # audio arrives as a float32 mono array at 16 kHz (see transcribe.py)
            aud = self._agent_audible(audio)
            kw = {} if aud is None else {"agent_audible": aud}
            return diarizer.process(audio, 16000, **kw).speaker
        except Exception as e:  # noqa: BLE001
            logger.warning(f"🖥️⚠️ diarization skipped: {e}")
            return None

    def _speaker_name(self, speaker):
        """Learned name for the UI label (never show raw S-tags to people); includes
        a name introduced in this very turn ("Hey Wren, it's Sam")."""
        if not speaker or speaker in ("self", "user", "you"):
            return None
        try:
            book = self.app.state.SpeechPipelineManager.people
            text = self.final_transcription or self.partial_transcription or ""
            return book.name_of(speaker) or book.peek_intro(text)
        except Exception:  # noqa: BLE001
            return None

    def on_before_final(self, audio: bytes, txt: str):
        """
        Callback invoked just before the final STT result for a user turn is confirmed.

        Sets flags indicating user finished, allows TTS if pending, interrupts microphone input,
        releases TTS stream to client, sends final user request and any pending partial
        assistant answer to the client, and adds user request to history.

        Args:
            audio: The raw audio bytes corresponding to the final transcription. (Currently unused)
            txt: The transcription text (might be slightly refined in on_final).
        """
        logger.info(Colors.apply('🖥️🏁 =================== USER TURN END ===================').light_gray)
        self.user_finished_turn = True
        self.user_history_committed = False
        self.user_interrupted = False # Reset connection-specific flag (user finished, not interrupted)
        # Access global manager state
        # Snapshot once: another thread may clear running_generation between
        # the validity check and the use (live 10-01 AttributeError on None).
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen is not None and self._drop_stale_draft(gen):
            gen = None
        if gen is not None and self.app.state.SpeechPipelineManager.is_valid_gen():
            logger.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED (before final)').blue}")
            gen.tts_quick_allowed_event.set()

        # first block further incoming audio (Audio processor's state)
        if not self.app.state.AudioInputProcessor.interrupted:
            logger.info(f"{Colors.apply('🖥️🎙️ ⏸️ Microphone interrupted (end of turn)').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time() # Set connection-specific flag

        logger.info(f"{Colors.apply('🖥️🔊 TTS STREAM RELEASED').blue}")
        self.tts_to_client = True # Set connection-specific flag

        # Send final user request (using the reliable final_transcription OR current partial if final isn't set yet)
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription

        # Diarize up front so the client event carries a speakerId and history
        # carries a [speaker] tag (echo/self is filtered).
        speaker = self._tag_speaker(audio)
        if speaker != "self" and self._text_echo(user_request_content):
            logger.info(f"🖥️🙉 own words echoed back (voiceprint said {speaker}) -> self")
            speaker = "self"
        if speaker:
            logger.info(f"🖥️🗣️ Speaker for turn: {speaker}")
            self._adopt_final_speaker(speaker)

        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request_content,
            "speakerId": speaker if speaker and speaker != "self" else None,
            "speakerName": self._speaker_name(speaker),
        })

        # Access global manager state
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen is not None and self.app.state.SpeechPipelineManager.is_valid_gen():
            # Send partial assistant answer (if available) to the client
            # Use connection-specific user_interrupted flag
            if gen.quick_answer and not self.user_interrupted:
                self.assistant_answer = gen.quick_answer
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer,
                    "personaId": self.app.state.SpeechPipelineManager.current_persona,
                })

        logger.info(f"🖥️🧠 Adding user request to history: '{user_request_content}'")
        if speaker != "self":
            _rec_event("user", spk=speaker or "user", text=user_request_content)
            try:   # Call summary plugin: in-memory session transcript
                import call_summary as _cs
                _cs.note(speaker or "user", user_request_content)
            except Exception:  # noqa: BLE001
                pass
        if speaker == "self":
            logger.info("🖥️🙉 Ignoring own voice (self) — not adding to history.")
        else:
            labeled = f"[{speaker}] {user_request_content}" if speaker else user_request_content
            mgr = self.app.state.SpeechPipelineManager
            mgr.history.append({"role": "user", "content": labeled})
            # Each bookkeeping step is isolated: one bug (e.g. a NameError in a
            # detector) must never silently skip the conversation log, name
            # learning or the held-fragment fallback for every later turn.
            steps = (
                ("trim_history", lambda: mgr.trim_history()),
                ("dynamics", lambda: mgr.dynamics.on_user_turn(user_request_content, speaker_id=speaker or "user")),
                ("floor", lambda: mgr.floor.on_user_turn(speaker)),
                ("convo_log", lambda: mgr.convo.add_user(speaker, user_request_content)),
                ("hypergraph", lambda: getattr(mgr, "hgmem", None) and mgr.hgmem.observe_user(
                    mgr.current_persona, (mgr.people.name_of(speaker) or ""),
                    user_request_content)),
                ("name_learning", lambda: mgr.people.learn(speaker, user_request_content)),
                ("name_record", lambda: _rec_name(speaker, mgr)),
                ("remember", lambda: mgr.remember(user_request_content)),
                ("room_vote", lambda: _passive_vote(speaker, user_request_content, mgr)),
                ("voice_mail", lambda: _mail_heard(speaker, mgr)),
                ("fact_check", lambda: _factcheck_heard(speaker, user_request_content, mgr)),
                ("translate", lambda: _translate_heard(speaker, user_request_content, mgr)),
                ("pacing", lambda: self._pace_turn(speaker, user_request_content, audio, mgr)),
            )
            for name, step in steps:
                try:
                    step()
                except Exception:  # noqa: BLE001
                    logger.exception("user-turn bookkeeping step '%s' failed", name)
            self.user_history_committed = True
            try:
                self._answer_held_fragment(labeled)
            except Exception:  # noqa: BLE001
                logger.exception("held-fragment fallback failed")
            self._draft_spk, self._draft_t = "<none>", 0.0  # next turn drafts immediately
        self.user_history_committed = True

    def _drop_stale_draft(self, gen) -> bool:
        """The reply was drafted from an early fragment of a turn that kept going:
        abort it and answer the final transcript instead (via the held-fragment
        path). Never cuts audio that's already audible."""
        from floor import stale_draft
        final = self.final_transcription or self.partial_transcription or ""
        draft = getattr(gen, "text", "") or ""
        if getattr(gen, "abortion_started", False) or not stale_draft(draft, final):
            return False
        bridge = getattr(self.app.state, "CallBridge", None)
        try:
            if bridge is not None and bridge.is_speaking():
                return False
        except Exception:  # noqa: BLE001
            pass
        logger.info(f"🖥️♻️ stale draft from '{draft[-40:]}' -> answering final '{final[-50:]}'")
        self.app.state.SpeechPipelineManager.abort_generation(
            wait_for_completion=False, reason="stale draft: turn kept going")
        self.held_fragment = final
        return True

    MONOLOGUE_S = float(os.environ.get("ATLAS_MONOLOGUE_S", "6.0"))

    def _monologue_hold(self, txt: str, speaker) -> bool:
        """Same speaker still going and didn't name us: don't re-draft every sentence
        (live 09-29: a new LLM call every ~3.5s through a monologue, none spoken).
        The first sentence of a turn drafts at once; later ones cancel the stale
        draft and are answered from the final transcript at turn end."""
        import time as _t
        from floor import names_agent
        mgr = self.app.state.SpeechPipelineManager
        now = _t.time()
        if (speaker == getattr(self, "_draft_spk", "<none>") and now - getattr(self, "_draft_t", 0.0) < self.MONOLOGUE_S
                and not names_agent(txt, tuple(getattr(getattr(mgr, "dynamics", None), "agent_names", ()) or ()))):
            self.held_fragment = txt
            self._draft_t = now  # rolling: stay held for the whole monologue
            run = getattr(mgr, "running_generation", None)
            if (run is not None and not getattr(run, "abortion_started", False)
                    and not getattr(run, "quick_answer_provided", False)):
                mgr.abort_generation(wait_for_completion=False, reason="monologue: answer at turn end")
            logger.info(f"🖥️🎙️ monologue: holding draft until turn end: '{txt[-50:]}'")
            return True
        self._draft_spk, self._draft_t = speaker, now
        return False

    def _answer_held_fragment(self, final_text: str) -> None:
        """A potential sentence was held as a fragment and the turn ended anyway:
        generate from the final transcript so the turn is never dropped."""
        held, self.held_fragment = getattr(self, "held_fragment", ""), ""
        mgr = self.app.state.SpeechPipelineManager
        run = mgr.running_generation
        # A draft we aborted for a monologue may still be tearing down; that's fine,
        # prepare_generation waits for the abort before starting the new turn.
        if not held or not final_text.strip() or (run is not None and not getattr(run, "abortion_started", False)):
            return
        if not mgr.requests_queue.empty():
            return  # a later potential sentence already queued a generation
        from floor import filler_only
        from conversation_dynamics import pre_llm_veto
        m = re.match(r'^\s*\[(S\d+)\]', final_text)
        if filler_only(final_text) or mgr.floor.turn_gate(
                final_text, m[1] if m else None, tuple(mgr.dynamics.agent_names)):
            return
        logger.info(f"🖥️⏳➡️ turn ended on held fragment; generating from final: '{final_text[-50:]}'")
        mgr.prepare_generation(final_text)

    def on_final(self, txt: str):
        """
        Callback invoked when the final transcription result for a user turn is available.

        Logs the final transcription and stores it.

        Args:
            txt: The final transcription text.
        """
        logger.info(f"\n{Colors.apply('🖥️✅ FINAL USER REQUEST (STT Callback): ').green}{txt}")
        if not self.final_transcription: # Store it if not already set by on_before_final logic
             self.final_transcription = txt

    def abort_generations(self, reason: str):
        """
        Triggers the abortion of any ongoing speech generation process.

        Logs the reason and calls the SpeechPipelineManager's abort method.

        Args:
            reason: A string describing why the abortion is triggered.
        """
        logger.info(f"{Colors.apply('🖥️🛑 Aborting generation:').blue} {reason}")
        # Access global manager state
        self.app.state.SpeechPipelineManager.abort_generation(reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        """
        Callback invoked when the silence detection state changes.

        Updates the internal silence_active flag.

        Args:
            silence_active: True if silence is currently detected, False otherwise.
        """
        # logger.debug(f"🖥️🎙️ Silence active: {silence_active}") # Optional: Can be noisy
        self.silence_active = silence_active
        self.last_voice_at = time.time()

    def on_partial_assistant_text(self, txt: str):
        """
        Callback invoked when a partial text result from the assistant (LLM) is available.

        Updates the internal assistant answer state and sends the partial answer to the client,
        unless the user has interrupted.

        Args:
            txt: The partial assistant text.
        """
        logger.info(f"{Colors.apply('🖥️💬 PARTIAL ASSISTANT ANSWER: ').green}{txt}")
        # Use connection-specific user_interrupted flag
        if not self.user_interrupted:
            self.assistant_answer = txt
            # Use connection-specific tts_to_client flag
            if self.tts_to_client:
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": txt,
                    "personaId": self.app.state.SpeechPipelineManager.current_persona,
                })

    def on_recording_start(self):
        """
        Callback invoked when the audio input processor starts recording user speech.

        If client-side TTS is playing, it triggers an interruption: stops server-side
        TTS streaming, sends stop/interruption messages to the client, aborts ongoing
        generation, sends any final assistant answer generated so far, and resets relevant state.
        """
        logger.info(f"{Colors.ORANGE}🖥️🎙️ Recording started.{Colors.RESET} TTS Client Playing: {self.tts_client_playing}")
        bridge = getattr(self.app.state, "CallBridge", None)
        if bridge is not None:
            # Discord: someone started talking while Max speaks. Don't cut him
            # off on "lol"/"yeah" -- decide in on_partial once there are words.
            self.barge_pending = bridge.is_speaking()
            if self.barge_pending:
                logger.info("🖥️👂 speech over agent -> barge-in pending (waiting for words)")
            return
        if self.tts_client_playing:
            self.interrupt_agent("on_recording_start, user interrupts, TTS Playing")

    def _check_barge_in(self, txt: str):
        if not getattr(self, "barge_pending", False):
            return
        bridge = getattr(self.app.state, "CallBridge", None)
        if bridge is None or not bridge.is_speaking():
            self.barge_pending = False
            return
        yes, why = self._yield_decision(txt)
        if yes:
            self.barge_pending = False
            self._record_cutoff(why)
            cut = bridge.flush_tts()
            logger.info(f"🖥️✋ BARGE-IN ({why}) on '{txt[:40]}' (cut {cut:.1f}s of agent audio)")
            self.interrupt_agent(f"barge-in: {txt[:40]}")

    def interrupt_agent(self, reason: str):
        """Stop the agent mid-reply (browser TTS or call audio)."""
        if True:
            self.tts_to_client = False # Stop server sending TTS
            self.user_interrupted = True # Mark connection as user interrupted
            logger.info(f"{Colors.apply('🖥️❗ INTERRUPTING TTS due to recording start').blue}")

            # Send final assistant answer *if* one was generated and not sent
            logger.info(Colors.apply("🖥️✅ Sending final assistant answer (forced on interruption)").pink)
            self.send_final_assistant_answer(forced=True)

            # Minimal reset for interruption:
            self.tts_chunk_sent = False # Reset chunk sending flag
            # self.assistant_answer = "" # Optional: Clear partial answer if needed

            logger.info("🖥️🛑 Sending stop_tts to client.")
            self.message_queue.put_nowait({
                "type": "stop_tts", # Client handles this to mute/ignore
                "content": ""
            })

            logger.info(f"{Colors.apply('🖥️🛑 RECORDING START ABORTING GENERATION').red}")
            self.abort_generations(reason)

            logger.info("🖥️❗ Sending tts_interruption to client.")
            self.message_queue.put_nowait({ # Tell client to stop playback and clear buffer
                "type": "tts_interruption",
                "content": ""
            })

            # Reset state *after* performing actions based on the old state
            # Be careful what exactly needs reset vs persists (like tts_client_playing)
            # self.reset_state() # Might clear too much, like user_interrupted prematurely

    def send_final_assistant_answer(self, forced=False):
        """
        Sends the final (or best available) assistant answer to the client.

        Constructs the full answer from quick and final parts if available.
        If `forced` and no full answer exists, uses the last partial answer.
        Cleans the text and sends it as 'final_assistant_answer' if not already sent.

        Args:
            forced: If True, attempts to send the last partial answer if no complete
                    final answer is available. Defaults to False.
        """
        final_answer = ""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            final_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer + self.app.state.SpeechPipelineManager.running_generation.final_answer

        if not final_answer: # Check if constructed answer is empty
            # If forced, try using the last known partial answer from this connection
            if forced and self.assistant_answer:
                 final_answer = self.assistant_answer
                 logger.warning(f"🖥️⚠️ Using partial answer as final (forced): '{final_answer}'")
            else:
                logger.warning(f"🖥️⚠️ Final assistant answer was empty, not sending.")
                return# Nothing to send

        logger.debug(f"🖥️✅ Attempting to send final answer: '{final_answer}' (Sent previously: {self.final_assistant_answer_sent})")

        if not self.final_assistant_answer_sent and final_answer:
            import re
            # Clean up the final answer text
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer: # Ensure it's not empty after cleaning
                logger.info(f"\n{Colors.apply('🖥️✅ FINAL ASSISTANT ANSWER (Sending): ').green}{cleaned_answer}")
                _rec_event("agent", persona=self.app.state.SpeechPipelineManager.current_persona, text=cleaned_answer)
                try:
                    import call_summary as _cs
                    _cs.note(self.app.state.SpeechPipelineManager.current_persona or "", cleaned_answer, agent=True)
                except Exception:  # noqa: BLE001
                    pass
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer,
                    "personaId": self.app.state.SpeechPipelineManager.current_persona,
                })
                generation = self.app.state.SpeechPipelineManager.running_generation
                target = generation.decision.target if generation else ''
                # Store the EXACT control header the model must emit. The old '[to=S1]' form
                # was copied by the model as its own header -> INVALID -> silent turns,
                # increasingly often as the call grew (tests/sim_history_format.py).
                app.state.SpeechPipelineManager.floor.on_agent_spoke(target, cleaned_answer)
                try:
                    from floor import promises_quiet
                    if promises_quiet(cleaned_answer):
                        app.state.SpeechPipelineManager.floor.set_quiet()
                        logger.info("🤫 agent promised quiet -> quiet mode until named")
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"self-quiet failed: {e}")
                # Barge-in can cut a reply mid-clause. Fragments saved as whole lines
                # make the model emit stubs ("Yeah, I") -- keep complete sentences only.
                from floor import model_safe_reply
                heard = getattr(self, "_cut_said", None) if forced else None
                self._cut_said = None
                safe_answer = model_safe_reply(heard if heard is not None else cleaned_answer)
                if not forced:
                    app.state.SpeechPipelineManager.cutoff = None  # thought delivered or dropped
                try:
                    from identity import scrub as _id_scrub
                    safe_answer = _id_scrub(safe_answer, app.state.SpeechPipelineManager.current_persona)
                except Exception as e:  # noqa: BLE001
                    logger.warning("identity scrub failed: %s", e)
                try:
                    guard = getattr(app.state, "EchoGuard", None)
                    if guard is not None:
                        guard.spoke(cleaned_answer)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"echo guard record failed: {e}")
                if safe_answer:
                    history_answer = f"[SPEAK to={target or 'user'}] {safe_answer}"
                    app.state.SpeechPipelineManager.convo.add_agent(target, safe_answer)
                    try:
                        _m = app.state.SpeechPipelineManager
                        if getattr(_m, "hgmem", None) is not None:
                            _m.hgmem.observe_reply(_m.current_persona, safe_answer)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"hypergraph reply hook failed: {e}")
                    try:
                        app.state.SpeechPipelineManager.people.note_agent_reply(target, safe_answer)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("name-ask tracking failed: %s", e)
                    self.app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": history_answer})
                    app.state.SpeechPipelineManager.trim_history()
                    try:
                        voiced = heard if heard is not None else ("" if forced else cleaned_answer)
                        told = app.state.SpeechPipelineManager.board().spoke(getattr(generation, "id", None), voiced)
                        if told:
                            logger.info(f"🖥️📋 task(s) {told} delivered")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("task delivery bookkeeping failed: %s", e)
                else:
                    logger.info(f"🖥️✂️ Cut-off reply kept out of model history: '{cleaned_answer}'")
                # Record that the agent spoke (feeds participation/floor-balance state).
                app.state.SpeechPipelineManager.dynamics.on_agent_turn(
                    duration_s=max(1.0, len(cleaned_answer) / 15.0)
                )
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer # Store the sent answer
            else:
                logger.warning(f"🖥️⚠️ {Colors.YELLOW}Final assistant answer was empty after cleaning.{Colors.RESET}")
                self.final_assistant_answer_sent = False # Don't mark as sent
                self.final_assistant_answer = "" # Clear the stored answer
        elif forced and not final_answer: # Should not happen due to earlier check, but safety
             logger.warning(f"🖥️⚠️ {Colors.YELLOW}Forced send of final assistant answer, but it was empty.{Colors.RESET}")
             self.final_assistant_answer = "" # Clear the stored answer


# --------------------------------------------------------------------
# Main WebSocket endpoint
# --------------------------------------------------------------------
def _wire_callbacks(app: FastAPI, callbacks: "TranscriptionCallbacks") -> None:
    aip = app.state.AudioInputProcessor
    aip.realtime_callback = callbacks.on_partial
    aip.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    aip.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    aip.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    aip.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    aip.transcriber.full_transcription_callback = callbacks.on_final
    aip.transcriber.before_final_sentence = callbacks.on_before_final
    aip.recording_start_callback = callbacks.on_recording_start
    aip.silence_active_callback = callbacks.on_silence_active
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text


# Finalized chat lines replayed to UIs that connect late (bounded).
import collections as _collections
_BACKLOG_TYPES = {"final_user_request", "final_assistant_answer"}
_CHAT_BACKLOG: "_collections.deque" = _collections.deque(maxlen=120)


async def _fanout(hub: asyncio.Queue, subscribers: set) -> None:
    """Copy every headless-session message to each connected UI (drop if none)."""
    while True:
        data = await hub.get()
        if data.get("type") != "tts_chunk":
            logger.info(Colors.apply(f"🖥️📤 →→UI x{len(subscribers)}: {data}").orange)
        if data.get("type") in _BACKLOG_TYPES:
            data.setdefault("t", time.time())
            _CHAT_BACKLOG.append(data)
        for q in list(subscribers):
            if q.qsize() < 500:
                q.put_nowait(data)


DELIVERY_GAP_S = float(os.environ.get("ATLAS_DELIVERY_GAP_S", "2.0"))


def delivery_ready(mgr, callbacks, bridge, now: float, gap_s: float = DELIVERY_GAP_S) -> bool:
    """A real opening in the call: nobody talking, agent idle, not stepped back."""
    if mgr.running_generation is not None or not mgr.requests_queue.empty():
        return False
    if not getattr(callbacks, "silence_active", True) or getattr(callbacks, "is_hot", False):
        return False
    if now - getattr(callbacks, "last_voice_at", 0.0) < gap_s:
        return False
    if bridge is not None and bridge.is_speaking():
        return False
    if mgr.floor.is_quiet():
        return False
    return True


RESUME_GAP_S = float(os.environ.get("ATLAS_RESUME_GAP_S", "1.2"))


def start_resume(mgr, callbacks) -> bool:
    """Nobody took the floor after cutting the agent off: let her finish."""
    cut = getattr(mgr, "cutoff", None)
    if cut is None or cut.gap_tried or not cut.live():
        return False
    cut.gap_tried = True
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True
    logger.info(f"🖥️↩️ resuming cut-off thought at a gap: '{cut.unsaid[:50]}'")
    mgr.prepare_generation(cut.cue())
    return True


def start_delivery(mgr, callbacks) -> Optional[int]:
    """Launch the cue generation for the oldest untold result (None if nothing)."""
    t = mgr.board().deliverable()
    if not t:
        return None
    cue = mgr.board().claim_delivery(t)
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True  # no user line: a HOLD retires cleanly
    logger.info(f"🖥️📋 delivering task #{t['id']} ({t['text'][:50]!r}) at a gap")
    mgr.prepare_generation(cue)
    return t["id"]


def _passive_vote(speaker, text: str, mgr) -> None:
    """Count 'put me down for tacos' while a poll is open even if the floor keeps the
    agent quiet on that line (Dice & polls plugin). One vote per person, so the
    model's own cast_vote on the same line can't double-count."""
    import room_tools
    import plugins as _plugins
    if not _plugins.is_enabled("dice_polls"):
        return
    pf = room_tools.intent(text)
    if pf and pf[0] == "cast_vote":
        who = mgr.people.name_of(speaker) or speaker or ""
        logger.info("🗳️ %s", room_tools.cast_vote(pf[1]["choice"], who)[:80])


def _factcheck_heard(speaker, text: str, mgr) -> None:
    """Quiet fact-check plugin: every line is remembered; checkable claims are looked up in
    the background (never spoken unless someone asks)."""
    import factcheck
    import plugins as _plugins
    if not _plugins.is_enabled("fact_check") or speaker == "self":
        return
    s = _plugins.settings_of("fact_check") or {}
    bg = (s.get("background") or "on") == "on" and _plugins.is_enabled("web_search")
    factcheck.observe(text, mgr.people.name_of(speaker) or speaker or "",
                      gap_s=float(s.get("gap") or factcheck.DEFAULT_GAP_S),
                      daily=int(s.get("daily") or factcheck.DEFAULT_DAILY), background=bg)


def _translate_heard(speaker, text: str, mgr) -> None:
    """Live translate plugin: remember each line with its detected language; in captions
    mode, translate foreign lines in the background (shown in the plugin, never spoken)."""
    import plugins as _plugins
    if not _plugins.is_enabled("live_translate") or speaker == "self":
        return
    import languages
    import translate
    det, prob = languages.ROOM.last_detect
    lang = det if (det and prob >= languages.CONFIDENT) else languages.guess_text(text, languages.ROOM.current())
    who = mgr.people.name_of(speaker) or speaker or ""
    translate.note(speaker or "", text, lang)
    s = _plugins.settings_of("live_translate") or {}
    if (s.get("mode") or "request") == "captions":
        translate.caption(who, text, lang, max(prob or 0.0, languages.CONFIDENT if not det else 0.0),
                          languages.primary(), daily=int(s.get("daily") or translate.CAPTION_DAILY))


def _mail_heard(speaker, mgr) -> None:
    """Voice mail plugin: note that a recognised person just spoke."""
    import voice_mail
    import plugins as _plugins
    if not _plugins.is_enabled("voice_mail") or not speaker or speaker == "self":
        return
    voice_mail.heard(mgr.people.name_of(speaker), speaker)


def start_mail(mgr, callbacks, bridge, now: float) -> Optional[int]:
    """Pass on a voice-addressed message once its recipient has spoken and the room pauses."""
    try:
        import voice_mail
        import plugins as _plugins
        if not _plugins.is_enabled("voice_mail"):
            return None
        days = float((_plugins.settings_of("voice_mail") or {}).get("days") or voice_mail.DEFAULT_DAYS)
        m = voice_mail.due(days)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"📬 mail check failed: {e}")
        return None
    if not m or not delivery_ready(mgr, callbacks, bridge, now, TIMER_GAP_S):
        return None
    cue = voice_mail.claim(m["id"])
    if not cue:
        return None
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True
    logger.info(f"🖥️📬 message #{m['id']} for {m['to']} ({m['text'][:50]!r}); delivering")
    mgr.prepare_generation(cue)
    return m["id"]


def start_bet(mgr, callbacks, bridge, now: float) -> Optional[int]:
    """Bring up a logged bet once it can be settled (Bet tracker plugin)."""
    try:
        import bets
        import plugins as _plugins
        if not _plugins.is_enabled("bets"):
            return None
        b = bets.due()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🎰 bet check failed: {e}")
        return None
    if not b or not delivery_ready(mgr, callbacks, bridge, now, TIMER_GAP_S):
        return None
    cue = bets.claim(b["id"])
    if not cue:
        return None
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True
    logger.info(f"🖥️🎰 bet #{b['id']} is due ({b['claim'][:50]!r}); bringing it up")
    mgr.prepare_generation(cue)
    return b["id"]


TIMER_GAP_S = float(os.environ.get("ATLAS_TIMER_GAP_S", "1.0"))
TIMER_OVERDUE_GAP_S = 0.5    # once a reminder is 20 s late, any short pause will do


def start_timer(mgr, callbacks, bridge, now: float) -> Optional[int]:
    """Announce a timer that went off, at the first real gap (room_tools / Timers plugin)."""
    try:
        import room_tools
        import plugins as _plugins
        if not _plugins.is_enabled("timers"):
            return None
        t = room_tools.due()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⏲️ timer check failed: {e}")
        return None
    if not t:
        return None
    gap = TIMER_OVERDUE_GAP_S if now - t["due"] > 20 else TIMER_GAP_S
    if not delivery_ready(mgr, callbacks, bridge, now, gap):
        return None
    cue = room_tools.claim(t["id"])
    if not cue:
        return None
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True
    logger.info(f"🖥️⏲️ timer #{t['id']} went off ({t['message'][:50]!r}); announcing")
    mgr.prepare_generation(cue)
    return t["id"]


async def _task_delivery(app: FastAPI, callbacks) -> None:
    """Bring finished background searches up when the call has an opening."""
    while True:
        await asyncio.sleep(0.5)
        try:
            mgr = app.state.SpeechPipelineManager
            bridge = getattr(app.state, "CallBridge", None)
            now = time.time()
            if start_timer(mgr, callbacks, bridge, now) is not None:
                continue
            if start_mail(mgr, callbacks, bridge, now) is not None:
                continue
            if start_bet(mgr, callbacks, bridge, now) is not None:
                continue
            if delivery_ready(mgr, callbacks, bridge, now, RESUME_GAP_S) and start_resume(mgr, callbacks):
                continue
            if delivery_ready(mgr, callbacks, bridge, now):
                start_delivery(mgr, callbacks)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"🖥️📋 delivery loop: {e}")


def _start_headless_session(app: FastAPI) -> dict:
    hub = asyncio.Queue()
    subscribers: set = set()
    callbacks = TranscriptionCallbacks(app, hub)
    _wire_callbacks(app, callbacks)
    tasks = [
        asyncio.create_task(send_tts_chunks(app, hub, callbacks)),
        asyncio.create_task(_fanout(hub, subscribers)),
        asyncio.create_task(_task_delivery(app, callbacks)),
    ]
    logger.info("🖥️🤖 Headless agent session started (bridge mode; UI optional).")
    return {"hub": hub, "subscribers": subscribers, "callbacks": callbacks, "tasks": tasks}


async def _ui_subscriber(ws: WebSocket, app: FastAPI) -> None:
    """Bridge-mode /ws: the UI observes the headless session and sends commands."""
    h = app.state.headless
    q = asyncio.Queue()
    q.put_nowait({"type": "chat_backlog", "items": list(_CHAT_BACKLOG),
                  "speed": getattr(app.state, "turn_speed", 0)})
    h["subscribers"].add(q)
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, asyncio.Queue(), h["callbacks"])),
        asyncio.create_task(send_text_messages(ws, q, quiet=True)),
    ]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        h["subscribers"].discard(q)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️❌ UI subscriber disconnected (agent keeps running).")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Handles the main WebSocket connection for real-time voice chat.

    Accepts a connection, sets up connection-specific state via `TranscriptionCallbacks`,
    initializes audio/message queues, and creates asyncio tasks for handling
    incoming data, audio processing, outgoing text messages, and outgoing TTS chunks.
    Manages the lifecycle of these tasks and cleans up on disconnect.

    Args:
        ws: The WebSocket connection instance provided by FastAPI.
    """
    await ws.accept()
    logger.info("🖥️✅ Client connected via WebSocket.")
    if getattr(app.state, "headless", None):
        await _ui_subscriber(ws, app)
        return

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # Set up callback manager - THIS NOW HOLDS THE CONNECTION-SPECIFIC STATE
    callbacks = TranscriptionCallbacks(app, message_queue)

    # Assign callbacks to the AudioInputProcessor (global component)
    # These methods within callbacks will now operate on its *instance* state
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active

    # Assign callback to the SpeechPipelineManager (global component)
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text

    # Create tasks for handling different responsibilities
    # Pass the 'callbacks' instance to tasks that need connection-specific state
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)), # Pass callbacks
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)), # Pass callbacks
    ]
    if not ENABLE_CALL_BRIDGE:
        # Browser/Electron mic is the audio source only in non-bridge mode.
        # In bridge mode the CallBridge feeds B2 directly; a second feed would
        # interleave silence with the call audio and corrupt VAD/STT.
        tasks.append(asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)))

    try:
        # Wait for any task to complete (e.g., client disconnect)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        # Await cancelled tasks to let them clean up if needed
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️💥 {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        logger.info("🖥️🧹 Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        # Ensure all tasks are awaited after cancellation
        # Use return_exceptions=True to prevent gather from stopping on first error during cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️❌ WebSocket session ended.")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
def _bind_host() -> str:
    """Public build listens on this computer only (setup/owner endpoints trust loopback).
    The dev build keeps LAN access. ATLAS_HOST overrides either."""
    h = os.environ.get("ATLAS_HOST", "").strip()
    if h:
        return h
    import user_settings as _us_host
    return "127.0.0.1" if _us_host.is_public() else "0.0.0.0"


if __name__ == "__main__":

    # Run the server without SSL
    if not USE_SSL:
        logger.info("🖥️▶️ Starting server without SSL.")
        uvicorn.run("server:app", host=_bind_host(), port=int(os.environ.get("ATLAS_PORT", "8000")), log_config=None)

    else:
        logger.info("🖥️🔒 Attempting to start server with SSL.")
        # Check if cert files exist
        cert_file = "127.0.0.1+1.pem"
        key_file = "127.0.0.1+1-key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
             logger.error(f"🖥️💥 SSL cert file ({cert_file}) or key file ({key_file}) not found.")
             logger.error("🖥️💥 Please generate them using mkcert:")
             logger.error("🖥️💥   choco install mkcert") # Assuming Windows based on earlier check, adjust if needed
             logger.error("🖥️💥   mkcert -install")
             logger.error("🖥️💥   mkcert 127.0.0.1 YOUR_LOCAL_IP") # Remind user to replace with actual IP if needed
             logger.error("🖥️💥 Exiting.")
             sys.exit(1)

        # Run the server with SSL
        logger.info(f"🖥️▶️ Starting server with SSL (cert: {cert_file}, key: {key_file}).")
        uvicorn.run(
            "server:app",
            host=_bind_host(),
            port=int(os.environ.get("ATLAS_PORT", "8000")),
            log_config=None,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
        )
