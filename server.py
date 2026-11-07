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


def clear_rec_names() -> None:
    """Call end (docs/designs/experience_memory.md, S4): the recorder's label -> name map belongs to the call."""
    _REC_NAMES.clear()


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
    _local_startup = _us.startup_local_config()
    if _local_startup is not None:
        LLM_START_MODEL = _local_startup["model"]
        os.environ.setdefault("OLLAMA_BASE_URL", _local_startup["url"])
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
    __import__("stall_watch").start()
    try:   # how often a thread running Python must hand the GIL to a waiting one (Python's default: 5 ms)
        _sw = float(os.environ.get("ATLAS_GIL_SWITCH_MS", "") or 0)
        if _sw > 0:
            sys.setswitchinterval(_sw / 1000.0)
            logger.info("🖥️🔀 GIL switch interval %g ms", _sw)
    except ValueError:
        pass
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

    try:   # Live screen (plugin, off by default) needs to know which model server replies
        import live_screen as _live_screen
        _live_screen.bind(app.state.SpeechPipelineManager)
    except Exception as e:  # noqa: BLE001
        logger.warning("live screen unavailable: %s", e)
    try:
        import screen as _screen
        _screen.bind_model(lambda: app.state.SpeechPipelineManager.llm)
    except Exception as e:  # noqa: BLE001
        logger.warning("model image capability unavailable: %s", e)

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

    # every agent or voice switch re-points the call diarizer at the new voice (on the switch's worker thread)
    app.state.SpeechPipelineManager.on_voice_changed = lambda: _sync_bridge_self_reference(app)

    _setup_owner_voice(app)
    _setup_call_boundary(app)

    # Refresh the selected remote transport and release any old local standby.
    # Startup already selects remote before prewarm; a failure retains that mode.
    try:
        import remote_llm as _rl_startup
        if _rl_startup.remote_selected():
            async def _restore_remote():
                res = await asyncio.to_thread(_switch_llm_backend, "remote")
                if res.get("error"):
                    logger.warning("Remote startup unavailable: %s; selected mode retained", res["error"])
            asyncio.create_task(_restore_remote())
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM backend restore skipped: %s", e)

    # Headless agent session: in bridge mode the agent must hear and answer the
    # call whether or not a UI is connected. UIs subscribe to its broadcast feed.
    app.state.headless = None
    if ENABLE_CALL_BRIDGE and app.state.CallBridge is not None:
        app.state.headless = _start_headless_session(app)
    app.state.call_ticker = asyncio.create_task(_call_ticker(app))

    yield

    if getattr(app.state, "headless", None):
        for t in app.state.headless["tasks"]:
            t.cancel()
    if getattr(app.state, "call_ticker", None) is not None:
        app.state.call_ticker.cancel()
    try:
        import call_boundary
        await asyncio.to_thread(call_boundary.BOUNDARY.end, "shutdown", True, 5.0)
    except Exception as e:  # noqa: BLE001
        logger.warning("call end at shutdown failed: %s", e)

    logger.info("🖥️⏹️ Server shutting down")
    try:
        import provider_accounts
        provider_accounts.close_logins()
    except Exception:
        logger.debug("account login cleanup failed",exc_info=True)
    __import__("turn_trace").flush_all()
    try:
        import live_screen as _live_screen
        _live_screen.shutdown()
    except Exception:  # noqa: BLE001
        pass
    if app.state.CallBridge is not None:
        app.state.CallBridge.stop()
    app.state.AudioInputProcessor.shutdown()
    try:
        from call_recorder import REC
        await asyncio.to_thread(REC.close)
    except Exception as e:  # noqa: BLE001
        logger.warning("recorder close at shutdown failed: %s", e)

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
    from setup_api import exit_for_restart
    threading.Thread(target=exit_for_restart, daemon=True).start()
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
    """Pushes `fast` frames (~30 Hz while anything moves: meters/spectrum/state), `snap` frames
    (~1 Hz: GPU/model/pipeline/room) and incremental events + log lines."""
    await ws.accept()
    ev_seq = max((e["seq"] for e in TELEMETRY.events_since(0)), default=0) - 60
    ln_seq = max((l["seq"] for l in TELEMETRY.lines_since(0)), default=0) - 80
    last_snap = last_fast = 0.0
    last_idle = False
    try:
        while True:
            now = time.time()
            # 30 Hz while anything moves (owner 11-04: a high refresh rate); a silent room sends 4 frames a second
            fast = TELEMETRY.fast()
            idle = (max(fast["in"], fast["clean"], fast["out"]) < 1e-3
                    and not (fast["speaking"] or fast["recording"] or fast["thinking"] or fast["partial"]))
            frames = [] if (idle and last_idle and now - last_fast < 0.25) else [fast]
            if frames:
                last_fast = now
            last_idle = idle
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
            await asyncio.sleep(1 / 30)
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
    if body.get("enabled"):
        import plugins as _screen_plugins
        if not _screen_plugins.is_enabled("eyes"):
            return JSONResponse({"error": "Install and enable Eyes in Plugins first."}, status_code=409)
    _scr.set_enabled(bool(body.get("enabled")))
    return _scr.status()

@app.get("/api/screen/log")
async def screen_log(request: Request):
    """Live screen's screen log (owner only): what the watcher saw this session, oldest first."""
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    import live_screen as _live_screen
    return {"entries": _live_screen.log_snapshot()}


@app.post("/api/screen/log/clear")
async def screen_log_clear(request: Request):
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    import live_screen as _live_screen
    _live_screen.log_clear()
    return {"ok": True}


@app.get("/api/screen/last")
async def screen_last(request: Request):
    import screen as _scr
    if not _is_owner(request):
        return JSONResponse({"error": "owner only"}, status_code=403)
    try:   # Live screen on: the newest live frame from memory (nothing is written to disk)
        import live_screen as _live_screen
        if _live_screen.enabled():
            jpg = await asyncio.to_thread(_live_screen.latest_jpeg)
            if not jpg:
                return JSONResponse({"error": "no live frame"}, status_code=404)
            from fastapi.responses import Response
            return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception as e:  # noqa: BLE001
        logger.debug(f"live frame: {e}")
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
            probe = _rl.make_session(cfg, None)
            try:
                if not probe.ping(timeout=4.0):
                    return {"error": "remote unreachable: " + probe.last_error}
            finally:
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
            _us3.remember_llm_backend(mode, cfg if mode == "remote" else None)
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


@app.post("/api/llm_backend/retry")
async def llm_backend_retry(request: Request):
    if not _is_owner(request):
        return JSONResponse({"error":"owner only"},status_code=403)
    llm=app.state.SpeechPipelineManager.llm
    session=getattr(llm,"remote",None)
    retry=getattr(session,"clear_usage_limit",None)
    if not callable(retry):
        return JSONResponse({"error":"The current provider has no paused account request."},status_code=400)
    retry()
    return {"ok":True,"message":"Ready for a new request. The previous turn was not replayed."}


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
    try:
        model_info = A.runtime_model_info(getattr(app.state.SpeechPipelineManager, "llm", None))
        model_error = None
    except A.DraftError as exc:
        model_info, model_error = None, str(exc)
    return {"fields": A.FIELDS, "voices": voice_names(), "used_voices": sorted(used),
            "model_info": model_info, "model_error": model_error,
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
    if getattr(app.state, "llm_switching", False):
        return JSONResponse({"error": "The current model is switching. Wait for it to finish, then retry drafting."}, status_code=409)
    llm = getattr(mgr, "llm", None)
    try:
        draft = await asyncio.to_thread(
            A.build_runtime_draft, llm, name, desc,
            body.get("voice") or (__import__("voices").DEFAULT_VOICE or ""))
        if getattr(mgr, "llm", None) is not llm or getattr(app.state, "llm_switching", False):
            raise A.DraftError("The current model changed while drafting. Retry with the model now loaded.")
        return draft
    except A.DraftError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


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
    ok, _, err = await _actions(app).run("pipeline", app.state.SpeechPipelineManager.refresh_registry)
    if not ok:
        return JSONResponse({"error": err}, status_code=500)
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
    mgr = app.state.SpeechPipelineManager

    def register():
        mgr.register_agent(rec, body["prompt"])
        return mgr.personas_info()
    ok, info, err = await _actions(app).run("pipeline", register)
    if not ok:
        return JSONResponse({"error": err}, status_code=500)
    return {"agent": rec, **info}


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
    ok, removed, err = await _actions(app).run("pipeline", lambda: mgr.remove_agent(aid))
    if not ok:
        return JSONResponse({"error": err}, status_code=500)
    if not removed:
        return JSONResponse({"error": "Can't delete a built-in or the active agent."}, status_code=400)
    await asyncio.to_thread(A.delete_agent, aid)
    return mgr.personas_info()


# --- Owner controls (owner_controls.py): mute, quiet mode, talkativeness, muted speakers ---
def _setup_call_boundary(app) -> None:
    """Register the server-owned call-end resets (the diarizer's voices, the owner adoptions, the recorder's name map)
    and end a call a stopped server left open (docs/designs/experience_memory.md, D2, S4)."""
    import call_boundary

    def forget_voices():
        bridge = getattr(app.state, "CallBridge", None)
        d = getattr(bridge, "diarizer", None) if bridge is not None else getattr(app.state, "Diarizer", None)
        if d is not None:
            with _DIARIZER_LOCK:
                d.forget_all()

    def end_adoptions():
        a = getattr(app.state, "OwnerAdopter", None)
        if a is not None:
            a.end_call()

    call_boundary.register("diarize.Diarizer.forget_all", forget_voices, order=11)
    call_boundary.register("owner_voice.Adopter.end_call", end_adoptions, order=12)
    call_boundary.register("server.clear_rec_names", clear_rec_names, order=10)
    call_boundary.BOUNDARY.set_owner_name_source(
        lambda: (app.state.OwnerAdopter.stored_name() if getattr(app.state, "OwnerAdopter", None) else None))
    for mod, fn in (("lore", "strip_legacy_names"), ("teach", "strip_legacy_names"),
                    ("self_tools", "scrub_existing_predictions")):
        try:   # one-time: names out of entries stored before the experience memory (design D8)
            n = getattr(__import__(mod), fn)()
            if n:
                logger.info("%s.%s: %d entries cleaned", mod, fn, n)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s.%s failed: %s", mod, fn, e)
    try:
        call_boundary.BOUNDARY.startup()
    except Exception as e:  # noqa: BLE001
        logger.warning("call boundary startup failed: %s", e)
    _setup_memory_runner(app)


def _llm_mode(app) -> dict:
    """What the pass talks to (D3): {"mode": "remote"|"local", "cloud": bool, "args": [...]}"""
    llm = getattr(getattr(app.state, "SpeechPipelineManager", None), "llm", None)
    remote = getattr(llm, "remote", None)
    if remote is not None:
        cfg = getattr(remote, "cfg", {}) or {}
        args = ["--mode", "remote"]
        if cfg.get("url") and cfg.get("model"):          # the model the live agent uses (D3); the key stays in its file
            args += ["--url", str(cfg["url"]), "--model", str(cfg["model"])]
        return {"mode": "remote", "cloud": bool(cfg.get("cloud")), "args": args}
    url = getattr(llm, "effective_ollama_url", None) or "http://127.0.0.1:11434"
    model = getattr(llm, "model", "") or ""
    return {"mode": "local", "cloud": False,
            "args": ["--mode", "local", "--local-url", str(url), "--local-model", str(model), "--ctx", "4096"]}


def _setup_memory_runner(app) -> None:
    """The post-call pass between calls (D3): started by the lifespan ticker, killed on speech activity."""
    try:
        import call_boundary
        import memory_writer
        if memory_writer.kill_leftover_run():
            logger.info("a memory pass left from the last quit was stopped")

        def done(code):        # any exit: a failed or killed run may still have written some calls (D3)
            mgr = getattr(app.state, "SpeechPipelineManager", None)
            if mgr is not None:
                mgr.memory_reloaded()
        runner = memory_writer.Runner(args_fn=lambda: _llm_mode(app)["args"],
                                      cloud_fn=lambda: _llm_mode(app)["cloud"], on_done=done)
        call_boundary.BOUNDARY.on_activity(runner.on_activity)
        app.state.MemoryRunner = runner
    except Exception as e:  # noqa: BLE001
        logger.warning("memory pass runner not set up: %s", e)
        app.state.MemoryRunner = None


def start_pass_at_quit(app) -> bool:
    """`start_pass` for POST /api/call/end (S3): pending files to do, not paused, the model is not on this PC, cloud
    use allowed or not needed, and the name tagger present."""
    try:
        import memory_writer
        m = _llm_mode(app)
        if m["mode"] == "local":
            return False
        return memory_writer.may_run(m["cloud"]) is None
    except Exception:  # noqa: BLE001
        return False


async def _call_ticker(app) -> None:
    """Heartbeat for current.json and the 30-minute call end, in browser and bridge mode alike."""
    import call_boundary
    while True:
        try:
            await asyncio.sleep(5.0)
            await asyncio.to_thread(call_boundary.BOUNDARY.tick)
            runner = getattr(app.state, "MemoryRunner", None)
            if runner is not None:
                b = call_boundary.BOUNDARY
                await asyncio.to_thread(lambda: runner.tick(b.in_progress_here(), b.last_activity))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("call ticker failed: %s", e)


def _setup_owner_voice(a) -> None:
    """Owner voice recognition (owner_voice.py, docs/designs/owner_voice.md): call-bridge mode only, since every
    measurement came from the Discord stream. Anything wrong leaves recognition off and Atlas as before."""
    a.state.OwnerAdopter = None
    mgr = getattr(a.state, "SpeechPipelineManager", None)
    try:
        if mgr is not None:
            mgr.people.owner = None
        bridge = getattr(a.state, "CallBridge", None)
        dz = getattr(bridge, "diarizer", None) if bridge is not None else None
        if dz is None or mgr is None:
            return
        import owner_voice
        store = owner_voice.Store(owner_voice.PATH, owner_voice.EMBEDDER,
                                  uses_center=getattr(dz, "center", None) is not None)
        if store.load():
            logger.info(f"👤 owner voice loaded ({len(store.samples)} sample(s))")
        adopter = owner_voice.Adopter(store)
        a.state.OwnerAdopter = adopter
        mgr.people.owner = adopter
    except Exception as e:  # noqa: BLE001
        logger.warning(f"owner voice recognition off: {e}")
        a.state.OwnerAdopter = None
        if mgr is not None:
            mgr.people.owner = None


def _owner_event(kind: str, **fields) -> None:
    """Owner adoptions, refusals, merges and "Forget my voice" go to the call's pending file (design D2, S1)."""
    try:
        import call_boundary
        call_boundary.owner_event(kind, **fields)
    except Exception as e:  # noqa: BLE001
        logger.debug("pending owner event skipped: %s", e)


def _store_said_fact(mgr, adopter, label, item) -> None:
    """Outcomes 1, 2 and 5 of "remember that" (D7): checks without the tagger; a confirmed `said` fact when the turn
    is a qualifying owner line, else "to confirm"; nothing when a check fails. Written through the writer thread."""
    import call_boundary
    import experience_memory
    from speech_pipeline_manager import said_fact
    fact, quote, problem = said_fact(item["clause"], adopter)
    if problem:
        logger.info("remember-that not kept (%s)", problem)
        return
    fid = experience_memory.new_fact_id()
    ids = mgr.said_facts.setdefault(label, [])
    ids.append(fid)
    call = call_boundary.BOUNDARY.call_id

    def store():
        rec = experience_memory.add_owner_fact(fact, quote, "said", bool(item["qualifying"]), call=call,
                                               src_t=item["t"], tagger_checked=False, fid=fid)
        if rec is not None and rec["id"] != fid and fid in ids:    # the same fact was already kept: a later
            ids[ids.index(fid)] = rec["id"]                       # "Not me" acts on the record that exists
    call_boundary.WRITER.submit(store)
    logger.info("remember-that kept for the owner (%s)", "confirmed" if item["qualifying"] else "to confirm")


def _release_held_facts(mgr, adopter, label) -> None:
    """A label was adopted as the owner: its held "remember that" turns are handled now (D7 outcome 3)."""
    for item in (getattr(mgr, "held_facts", {}) or {}).pop(label, []):
        try:
            _store_said_fact(mgr, adopter, label, item)
        except Exception as e:  # noqa: BLE001
            logger.warning("held remember-that failed: %s", e)


def _said_facts_to_confirm(mgr, label) -> None:
    """A later "Not me": the facts said under that label this call go back to "to confirm" (D7)."""
    ids = (getattr(mgr, "said_facts", {}) or {}).get(label) or []
    if ids:
        import call_boundary
        import experience_memory
        call_boundary.WRITER.submit(lambda: experience_memory.to_confirm(list(ids)))


def _merge_said_facts(mgr, adopter, old, new) -> None:
    said = getattr(mgr, "said_facts", None)
    if said is None:
        return
    moved = said.pop(old, [])
    if moved:
        said.setdefault(new, []).extend(moved)
        try:
            if new in adopter.refused:
                _said_facts_to_confirm(mgr, new)
        except Exception:  # noqa: BLE001
            pass


_IDENTS: dict = {}


def _agent_ident(agent: str) -> str:
    """The agent folder's identity (agent_memory.ident), cached per agent; a deleted and re-created agent gets a new
    one, so the pass never writes its old calls into the new folder (design D1)."""
    if not agent:
        return ""
    try:
        import agent_memory
        v = agent_memory.ident(agent, create=False)
    except Exception:  # noqa: BLE001
        return _IDENTS.get(agent, "")
    _IDENTS[agent] = v
    return v


def _owner_voice_action(a, field) -> JSONResponse | dict:
    """POST /api/owner {"owner_voice": ...} (runs on a worker thread): enroll {"label", "name"}, {"not_me": label}, or
    null to forget. Returns the owner status, or a JSONResponse with {"error"} and 400/409/500."""
    import owner_voice
    adopter = getattr(a.state, "OwnerAdopter", None)
    if adopter is None:
        return JSONResponse({"error": "Voice recognition works in Discord calls only"}, status_code=409)
    mgr = a.state.SpeechPipelineManager
    book = mgr.people
    if field is None:                                        # Forget my voice
        name, voice = adopter.store.name, adopter.store.voice
        try:
            adopter.store.forget()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"owner voice not deleted: {type(e).__name__}")
            return JSONResponse({"error": "Couldn't delete the saved voice"}, status_code=500)
        adopter.clear()
        book.drop_prints(name, voice, 0.70)
        _owner_event("forget_voice")
        logger.info("👤 owner voice forgotten")
        return _owner_status(a)
    if not isinstance(field, dict):
        return JSONResponse({"error": "owner_voice must be an object or null"}, status_code=400)
    if "not_me" in field:
        label = book._resolve(str(field.get("not_me") or ""))
        if not owner_voice.is_label(label):
            return JSONResponse({"error": "unknown voice"}, status_code=400)
        adopter.refuse(label)
        _owner_event("refuse", label=label)
        _said_facts_to_confirm(mgr, label)
        logger.info(f"👤 {label} is not the owner (this session)")
        return _owner_status(a)
    label = book._resolve(str(field.get("label") or ""))
    if not owner_voice.is_label(label):
        return JSONResponse({"error": "unknown voice"}, status_code=400)
    agents = set(getattr(book, "agent_names", set()) or set())
    try:
        import agent_registry as _ar
        agents |= set(_ar.all_agent_words())
    except Exception:  # noqa: BLE001
        pass
    name = owner_voice.valid_name(field.get("name"), agents)
    if not name:
        return JSONResponse({"error": "Use a name of up to 40 letters, not an agent's name"}, status_code=400)
    centroid, n = adopter.sample_for(label)
    if n < owner_voice.ENROLL_TURNS:
        return JSONResponse({"error": f"Talk a little more first ({n} of {owner_voice.ENROLL_TURNS} lines heard)"},
                            status_code=409)
    if centroid is None:
        return JSONResponse({"error": "This row mixes voices; try again later"}, status_code=409)
    try:
        adopter.store.enroll(name, centroid, n, adopter.session)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"owner voice not saved: {type(e).__name__}")
        return JSONResponse({"error": "Couldn't save the voice"}, status_code=500)
    adopter.restart_scores()
    adopter.adopt(label)
    book.owner_adopted(label)
    _owner_event("adopt", label=label, how="manual")
    _release_held_facts(mgr, adopter, label)
    logger.info(f"👤 owner enrolled from {label} ({len(adopter.store.samples)} sample(s))")
    return _owner_status(a)


def _owner_status(a=None) -> dict:
    import owner_controls as _owner
    a = a if a is not None else app
    out = _owner.snapshot()
    try:
        import owner_voice
        out["voice"] = owner_voice.voice_status(getattr(a.state, "OwnerAdopter", None))
    except Exception:  # noqa: BLE001
        out["voice"] = None
    mgr = getattr(a.state, "SpeechPipelineManager", None)
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
    if isinstance(body, dict) and "owner_voice" in body:
        field = body["owner_voice"]
        ok, out, err = await _actions(request.app).run("owner_voice",
                                                       lambda: _owner_voice_action(request.app, field))
        return out if ok else JSONResponse({"error": err or "failed"}, status_code=500)
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
    """Point the bridge diarizer's self-profile at the current active voice. Called after every agent or voice switch
    (SpeechPipelineManager.on_voice_changed), on the worker thread that ran the switch."""
    bridge = app.state.CallBridge
    if bridge is None:
        return
    from voices import voice_wav
    voice = getattr(app.state.SpeechPipelineManager.audio, "current_voice", None)
    wav = voice_wav(voice)
    if not wav:      # never hand the diarizer None: that would erase the agent's own voice profile
        logger.warning(f"🖥️⚠️ no reference audio for voice {voice!r}; diarizer self-reference unchanged")
        return
    try:
        bridge.update_self_reference(wav)
        logger.info(f"🖥️🎙️ diarizer self-reference -> {voice}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🖥️⚠️ self-reference update failed: {e}")


# ---- slow dashboard actions, off the event loop (owner 11-04: an agent click froze the dashboard) ----
# Agent switch, voice switch, clear history and audio devices are REST endpoints for the dashboard; the same /ws
# messages stay for older pages and tools. Both run through dash_actions.ActionGroups: one action at a time per group
# on a worker thread; a newer agent switch replaces one still waiting.
from dash_actions import ActionFailed, REPLACED  # noqa: E402

_DASH_MESSAGES = {"set_persona", "set_voice", "clear_history", "list_devices", "set_audio_devices"}
_OWNER_MESSAGES = {"set_persona", "set_voice", "clear_history", "set_audio_devices"}
_WS_TASKS: set = set()


def _actions(app: FastAPI):
    """The app's ActionGroups, created on first use on the running event loop."""
    acts = getattr(app.state, "dash_actions", None)
    if acts is None:
        from dash_actions import ActionGroups
        acts = app.state.dash_actions = ActionGroups()
    return acts


def _current(spm) -> dict:
    return {"persona": spm.current_persona, "voice": getattr(spm.audio, "current_voice", None)}


def _hub(app: FastAPI):
    h = getattr(app.state, "headless", None)
    return h["hub"] if h else None


def _switch_work(spm, kind: str, name: str):
    """The work of an agent or voice switch (runs on a worker thread). Returns the agent list for the other dashboards,
    or None when the agent was already current (nothing changed)."""
    def work():
        if kind == "agent":
            if name == spm.current_persona:
                return None
            ok = spm.set_persona(name)
        else:
            ok = spm.set_voice(name)
        logger.info(f"🖥️🎭 {kind} switch '{name}' -> {'ok' if ok else 'failed'}")
        if not ok:
            raise ActionFailed(f"couldn't switch to {name}" if kind == "agent" else f"couldn't switch to voice {name}")
        return spm.personas_info()
    return work


def _reroute_work(app: FastAPI, inp, out):
    def work():
        bridge = app.state.CallBridge
        if bridge is None:
            logger.warning("🖥️⚠️ audio device change ignored: no CallBridge (browser mode)")
        else:
            bridge.reconfigure(input_device=inp, output_device=out)
        return _audio_devices_info(app)
    return work


async def _json_body(request: Request) -> dict:
    try:
        d = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return d if isinstance(d, dict) else {}


def _refuse() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "owner only"}, status_code=403)


async def _switch_endpoint(request: Request, kind: str, field: str):
    if not _is_owner(request):
        return _refuse()
    name = str((await _json_body(request)).get(field) or "").strip()
    if kind == "agent":
        name = name.lower()
    if not name:
        return JSONResponse({"ok": False, "error": f"{field} required"}, status_code=400)
    a = request.app
    spm = a.state.SpeechPipelineManager
    ok, info, err = await _actions(a).run("pipeline", _switch_work(spm, kind, name), replaceable=kind == "agent")
    cur = _current(spm)
    if not ok:
        return JSONResponse({"ok": False, "error": err, "current": cur}, status_code=409 if err == REPLACED else 500)
    hub = _hub(a)
    if hub is not None and isinstance(info, dict):
        hub.put_nowait({"type": "personas", **info})      # the other dashboards
    return {"ok": True, "current": cur}


@app.post("/api/active_agent")
async def api_active_agent(request: Request):
    return await _switch_endpoint(request, "agent", "persona")


@app.post("/api/active_voice")
async def api_active_voice(request: Request):
    return await _switch_endpoint(request, "voice", "voice")


@app.post("/api/call/end")
async def api_call_end(request: Request):
    """End the call in progress (the desktop app calls this before it stops the server). Runs in its own action
    group so it never waits behind an agent switch; the resets continue after the answer (design D2, S3)."""
    if not _is_owner(request):
        return _refuse()
    import call_boundary
    ok, res, err = await _actions(request.app).run("call", lambda: call_boundary.BOUNDARY.end("api"))
    if not ok:
        return JSONResponse({"ended": False, "start_pass": False, "error": err}, status_code=500)
    await asyncio.to_thread(call_boundary.WRITER.flush, 2.0)     # the end record is on disk before the answer
    start = await asyncio.to_thread(start_pass_at_quit, request.app)
    out = {"ended": bool(res.get("ended")), "start_pass": bool(start)}
    if start:
        out["pass_args"] = _llm_mode(request.app)["args"]
    return out


@app.post("/api/clear_history")
async def api_clear_history(request: Request):
    if not _is_owner(request):
        return _refuse()
    ok, _, err = await _actions(request.app).run("pipeline", request.app.state.SpeechPipelineManager.reset)
    return {"ok": True} if ok else JSONResponse({"ok": False, "error": err}, status_code=500)


@app.get("/api/audio_devices")
async def api_audio_devices(request: Request):
    a = request.app
    ok, info, err = await _actions(a).run("devices", lambda: _audio_devices_info(a))
    return {"ok": True, **info} if ok else JSONResponse({"ok": False, "error": err}, status_code=500)


@app.post("/api/audio_devices")
async def api_set_audio_devices(request: Request):
    if not _is_owner(request):
        return _refuse()
    body = await _json_body(request)
    a = request.app
    ok, info, err = await _actions(a).run("devices", _reroute_work(a, body.get("input"), body.get("output")))
    return {"ok": True, **info} if ok else JSONResponse({"ok": False, "error": err}, status_code=500)


def _ws_action(app: FastAPI, msg_type: str, data: dict, reply_queue: asyncio.Queue) -> None:
    """The /ws form of the same actions: run off the loop and reply as before, on the requester's queue; a successful
    switch goes on the hub instead in bridge mode, so every dashboard sees it."""
    spm = app.state.SpeechPipelineManager

    async def go():
        acts = _actions(app)
        if msg_type in ("set_persona", "set_voice"):
            kind = "agent" if msg_type == "set_persona" else "voice"
            name = str(data.get("persona" if kind == "agent" else "voice") or "").strip()
            name = name.lower() if kind == "agent" else name
            ok, info, err = await acts.run("pipeline", _switch_work(spm, kind, name), replaceable=kind == "agent")
            if not isinstance(info, dict):
                info = await asyncio.to_thread(spm.personas_info)
            msg = {"type": "personas", "kind": kind, "request": name, **info}
            if err:
                msg["error"] = err
            hub = _hub(app)
            (hub if ok and hub is not None else reply_queue).put_nowait(msg)
        elif msg_type == "clear_history":
            ok, _, err = await acts.run("pipeline", spm.reset)
            if not ok:
                reply_queue.put_nowait({"type": "action_error", "action": msg_type, "error": err})
        elif msg_type == "list_devices":
            ok, info, err = await acts.run("devices", lambda: _audio_devices_info(app))
            reply_queue.put_nowait(info if ok else {"type": "action_error", "action": msg_type, "error": err})
        elif msg_type == "set_audio_devices":
            ok, info, err = await acts.run("devices", _reroute_work(app, data.get("input"), data.get("output")))
            reply_queue.put_nowait(info if ok else {"type": "action_error", "action": msg_type, "error": err})

    task = asyncio.get_running_loop().create_task(go())
    _WS_TASKS.add(task)
    task.add_done_callback(_WS_TASKS.discard)


async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks',
                                reply_queue: asyncio.Queue | None = None) -> None:
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
                elif msg_type in _DASH_MESSAGES:
                    if msg_type in _OWNER_MESSAGES and not _is_owner(ws):
                        await ws.send_json({"type": "action_error", "action": msg_type, "error": "owner only"})
                    else:
                        _ws_action(app, msg_type, data, reply_queue or callbacks.message_queue)
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    app.state.turn_speed = int(speed_value)
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        logger.info(f"🖥️⚙️ Updated turn detection settings to factor: {speed_factor:.2f}")
                elif msg_type == "list_personas":     # to the requester only (in bridge mode the hub reaches everyone)
                    (reply_queue or callbacks.message_queue).put_nowait({
                        "type": "personas",
                        **app.state.SpeechPipelineManager.personas_info(),
                    })


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

IDLE_POLL_S = 0.005


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

            # Use connection-specific state via callbacks. Nothing to send: check every 5 ms, not every 1 ms (a
            # thousand wakeups a second on the event loop, each taking the GIL, between replies; 11-04). At most 5 ms
            # between the turn's release and its first chunk going out, which usually comes hundreds of ms later.
            if not callbacks.tts_to_client:
                await asyncio.sleep(IDLE_POLL_S)
                log_status()
                continue

            _g = app.state.SpeechPipelineManager.running_generation
            if not _g:
                await asyncio.sleep(IDLE_POLL_S)
                log_status()
                continue

            if _g.abortion_started or getattr(_g, 'resume_cancelled', False):
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
                with app.state.SpeechPipelineManager.__dict__.setdefault('_request_lock', threading.RLock()):
                    if (app.state.SpeechPipelineManager.running_generation is gen
                            and not getattr(gen, 'resume_cancelled', False)):
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
                    with app.state.SpeechPipelineManager.__dict__.setdefault('_request_lock', threading.RLock()):
                        if (app.state.SpeechPipelineManager.running_generation is not _g
                                or getattr(_g, 'resume_cancelled', False)):
                            continue
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

                    with app.state.SpeechPipelineManager.__dict__.setdefault('_request_lock', threading.RLock()):
                        if (app.state.SpeechPipelineManager.running_generation is _g
                                and not getattr(_g, 'resume_cancelled', False)):
                            app.state.SpeechPipelineManager.running_generation = None
                            callbacks.tts_chunk_sent = False
                            callbacks.reset_state()

                await asyncio.sleep(0.001)
                log_status()
                continue

            # Route synthesized audio to VAIO3 (into the call) in addition to the
            # browser monitor. AEC uses this same audio as its reference.
            with app.state.SpeechPipelineManager.__dict__.setdefault('_request_lock', threading.RLock()):
                if (app.state.SpeechPipelineManager.running_generation is not _g
                        or getattr(_g, 'resume_cancelled', False)):
                    continue
                if app.state.CallBridge is not None:
                    try:
                        app.state.CallBridge.play_tts(chunk)
                        app.state.SpeechPipelineManager.note_audio_committed(_g)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"🖥️⚠️ CallBridge.play_tts error: {e}")

                if not ENABLE_CALL_BRIDGE:
                    # Bridge mode routes TTS to VAIO3 only (play_tts above).
                    base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
                    message_queue.put_nowait({
                        "type": "tts_chunk",
                        "content": base64_chunk
                    })
                    app.state.SpeechPipelineManager.note_audio_committed(_g)
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

def reply_in_flight(gen, bridge=None) -> bool:
    """True once a reply is committed to audio (allowed to play, quick answer out,
    or the call bridge is speaking). Only barge-in may stop those (live 10-06)."""
    try:
        if getattr(gen, "quick_answer_provided", False):
            return True
        ev = getattr(gen, "tts_quick_allowed_event", None)
        if ev is not None and ev.is_set():
            return True
        if bridge is not None and bridge.is_speaking():
            return True
    except Exception:  # noqa: BLE001
        return False
    return False

# One diarizer serves the app; calls come from the STT callbacks and (think while listening) the
# thinker's label lookup, so every diarizer call takes this lock.
_DIARIZER_LOCK = threading.Lock()


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
        try:   # think while listening: no thinking while the agent's audio plays through the bridge
            self.app.state.SpeechPipelineManager.think_busy_fn = self._agent_playing
        except Exception:  # noqa: BLE001
            pass

    def _agent_playing(self) -> bool:
        bridge = getattr(self.app.state, "CallBridge", None)
        try:
            return bool(bridge is not None and bridge.is_speaking())
        except Exception:  # noqa: BLE001
            return False


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
                text = self.abort_text          # read once: on_partial may replace it meanwhile
                if self.last_abort_text != text:
                    self.last_abort_text = text
                    if not re.search(r"\w", text or ""):
                        # A blank partial (a recognizer restarting on a new stretch of audio, e.g.
                        # Moonshine re-decoding a noise-only buffer) carries no words: it is not new
                        # speech and must not abort the reply (live 11-02: it cut the agent off mid-turn).
                        continue
                    logger.debug(f"🖥️🧠 Abort check triggered by partial: '{text}'")
                    mgr = self.app.state.SpeechPipelineManager
                    spk = getattr(self, "live_speaker", None)
                    act = getattr(mgr, "threads", None) and mgr.threads.active
                    if act is not None and spk and spk != act.speaker and text:
                        try:
                            if not mgr.threads.should_preempt(
                                    mgr.threads.note(f"[{spk}] {text}", mgr.floor.active_partner())):
                                continue
                        except Exception:  # noqa: BLE001
                            pass
                    if self._backchannel_over_reply(mgr, text):
                        continue
                    if self._reaction_over_named_draft(mgr, text):
                        continue
                    # The line kept going past the draft this cancels. If the turn ends before another
                    # draft starts, answer it from the final transcript (_answer_held_fragment, gated).
                    # Live 11-03: "All right." was drafted, cancelled by "...what do you got?", and
                    # the question was never answered. Marked BEFORE the abort: the abort runs here,
                    # synchronously (~0.5 s while the LLM stops), and the turn can end meanwhile (live
                    # 11-04: "Did you feel that with this woman?" committed mid-abort, never answered).
                    prev_held = getattr(self, "held_fragment", "")     # unset until the first held line
                    self.held_fragment = text
                    if not mgr.check_abort(text, False, "on_partial") and self.held_fragment == text:
                        self.held_fragment = prev_held     # nothing was cancelled

    @staticmethod
    def _more_audio_coming(mgr) -> bool:
        """Is more of the current reply still being generated or synthesized? With streaming
        TTS the playout buffer often holds under 2s mid-reply, so "about to finish" read from
        it alone let the agent talk over real takeovers."""
        gen = getattr(mgr, "running_generation", None)
        if gen is None or getattr(gen, "abortion_started", False):
            return False
        if not getattr(gen, "llm_finished", True):
            return True
        if getattr(gen, "tts_quick_started", False) and not getattr(gen, "audio_quick_finished", True):
            return True
        if getattr(gen, "tts_final_started", False) and not getattr(gen, "audio_final_finished", True):
            return True
        return bool(getattr(gen, "quick_answer_overhang", "")) and not getattr(gen, "audio_final_finished", True)

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
            if left is not None and TranscriptionCallbacks._more_audio_coming(mgr):
                left = None   # the playout buffer is not the end of her line: more is still being made
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

    _REACTION_WORDS = frozenset({"yes", "yeah", "yea", "yep", "yup", "no", "nah", "nope", "oh", "ok", "okay",
                                 "mm", "mhm", "hm", "hmm", "ah", "uh", "um", "lol", "haha", "right", "sure", "true"})

    def _reaction_over_named_draft(self, mgr, txt: str) -> bool:
        """Live: "Hey Max, can you hear me?" was aborted by another voice's "Yes." before
        its reply started, and the "Yes." was then held, so nobody was answered (119 named
        turns). A short reaction never aborts a draft answering someone who named the agent;
        a partial that grows into real words re-checks on the next change."""
        gen = getattr(mgr, "running_generation", None)
        if gen is None or getattr(gen, "abortion_started", False) or not txt or "?" in txt:
            return False
        from floor import names_agent, strip_label
        words = re.findall(r"[a-z']+", strip_label(txt).lower())
        if not words or len(words) > 3 or not all(w in self._REACTION_WORDS for w in words):
            return False
        names = tuple(getattr(getattr(mgr, "dynamics", None), "agent_names", ()) or ())
        if not names or not names_agent(getattr(gen, "text", "") or "", names):
            return False
        logger.info(f"🖥️👂 keeping the reply to a named turn over a reaction: '{txt[:40]}'")
        return True

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
        if any(ch.isalnum() for ch in txt or ""):
            try:   # speech activity stops a running memory pass at once (docs/designs/experience_memory.md, D3)
                import call_boundary
                call_boundary.activity("partial")
            except Exception:  # noqa: BLE001
                pass
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt # Update text used for abort check
        self.abort_request_event.set() # Signal the abort worker
        self._check_barge_in(txt)
        try:   # Live screen: judge a question to the agent while its last words come in (inert unless it's on)
            self.app.state.SpeechPipelineManager.prefetch_screen_question(txt)
        except Exception:  # noqa: BLE001
            pass
        try:   # think while listening: the thinker sees the partial (inert unless the plugin is on)
            _m = self.app.state.SpeechPipelineManager
            if getattr(_m, "thinker", None) is not None:
                _m.thinker.hear(txt, self._live_speaker)
        except Exception:  # noqa: BLE001
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
            # Only an UNSPOKEN draft is retired. A reply already committed to
            # audio belongs to an earlier turn: a "Yeah." / side remark over it
            # must not cut it off (live 10-06: 'Yeah.' -> HOLD(acknowledgement)
            # aborted Max mid-sentence; barge-in had already said talk over).
            run = mgr.running_generation
            if run is not None and not reply_in_flight(run, getattr(getattr(getattr(self, "app", None), "state", None), "CallBridge", None)):
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
        if bridge is not None and bridge.is_speaking() and not _only_filler_playing(mgr):
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
            with _DIARIZER_LOCK:
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
        self._turn_meta = {"dur": rec.get("dur_s"), "audible": ov}
        # Dev builds only: the turn's voiceprint, so diarization can be replayed
        # and tuned on real calls (biometric -> never in public recordings). The owner's turns only (owner 11-04:
        # "no permanent voiceprints" for friends): a label adopted as the owner by owner_voice.
        try:
            import user_settings as _us
            adopter = getattr(self.app.state, "OwnerAdopter", None)
            owners_turn = adopter is not None and bool(speaker) and adopter.is_adopted(speaker)
            if (owners_turn and not _us.is_public()
                    and os.environ.get("ATLAS_LOG_VOICEPRINTS", "1") != "0"):
                bridge = getattr(self.app.state, "CallBridge", None)
                dz = getattr(self.app.state, "Diarizer", None) or getattr(bridge, "diarizer", None)
                vp = getattr(self, "_turn_vp", None)
                if vp is None:
                    vp = getattr(dz, "last_embedding", None)
                if vp is not None:
                    rec["vp"] = [round(float(x), 4) for x in vp]
        except Exception:  # noqa: BLE001
            pass
        _rec_event("pace", **rec)

    def _tag_speaker(self, audio) -> str | None:
        """Diarize one turn using the app-level Diarizer (browser or bridge)."""
        self._turn_vp = None
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
            with _DIARIZER_LOCK:
                spk = diarizer.process(audio, 16000, **kw).speaker
                vp = getattr(diarizer, "last_embedding", None)
                # this turn's voiceprint, for owner recognition and the dev-build record (read under the lock: the
                # next turn's process() replaces last_embedding)
                self._turn_vp = vp.copy() if vp is not None else None
            try:   # fragments folded into a person keep any name learned for them
                merges = diarizer.drain_merges() if hasattr(diarizer, "drain_merges") else []
                if merges:
                    book = self.app.state.SpeechPipelineManager.people
                    adopter = getattr(self.app.state, "OwnerAdopter", None)
                    for old, new in merges:
                        book.merge(old, new)
                        if adopter is not None:
                            adopter.merge(old, new)
                        _owner_event("merge", old=old, new=new)
                        _merge_said_facts(self.app.state.SpeechPipelineManager, adopter, old, new)
                        logger.info(f"🖥️👥 {old} folded into {new}")
            except Exception as e:  # noqa: BLE001
                logger.debug(f"merge carry-over skipped: {e}")
            return spk
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
        try:   # think while listening: this utterance is committed; the next partial starts a new one
            _tm = self.app.state.SpeechPipelineManager
            if getattr(_tm, "thinker", None) is not None:
                _tm.thinker.end()
        except Exception:  # noqa: BLE001
            pass
        self.user_history_committed = False
        self.user_interrupted = False # Reset connection-specific flag (user finished, not interrupted)
        # Access global manager state
        # Snapshot once: another thread may clear running_generation between
        # the validity check and the use (live 10-01 AttributeError on None).
        gen = self.app.state.SpeechPipelineManager.running_generation
        if gen is not None and self._drop_stale_draft(gen):
            gen = None
        elif gen is not None and getattr(gen, "abortion_started", False) and not getattr(self, "held_fragment", ""):
            # The turn ended while its draft was being cancelled: nothing else will answer it, so the final
            # transcript is (gated, in _answer_held_fragment). A turn is never dropped.
            self.held_fragment = self.final_transcription or self.partial_transcription or ""
        __import__("turn_trace").turn_end(getattr(gen, "id", None))
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

        if not (user_request_content or "").strip():
            # Live: the turn often ends before its final transcript exists (no realtime
            # text yet), and committing '' lost 1,917 turns, 901 never answered ("Hey
            # Pup, can you hear me alright?" 10-31). Commit the final text when it
            # lands (on_final) and answer it from there.
            self._deferred_turn = (audio, speaker, time.time())
            self._deferred_vp = getattr(self, "_turn_vp", None)
            logger.info("🖥️⏳ turn ended before its text; committing it when the final transcript lands")
            return
        self._commit_turn(audio, speaker, user_request_content)

    def _commit_turn(self, audio, speaker, user_request_content: str) -> None:
        """Send the finished user turn to the UI and commit it to history and the room state."""
        if speaker != "self":   # a human turn starts or continues the call (docs/designs/experience_memory.md, D2);
            try:                # the agent's own echoed voice does not
                import call_boundary
                call_boundary.BOUNDARY.turn()
            except Exception as e:  # noqa: BLE001
                logger.warning("call boundary turn failed: %s", e)
        self._observe_owner(speaker)
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
                ("floor", lambda: mgr.floor.on_user_turn(speaker, user_request_content)),
                ("convo_log", lambda: mgr.convo.add_user(speaker, user_request_content)),
                ("name_learning", lambda: mgr.people.learn(speaker, user_request_content)),
                ("name_record", lambda: _rec_name(speaker, mgr)),
                ("room_vote", lambda: _passive_vote(speaker, user_request_content, mgr)),
                ("fact_check", lambda: _factcheck_heard(speaker, user_request_content, mgr)),
                ("translate", lambda: _translate_heard(speaker, user_request_content, mgr)),
                ("live_screen", lambda: _live_screen_heard(speaker, user_request_content, mgr)),
                ("pacing", lambda: self._pace_turn(speaker, user_request_content, audio, mgr)),
                ("memory_line", lambda: self._memory_line(speaker, user_request_content, mgr)),
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

    def _memory_line(self, speaker, text, mgr) -> None:
        """The committed turn goes to the call's pending file (docs/designs/experience_memory.md, D2, S1). A turn the
        diarizer couldn't place yet (no label) still triggers off the record (D7: anyone can) and is written without a
        label (the pass shows it as "someone", never as one of the call's people)."""
        if speaker == "self":
            return
        import call_boundary
        adopter = getattr(self.app.state, "OwnerAdopter", None)
        meta = getattr(self, "_turn_meta", None) or {}
        self._turn_meta = None
        agent = getattr(mgr, "current_persona", "") or ""
        owner = bool(speaker and adopter is not None and adopter.is_adopted(speaker))
        import memory_redact as _MR
        agents = tuple(getattr(getattr(mgr, "people", None), "agent_names", ()) or ())
        kind = _MR.off_record_kind(text, agents)
        if kind:   # off the record (docs/designs/experience_memory.md, D7)
            call_boundary.apply_off_record(kind)
        marked = kind in ("span", "back") or call_boundary.OFF.is_open()
        if not marked and speaker:
            try:
                self._remember_that(speaker, text, meta, mgr, adopter)
            except Exception as e:  # noqa: BLE001
                logger.warning("remember that failed: %s", e)
        if owner:
            try:
                mgr.owner_spoke()
            except Exception as e:  # noqa: BLE001
                logger.debug("owner presence not noted: %s", e)
        call_boundary.WRITER.turn(
            label=speaker or None, agent=agent, ident=_agent_ident(agent),
            owner=owner,
            score=(adopter.last_score(speaker) if adopter is not None and speaker else None),
            dur=meta.get("dur"), audible=meta.get("audible"), text=text, write=not marked)

    def _remember_that(self, speaker, text, meta, mgr, adopter) -> None:
        """'Remember that ...' at commit (D7): outcomes 1 to 5. The cue was already chosen from the draft."""
        import memory_redact as _MR
        agents = tuple(getattr(getattr(mgr, "people", None), "agent_names", ()) or ())
        clause = _MR.remember_clause(text, agents)
        if not clause or adopter is None:
            return                                         # not a request, or browser mode (outcome 4)
        import owner_voice as _ov
        score = adopter.last_score(speaker)
        dur, audible = meta.get("dur"), meta.get("audible")
        qualifying = bool(dur is not None and dur >= _ov.QUALIFY_MIN_S and audible is False
                          and score is not None and score >= _ov.TURN_SIM)
        item = {"clause": clause, "t": time.time(), "qualifying": qualifying}
        if adopter.is_adopted(speaker):
            _store_said_fact(mgr, adopter, speaker, item)
        elif not adopter.labels() and adopter.has_voice():
            mgr.held_facts.setdefault(speaker, []).append(item)        # outcome 3: wait for the label's adoption
            logger.info("remember-that held until %s is recognized", speaker)

    def _observe_owner(self, speaker) -> None:
        """Owner voice recognition (owner_voice.py): every committed, labeled turn, after both echo checks and before
        its dashboard line, so the adopting turn already shows the owner's name. Never blocks a turn."""
        try:
            adopter = getattr(self.app.state, "OwnerAdopter", None)
            if adopter is None or not speaker or speaker == "self":
                return
            if adopter.observe(speaker, getattr(self, "_turn_vp", None)):
                logger.info(f"👤 {speaker} recognized as the owner by voice")
                self.app.state.SpeechPipelineManager.people.owner_adopted(speaker)
                _owner_event("adopt", label=speaker, how="auto")
                _release_held_facts(self.app.state.SpeechPipelineManager, adopter, speaker)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"owner voice check skipped: {e}")

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

    def _reply_in_flight(self, gen) -> bool:
        return reply_in_flight(gen, getattr(getattr(getattr(self, "app", None), "state", None), "CallBridge", None))

    MONOLOGUE_S = float(os.environ.get("ATLAS_MONOLOGUE_S", "6.0"))
    DEFERRED_TURN_S = 15.0   # a final transcript later than this belongs to no turn

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
                    and not reply_in_flight(run, getattr(getattr(getattr(self, "app", None), "state", None), "CallBridge", None))):
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
        try:   # think while listening: the utterance is over
            _tm = self.app.state.SpeechPipelineManager
            if getattr(_tm, "thinker", None) is not None:
                _tm.thinker.end()
        except Exception:  # noqa: BLE001
            pass
        if not self.final_transcription: # Store it if not already set by on_before_final logic
             self.final_transcription = txt
        deferred, self._deferred_turn = getattr(self, "_deferred_turn", None), None
        if deferred is None:
            return
        audio, speaker, ended_at = deferred
        text = (txt or "").strip()
        if not text or time.time() - ended_at > self.DEFERRED_TURN_S:
            self.user_history_committed = True
            return
        if speaker != "self" and self._text_echo(text):
            logger.info(f"🖥️🙉 own words echoed back (voiceprint said {speaker}) -> self")
            speaker = "self"
        logger.info(f"🖥️⏳ committing the final transcript of a turn that ended early: '{text[-60:]}'")
        if speaker != "self":
            self.held_fragment = text   # no draft exists for it: answer from the final (gated)
        self._turn_vp = getattr(self, "_deferred_vp", None)
        self._commit_turn(audio, speaker, text)

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
        try:   # Live screen: no watcher look between the end of speech and the reply
            import live_screen as _live_screen
            _live_screen.on_silence(silence_active)
        except Exception:  # noqa: BLE001
            pass

    def on_partial_assistant_text(self, txt: str):
        """
        Callback invoked when a partial text result from the assistant (LLM) is available.

        Updates the internal assistant answer state and sends the partial answer to the client,
        unless the user has interrupted.

        Args:
            txt: The partial assistant text.
        """
        logger.info(f"{Colors.apply('🖥️💬 PARTIAL ASSISTANT ANSWER: ').green}{txt}")
        try:   # Live screen: the reply has started, watcher looks may run again
            import live_screen as _live_screen
            _live_screen.on_reply_text()
        except Exception:  # noqa: BLE001
            pass
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
        if self._text_echo(txt):
            # Live: a takeover fired on the agent's own line ("Start with the data you
            # already have") and cut 2.8s of its audio; the echo check never ran here.
            logger.info(f"🖥️🙉 not a barge-in, own words echoed: '{txt[:40]}'")
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
        interrupted_generation = self.app.state.SpeechPipelineManager.running_generation
        if True:
            self.tts_to_client = False # Stop server sending TTS
            self.user_interrupted = True # Mark connection as user interrupted
            logger.info(f"{Colors.apply('🖥️❗ INTERRUPTING TTS due to recording start').blue}")

            # Send final assistant answer *if* one was generated and not sent
            logger.info(Colors.apply("🖥️✅ Sending final assistant answer (forced on interruption)").pink)
            if interrupted_generation is not None:
                self.send_final_assistant_answer(forced=True, generation=interrupted_generation)

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

    def send_final_assistant_answer(self, forced=False, *, generation=None):
        """Commit only the captured live owner, for normal and forced callers alike."""
        mgr = self.app.state.SpeechPipelineManager
        with mgr.__dict__.setdefault('_request_lock', threading.RLock()):
            owner = mgr.running_generation
            if owner is None or (generation is not None and owner is not generation):
                return False
            if getattr(owner, 'resume_cancelled', False):
                self._cut_said = None
                return False
            if not forced and getattr(owner, 'abortion_started', False):
                return False
            return self._commit_final_assistant_answer(owner, forced=forced)

    def _commit_final_assistant_answer(self, generation, forced=False):
        """
        Sends the final (or best available) assistant answer to the client.

        Constructs the full answer from quick and final parts if available.
        A forced answer uses this generation's available partial buffers.
        Cleans the text and sends it as 'final_assistant_answer' if not already sent.

        Args:
            forced: If True, attempts to send the last partial answer if no complete
                    final answer is available. Defaults to False.
        """
        final_answer = generation.quick_answer + generation.final_answer

        if not final_answer: # Check if constructed answer is empty
            # A connection-level partial may belong to an older generation.
            logger.warning(f"🖥️⚠️ Final assistant answer was empty, not sending.")
            return

        logger.debug(f"🖥️✅ Attempting to send final answer: '{final_answer}' (Sent previously: {self.final_assistant_answer_sent})")

        if not getattr(generation, 'final_answer_committed', False) and final_answer:
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
                target = generation.decision.target if generation else ''
                # Store the EXACT control header the model must emit. The old '[to=S1]' form
                # was copied by the model as its own header -> INVALID -> silent turns,
                # increasingly often as the call grew (tests/sim_history_format.py).
                app.state.SpeechPipelineManager.floor.on_agent_spoke(target, cleaned_answer)
                try:   # "show me" / "I'd need to see it" -> it glances (owner 11-01)
                    app.state.SpeechPipelineManager.note_reply_for_glance(target, cleaned_answer, generation)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"glance note failed: {e}")
                try:   # "I'll look into it" with no search -> search for real (live 11-01)
                    app.state.SpeechPipelineManager.note_reply_for_promise(target, cleaned_answer, generation)
                except Exception as e:  # noqa: BLE001
                    logger.debug(f"promise keeper failed: {e}")
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
                        app.state.SpeechPipelineManager.people.note_agent_reply(target, safe_answer)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("name-ask tracking failed: %s", e)
                    self.app.state.SpeechPipelineManager.history.append({"role": "assistant", "content": history_answer})
                    app.state.SpeechPipelineManager.trim_history()
                    try:   # the spoken reply: pending file, and its memory snippets count as used (design D2, D4)
                        _spm = app.state.SpeechPipelineManager
                        _spm.reply_spoken(generation, safe_answer)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("spoken reply bookkeeping failed: %s", e)
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
                generation.final_answer_committed = True
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
MONO_TAIL_S = float(os.environ.get("ATLAS_MONO_TAIL_S", "2.0"))


def start_monologue_tail(mgr, callbacks, bridge, now: float):
    """A monologue's last line was held; the speaker has now really stopped.
    Answer it if the normal gate (engagement, passive, pacing) lets it through."""
    floor = mgr.floor
    if not getattr(floor, "mono_pending", None) or not delivery_ready(mgr, callbacks, bridge, now, MONO_TAIL_S):
        return None
    got = floor.take_monologue_pending()
    if not got:
        return None
    speaker, text = got
    if not text.lstrip().startswith("["):
        text = f"[{speaker}] {text}"
    why = floor.turn_gate(text, speaker, tuple(mgr.dynamics.agent_names))
    if why:
        logger.info(f"🖥️🎙️ monologue ended; still holding ({why}): '{text[-50:]}'")
        return False
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True
    logger.info(f"🖥️🎙️➡️ monologue ended; answering: '{text[-60:]}'")
    mgr.prepare_generation(text)
    return True


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
    mgr.prepare_generation(cut.cue(), resume_cut=cut)
    return True


GLANCE_GAP_S = float(os.environ.get("ATLAS_GLANCE_GAP_S", "2.0"))


def start_glance(mgr, callbacks, bridge, now: float):
    """It said "show me" and nobody answered: glance at the screen and react (owner
    11-01: agents look when they want to see). None = nothing pending."""
    p = getattr(mgr, "glance_pending", None)
    if not p:
        return None
    import speech_pipeline_manager as _spm
    if now - p["at"] > _spm.GLANCE_ANSWER_S or getattr(callbacks, "last_voice_at", 0.0) > p["end"]:
        mgr.glance_pending = None      # too late, or someone spoke: their line had its chance
        return None
    if now < p["end"] + GLANCE_GAP_S or not delivery_ready(mgr, callbacks, bridge, now, GLANCE_GAP_S):
        return False
    mgr.glance_pending = None
    import capability as _cap
    import screen as _screen
    if not _screen.status().get("enabled") or not _screen.can_glance(_cap.glance_mode()):
        return False
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True   # no user line: a HOLD retires cleanly
    logger.info(f"🖥️👁️ it wanted to see and nobody answered: glancing ({p['who'] or 'room'})")
    mgr.prepare_generation(_cap.glance_cue(p["who"]))
    return True


def _only_filler_playing(mgr) -> bool:
    """The audio out is just a "give me a sec" clip, no reply started yet: someone finishing their
    question is not talking over her (live 11-02: "Can you search for the VPN apps on the system?"
    landed during the clip, was dropped as "finishing her line", and never got an answer)."""
    try:
        if time.monotonic() >= float(getattr(mgr, "filler_until", 0.0) or 0.0):
            return False
        gen = getattr(mgr, "running_generation", None)
        return not (gen is not None and getattr(gen, "quick_answer", ""))
    except Exception:  # noqa: BLE001
        return False


def start_delivery(mgr, callbacks) -> Optional[int]:
    """Launch the cue generation for the oldest untold result (None if nothing)."""
    import plugins
    if not plugins.usable("web_search"):
        return None
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
    bg = (s.get("background") or "on") == "on" and _plugins.usable("web_search")
    factcheck.observe(text, mgr.people.name_of(speaker) or speaker or "",
                      gap_s=float(s.get("gap") or factcheck.DEFAULT_GAP_S),
                      daily=int(s.get("daily") or factcheck.DEFAULT_DAILY), background=bg)


def _live_screen_heard(speaker, text: str, mgr) -> None:
    """Live screen: every committed line someone says (never the agent's own) is checked for "forget what you
    saw" before any reply decision (live_screen.hear)."""
    import live_screen as _live_screen
    _live_screen.hear(text, (mgr.people.name_of(speaker) if speaker else "") or "")


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


TIMER_GAP_S = float(os.environ.get("ATLAS_TIMER_GAP_S", "1.0"))
TIMER_OVERDUE_GAP_S = 0.5    # once a reminder is 20 s late, any short pause will do


def start_tip(mgr, callbacks, bridge, now: float) -> bool:
    """Live screen tips (live_screen/tips.py): a prepared tip, at a real gap in the talk, said as a cue turn."""
    try:
        import live_screen as _live_screen
        from live_screen import tips as _tips
        if not _live_screen.enabled() or not _tips.ready(now):
            return False
    except Exception as e:  # noqa: BLE001
        logger.debug(f"tip check failed: {e}")
        return False
    if not delivery_ready(mgr, callbacks, bridge, now):
        return False
    cue = _live_screen.start_tip_cue(now)
    if not cue:
        return False
    callbacks.reset_state()
    callbacks.tts_to_client = True
    callbacks.user_finished_turn = True
    callbacks.user_history_committed = True   # no user line: a HOLD retires cleanly
    logger.info(f"🖥️💡 speaking a tip: {cue[:90]!r}")
    mgr.prepare_generation(cue)
    return True


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
    try:
        import turndetect as _td
        _td.EXTRA_WAIT_HOOK = lambda: app.state.SpeechPipelineManager.floor.monologue_extra_wait()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"🖥️🎙️ monologue wait hook: {e}")
    while True:
        await asyncio.sleep(0.5)
        try:   # Live screen: start/stop the capture process to match the switches (off the audio loop)
            import live_screen as _live_screen
            await asyncio.to_thread(_live_screen.reconcile)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"live screen reconcile: {e}")
        try:
            mgr = app.state.SpeechPipelineManager
            bridge = getattr(app.state, "CallBridge", None)
            now = time.time()
            if start_timer(mgr, callbacks, bridge, now) is not None:
                continue
            if start_monologue_tail(mgr, callbacks, bridge, now) is not None:
                continue
            if delivery_ready(mgr, callbacks, bridge, now, RESUME_GAP_S) and start_resume(mgr, callbacks):
                continue
            if start_glance(mgr, callbacks, bridge, now):
                continue
            if start_tip(mgr, callbacks, bridge, now):
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
        asyncio.create_task(process_incoming_data(ws, app, asyncio.Queue(), h["callbacks"], reply_queue=q)),
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
