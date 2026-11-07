"""Live screen (docs/designs/live_screen.md): when the `live_screen` plugin is on, every look reads frames from a
capture process watching the owner's screen (or one program the owner picked) instead of taking a screenshot.

enabled()   the plugin is on, the Eyes plugin is on, its live toggle is on, and replies go to the remote vLLM server
reconcile() called every 0.5 s from the server's delivery loop (in a worker thread): starts or stops the capture
            process to match enabled(); ending a session clears everything it saw
look(plan)  the tool result for one look: frames (base64 JPEG) and the text the model is told, or "can't see"
Only in-memory state is read on the hot path; plugin settings are cached and refreshed on plugin changes.
"""
from __future__ import annotations

import logging
import threading
import time

from live_screen import config as CFG

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_mgr = None
_client = None
_status: dict = {}
_restarts: list = []
_given_up = False
_next_start = 0.0
_target_sent = None
_flags = {"loaded": False, "on": False, "settings": {}, "eyes": True, "monitor": 1}
_flags_lock = threading.Lock()


def bind(mgr) -> None:
    """The pipeline manager (its llm.remote tells which model server replies)."""
    global _mgr
    _mgr = mgr


def _read_flags():
    try:
        import plugins
        st = plugins._load()
        _flags.update(on=bool(plugins.is_enabled("live_screen", st)),
                      settings=dict(plugins.settings_of("live_screen", st) or {}),
                      eyes=bool(plugins.is_enabled("eyes", st)),
                      monitor=int((plugins.settings_of("eyes", st) or {}).get("monitor") or 1))
    except Exception as e:  # noqa: BLE001
        logger.debug("live screen flags: %s", e)
        _flags.update(on=False)


def _load_flags() -> None:
    with _flags_lock:
        if _flags["loaded"]:
            return
        _flags["loaded"] = True
        try:
            import plugins
            plugins.on_change(_read_flags)
        except Exception:  # noqa: BLE001
            pass
        _read_flags()


def settings() -> dict:
    _load_flags()
    return dict(_flags["settings"])


def target() -> dict:
    """What to watch: {"mode": "screen", "monitor": n} or {"mode": "app", "process": "game.exe"}."""
    _load_flags()
    watch = (_flags["settings"].get("watch") or "screen").strip()
    if watch and watch != "screen":
        return {"mode": "app", "process": watch.lower(), "monitor": _flags["monitor"]}
    return {"mode": "screen", "monitor": _flags["monitor"], "process": ""}


def _eyes_toggle() -> bool:
    try:
        import screen
        return bool(screen.status().get("enabled"))
    except Exception:  # noqa: BLE001
        return False


def _remote():
    return getattr(getattr(_mgr, "llm", None), "remote", None)


def _transport_ok() -> bool:
    rs = _remote()
    cfg = getattr(rs, "cfg", None) or {}
    return rs is not None and bool(cfg.get("vllm", True)) and not cfg.get("cloud")


def remote_up() -> bool:
    rs = _remote()
    try:
        return rs is not None and bool(rs._remote_ok())
    except Exception:  # noqa: BLE001
        return False


def enabled() -> bool:
    _load_flags()
    return bool(_flags["on"] and _flags["eyes"] and _eyes_toggle() and _transport_ok())


_scheduler = None


def _on_event(ev: dict) -> None:
    global _status
    if ev.get("type") == "status":
        _status = dict(ev.get("status") or {})
    elif ev.get("type") == "change" and _scheduler is not None:
        _scheduler.on_change(ev.get("t"))


def _request(op: str, **args):
    c = _client
    return c.request(op, **args) if c is not None and c.running else None


def _start_scheduler() -> None:
    global _scheduler
    if _scheduler is None:
        from live_screen.scheduler import Scheduler
        _scheduler = Scheduler(_request, lambda: dict(getattr(_remote(), "cfg", None) or {}), watching_label)
        _scheduler.start()


def _end_session(reason: str) -> None:
    global _client, _status, _given_up, _next_start, _target_sent, _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None
    if _client is not None:
        _client.stop()
        logger.info("👁️ live screen session ended (%s)", reason)
    _client, _status, _given_up, _next_start, _target_sent = None, {}, False, 0.0, None
    _restarts.clear()
    try:
        from live_screen import memory, tips, watcher
        memory.clear()
        watcher.reset()
        tips.reset()
    except Exception:  # noqa: BLE001
        pass


# Phase 2 signals and the screen log ---------------------------------------
def on_silence(active: bool) -> None:
    if _scheduler is not None:
        _scheduler.on_silence(active)


def on_reply_text() -> None:
    if _scheduler is not None:
        _scheduler.on_reply_text()


FORGET_RE = None


def hear(line: str, speaker: str = "") -> bool:
    """Every line someone in the call says (not the agent's own voice): "forget what you saw" drops the last two
    minutes of the screen log. "forget that" stays the long-term memory command."""
    global FORGET_RE
    if not enabled():
        return False
    try:                                              # Phase 3: "no tips", "quiet for 20 minutes", reactions
        from live_screen import tips
        did = tips.hear(line, speaker)
        if did:
            logger.info("💡 tips: %s (%s)", did, speaker or "someone")
    except Exception as e:  # noqa: BLE001
        logger.debug("tips hear failed: %s", e)
    import re
    if FORGET_RE is None:
        FORGET_RE = re.compile(r"\bforget\s+(?:what\s+you\s+(?:just\s+)?saw|what\s+was\s+on\s+(?:my|the)\s+screen|"
                               r"(?:my|the)\s+screen)\b", re.IGNORECASE)
    if not FORGET_RE.search(line or ""):
        return False
    from live_screen import memory
    n = memory.forget()
    logger.info("👁️ screen log: forgot the last %d entries (%s asked)", n, speaker or "someone")
    return True


def log_snapshot() -> list:
    from live_screen import memory
    return memory.snapshot()


def log_clear() -> None:
    from live_screen import memory
    memory.clear()


def reconcile(backend: str = "real") -> None:
    """Match the capture process to enabled(); restart after a crash with backoff, give up past the cap."""
    global _client, _given_up, _next_start, _target_sent
    with _lock:
        if not enabled():
            if _client is not None or _status or _restarts:
                _end_session("switched off")
            return
        now = time.time()
        if _client is not None and not _client.running:
            logger.warning("👁️ live screen capture process exited")
            _client.stop()
            _client, _target_sent = None, None
            _restarts.append(now)
            _restarts[:] = [t for t in _restarts if now - t < CFG.RESTART_WINDOW_S]
            if len(_restarts) > CFG.RESTART_CAP:
                _given_up = True
                logger.warning("👁️ live screen: capture keeps failing; off until Live screen is switched off and on")
            else:
                _next_start = now + CFG.RESTART_BACKOFF_S[min(len(_restarts) - 1, len(CFG.RESTART_BACKOFF_S) - 1)]
        if _client is None and not _given_up and now >= _next_start:
            from live_screen.sidecar_client import SidecarClient
            c = SidecarClient(on_event=_on_event, backend=backend)
            if c.start():
                _client = c
                logger.info("👁️ live screen capture started")
                _start_scheduler()
            else:
                _restarts.append(now)
                _next_start = now + CFG.RESTART_BACKOFF_S[-1]
        if _client is not None:
            tgt = target()
            if tgt != _target_sent and _client.request("set_target", **tgt) is not None:
                _target_sent = tgt


def shutdown() -> None:
    with _lock:
        _end_session("server shutting down")


def available() -> bool:
    return _client is not None and _client.running and _status.get("state") == "capturing"


def watching_label() -> str:
    """How the agent refers to what it watches."""
    name = (settings().get("name") or "").strip()
    owner = f"{name}'s screen" if name else "the owner's screen"
    t = target()
    if t["mode"] == "app":
        title = (_status.get("title") or "").strip()
        return f"{title or t['process']} on {owner}"
    return owner


def status() -> dict:
    """For the status line, telemetry and the Atlas window."""
    on = enabled()
    st = dict(_status)
    try:
        from live_screen import memory, tips
        remembered, tip_state = memory.size(), tips.status()
    except Exception:  # noqa: BLE001
        remembered, tip_state = 0, {}
    return {"on": on, "running": bool(_client is not None and _client.running), "state": st.get("state", "off"),
            "mode": target()["mode"] if on else "off", "title": st.get("title", ""), "watching": watching_label(),
            "given_up": _given_up, "remembered": remembered, "tips": tip_state}


def start_tip_cue(now: float) -> str:
    """The delivery loop asks: a prepared tip to speak now? Its cue line, or ''."""
    if not enabled():
        return ""
    from live_screen import tips
    return tips.claim(now) if tips.ready(now) else ""


def plan_look(line: str = "", reason: str = "", asked: bool = True):
    from live_screen.plan import plan_look as _plan
    return _plan(line, reason, asked)


def look(plan, speaker: str = "") -> dict:
    """The look_at_screen tool result for one look with Live screen on (never a screenshot). One set of eyes (owner,
    2026-11-03): Model itself by default, which gets the frames; with a separate watcher server, the watcher
    gets the frames with what was said, and the agent gets its report."""
    import base64
    from live_screen import plan as P
    from live_screen import watcher
    if not CFG.IMAGES:
        return P.cant_see(plan, "images")
    if not remote_up():
        return P.cant_see(plan, "offline")
    c = _client
    if c is None or not c.running:
        return P.cant_see(plan, "starting" if not _given_up else "error")
    res = c.look(plan.kind)
    if res is None:
        return P.cant_see(plan, "not_responding")
    if not res["ok"] or not res["images"]:
        return P.cant_see(plan, res.get("error") or "error")
    if not CFG.SEPARATE_WATCHER:
        return P.seen(plan, res, speaker, watching_label())
    jpegs = [base64.b64decode(b) for b in res["images"]]
    ages = [float(f.get("age_s") or 0.0) for f in res["frames"]]
    report = watcher.answer(jpegs, P.question_of(plan), res.get("title") or "", ages,
                            dict(getattr(_remote(), "cfg", None) or {}), glance=plan.glance)
    if not report:
        return P.cant_see(plan, "eyes_offline")
    return P.reported(plan, report, speaker, watching_label(), res.get("title") or "", res["frames"])


def latest_jpeg() -> bytes | None:
    """The newest frame (asked-look size) for the owner-only page; None when there is none."""
    c = _client
    if c is None or not c.running:
        return None
    got = c.request("look", kind="now")
    if not got or not got[0].get("ok") or not got[1]:
        return None
    return got[1][-1]


def end_call() -> None:
    """Call end (docs/designs/experience_memory.md, S4): what the agent saw and the tips it gave stay in that call."""
    try:
        from live_screen import memory, tips, watcher
        memory.clear()
        watcher.reset()
        tips.reset()
    except Exception:  # noqa: BLE001
        pass


try:
    import call_boundary as _cb
    _cb.register("live_screen.end_call", end_call, order=9)
except Exception:  # noqa: BLE001
    pass
