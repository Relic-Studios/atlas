"""Screen-look tool: the agent decides to look, we capture, the model sees pixels.

Native vision: local-model = the same Model weights + the Qwen3.8-27B
mmproj (vision tower + merger). The screenshot goes back to the model as an
image on the tool message, exactly like web_search returns text.

Budget: a 1024-px long side costs ~640 prompt tokens (measured), which fits the
4096 context next to persona + rules + room log. Larger shots would push the
room log out, so we downscale.

Safety: anyone in the call can ask the agent to look. The owner switch
(`set_enabled`, ATLAS "Eyes" toggle, or ATLAS_SCREEN_TOOL=0) is the kill switch,
every capture is logged, and the last capture is kept at private/last_screen.jpg
so the owner can see exactly what was looked at.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_SIDE = int(os.environ.get("ATLAS_SCREEN_MAX_SIDE", "1600"))
# Area budget = 1024x576 worth of pixels (~640 tokens, measured). Area, not long
# side, so ultrawide monitors keep readable text instead of a 1024x428 sliver.
MAX_PIXELS = int(os.environ.get("ATLAS_SCREEN_MAX_PIXELS", str(1024 * 576)))
# NOT under static/: the server binds 0.0.0.0, so anything in static/ is readable
# from the LAN/ZeroTier. Served only via the loopback-guarded /api/screen/last.
LAST_PATH = Path(__file__).resolve().parent / "private" / "last_screen.jpg"

SCREEN_TOOL = {
    "type": "function",
    "function": {
        "name": "look_at_screen",
        "description": (
            "Take a screenshot of the owner's computer screen and look at it. Call this when "
            "someone says 'look!', shows you something, or asks you to check, read, or react to "
            "what's on screen (an error, a game, a picture, a video, a website), or when people are "
            "reacting to something on screen and you'd need to see it to join in. You may also "
            "glance on your own, now and then, when you'd like to see the person you're talking "
            "with (their face or reaction on camera) or something they mention about themselves "
            "(an outfit, a haircut, a pet, their setup). Not every turn, and never for things "
            "that only need to be heard."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "what you're looking for"},
            },
            "required": [],
        },
    },
}

_lock = threading.Lock()
_state = {
    "enabled": os.environ.get("ATLAS_SCREEN_TOOL", "1") != "0",
    "count": 0,
    "last_ts": 0.0,
    "last_reason": "",
    "last_size": None,
    "monitor": int(os.environ.get("ATLAS_SCREEN_MONITOR", "1")),  # mss: 1 = primary
    "glances": [],                   # times of looks nobody asked for (rate limit)
}

# Looks the agent takes on its own (owner 11-01: "when they feel like they need or
# want to see them"). Eyes plugin setting "glance": cooldown since the last look of
# any kind, and a cap per 10 minutes. Looks someone asks for are never limited.
GLANCE_LIMITS = {"sometimes": (90.0, 4), "often": (30.0, 10)}
GLANCE_WINDOW_S = 600.0


def can_glance(mode: str = "sometimes", now: float | None = None) -> bool:
    """Would a look nobody asked for be allowed right now?"""
    lim = GLANCE_LIMITS.get(mode)
    if lim is None:
        return False
    now = time.time() if now is None else now
    with _lock:
        if not _state["enabled"]:
            return False
        recent = [t for t in _state["glances"] if now - t < GLANCE_WINDOW_S]
        return now - _state["last_ts"] >= lim[0] and len(recent) < lim[1]


def set_enabled(on: bool) -> None:
    with _lock:
        _state["enabled"] = bool(on)
    logger.info("👁️ screen tool %s", "ENABLED" if on else "DISABLED")


def status() -> dict:
    with _lock:
        return dict(_state)


def _grab():
    """Return a PIL RGB image of the configured monitor."""
    from PIL import Image
    try:
        import mss
        with mss.mss() as s:
            mons = s.monitors
            idx = _state["monitor"] if 0 < _state["monitor"] < len(mons) else 1
            raw = s.grab(mons[idx])
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as e:  # noqa: BLE001 - fall back to PIL
        logger.debug("mss grab failed (%s); using ImageGrab", e)
        from PIL import ImageGrab
        return ImageGrab.grab().convert("RGB")


def encode(img, max_side: int = MAX_SIDE) -> tuple[str, bytes, tuple[int, int]]:
    """Downscale to max_side, JPEG-encode. Returns (b64, jpeg_bytes, size)."""
    from PIL import Image
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)), (MAX_PIXELS / float(w * h)) ** 0.5)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()
    return base64.b64encode(data).decode("ascii"), data, img.size


def capture(reason: str = "", grab=None, asked: bool = True, glance: str = "sometimes", speaker: str = "") -> dict:
    """Execute the tool. Returns {"text": ..., "images": [b64]} or {"text": refusal}.

    asked=False is a look the agent chose on its own: allowed by the Eyes "glance"
    setting and its rate limit, and answered as a glance rather than a request."""
    with _lock:
        enabled = _state["enabled"]
    if not enabled:
        logger.info("👁️ screen look refused (tool disabled by owner)")
        return {"text": "Screen looking is switched off by the owner right now. Say you can't see the screen at the moment."}
    if not asked:
        if glance not in GLANCE_LIMITS:
            logger.info("👁️ unasked look refused (glances off)")
            return {"text": "Right now you only look at the screen when someone asks you to. Carry on "
                            "with the conversation without it."}
        if not can_glance(glance):
            logger.info("👁️ unasked look skipped (looked %.0fs ago)", time.time() - _state["last_ts"])
            return {"text": "You looked at the screen a moment ago; don't look again yet. Carry on "
                            "with the conversation from what you heard."}
    t0 = time.perf_counter()
    try:
        img = (grab or _grab)()
        b64, jpeg, size = encode(img)
    except Exception as e:  # noqa: BLE001
        logger.warning("👁️ screen capture failed: %s", e)
        return {"text": f"Screenshot failed ({e}). Say you couldn't get a look."}
    try:
        LAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        LAST_PATH.write_bytes(jpeg)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not save last screen: %s", e)
    with _lock:
        _state["count"] += 1
        _state["last_ts"] = time.time()
        _state["last_reason"] = (reason or "")[:120]
        _state["last_size"] = list(size)
        if not asked:
            _state["glances"] = [t for t in _state["glances"] if _state["last_ts"] - t < GLANCE_WINDOW_S]
            _state["glances"].append(_state["last_ts"])
    ms = (time.perf_counter() - t0) * 1000
    logger.info("👁️ screen captured %sx%s (%d KB) in %.0f ms — %s: %s",
                size[0], size[1], len(jpeg) // 1024, ms, "reason" if asked else "glance (unasked)", reason or "-")
    # Which camera tile is theirs? The agent hears "S1", the screen shows handles (sim 11-01:
    # 3 of 4 agents said "you're not in it" with the speaker's tile right there).
    who = (f" The person who just spoke to you is {speaker}." if speaker else "") + (
        " If camera tiles are on screen, the one outlined in green is whoever is talking; if no "
        "tile is outlined or there are no tiles, you can't tell who they are on screen.")
    if not asked:
        return {
            "text": ("You glanced at the screen on your own; nobody asked you to. Use what you see only "
                     "if it fits what's being said, the way someone who looked over would: a quick, "
                     "natural reaction, not a description of the screenshot, and don't announce that "
                     "you looked. Don't invent anything that isn't visible: if the person or thing they "
                     "mentioned isn't in it, don't pretend you saw it (say you can't spot it, or just "
                     "answer what they said)." + who),
            "images": [b64],
            "glance": True,
        }
    return {
        "text": ("Here is the screenshot of the screen you just took. Describe or react to what "
                 "you actually see in it; don't invent anything that isn't visible. If what they "
                 "asked about (them on camera, a thing they're showing) isn't in it, say plainly "
                 "that you can't see it." + who),
        "images": [b64],
    }
