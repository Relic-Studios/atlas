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
            "someone asks you to look at, check, read, or react to what's on the screen "
            "(an error, a game, a picture, a video, a website). Don't call it otherwise."
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
}


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


def capture(reason: str = "", grab=None) -> dict:
    """Execute the tool. Returns {"text": ..., "images": [b64]} or {"text": refusal}."""
    with _lock:
        enabled = _state["enabled"]
    if not enabled:
        logger.info("👁️ screen look refused (tool disabled by owner)")
        return {"text": "Screen looking is switched off by the owner right now. Say you can't see the screen at the moment."}
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
    ms = (time.perf_counter() - t0) * 1000
    logger.info("👁️ screen captured %sx%s (%d KB) in %.0f ms — reason: %s",
                size[0], size[1], len(jpeg) // 1024, ms, reason or "-")
    return {
        "text": ("Here is the screenshot of the screen you just took. Describe or react to what "
                 "you actually see in it; don't invent anything that isn't visible."),
        "images": [b64],
    }
