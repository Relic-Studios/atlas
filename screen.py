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
import re
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


# What a program on screen is, in plain words, where a screenshot is easy to misread.
_APP_KIND = {
    "discord": "the Discord call app; its big tiles with a round picture in the middle are people in "
               "the call, not game characters",
    "chrome": "a web browser", "msedge": "a web browser", "firefox": "a web browser", "brave": "a web browser",
    "steam": "the Steam game launcher", "steamwebhelper": "the Steam game launcher",
    "obs64": "OBS, streaming software", "code": "a code editor",
}


_CALL_CACHE = {"at": 0.0, "val": None}


def _window_titles(prog_substr: str) -> list[str]:
    """Titles of visible top-level windows whose program name contains prog_substr (Windows)."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    out = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if not n:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        if h:
            try:
                size = wintypes.DWORD(260)
                pb = ctypes.create_unicode_buffer(260)
                if (kernel32.QueryFullProcessImageNameW(h, 0, pb, ctypes.byref(size))
                        and prog_substr in os.path.basename(pb.value).lower()):
                    out.append(buf.value)
            finally:
                kernel32.CloseHandle(h)
        return True
    user32.EnumWindows(proc(cb), 0)
    return out


def _discord_hwnd():
    """Handle of Discord's main window (title ends in " - Discord"), else None."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    found = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        n = user32.GetWindowTextLengthW(hwnd)
        if not n or not user32.IsWindowVisible(hwnd):
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if not buf.value.rstrip().endswith("- Discord") and buf.value.strip() != "Discord":
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)
        if h:
            try:
                size = wintypes.DWORD(260)
                pb = ctypes.create_unicode_buffer(260)
                if (kernel32.QueryFullProcessImageNameW(h, 0, pb, ctypes.byref(size))
                        and "discord" in os.path.basename(pb.value).lower()):
                    found.append(hwnd)
            finally:
                kernel32.CloseHandle(h)
        return True
    user32.EnumWindows(proc(cb), 0)
    return found[0] if found else None


def grab_discord():
    """Just the Discord window as a PIL image, even when other windows cover it (PrintWindow with
    PW_RENDERFULLCONTENT). None if Discord isn't open, is minimized, or renders blank. Owner 11-02:
    "can our agent screenshot the discord application specifically? not the whole screen?" -- on the
    3440-wide screen, the call was a small patch of a 1187x496 shot."""
    import ctypes
    from ctypes import wintypes
    from PIL import Image
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    hwnd = _discord_hwnd()
    if not hwnd or user32.IsIconic(hwnd):
        return None
    try:
        user32.SetProcessDPIAware()
    except Exception:  # noqa: BLE001
        pass
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w < 200 or h < 150:
        return None
    hdc_win = user32.GetWindowDC(hwnd)
    hdc = gdi32.CreateCompatibleDC(hdc_win)
    bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    old = gdi32.SelectObject(hdc, bmp)
    try:
        if not user32.PrintWindow(hwnd, hdc, 2):          # 2 = PW_RENDERFULLCONTENT
            return None

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]
        bi = BITMAPINFOHEADER()
        bi.biSize, bi.biWidth, bi.biHeight = ctypes.sizeof(BITMAPINFOHEADER), w, -h   # top-down
        bi.biPlanes, bi.biBitCount, bi.biCompression = 1, 32, 0
        buf = ctypes.create_string_buffer(w * h * 4)
        if not gdi32.GetDIBits(hdc, bmp, 0, h, buf, ctypes.byref(bi), 0):
            return None
        img = Image.frombuffer("RGB", (w, h), buf.raw, "raw", "BGRX", 0, 1)
    finally:
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(hdc)
        user32.ReleaseDC(hwnd, hdc_win)
    lo, hi = img.convert("L").getextrema()
    return None if hi - lo < 8 else img                   # all one colour = it rendered blank


def discord_call(fresh: bool = False) -> dict | None:
    """{"channel": "5", "server": "ChillZone"} from the Discord window's title ("➣ 5 | ChillZone |
    Active • Social ... - Discord"), else None. Live 11-02: "I'm not on Discord, I live in calls
    like this one" / "I don't have a call number for this one". Cached 15 s unless fresh."""
    now = time.time()
    if not fresh and now - _CALL_CACHE["at"] < 15.0:
        return _CALL_CACHE["val"]
    val = None
    try:
        for title in _window_titles("discord"):
            val = parse_discord_title(title)
            if val:
                break
    except Exception:  # noqa: BLE001
        val = None
    _CALL_CACHE.update(at=now, val=val)
    return val


def parse_discord_title(title: str) -> dict | None:
    """'➣ 5 | ChillZone | Active • Social ... - Discord' -> {'channel': '5', 'server': 'ChillZone'}."""
    t = re.sub(r"\s*-\s*Discord\s*$", "", title or "").strip()
    if t == (title or "").strip():
        return None
    parts = [re.sub(r"^[^\w#@]+", "", p).strip() for p in t.split(" | ")]
    if len(parts) >= 2 and parts[0] and parts[1]:
        return {"channel": parts[0].lstrip("#"), "server": parts[1][:60]}
    return None


def front_window() -> str:
    """'Discord: "ChillZone | Active ..."' for the window in front (Windows; "" elsewhere or on any
    error). Live 11-02: on a 3440-wide screen shrunk to 1187x496 Ivy read a Discord call's two
    avatar tiles as "two pixel-art characters on a grassy meadow" -- the game from her notes -- and
    couldn't read the title bar when asked for the window's name."""
    try:
        import ctypes
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        n = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.strip()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        prog = ""
        h = kernel32.OpenProcess(0x1000, False, pid.value)       # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            try:
                size = wintypes.DWORD(260)
                pbuf = ctypes.create_unicode_buffer(260)
                if kernel32.QueryFullProcessImageNameW(h, 0, pbuf, ctypes.byref(size)):
                    prog = os.path.splitext(os.path.basename(pbuf.value))[0]
            finally:
                kernel32.CloseHandle(h)
        if not title and not prog:
            return ""
        kind = _APP_KIND.get(prog.lower(), "")
        name = f"{prog or 'a program'}{f' ({kind})' if kind else ''}"
        return f'{name}: "{title[:120]}"' if title else name
    except Exception:  # noqa: BLE001
        return ""


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


def _capture_live(live, plan, reason: str, asked: bool, speaker: str) -> dict:
    """A look with Live screen on: frames from the live capture (RAM only, never written to disk)."""
    t0 = time.perf_counter()
    res = live.look(plan, speaker)
    if res.get("seen") or res.get("images"):
        with _lock:
            _state["count"] += 1
            _state["last_ts"] = time.time()
            _state["last_reason"] = (reason or "")[:120]
            _state["last_size"] = None
            if not asked:
                _state["glances"] = [t for t in _state["glances"] if _state["last_ts"] - t < GLANCE_WINDOW_S]
                _state["glances"].append(_state["last_ts"])
    logger.info("👁️ live look (%s, %s) in %.0f ms: %s%s", plan.kind, "asked" if asked else "glance",
                (time.perf_counter() - t0) * 1000,
                f"watcher report on {len(res.get('frames') or [])} frame(s)" if res.get("seen")
                else f"can't see ({res.get('why')})", f" — {reason}" if reason else "")
    return res


def capture(reason: str = "", grab=None, asked: bool = True, glance: str = "sometimes", speaker: str = "",
            plan=None, line: str = "") -> dict:
    """Execute the tool. Returns {"text": ..., "images": [b64]} or {"text": refusal}.

    asked=False is a look the agent chose on its own: allowed by the Eyes "glance"
    setting and its rate limit, and answered as a glance rather than a request.
    With Live screen on (live_screen.enabled()) the frames come from the live capture, never a screenshot;
    `plan` (live_screen.plan.Plan) or `line` decides the kind of look."""
    with _lock:
        enabled = _state["enabled"]
    if not enabled:
        logger.info("👁️ screen look refused (tool disabled by owner)")
        return {"text": "Screen looking is switched off by the owner right now. Say you can't see the screen at the moment."}
    live = None
    if grab is None:
        try:
            import live_screen as _ls
            live = _ls if _ls.enabled() else None
        except Exception:  # noqa: BLE001
            live = None
    if live is not None:
        if plan is None:
            plan = live.plan_look(line, reason, asked)
        asked = plan.asked
    if not asked:
        if glance not in GLANCE_LIMITS:
            logger.info("👁️ unasked look refused (glances off)")
            return {"text": "Right now you only look at the screen when someone asks you to. Carry on "
                            "with the conversation without it."}
        if not can_glance(glance):
            logger.info("👁️ unasked look skipped (looked %.0fs ago)", time.time() - _state["last_ts"])
            return {"text": "You looked at the screen a moment ago; don't look again yet. Carry on "
                            "with the conversation from what you heard."}
    if live is not None:
        return _capture_live(live, plan, reason, asked, speaker)
    t0 = time.perf_counter()
    extra, discord_only = [], False
    try:
        img = None
        if grab is None and (reason or "").startswith("call:"):
            # a question about the call: just the Discord window, plus its left side up close
            # (server, channels, who's in each voice channel, 'Voice Connected')
            try:
                img = grab_discord()
            except Exception as e:  # noqa: BLE001
                logger.debug("discord window grab failed: %s", e)
                img = None
            if img is not None:
                discord_only = True
                side = img.crop((0, 0, min(img.size[0], max(360, int(img.size[0] * 0.14))), img.size[1]))
                extra.append(encode(side)[0])
        if img is None:
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
    front = front_window()
    who += ((f" The window in front is {front} (that's its name if they ask)." if front else "")
            + " This is the screen right now: describe only what this image shows, not a game or "
              "topic from your notes or from earlier in the call.")
    if (reason or "").startswith("call:"):
        # owner 11-02: a question about the call gets a look with a job, not a guess
        c = discord_call(fresh=True)
        where = (f" Right now the Discord window's title shows the server \"{c['server']}\", channel "
                 f"\"{c['channel']}\" (the channel open in Discord; the 'Voice Connected' panel shows the "
                 f"call itself)." if c else "")
        shots = (" These screenshots are just the Discord window (covered or not): the whole window, then "
                 "its left side up close so names are readable." if discord_only else
                 " Discord's window couldn't be captured on its own (closed or minimized), so this is the "
                 "whole screen.")
        who += (" They asked about the call itself." + shots + " Find the answer in the Discord window: the "
                "server name at the top left, the voice channel with the speaker icon and the people listed "
                "under it, the 'Voice Connected' panel at the bottom left (that's the call you're in), and the "
                "people's tiles in the middle. Answer exactly what they asked (which channel, who's in it, how "
                "many) from that; if Discord isn't on screen, say so and tell them what you do know." + where)
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
        "images": [b64] + extra,
    }
