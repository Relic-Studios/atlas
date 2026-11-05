"""Screen capture for Live screen (docs/designs/live_screen.md, implementation plan item 1).

Windows only. Used by the Phase 0 tools (tools/record_screen_corpus.py, tests/check_capture.py) and,
from Phase 1, by screen_sidecar.py. Never imported by the Atlas server process: loading dxcam switches
the whole process to per-monitor DPI awareness, so the backends import it lazily.

All rectangles are (left, top, right, bottom) in physical desktop pixels; call set_dpi_aware() first so
window and monitor rectangles agree. Frames are numpy RGB arrays (height, width, 3), uint8.
"""
from __future__ import annotations

import ctypes
import io
import re
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

IS_WINDOWS = sys.platform == "win32"

Rect = tuple  # (left, top, right, bottom)

# Windows blacked out of full-screen frames (D5): (process name contains, window class or None). Only Windows
# notification pop-ups (they can show DMs); everything else on the screen is seen as the owner sees it.
MASK_RULES = (
    ("shellexperiencehost", "Windows.UI.Core.CoreWindow"),    # notification pop-ups, action center
)
# Desktop and taskbar windows never count as "the game" (largest window on the monitor).
SHELL_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})


class CaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Window:
    hwnd: int
    title: str
    cls: str
    rect: Rect
    minimized: bool
    process: str            # executable name, lower case ("discord.exe")
    pid: int

    @property
    def area(self) -> int:
        return max(0, self.rect[2] - self.rect[0]) * max(0, self.rect[3] - self.rect[1])


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def intersect(a: Rect, b: Rect) -> Rect | None:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    return (left, top, right, bottom) if right > left and bottom > top else None


def to_local(r: Rect, origin: Rect) -> Rect:
    return (r[0] - origin[0], r[1] - origin[1], r[2] - origin[0], r[3] - origin[1])


def area(r: Rect | None) -> int:
    return 0 if not r else max(0, r[2] - r[0]) * max(0, r[3] - r[1])


# ---------------------------------------------------------------------------
# Windows and monitors
# ---------------------------------------------------------------------------
def set_dpi_aware() -> None:
    """Per-monitor DPI awareness, so rectangles are physical pixels (as DXGI and mss report them)."""
    if not IS_WINDOWS:
        return
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):   # per-monitor v2
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        pass


def _process_name(pid: int, cache: dict) -> str:
    if pid in cache:
        return cache[pid]
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    name = ""
    h = kernel32.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
    if h:
        try:
            size = wintypes.DWORD(260)
            buf = ctypes.create_unicode_buffer(260)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                name = buf.value.replace("/", "\\").rsplit("\\", 1)[-1].lower()
        finally:
            kernel32.CloseHandle(h)
    cache[pid] = name
    return name


def _window_rect(hwnd) -> Rect:
    """The visible frame (DWM extended frame bounds, without the invisible resize border)."""
    from ctypes import wintypes
    r = wintypes.RECT()
    try:
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 9, ctypes.byref(r),
                                                      ctypes.sizeof(r)) == 0:
            return (r.left, r.top, r.right, r.bottom)
    except (AttributeError, OSError):
        pass
    ctypes.windll.user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r))
    return (r.left, r.top, r.right, r.bottom)


def _cloaked(hwnd) -> bool:
    """Hidden by the shell although 'visible' (other virtual desktops, suspended UWP windows)."""
    from ctypes import wintypes
    val = wintypes.DWORD(0)
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 14, ctypes.byref(val), ctypes.sizeof(val))
    except (AttributeError, OSError):
        return False
    return bool(val.value)


def list_windows(process: str | None = None, include_cloaked: bool = False) -> list[Window]:
    """Visible top-level windows, topmost first. process: keep only executables whose name contains it."""
    if not IS_WINDOWS:
        return []
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    names: dict = {}
    out: list[Window] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _lp):
        if not hwnd or not user32.IsWindowVisible(hwnd):
            return True
        if not include_cloaked and _cloaked(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        prog = _process_name(pid.value, names)
        if process and process.lower() not in prog:
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        title = ""
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value
        cbuf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cbuf, 256)
        out.append(Window(hwnd=int(hwnd), title=title, cls=cbuf.value, rect=_window_rect(hwnd),
                          minimized=bool(user32.IsIconic(hwnd)), process=prog, pid=int(pid.value)))
        return True

    user32.EnumWindows(proto(cb), 0)
    return out


def monitors() -> list[Rect]:
    """Monitor rectangles in the order the Eyes plugin numbers them (mss: index 1 = first monitor)."""
    import mss
    with mss.mss() as s:
        return [(m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"]) for m in s.monitors[1:]]


def monitor_rect(eyes_index: int) -> Rect | None:
    mons = monitors()
    return mons[eyes_index - 1] if 1 <= eyes_index <= len(mons) else None


def dxgi_output_for(rect: Rect) -> tuple[int, int] | None:
    """(device index, output index) of the DXGI output covering exactly this desktop rectangle."""
    import dxcam
    fac = vars(dxcam).get("__factory") or dxcam.DXFactory()   # the factory dxcam built at import
    for d, outs in enumerate(fac.outputs):
        for o, out in enumerate(outs):
            out.update_desc()
            c = out.desc.DesktopCoordinates
            if (c.left, c.top, c.right, c.bottom) == tuple(rect):
                return d, o
    return None


def hdr_enabled(monitor: Rect) -> bool | None:
    """Is Windows HDR ("advanced color") on for this monitor? None when it can't be told. Captured HDR frames
    look washed out until tone mapping is added, so HDR monitors are reported as unsupported."""
    if not IS_WINDOWS:
        return None
    from ctypes import wintypes

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class RATIONAL(ctypes.Structure):
        _fields_ = [("Numerator", ctypes.c_uint32), ("Denominator", ctypes.c_uint32)]

    class SOURCE(ctypes.Structure):
        _fields_ = [("adapterId", LUID), ("id", ctypes.c_uint32), ("modeInfoIdx", ctypes.c_uint32),
                    ("statusFlags", ctypes.c_uint32)]

    class TARGET(ctypes.Structure):
        _fields_ = [("adapterId", LUID), ("id", ctypes.c_uint32), ("modeInfoIdx", ctypes.c_uint32),
                    ("outputTechnology", ctypes.c_uint32), ("rotation", ctypes.c_uint32),
                    ("scaling", ctypes.c_uint32), ("refreshRate", RATIONAL), ("scanLineOrdering", ctypes.c_uint32),
                    ("targetAvailable", wintypes.BOOL), ("statusFlags", ctypes.c_uint32)]

    class PATH(ctypes.Structure):
        _fields_ = [("sourceInfo", SOURCE), ("targetInfo", TARGET), ("flags", ctypes.c_uint32)]

    class SOURCE_MODE(ctypes.Structure):
        _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32), ("pixelFormat", ctypes.c_uint32),
                    ("x", ctypes.c_long), ("y", ctypes.c_long)]

    class MODE(ctypes.Structure):
        _fields_ = [("infoType", ctypes.c_uint32), ("id", ctypes.c_uint32), ("adapterId", LUID),
                    ("data", ctypes.c_byte * 48)]

    class COLOR_INFO(ctypes.Structure):
        _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32), ("adapterId", LUID),
                    ("id", ctypes.c_uint32), ("value", ctypes.c_uint32), ("colorEncoding", ctypes.c_uint32),
                    ("bitsPerColorChannel", ctypes.c_uint32)]

    try:
        user32 = ctypes.windll.user32
        n_paths, n_modes = ctypes.c_uint32(), ctypes.c_uint32()
        if user32.GetDisplayConfigBufferSizes(2, ctypes.byref(n_paths), ctypes.byref(n_modes)) != 0:
            return None
        paths, modes = (PATH * n_paths.value)(), (MODE * n_modes.value)()
        if user32.QueryDisplayConfig(2, ctypes.byref(n_paths), paths, ctypes.byref(n_modes), modes, None) != 0:
            return None
        for p in paths[:n_paths.value]:
            idx = p.sourceInfo.modeInfoIdx
            if idx >= n_modes.value or modes[idx].infoType != 1:          # 1 = source mode
                continue
            sm = SOURCE_MODE.from_buffer_copy(bytes(modes[idx].data)[:ctypes.sizeof(SOURCE_MODE)])
            if (sm.x, sm.y, sm.x + sm.width, sm.y + sm.height) != tuple(monitor):
                continue
            info = COLOR_INFO(type=9, size=ctypes.sizeof(COLOR_INFO), adapterId=p.targetInfo.adapterId,
                              id=p.targetInfo.id)
            if user32.DisplayConfigGetDeviceInfo(ctypes.byref(info)) != 0:
                return None
            return bool((info.value >> 1) & 1)
    except (AttributeError, OSError, ValueError):
        return None
    return None


def tray_rects() -> list[Rect]:
    """The taskbar's notification area (clock, tray icons) on the main taskbar."""
    if not IS_WINDOWS:
        return []
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.FindWindowW.restype = wintypes.HWND
    user32.FindWindowExW.restype = wintypes.HWND
    out = []
    tray = user32.FindWindowW("Shell_TrayWnd", None)
    if tray:
        notify = user32.FindWindowExW(tray, None, "TrayNotifyWnd", None)
        if notify:
            r = wintypes.RECT()
            user32.GetWindowRect(notify, ctypes.byref(r))
            if r.right > r.left and r.bottom > r.top:
                out.append((r.left, r.top, r.right, r.bottom))
    return out


def mask_rects(monitor: Rect, windows: list[Window] | None = None, tray: list[Rect] | None = None) -> list[Rect]:
    """Rectangles to black out of a frame of `monitor`, relative to the monitor's top-left corner."""
    wins = list_windows() if windows is None else windows
    out = []
    for w in wins:
        if w.minimized:
            continue
        if not any(p in w.process and (c is None or w.cls == c) for p, c in MASK_RULES):
            continue
        r = intersect(w.rect, monitor)
        if r:
            out.append(to_local(r, monitor))
    for t in (tray or []):
        r = intersect(t, monitor)
        if r:
            out.append(to_local(r, monitor))
    return out


def apply_masks(frame: np.ndarray, rects: list[Rect]) -> np.ndarray:
    """Black out rects (frame coordinates) in place; returns the frame."""
    h, w = frame.shape[:2]
    for left, top, right, bottom in rects:
        left, top = max(0, int(left)), max(0, int(top))
        right, bottom = min(w, int(right)), min(h, int(bottom))
        if right > left and bottom > top:
            frame[top:bottom, left:right] = 0
    return frame


def largest_window(monitor: Rect, windows: list[Window] | None = None, exclude_pids=()) -> Window | None:
    """The window covering most of the monitor (the game), ignoring the desktop and taskbars."""
    best, best_area = None, 0
    for w in (list_windows() if windows is None else windows):
        if w.minimized or w.cls in SHELL_CLASSES or w.pid in exclude_pids or not w.title:
            continue
        a = area(intersect(w.rect, monitor))
        if a > best_area:
            best, best_area = w, a
    return best


def program_choices(exclude_processes=("explorer.exe", "textinputhost.exe", "applicationframehost.exe",
                                       "shellexperiencehost.exe", "searchhost.exe", "startmenuexperiencehost.exe",
                                       "systemsettings.exe", "lockapp.exe")) -> list[dict]:
    """Open programs the owner can pick to watch: one per process, the largest titled window's title as label."""
    best: dict = {}
    for w in list_windows():
        if not w.title or not w.process or w.cls in SHELL_CLASSES or w.process in exclude_processes:
            continue
        if w.area < 300 * 200 and not w.minimized:
            continue
        cur = best.get(w.process)
        if cur is None or w.area > cur.area:
            best[w.process] = w
    out = [{"value": p, "label": f"{w.title[:60]} ({p})"} for p, w in best.items()]
    return sorted(out, key=lambda o: o["label"].lower())


def match_window(w: Window, pattern: dict | None) -> bool:
    """pattern: {"title": regex, "cls": regex} from live_screen/capture_config.json (either may be absent)."""
    if not pattern:
        return False
    t, c = pattern.get("title"), pattern.get("cls")
    if not t and not c:
        return False
    return bool((not t or re.search(t, w.title)) and (not c or re.search(c, w.cls)))


def streamer_name(title: str, pattern: dict | None) -> str | None:
    """The streamer's name from a stream window's title, via the pattern's named group `name`."""
    t = (pattern or {}).get("title")
    if not t:
        return None
    m = re.search(t, title or "")
    if not m or "name" not in m.groupdict():
        return None
    return (m.group("name") or "").strip() or None


# ---------------------------------------------------------------------------
# Backends: grab() -> RGB frame or None; release()
# ---------------------------------------------------------------------------
class DxgiMonitor:
    """The Eyes monitor through DXGI Desktop Duplication (dxcam). The game runs untouched."""
    name = "dxgi"

    def __init__(self, eyes_index: int):
        set_dpi_aware()
        self.rect = monitor_rect(eyes_index)
        if self.rect is None:
            raise CaptureError(f"no monitor {eyes_index}")
        where = dxgi_output_for(self.rect)
        if where is None:
            raise CaptureError(f"monitor {eyes_index} {self.rect} matches no DXGI output")
        import dxcam
        self.cam = dxcam.create(device_idx=where[0], output_idx=where[1], output_color="RGB")

    def grab(self) -> np.ndarray | None:
        """Newest frame; the previous one again when the screen hasn't changed."""
        try:
            return self.cam.grab(new_frame_only=False)
        except Exception:  # noqa: BLE001 - access lost (UAC, mode switch): dxcam recovers on the next call
            return None

    def release(self) -> None:
        try:
            self.cam.release()
        except Exception:  # noqa: BLE001
            pass


def print_window(hwnd: int) -> np.ndarray | None:
    """One window, even when covered (PrintWindow with PW_RENDERFULLCONTENT; the method of
    screen.grab_discord). None when minimized, tiny or the grab fails."""
    from ctypes import wintypes
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    h_wnd = wintypes.HWND(hwnd)
    if not user32.IsWindow(h_wnd) or user32.IsIconic(h_wnd):
        return None
    r = wintypes.RECT()
    user32.GetWindowRect(h_wnd, ctypes.byref(r))
    w, h = r.right - r.left, r.bottom - r.top
    if w < 200 or h < 150:
        return None
    hdc_win = user32.GetWindowDC(h_wnd)
    hdc = gdi32.CreateCompatibleDC(hdc_win)
    bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    old = gdi32.SelectObject(hdc, bmp)
    try:
        if not user32.PrintWindow(h_wnd, hdc, 2):          # 2 = PW_RENDERFULLCONTENT
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
        bgra = np.frombuffer(buf.raw, dtype=np.uint8).reshape(h, w, 4)
        return bgra[:, :, 2::-1].copy()
    finally:
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(hdc)
        user32.ReleaseDC(h_wnd, hdc_win)


class WindowGrab:
    """A window through PrintWindow, grabbed on each call."""
    name = "printwindow"

    def __init__(self, hwnd: int):
        set_dpi_aware()
        self.hwnd = int(hwnd)

    def grab(self) -> np.ndarray | None:
        try:
            return print_window(self.hwnd)
        except Exception:  # noqa: BLE001
            return None

    def release(self) -> None:
        pass


class WgcWindow:
    """A window through Windows Graphics Capture (windows-capture); frames arrive on its own thread."""
    name = "wgc"

    def __init__(self, hwnd: int, min_interval_ms: int = 100):
        set_dpi_aware()
        from windows_capture import WindowsCapture
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.frame_time = 0.0
        self.closed = False
        self._control = None
        last_err = None
        for border in (False, None):          # no yellow border if Windows allows it
            try:
                cap = WindowsCapture(cursor_capture=False, draw_border=border,
                                     minimum_update_interval=min_interval_ms, window_hwnd=int(hwnd))
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

            @cap.event
            def on_frame_arrived(frame, control):
                rgb = frame.frame_buffer[:, :, 2::-1].copy()          # BGRA view -> owned RGB
                with self._lock:
                    self._frame, self.frame_time = rgb, time.time()

            @cap.event
            def on_closed():
                self.closed = True

            try:
                self._control = cap.start_free_threaded()
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise CaptureError(f"windows graphics capture failed: {last_err}")

    def grab(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def release(self) -> None:
        try:
            if self._control is not None:
                self._control.stop()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------
def luminance(frame: np.ndarray) -> float:
    """Mean luminance 0-255 on a subsample (cheap)."""
    s = frame[::8, ::8, :3].astype(np.float32)
    return float((s[..., 0] * 0.299 + s[..., 1] * 0.587 + s[..., 2] * 0.114).mean())


def is_black(frame: np.ndarray | None, level: float = 8.0) -> bool:
    return frame is None or luminance(frame) < level


def crop(frame: np.ndarray, rect: Rect) -> np.ndarray:
    left, top, right, bottom = (int(v) for v in rect)
    return frame[max(0, top):bottom, max(0, left):right]


def video_area(frames: list, method: str = "b", bar_px: int = 0, min_frac: float = 0.3) -> Rect | None:
    """Where the video is inside a stream window (frame coordinates), or None (then nothing is captured).

    method "a": the whole window minus Discord's control bar (bottom strip of bar_px pixels).
    method "b": the box whose contents change over the frames (at least 2 s of them) while its
    edges stay put; must cover min_frac of the window and look like video (aspect 1.0-2.6)."""
    if not frames:
        return None
    h, w = frames[-1].shape[:2]
    if method == "a":
        return (0, 0, w, h - bar_px) if 0 <= bar_px < h // 2 else None
    same = [f for f in frames if f.shape[:2] == (h, w)]
    if len(same) < 3:
        return None
    step = 8
    gray = np.stack([f[::step, ::step, :3].astype(np.float32).mean(axis=2) for f in same])
    moving = gray.std(axis=0) > 6.0
    if moving.mean() < 0.02:
        return None                                   # nothing changes: can't tell
    rows = np.where(moving.mean(axis=1) > 0.15)[0]
    cols = np.where(moving.mean(axis=0) > 0.15)[0]
    if not rows.size or not cols.size:
        return None
    top, bottom = int(rows[0]) * step, min(h, (int(rows[-1]) + 1) * step)
    left, right = int(cols[0]) * step, min(w, (int(cols[-1]) + 1) * step)
    bw, bh = right - left, bottom - top
    if bw * bh < min_frac * w * h or not (1.0 <= bw / max(1, bh) <= 2.6):
        return None
    return (left, top, right, bottom)


def fit_area(img, max_area: int):
    """Downscale a PIL image (keeping its aspect ratio) so width x height <= max_area."""
    from PIL import Image
    w, h = img.size
    scale = min(1.0, (max_area / float(w * h)) ** 0.5)
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def dhash(frame: np.ndarray) -> int:
    """64-bit difference hash of a frame (grayscale 9x8: is each pixel brighter than its right neighbour)."""
    g = frame[..., :3].astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
    try:
        import cv2
        small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    except ImportError:
        h, w = g.shape
        ys = (np.arange(8) * h // 8)
        xs = (np.arange(9) * w // 9)
        small = g[np.ix_(ys, xs)]
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def fast_jpeg(frame: np.ndarray, max_area: int, quality: int = 85) -> tuple[bytes, int, int]:
    """(jpeg, width, height) of an RGB frame scaled down (aspect kept) to at most max_area pixels; OpenCV when
    available (several times faster than PIL on a 3440x1440 frame), PIL otherwise."""
    h, w = frame.shape[:2]
    scale = min(1.0, (max_area / float(w * h)) ** 0.5)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        import cv2
        small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            return buf.tobytes(), nw, nh
    except ImportError:
        pass
    return to_jpeg(frame, max_area=max_area, quality=quality), nw, nh


def to_jpeg(frame_or_img, max_area: int | None = None, quality: int = 85, max_width: int | None = None) -> bytes:
    from PIL import Image
    img = Image.fromarray(frame_or_img) if isinstance(frame_or_img, np.ndarray) else frame_or_img
    if max_area:
        img = fit_area(img, max_area)
    if max_width and img.size[0] > max_width:
        img = img.resize((max_width, max(1, round(img.size[1] * max_width / img.size[0]))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
