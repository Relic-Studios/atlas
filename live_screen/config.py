"""Live screen runtime settings (docs/designs/live_screen.md, "Runtime settings"): the only place the numbers are
set. capture_config.json is written by tests/check_capture.py (the owner's Phase 0 capture check)."""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURE_CONFIG = HERE / "capture_config.json"

CAPTURE_FPS = 2.0
RING_S = 10.0
RING_JPEG_QUALITY = 80
ASKED_AREA = 1024 * 576            # ~640 tokens
SMALL_AREA = 448 * 252             # ~112 tokens
WATCHER_AREA = 768 * 432           # ~336 tokens
RECENT_OFFSETS = (7.5, 5.0, 2.5)   # seconds before the newest frame, oldest first
NOW_MAX_AGE_S = 1.0
REQUEST_TIMEOUT_S = 0.5
RESTART_BACKOFF_S = (1.0, 5.0, 30.0)
RESTART_CAP = 3
RESTART_WINDOW_S = 600.0
BLACK_LEVEL = 8.0
BLACK_TIME_S = 5.0
RECONCILE_S = 0.5

# Phase 2: the screen log. Owner 11-03: one set of eyes, and it is Model (neither small model fit beside it on
# the 5090, and both invented text on a real screen). Model sees the frames itself on screen turns and writes
# the log. ATLAS_WATCHER_PORT=<port> points the log and the looks at a separate watcher server instead (its report
# then replaces the frames on screen turns; tools/setup_watcher.sh installs one).
WATCHER_PORT = int(os.environ.get("ATLAS_WATCHER_PORT", "0") or 0)
SEPARATE_WATCHER = WATCHER_PORT > 0
CHANGE_BITS = 6                    # a change = difference-hash distance of 6 or more of 64, then held one tick
WATCH_EVERY_S = 2.0 if SEPARATE_WATCHER else 4.0     # at most one log look per this many seconds
HEARTBEAT_S = 10.0 if SEPARATE_WATCHER else 20.0     # a log look anyway after this long without a change
REPLY_WAIT_S = 2.5                 # no log look from speech end until the reply's first text, at most this
WATCHER_TIMEOUT_S = 3.0
WATCHER_MAX_TOKENS = 300           # 160 cut most JSON looks off mid-object in the 11-03 measurement
ANSWER_TIMEOUT_S = 4.0             # a look on the reply path: the watcher's report or "can't see right now"
ANSWER_MAX_TOKENS = 120
DESCRIPTION_FRESH_S = 10.0

# Phase 3: tips (G4: at most 4 an hour, 10 minutes apart; the owner rates them with tools/rate_tips.py).
TIPS_PER_HOUR = 4
TIP_MIN_GAP_S = 600.0              # doubled after each ignored tip (up to 8x), reset by a thank-you
TIP_EXPIRY_S = 60.0                # a prepared tip not spoken within this is dropped
TIP_PAUSE_WAIT_S = 20.0            # wait this long for a loading screen, menu, map or death screen, then any gap
TIP_REACTION_S = 30.0              # a reaction within this counts for (or against) the last tip
TIP_QUIET_S = 1800.0               # quiet after a tip got "stop" / "I know" / "not now"
TIP_SEARCH_CAP = 10                # web searches for tips per session
WORTH_THRESHOLD = float(os.environ.get("ATLAS_TIP_WORTH", "0.7"))
LOG_CAP = 5000
MERGE_WINDOW_S = 60.0
FORGET_S = 120.0
SEARCH_RESULTS = 5

# Q2 fallback (Phase 0 passed: images on). ATLAS_LIVE_SCREEN_IMAGES=0 makes every look return "can't see".
IMAGES = os.environ.get("ATLAS_LIVE_SCREEN_IMAGES", "1") != "0"
# The screen-question judge (glance_judge.p_screen): "on" = a yes counts as an asked look, "log" = logged only,
# "off". On since 11-03 (owner): on that day's calls it said yes to 4 of 19 lines, all really about the screen.
SCREEN_DETECT = os.environ.get("ATLAS_SCREEN_DETECT", "on").strip().lower()
SCREEN_DETECT_THRESHOLD = float(os.environ.get("ATLAS_SCREEN_DETECT_THRESHOLD", "0.75"))


def capture_config() -> dict:
    """Backends, Discord window patterns and video-area methods from the owner's capture check ({} if not run)."""
    try:
        d = json.loads(CAPTURE_CONFIG.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}
