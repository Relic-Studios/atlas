"""When the eyes take a screen-log look (docs/designs/live_screen.md, section 6): on a change event from the capture
process, or after the heartbeat interval with no change; at most one look per WATCH_EVERY_S; never between someone
finishing speaking and the reply's first text (or REPLY_WAIT_S if no reply comes), and a look already running is
cancelled when someone stops talking, so log looks never sit in front of a reply on the 5090. Identical frames are
skipped; a failed or cancelled look is dropped, never retried."""
from __future__ import annotations

import logging
import threading
import time

from live_screen import config as CFG

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, request, remote_cfg, label):
        """request(op) -> (header, [jpeg]) from the capture process; remote_cfg() -> Model's connection
        settings; label() -> whose screen it is."""
        self.request, self.remote_cfg, self.label = request, remote_cfg, label
        self.pending = threading.Event()
        self.stop_ev = threading.Event()
        self.cancel = threading.Event()          # set when someone stops talking: drop the look in flight
        self.block_until = 0.0
        self.last_look = 0.0
        self.last_hash = None
        self.looks = self.skipped_same = self.blocked = 0
        self.thread = None

    # signals -------------------------------------------------------------
    def on_change(self, _t=None):
        self.pending.set()

    def on_silence(self, active: bool):
        """Silence started = someone finished speaking: drop the log look in flight (the reply needs the GPU) and hold
        new ones until the reply starts."""
        if active:
            self.block_until = time.time() + CFG.REPLY_WAIT_S
            self.cancel.set()

    def on_reply_text(self):
        self.block_until = 0.0

    # loop ----------------------------------------------------------------
    def start(self):
        self.thread = threading.Thread(target=self._run, name="live-screen-watcher", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_ev.set()
        self.cancel.set()
        self.pending.set()

    def due(self, now: float) -> bool:
        if now - self.last_look < CFG.WATCH_EVERY_S:
            return False
        if now < self.block_until:
            self.blocked += 1
            return False
        return self.pending.is_set() or now - self.last_look >= CFG.HEARTBEAT_S

    def _run(self):
        while not self.stop_ev.is_set():
            self.pending.wait(0.25)
            if self.stop_ev.is_set():
                break
            now = time.time()
            if not self.due(now):
                # A change stays pending until a look is allowed (WATCH_EVERY_S, the reply wait), so pending.wait
                # returns at once: this loop spun a core and held the GIL whenever the screen moved, slowing speech,
                # TTS and replies (heavy-call test 11-04: 85% more GIL time with Live screen on)
                self.stop_ev.wait(0.25)
                continue
            self.pending.clear()
            self.last_look = now
            try:
                self.look_once()
            except Exception as e:  # noqa: BLE001 - never die
                logger.debug("watcher look failed: %s", e)

    def look_once(self) -> int:
        from live_screen import filters, memory, watcher
        if watcher._client(self.remote_cfg())[0] is None:
            return 0                                   # no watcher server (yet): nothing to do
        got = self.request("watch")
        if not got or not got[0].get("ok") or not got[1]:
            return 0
        header, jpegs = got
        if header.get("hash") and header["hash"] == self.last_hash:
            self.skipped_same += 1
            return 0
        self.last_hash = header.get("hash")
        title, process = header.get("title") or "", header.get("process") or ""
        t = float(header.get("t") or time.time())
        if filters.skip(process):
            memory.note_app(title, process, t, self.label())
            return 0
        self.cancel.clear()
        res = watcher.look(jpegs[0], filters.clean(title, 80), self.remote_cfg(), cancel=self.cancel)
        if res is None:
            return 0
        self.looks += 1
        added = memory.add(res, t, title, self.label())
        try:                                          # Phase 3: is this a moment worth a tip?
            import live_screen
            from live_screen import tips
            tips.on_look(res, t, title, process, self.label(), (live_screen.settings().get("name") or "").strip(),
                         self.remote_cfg())
        except Exception as e:  # noqa: BLE001
            logger.debug("tip check failed: %s", e)
        return added
