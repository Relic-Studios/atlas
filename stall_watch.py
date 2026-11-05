"""Event-loop stall watch (owner 11-03: "sometimes it freezes"). A heartbeat task on the server's event loop ticks
every HEARTBEAT_S; a plain thread watches it, and when the loop has been stuck for STALL_S it writes every thread's
stack to logs/stalls.log while the stall is still happening (so the loop thread's stack shows what is blocking it),
once per stall, and logs how long it lasted when the loop comes back. ATLAS_STALL_S=0 turns it off.

GIL wait probe (owner 11-04: "GIL thread contention is really hurting the application"): a thread sleeps 10 ms over
and over and measures how late it wakes. Past the timer's own slack, that lateness is time spent waiting for the GIL,
the same wait the audio threads see. Every 10 s the window's p50/p99/max go to logs/gil_wait.jsonl."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_S = 0.25
STALL_S = float(os.environ.get("ATLAS_STALL_S", "2.0"))
LOG = Path(__file__).resolve().parent / "logs" / "stalls.log"
MAX_LOG_BYTES = 5_000_000

_beat = [time.monotonic()]
_stats = {"stalls": 0, "worst_s": 0.0}
GIL_PROBE_S = 0.010
GIL_LOG = LOG.parent / "gil_wait.jsonl"
_gil = {"p50_ms": None, "p99_ms": None, "max_ms": None}


def gil_window(late_ms: list) -> dict:
    """p50/p99/max of one window's wake-up lateness (ms)."""
    xs = sorted(late_ms)
    if not xs:
        return {"n": 0}
    q = lambda p: round(xs[min(len(xs) - 1, int(len(xs) * p))], 2)  # noqa: E731
    return {"n": len(xs), "p50_ms": q(.5), "p99_ms": q(.99), "max_ms": round(xs[-1], 2)}


def _gil_probe() -> None:
    import json
    late, t_win = [], time.monotonic()
    while True:
        t0 = time.perf_counter()
        time.sleep(GIL_PROBE_S)
        late.append(max(0.0, (time.perf_counter() - t0 - GIL_PROBE_S) * 1000))
        now = time.monotonic()
        if now - t_win >= 10.0:
            w = gil_window(late)
            _gil.update({k: w.get(k) for k in _gil})
            late, t_win = [], now
            try:
                GIL_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(GIL_LOG, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"t": round(time.time(), 1), **w}) + "\n")
            except OSError:
                pass


async def _heartbeat() -> None:
    while True:
        _beat[0] = time.monotonic()
        await asyncio.sleep(HEARTBEAT_S)


def stacks(loop_thread: int | None = None) -> str:
    """Every live thread's stack, the event loop's first."""
    names = {t.ident: t.name for t in threading.enumerate()}
    frames = sys._current_frames()  # noqa: SLF001 - the only way to see a stuck thread's stack
    order = sorted(frames, key=lambda tid: tid != loop_thread)
    parts = []
    for tid in order:
        tag = " (event loop)" if tid == loop_thread else ""
        parts.append(f"--- thread {names.get(tid, '?')} [{tid}]{tag}\n" + "".join(traceback.format_stack(frames[tid])))
    return "\n".join(parts)


def _write(text: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        if LOG.exists() and LOG.stat().st_size > MAX_LOG_BYTES:
            LOG.replace(LOG.with_suffix(".log.old"))
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as e:
        logger.debug("stall log write failed: %s", e)


def check(now: float, state: dict, loop_thread: int | None) -> None:
    """One watcher step (also used by the tests): start/end of a stall."""
    age = now - _beat[0]
    if age >= STALL_S and not state.get("in"):
        state.update({"in": True, "since": _beat[0]})
        _stats["stalls"] += 1
        logger.warning("⏱️🧊 server event loop stuck for %.1f s: thread stacks -> logs/stalls.log", age)
        _write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} event loop stuck for {age:.1f} s ===\n"
               + stacks(loop_thread))
    elif age < STALL_S and state.get("in"):
        took = _beat[0] - state["since"]
        _stats["worst_s"] = max(_stats["worst_s"], took)
        state["in"] = False
        logger.warning("⏱️ server event loop back after %.1f s", took)
        _write(f"=== recovered after {took:.1f} s ===\n")


def start() -> None:
    """Call from the server's startup, on the event loop."""
    if STALL_S <= 0:
        return
    loop = asyncio.get_running_loop()
    loop.create_task(_heartbeat())
    loop_thread = threading.get_ident()

    def watch():
        state: dict = {}
        while True:
            time.sleep(0.5)
            try:
                check(time.monotonic(), state, loop_thread)
            except Exception as e:  # noqa: BLE001
                logger.debug("stall watch failed: %s", e)
    threading.Thread(target=watch, name="StallWatch", daemon=True).start()
    threading.Thread(target=_gil_probe, name="GilProbe", daemon=True).start()
    logger.info("⏱️ stall watch on (event loop stuck >= %.1f s is logged with thread stacks)", STALL_S)


def stats() -> dict:
    return {**_stats, "gil_wait": dict(_gil)}
