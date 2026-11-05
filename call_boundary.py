"""The one definition of a call (docs/designs/experience_memory.md, D2, S1, S4).

A call starts with the first committed turn while no call is in progress and ends 30 minutes after the last committed
turn, when the desktop app quits (`POST /api/call/end`), or when the server shuts down. At the end, every piece of
per-call state that could carry a non-owner into the next call is reset, in registry order (S4), on a worker thread;
a turn committed meanwhile waits for the resets before it starts the next call.

`_pending/current.json` (under the agent memory root) names the call in progress with this process's id, creation
time and a heartbeat, so other processes (the memory pass, tools) can ask `in_progress()`; a file left by a dead
process or with an old heartbeat does not count.

Also here: speech activity (partial transcripts with a word, committed turns, replies being generated), which the
memory pass runner listens to, and `call_names()`, the names the current call has learned or heard.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

CALL_GAP_S = float(os.environ.get("ATLAS_CALL_GAP_S", str(30 * 60)))   # S6: a call ends after 30 quiet minutes
HEARTBEAT_S = 30.0
STALE_S = 120.0
TOO_SMALL_TURNS = 6
TOO_SMALL_S = 120.0
EXPIRE_S = 2 * 86400.0          # S6: a pending file goes 2 days after its call's last line
BACKWARD_S = 60.0               # S6: a backward off-the-record trigger reaches back 60 seconds

# Every reset or clear function in the code that is not registered, with the reason (S4). The registry test scans the
# top-level modules and live_screen/ and fails on any function named reset, reset_*, clear, clear_*, end_call or
# forget_all that is neither registered (REGISTERED) nor listed here.
EXEMPT = {
    "factcheck.reset": "also clears the daily check cap; call end uses factcheck.end_call",
    "translate.reset": "also clears the daily caption cap; call end uses translate.end_call",
    "speech_pipeline_manager.SpeechPipelineManager.reset": "the Clear history action; call end uses end_call",
    "tasks.TaskBoard.clear": "removes one task (a tool action)",
    "tasks.TaskBoard.reset": "covered by tasks.Boards.reset_all",
    "live_screen.memory.clear": "covered by live_screen.end_call",
    "live_screen.tips.reset": "covered by live_screen.end_call",
    "live_screen.watcher.reset": "covered by live_screen.end_call",
    "agent_runtime.AgentRuntime.reset_conversation": "covered by agent_runtime.AgentRegistry.reset_all",
    "agent_runtime.AgentRuntime.reset_session": "covered by agent_runtime.AgentRegistry.reset_all",
    "floor.ConversationFloor.reset": "covered by agent_runtime.AgentRegistry.reset_all (reset_session)",
    "pacing.Pacing.reset": "covered by floor.ConversationFloor.reset",
    "threads.ThreadArbiter.reset": "covered by agent_runtime.AgentRegistry.reset_all (reset_session)",
    "loopbait.LoopBaitGuard.reset": "covered by agent_runtime.AgentRegistry.reset_all (reset_session)",
    "loopbait.DictationMemory.reset": "covered by speech_pipeline_manager.SpeechPipelineManager.end_call",
    "conversation_log.ConversationLog.reset": "covered by agent_runtime.AgentRegistry.reset_all (reset_conversation)",
    "think_listen.Thinker.reset": "covered by speech_pipeline_manager.SpeechPipelineManager.end_call",
    "turndetect.TurnDetection.reset": "audio turn state, no names or labels",
    "echo_guard.EchoGuard.reset": "echo state of the agent's own audio, no names or labels",
    "tts_conditioning.Upsampler2x.reset": "audio upsampler state, no names or labels",
    "server.TranscriptionCallbacks.reset_state": "per-turn state",
    "owner_voice.Adopter.clear": "used by Forget my voice; call end uses owner_voice.Adopter.end_call",
    "experience_memory.clear_owner_facts": "the owner's Clear owner facts control, not per-call state",
    "experience_memory.CallLines.clear": "covered by speech_pipeline_manager.SpeechPipelineManager.end_call",
}

# Names registered by the code that wires the resets (module imports, the server's lifespan and
# SpeechPipelineManager.__init__); the registry test checks the scan against this set plus EXEMPT.
REGISTERED = {
    "room_tools.end_call", "tasks.Boards.reset_all", "speech_pipeline_manager.SpeechPipelineManager.end_call",
    "agent_runtime.AgentRegistry.reset_all", "people.NameBook.reset", "conversation_dynamics.clear_room_names",
    "call_summary.clear", "factcheck.end_call", "translate.end_call", "live_screen.end_call",
    "server.clear_rec_names", "owner_controls.clear_muted", "diarize.Diarizer.forget_all",
    "owner_voice.Adopter.end_call",
}


def _root() -> Path:
    import agent_memory
    return Path(agent_memory.ROOT)


def pending_dir() -> Path:
    return _root() / "_pending"


def current_path() -> Path:
    return pending_dir() / "current.json"


def _proc_created(pid: int) -> Optional[float]:
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, path)


def read_current() -> Optional[dict]:
    try:
        return json.loads(current_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def in_progress(now: Optional[float] = None) -> bool:
    """True while a call is in progress in a live Atlas server (works from any process)."""
    cur = read_current()
    if not cur or not cur.get("call"):
        return False
    now = time.time() if now is None else now
    if now - float(cur.get("heartbeat") or 0) > STALE_S:
        return False
    pid, created = cur.get("pid"), cur.get("created")
    if pid is None:
        return False
    actual = _proc_created(int(pid))
    return actual is not None and created is not None and abs(actual - float(created)) < 1.0   # D2: same process


class _Reset:
    __slots__ = ("order", "name", "fn", "needs_names")

    def __init__(self, order, name, fn, needs_names):
        self.order, self.name, self.fn, self.needs_names = order, name, fn, needs_names


class CallBoundary:
    """Call state for this server process. Thread-safe; the clock is injectable for tests."""

    def __init__(self, clock: Callable[[], float] = time.time, write_files: bool = True):
        self.clock = clock
        self.write_files = write_files
        self.lock = threading.RLock()
        self.call_id: Optional[str] = None
        self.started: Optional[float] = None
        self.last_turn: Optional[float] = None
        self.first_turn: Optional[float] = None
        self._heartbeat_at = 0.0
        self.turns = 0
        self.last_activity = 0.0
        self._resets: dict = {}                     # name -> _Reset
        self._name_sources: dict = {}               # name -> callable returning names
        self._activity_cbs: list = []
        self._end_cbs: list = []                    # called with (call_id, info) under the lock, before resets
        self._start_cbs: list = []                  # called with (call_id, time) under the lock when a call starts
        self._resets_done = threading.Event()
        self._resets_done.set()
        self._pid = os.getpid()
        self._created = _proc_created(self._pid)

    # ------------------------------------------------------------ registry
    def register(self, name: str, fn: Callable, order: int = 50, needs_names: bool = False) -> None:
        """Register a call-end reset. `fn()` or, with needs_names, `fn(names)`. Re-registering a name replaces it
        (the pipeline is rebuilt in some tests and on reload)."""
        with self.lock:
            self._resets[name] = _Reset(order, name, fn, needs_names)

    def unregister(self, name: str) -> None:
        with self.lock:
            self._resets.pop(name, None)

    def registered(self) -> list:
        with self.lock:
            return [r.name for r in sorted(self._resets.values(), key=lambda r: (r.order, r.name))]

    def set_owner_name_source(self, fn: Callable[[], Optional[str]]) -> None:
        self._owner_name_fn = fn

    def owner_name(self) -> Optional[str]:
        """The owner's saved name (owner_voice), or None (browser mode, nothing enrolled)."""
        fn = getattr(self, "_owner_name_fn", None)
        try:
            return (fn() or None) if fn else None
        except Exception:  # noqa: BLE001
            return None

    def add_name_source(self, name: str, fn: Callable[[], Iterable[str]]) -> None:
        with self.lock:
            self._name_sources[name] = fn

    def on_activity(self, cb: Callable[[str], None]) -> None:
        self._activity_cbs.append(cb)

    def on_end(self, cb: Callable[[str, dict], None]) -> None:
        self._end_cbs.append(cb)

    def on_start(self, cb: Callable[[str, float], None]) -> None:
        self._start_cbs.append(cb)

    # ------------------------------------------------------------ names
    def call_names(self) -> list:
        """Every name the current call has learned or heard (NameBook, room names), de-duplicated; empty while no
        call is in progress (whatever the sources still hold belongs to no call)."""
        if self.call_id is None:
            return []
        out, seen = [], set()
        for src in list(self._name_sources.values()):
            try:
                for n in src() or ():
                    k = (n or "").strip()
                    if k and k.lower() not in seen:
                        seen.add(k.lower())
                        out.append(k)
            except Exception as e:  # noqa: BLE001
                logger.warning("call names source failed: %s", e)
        return out

    # ------------------------------------------------------------ activity and turns
    def activity(self, kind: str) -> None:
        """Speech activity: 'partial' (a partial transcript with a word), 'turn' or 'generation'. Cheap; the
        listeners must not block (the pass runner only sets a flag)."""
        self.last_activity = self.clock()
        for cb in list(self._activity_cbs):
            try:
                cb(kind)
            except Exception as e:  # noqa: BLE001
                logger.debug("activity listener failed: %s", e)

    def in_progress_here(self) -> bool:
        with self.lock:
            return self.call_id is not None

    def turn(self) -> str:
        """A committed human turn. Starts a call when none is in progress (after any running resets finish, also
        resets of an end that began while this turn waited)."""
        deadline = time.monotonic() + 30.0
        while True:
            self._resets_done.wait(timeout=max(0.0, deadline - time.monotonic()))
            with self.lock:
                if not self._resets_done.is_set() and time.monotonic() < deadline:
                    continue              # an end slipped in after the wait: its resets must not wipe this turn
                now = self.clock()        # checked and started under one hold of the lock
                if self.call_id is None:
                    self.call_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + "-" + os.urandom(2).hex()
                    self.started = self.first_turn = now
                    self.turns = 0
                    logger.info("call %s started", self.call_id)
                    for cb in list(self._start_cbs):
                        try:
                            cb(self.call_id, now)
                        except Exception as e:  # noqa: BLE001
                            logger.warning("call start listener failed: %s", e)
                self.turns += 1
                self.last_turn = now
                self._write_current()
                call = self.call_id
                break
        self.activity("turn")
        return call

    def too_small(self) -> bool:
        with self.lock:
            if self.call_id is None:
                return True
            span = (self.last_turn or 0) - (self.first_turn or 0)
            return self.turns < TOO_SMALL_TURNS or span < TOO_SMALL_S

    def _write_current(self) -> None:
        self._heartbeat_at = self.clock()
        if not self.write_files:
            return
        try:
            _write_json(current_path(), {"call": self.call_id, "started": self.started, "last_turn": self.last_turn,
                                         "pid": self._pid, "created": self._created, "heartbeat": self.clock()})
        except OSError as e:
            logger.warning("current call file not written: %s", e)

    def tick(self) -> Optional[dict]:
        """Lifespan task, every few seconds: heartbeat, the 30-minute end, and the hourly pending-file expiry."""
        now = self.clock()
        if self.write_files and now - getattr(self, "_last_expiry", 0.0) >= 3600.0:
            self._last_expiry = now
            try:
                expire(now, current=self.call_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("pending expiry failed: %s", e)
        with self.lock:
            if self.call_id is None:
                return None
            if self.clock() - (self.last_turn or 0) >= CALL_GAP_S:
                return self.end("timer")
            if self.clock() - self._heartbeat_at >= HEARTBEAT_S:      # S6: rewritten every 30 seconds
                self._write_current()
        return None

    # ------------------------------------------------------------ end
    def end(self, how: str, wait: bool = False, timeout: float = 10.0) -> dict:
        """End the call in progress: snapshot the names, tell the end listeners, remove current.json, answer, and
        run the resets on a worker thread (or in this thread up to `timeout` when wait=True)."""
        with self.lock:
            if self.call_id is None:
                return {"ended": False}
            names = self.call_names()
            info = {"call": self.call_id, "how": how, "first": self.first_turn, "last": self.last_turn,
                    "turns": self.turns, "too_small": self.too_small(), "names": names}
            for cb in list(self._end_cbs):
                try:
                    cb(self.call_id, info)
                except Exception as e:  # noqa: BLE001
                    logger.warning("call end listener failed: %s", e)
            logger.info("call %s ended (%s, %d turns)", self.call_id, how, self.turns)
            self.call_id = None
            self.started = self.first_turn = self.last_turn = None
            self.turns = 0
            if self.write_files:
                try:
                    current_path().unlink()
                except OSError:
                    pass
            self._resets_done.clear()
        t = threading.Thread(target=self._run_resets, args=(names,), name="CallEndResets", daemon=True)
        t.start()
        if wait:
            t.join(timeout)
        return {"ended": True, **{k: info[k] for k in ("call", "how", "turns", "too_small")}}

    def run_resets(self, names: Iterable[str] = ()) -> list:
        """Run every registered reset in order; returns the names that failed (each failure is logged)."""
        self._resets_done.clear()
        try:
            return self._run_resets(list(names))
        finally:
            self._resets_done.set()

    def _run_resets(self, names: list) -> list:
        failed = []
        try:
            with self.lock:
                resets = sorted(self._resets.values(), key=lambda r: (r.order, r.name))
            for r in resets:
                try:
                    r.fn(list(names)) if r.needs_names else r.fn()
                except Exception as e:  # noqa: BLE001 - one failing reset must not stop the others
                    failed.append(r.name)
                    logger.warning("call-end reset %s failed: %s", r.name, e)
        finally:
            self._resets_done.set()
        return failed

    def startup(self) -> bool:
        """At server start: a call a dead process never ended (its current.json, or a pending file with no end
        record) is ended now: the resets run (in-memory state is fresh anyway; this clears what was kept on disk) with
        the names recorded in those pending files as the name list (D2), read before a too-small file is deleted."""
        names, closed = [], []
        if self.write_files:
            try:
                d = pending_dir()
                for f in sorted(d.glob("*.jsonl")) if d.exists() else []:
                    if f.stem != self.call_id and not any(r.get("k") == "end" for r in read_records(f)):
                        names += names_in(f.stem)
                closed = close_leftovers(current=self.call_id)
                for call in closed:
                    logger.info("pending call %s closed at startup", call)
                expire(self.clock(), current=self.call_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("pending files not tidied at startup: %s", e)
        cur = read_current() if self.write_files else None
        if cur and cur.get("pid") == self._pid:
            cur = None
        if cur:
            logger.info("call %s was left open by a stopped server; ending it now", cur.get("call"))
            try:
                current_path().unlink()
            except OSError:
                pass
        if not cur and not closed:
            return False
        self.run_resets(sorted(set(names)))
        return True


# ------------------------------------------------------------------ the pending-file writer (D2, S1)
def pending_file(call: str) -> Path:
    return pending_dir() / f"{call}.jsonl"


def read_records(path: Path) -> list:
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    try:
                        out.append(json.loads(raw))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def _paused() -> bool:
    try:
        import experience_memory
        return bool(experience_memory.settings()["paused"])
    except Exception:  # noqa: BLE001
        return False


class PendingWriter:
    """Owns the pending files. Callers on the commit path only put work on a queue; one thread writes, so appends,
    the backward rewrite and server-side store writes never race. Turn and reply numbers (`n`, the turn ids the
    model cites) are assigned at once, in commit order."""

    def __init__(self, boundary: "CallBoundary", paused: Callable[[], bool] = _paused,
                 clock: Callable[[], float] = time.time):
        self.boundary = boundary
        self.paused = paused
        self.clock = clock
        self.q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._call: Optional[str] = None
        self._n = 0
        self._paused_at_start = False
        self._thread: Optional[threading.Thread] = None
        self._pause_cache = (0.0, False)

    # ------------------------------------------------------------ queue
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="PendingWriter", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            job = self.q.get()
            try:
                job()
            except Exception as e:  # noqa: BLE001
                logger.warning("pending writer job failed: %s", e)
            finally:
                self.q.task_done()

    def _put(self, job: Callable[[], None]) -> None:
        self._ensure_thread()
        self.q.put(job)

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait until every queued job has run (tests, shutdown)."""
        end = time.monotonic() + timeout
        while self.q.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self.q.unfinished_tasks

    def submit(self, fn: Callable[[], None], retry_s: float = 0.5, max_tries: int = 1000) -> None:
        """A server-side store write (a "remember that" fact): run on the writer thread, retried until it succeeds."""
        def job():
            for i in range(max_tries):
                try:
                    fn()
                    return
                except Exception as e:  # noqa: BLE001
                    if i == 0:
                        logger.warning("memory write retrying: %s", e)
                    time.sleep(retry_s)
        self._put(job)

    # ------------------------------------------------------------ records
    def _is_paused(self) -> bool:
        t, v = self._pause_cache
        now = time.monotonic()
        if now - t > 2.0:
            v = bool(self.paused())
            self._pause_cache = (now, v)
        return v

    def _append(self, call: str, rec: dict) -> None:
        path = pending_file(call)

        def job():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._put(job)

    def begin(self, call: str, t: float) -> None:
        with self._lock:
            self._call, self._n = call, 0
        self._append(call, {"k": "start", "call": call, "t": t})

    def _record(self, rec: dict, numbered: bool) -> Optional[int]:
        with self._lock:
            call = self._call
            if call is None or call != self.boundary.call_id:
                return None
            n = None
            if numbered:
                self._n += 1
                n = rec["n"] = self._n
        if self._is_paused():
            return n
        self._append(call, rec)
        return n

    def turn(self, *, label, agent, ident, owner, score, dur, audible, text, t=None, write=True) -> Optional[int]:
        """A committed human turn (S1). `write=False` (inside an off-the-record span): numbered, never written."""
        rec = {"k": "turn", "t": self.clock() if t is None else t, "label": label, "agent": agent, "ident": ident,
               "owner": bool(owner), "score": score, "dur": dur, "audible": audible, "text": text}
        if not write:
            with self._lock:
                self._n += 1
                return self._n
        n = self._record(rec, numbered=True)
        try:
            import memory_redact
            for h in memory_redact.handles(text or ""):
                self.event("name", name=h, kind="handle")
        except Exception:  # noqa: BLE001
            pass
        return n

    def reply(self, *, agent, ident, recalled, text, t=None, write=True) -> Optional[int]:
        rec = {"k": "reply", "t": self.clock() if t is None else t, "agent": agent, "ident": ident,
               "recalled": bool(recalled), "text": text}
        if not write:
            with self._lock:
                self._n += 1
                return self._n
        return self._record(rec, numbered=True)

    def event(self, rec_kind: str, **fields) -> None:
        """name / namesake / adopt / refuse / forget_voice / merge (S1)."""
        rec = {"k": rec_kind}
        rec.update(fields)
        self._record(rec, numbered=False)

    def finish(self, call: str, info: dict) -> None:
        """Call end: the end record, then delete the file of a too-small call."""
        with self._lock:
            if self._call == call:
                self._call = None
        rec = {"k": "end", "t": info.get("last") or self.clock(), "how": info.get("how")}
        path = pending_file(call)
        too_small = bool(info.get("too_small"))

        def job():
            if too_small:
                try:
                    path.unlink()
                except OSError:
                    pass
                return
            if path.exists():
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        self._put(job)

    def drop_since(self, t_from: float, t_to: Optional[float] = None) -> None:
        """Backward off-the-record trigger (D7): remove this call's turns and replies committed from `t_from` to
        `t_to` from the pending file at once (the file is rewritten on the writer thread)."""
        with self._lock:
            call = self._call
        if call is None:
            return
        t_to = self.clock() if t_to is None else t_to
        path = pending_file(call)

        def job():
            recs = read_records(path)
            keep = [r for r in recs if not (r.get("k") in ("turn", "reply") and t_from <= float(r.get("t", 0)) <= t_to)]
            if len(keep) == len(recs):
                return
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                for r in keep:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        # retried until it succeeds (an antivirus or indexer can hold the file for a moment on Windows): the marked
        # lines must not reach the pass
        self.submit(job, retry_s=0.2, max_tries=300)


def close_leftovers(now: Optional[float] = None, current: Optional[str] = None) -> list:
    """Startup (D2): pending files without an end record belong to calls a stopped server never ended; each gets an
    end record dated at its last line (a too-small one is deleted). Returns the calls closed."""
    out = []
    d = pending_dir()
    if not d.exists():
        return out
    for f in sorted(d.glob("*.jsonl")):
        call = f.stem
        if call == current:
            continue
        recs = read_records(f)
        if any(r.get("k") == "end" for r in recs):
            continue
        times = [float(r.get("t", 0)) for r in recs if r.get("k") in ("turn", "reply")]
        turns = [float(r.get("t", 0)) for r in recs if r.get("k") == "turn"]
        last = max(times) if times else (f.stat().st_mtime if f.exists() else 0.0)
        if len(turns) < TOO_SMALL_TURNS or (max(turns) - min(turns) if turns else 0) < TOO_SMALL_S:
            try:
                f.unlink()
            except OSError:
                pass
        else:
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": "end", "t": last, "how": "startup"}) + "\n")
        out.append(call)
    return out


def names_in(call: str) -> list:
    """The `name` records of one pending file (the name list for a call ended at startup, D2)."""
    return [r.get("name") for r in read_records(pending_file(call)) if r.get("k") == "name" and r.get("name")]


def expire(now: Optional[float] = None, current: Optional[str] = None) -> list:
    """Delete pending files whose call's last line is 2 or more days old (D2). Returns the calls deleted."""
    now = time.time() if now is None else now
    out = []
    d = pending_dir()
    if not d.exists():
        return out
    for f in sorted(d.glob("*.jsonl")):
        if f.stem == current:
            continue
        recs = read_records(f)
        times = [float(r.get("t", 0)) for r in recs if r.get("k") in ("turn", "reply", "end", "start")]
        last = max(times) if times else f.stat().st_mtime
        if now - last >= EXPIRE_S:
            for g in (f, f.with_suffix(".fails")):
                try:
                    g.unlink()
                except OSError:
                    pass
            out.append(f.stem)
    return out


BOUNDARY = CallBoundary()
WRITER = PendingWriter(BOUNDARY)
BOUNDARY.on_start(WRITER.begin)
BOUNDARY.on_end(WRITER.finish)


def name_event(kind: str, name: str) -> None:
    """people.py and conversation_dynamics report names as the call learns, hears or uses them; the pending file
    keeps them for the pass's name list (D2). kind: learned, heard, room, handle."""
    if name:
        WRITER.event("name", name=name, kind=kind)


def owner_event(rec_kind: str, **fields) -> None:
    """namesake / adopt / refuse / forget_voice / merge (D2, S1)."""
    WRITER.event(rec_kind, **fields)


def register(name: str, fn: Callable, order: int = 50, needs_names: bool = False) -> None:
    BOUNDARY.register(name, fn, order, needs_names)


def call_names() -> list:
    return BOUNDARY.call_names()


def activity(kind: str) -> None:
    BOUNDARY.activity(kind)


def owner_name() -> Optional[str]:
    return BOUNDARY.owner_name()


# ------------------------------------------------------------------ off the record (D7)
SPAN_S = 600.0                  # S6: a span ends 10 minutes after its trigger


class OffRecord:
    """The open off-the-record span of the current call (closed by "back on the record", 10 minutes, or call end)."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self._until = 0.0

    def open(self, now: Optional[float] = None) -> None:
        self._until = (self.clock() if now is None else now) + SPAN_S

    def close(self) -> None:
        self._until = 0.0

    def is_open(self, now: Optional[float] = None) -> bool:
        return (self.clock() if now is None else now) < self._until


OFF = OffRecord()


def apply_off_record(kind: str, now: Optional[float] = None) -> None:
    """A trigger was committed (D7): "end" closes the span; "back" and "span" mark the last 60 seconds (removed from
    the pending file, masked in the call recording, and every said or held fact, lore entry, lesson or prediction
    from those lines deleted); "span" also opens a span."""
    now = time.time() if now is None else now
    if kind == "end":
        OFF.close()
        return
    if kind not in ("back", "span"):
        return
    t_from = now - BACKWARD_S
    _MARKED[:] = [w for w in _MARKED if now - w[1] < _MARKED_KEEP_S] + [(t_from, now + 1.0)]
    if kind == "back":           # a span covers its own replies until "back on the record"
        _REPLY_OFF_UNTIL[0] = max(_REPLY_OFF_UNTIL[0], now + REPLY_AFTER_S)
    WRITER.drop_since(t_from, now + 1.0)
    for fn in list(_WINDOW_LISTENERS):             # in-memory items from those lines (held "remember that" turns)
        try:
            fn(t_from, now + 1.0)
        except Exception as e:  # noqa: BLE001
            logger.warning("off-the-record listener failed: %s", e)

    def mask():
        from call_recorder import REC
        REC.mask_since(t_from, now + 1.0)

    def forget():
        import experience_memory
        experience_memory.forget_window(t_from, now + 1.0)
    WRITER.submit(mask)                            # off the commit path (D7: the writer thread masks)
    WRITER.submit(forget)
    if kind == "span":
        OFF.open(now)


_WINDOW_LISTENERS: list = []
_MARKED: list = []                 # [(t_from, t_to)] of backward triggers in the last hour
_MARKED_KEEP_S = 3600.0


def source_time(now: Optional[float] = None) -> float:
    """The time of the line a tool write comes from (D7: stored items record their source line's time): the last
    committed turn of the call in progress (a tool runs while the agent answers it), else now."""
    now = time.time() if now is None else now
    last = BOUNDARY.last_turn if BOUNDARY.call_id is not None else None
    return float(last) if last is not None and 0 <= now - float(last) < 300 else now


REPLY_AFTER_S = 30.0               # replies to a backward trigger (often echoing what was retracted) aren't kept
_REPLY_OFF_UNTIL = [0.0]


def reply_off_record(t: Optional[float] = None) -> bool:
    """An agent reply spoken now is not kept: a span is open, or a backward trigger came in the last 30 seconds."""
    t = time.time() if t is None else t
    return OFF.is_open() or t <= _REPLY_OFF_UNTIL[0]


def marked(t: float) -> bool:
    """A time inside a window a backward trigger marked (so a write from those lines that arrives after the delete
    is refused too)."""
    return any(a <= t <= b for a, b in _MARKED)


def off_record_at(t: Optional[float] = None) -> bool:
    """Writes are refused while a span is open or when their source line was marked (D7)."""
    return OFF.is_open() or marked(source_time() if t is None else t)


def on_forget_window(fn) -> None:
    """fn(t_from, t_to) runs when a backward trigger marks the lines from t_from to t_to (D7)."""
    if fn not in _WINDOW_LISTENERS:
        _WINDOW_LISTENERS.append(fn)
