"""call_boundary and the call-end resets (design docs/designs/experience_memory.md, D2, S4; acceptance A2a)."""
import ast
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent_memory  # noqa: E402
import call_boundary as CB  # noqa: E402
import importlib.util  # noqa: E402
SPEECH = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "huggingface_hub"))
NEEDS_SPEECH = unittest.skipUnless(SPEECH, "imports the server, which needs the speech stack "
                                   "(not in the light CI deps)")


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._root = agent_memory.ROOT
        agent_memory.ROOT = Path(self.tmp.name) / "memory"
        self.clock = Clock()
        self.b = CB.CallBoundary(clock=self.clock)

    def tearDown(self):
        agent_memory.ROOT = self._root
        self.tmp.cleanup()


class StartAndEnd(Base):
    def test_first_turn_starts_and_quiet_ends(self):
        self.assertFalse(self.b.in_progress_here())
        call = self.b.turn()
        self.assertTrue(self.b.in_progress_here())
        cur = json.loads(CB.current_path().read_text(encoding="utf-8"))
        self.assertEqual((cur["call"], cur["pid"]), (call, os.getpid()))
        self.assertEqual(self.b.turn(), call)                      # same call
        self.clock.t += CB.CALL_GAP_S - 1
        self.assertIsNone(self.b.tick())
        self.clock.t += 2
        res = self.b.tick()
        self.assertEqual((res["ended"], res["how"], res["call"]), (True, "timer", call))
        self.assertFalse(CB.current_path().exists())
        self.assertNotEqual(self.b.turn(), call)                   # the next turn starts a new call

    def test_api_and_shutdown_end(self):
        self.b.turn()
        self.assertEqual(self.b.end("api")["how"], "api")
        self.assertEqual(self.b.end("api"), {"ended": False})      # nothing in progress
        self.b.turn()
        self.assertTrue(self.b.end("shutdown", wait=True)["ended"])

    def test_too_small(self):
        self.b.turn()
        self.assertTrue(self.b.too_small())
        for _ in range(6):
            self.clock.t += 30
            self.b.turn()
        self.assertFalse(self.b.too_small())


class Liveness(Base):
    def test_in_progress_needs_a_live_process_and_a_fresh_heartbeat(self):
        self.b.turn()
        self.assertTrue(CB.in_progress(now=self.clock.t))
        self.assertFalse(CB.in_progress(now=self.clock.t + CB.STALE_S + 1))     # old heartbeat
        cur = json.loads(CB.current_path().read_text(encoding="utf-8"))
        cur["created"] = (cur["created"] or 0) + 100                            # a reused pid
        CB.current_path().write_text(json.dumps(cur), encoding="utf-8")
        self.assertFalse(CB.in_progress(now=self.clock.t))
        CB.current_path().unlink()
        self.assertFalse(CB.in_progress(now=self.clock.t))

    def test_startup_ends_a_call_left_by_a_dead_server(self):
        ran = []
        self.b.register("t.reset", lambda: ran.append(1))
        CB.current_path().parent.mkdir(parents=True, exist_ok=True)
        CB.current_path().write_text(json.dumps({"call": "x", "pid": 999999, "created": 1.0,
                                                 "heartbeat": self.clock.t}), encoding="utf-8")
        self.assertTrue(self.b.startup())
        self.assertEqual(ran, [1])
        self.assertFalse(CB.current_path().exists())
        self.assertFalse(self.b.startup())

    def test_startup_resets_use_the_dead_calls_names(self):
        """A call a crash left unended gets its disk resets with the names its pending file recorded (D2), even when
        the file is too small to keep."""
        got = []
        self.b.register("t.names", lambda names: got.append(names), needs_names=True)
        CB.pending_dir().mkdir(parents=True, exist_ok=True)
        recs = [{"k": "start", "t": self.clock.t}, {"k": "turn", "n": 1, "t": self.clock.t, "label": "S2"},
                {"k": "name", "label": "S2", "name": "Zorvath", "kind": "said"}]
        CB.pending_file("dead").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        self.assertTrue(self.b.startup())
        self.assertEqual(got, [["Zorvath"]])
        self.assertFalse(CB.pending_file("dead").exists())            # too small: deleted after its names were read


class Registry(Base):
    def test_order_names_snapshot_and_failures(self):
        order = []
        self.b.add_name_source("nb", lambda: ["Sam", "sam", "Riley"])
        self.assertEqual(self.b.call_names(), [])                    # no call in progress: no names
        self.b.register("late", lambda: order.append("late"), order=9)
        self.b.register("names", lambda names: order.append(("names", names)), order=1, needs_names=True)
        self.b.register("boom", lambda: 1 / 0, order=5)
        self.b.turn()
        self.b.end("api", wait=True)
        self.assertEqual(order, [("names", ["Sam", "Riley"]), "late"])

    def test_a_turn_waits_for_running_resets(self):
        gate = threading.Event()
        self.b.register("slow", lambda: gate.wait(5))
        self.b.turn()
        self.b.end("api")
        started = []
        t = threading.Thread(target=lambda: started.append(self.b.turn()))
        t.start()
        time.sleep(0.2)
        self.assertEqual(started, [])          # still waiting for the resets
        gate.set()
        t.join(5)
        self.assertEqual(len(started), 1)

    def test_activity_listeners(self):
        seen = []
        self.b.on_activity(seen.append)
        self.b.activity("partial")
        self.b.turn()
        self.assertEqual(seen, ["partial", "turn"])


def _scan() -> set:
    """Qualified names of every reset/clear function in the top-level modules and live_screen/ (S4 scan scope)."""
    pat = ("reset", "clear", "end_call", "forget_all")
    found = set()
    files = [p for p in ROOT.glob("*.py")] + [p for p in (ROOT / "live_screen").glob("*.py")]
    for f in files:
        mod = f.stem if f.parent == ROOT else f"live_screen.{f.stem}"
        if mod == "live_screen.__init__":
            mod = "live_screen"
        tree = ast.parse(f.read_text(encoding="utf-8"))

        def visit(node, prefix):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.ClassDef):
                    visit(ch, prefix + [ch.name])
                elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n = ch.name
                    if n in pat or n.startswith(("reset_", "clear_")):
                        found.add(".".join([mod] + prefix + [n]))
        visit(tree, [])
    return found


class Scan(unittest.TestCase):
    def test_every_reset_function_is_registered_or_exempt(self):
        missing = sorted(n for n in _scan() if n not in CB.REGISTERED and n not in CB.EXEMPT)
        self.assertEqual(missing, [], "register these with call_boundary or list them in EXEMPT with a reason")

    def test_exempt_names_exist(self):
        found = _scan()
        self.assertEqual(sorted(n for n in CB.EXEMPT if n not in found), [])

    def test_registered_names_exist(self):
        found = _scan()
        for n in CB.REGISTERED:
            if n in ("server.clear_rec_names",):
                continue
            self.assertIn(n, found, n)


class Resets(unittest.TestCase):
    """The functions S4 registers clear what they must and keep what they must."""

    def test_module_registrations(self):
        import call_summary  # noqa: F401
        import conversation_dynamics  # noqa: F401
        import factcheck  # noqa: F401
        import live_screen  # noqa: F401
        import owner_controls  # noqa: F401
        import room_tools  # noqa: F401
        import translate  # noqa: F401
        reg = set(CB.BOUNDARY.registered())
        for n in ("room_tools.end_call", "conversation_dynamics.clear_room_names", "call_summary.clear",
                  "factcheck.end_call", "translate.end_call", "live_screen.end_call", "owner_controls.clear_muted"):
            self.assertIn(n, reg)

    @NEEDS_SPEECH
    def test_pipeline_registers_its_resets(self):
        import speech_pipeline_manager as SPM
        fake = SimpleNamespace(boards=SimpleNamespace(reset_all=lambda: None),
                               agents=SimpleNamespace(reset_all=lambda: None),
                               people=SimpleNamespace(reset=lambda: None, call_names=lambda: ["Zed"]),
                               end_call=lambda: None)
        b = CB.CallBoundary(write_files=False)
        with mock.patch.object(CB, "BOUNDARY", b), mock.patch.object(CB, "register", b.register):
            SPM.SpeechPipelineManager._register_call_resets(fake)
        self.assertEqual(set(b.registered()), {"tasks.Boards.reset_all",
                                               "speech_pipeline_manager.SpeechPipelineManager.end_call",
                                               "agent_runtime.AgentRegistry.reset_all", "people.NameBook.reset"})
        b.call_id = "c"                       # a call in progress
        self.assertEqual(b.call_names(), ["Zed"])

    @NEEDS_SPEECH
    def test_live_registry_matches_registered(self):
        """S4: the static REGISTERED set the scan relies on is what the code actually registers at runtime (module
        imports, the pipeline and the server lifespan), so the scan can't pass on a name nobody registers."""
        import server
        import speech_pipeline_manager as SPM
        b = CB.CallBoundary(write_files=False)
        fake = SimpleNamespace(boards=SimpleNamespace(reset_all=lambda: None), agents=SimpleNamespace(reset_all=lambda: None),
                               people=SimpleNamespace(reset=lambda: None, call_names=lambda: []), end_call=lambda: None)
        app = SimpleNamespace(state=SimpleNamespace())
        # imported (registering with the real registry) before the patch, so other tests keep their registrations
        mods = {m: __import__(m) for m in ("call_summary", "conversation_dynamics", "factcheck", "live_screen",
                                            "owner_controls", "room_tools", "translate")}
        with mock.patch.object(CB, "BOUNDARY", b), mock.patch.object(CB, "register", b.register), \
                mock.patch.object(b, "startup"), mock.patch.object(server, "_setup_memory_runner"), \
                mock.patch("lore.strip_legacy_names", return_value=0), \
                mock.patch("teach.strip_legacy_names", return_value=0), \
                mock.patch("self_tools.scrub_existing_predictions", return_value=0):
            for mod, m in mods.items():
                fn = {"call_summary": "clear", "conversation_dynamics": "clear_room_names",
                      "owner_controls": "clear_muted"}.get(mod, "end_call")
                b.register(f"{mod}.{fn}", getattr(m, fn))          # what each module registers on import
            SPM.SpeechPipelineManager._register_call_resets(fake)
            server._setup_call_boundary(app)
        self.assertEqual(set(b.registered()), CB.REGISTERED)

    def test_names_and_room_state_are_cleared(self):
        import conversation_dynamics as CD
        from people import NameBook
        book = NameBook(agent_names=("ivy",), store=None)
        book.learn("S2", "hey I'm Zarvik")
        self.assertIn("Zarvik", book.call_names())
        CD.note_room_name("Zarvik")
        book.reset()
        CD.clear_room_names()
        self.assertEqual(book.call_names(), [])
        self.assertNotIn("zarvik", CD.room_names())

    def test_room_poll_and_timers(self):
        import room_tools as RT
        with tempfile.TemporaryDirectory() as d:
            saved = RT.STATE_PATH
            RT.STATE_PATH = Path(d) / "room.json"
            try:
                RT.start_poll("Pizza or tacos?", ["pizza", "tacos"])
                RT.set_timer(10, "tell Zarvik the raid starts", asker="S2")
                RT.set_timer(20, "remind Zarvik about the boss")
                RT.end_call(["Zarvik"])
                d2 = json.loads(RT.STATE_PATH.read_text(encoding="utf-8"))
                self.assertNotIn("poll", d2)
                self.assertEqual([t["message"] for t in d2["timers"]], ["remind someone about the boss"])
            finally:
                RT.STATE_PATH = saved

    def test_task_boards_all_files(self):
        import tasks
        with tempfile.TemporaryDirectory() as d:
            for p in ("ivy", "fae"):
                Path(d, f"tasks_{p}.json").write_text(json.dumps({"tasks": [{"id": 1, "kind": "note", "text": "ask S2"}],
                                                                  "next_id": 2}), encoding="utf-8")
            boards = tasks.Boards(root=d)
            boards.get("ivy")
            boards.reset_all()
            for p in ("ivy", "fae"):
                data = json.loads(Path(d, f"tasks_{p}.json").read_text(encoding="utf-8"))
                self.assertEqual(data.get("tasks"), [], p)

    def test_daily_caps_survive(self):
        import factcheck
        import translate
        factcheck._day.update(day=5, n=7)
        translate._day.update(d="2026-11-04", n=3)
        factcheck.end_call()
        translate.end_call()
        self.assertEqual(factcheck._day["n"], 7)
        self.assertEqual(translate._day["n"], 3)

    def test_diarizer_forgets_voices_keeps_counter(self):
        from diarize import Diarizer
        d = Diarizer(self_reference_audio=None, device="cpu")
        d._profiles = [object()]
        d._next_label = 7
        d._aliases = {"S3": "S2"}
        d.forget_all()
        self.assertEqual((d._profiles, d._aliases, d._next_label), ([], {}, 7))

    def test_adopter_end_call(self):
        import numpy as np
        import owner_voice as ov
        with tempfile.TemporaryDirectory() as d:
            store = ov.Store(Path(d) / "owner_voice.json")
            v = np.ones(192, dtype=np.float32) / np.sqrt(192)
            store.enroll("Relic", v, 6, "s0")
            a = ov.Adopter(store, session="s1")
            for _ in range(5):
                a.observe("S1", v)
            self.assertTrue(a.is_adopted("S1"))
            self.assertIsNotNone(a.last_score("S1"))
            a.refuse("S4")
            a.end_call()
            self.assertEqual((a.labels(), a.refused, a.vecs, a.last_score("S1")), ([], set(), {}, None))
            self.assertEqual(store.name, "Relic")



@NEEDS_SPEECH
class CallEndApi(Base):
    """POST /api/call/end (design D2, S3): owner only; answers at once; no pass is started before phase 3."""

    class Req:
        def __init__(self, app, host="127.0.0.1"):
            self.app, self.client = app, SimpleNamespace(host=host)

    def test_owner_only_and_answer(self):
        import asyncio
        import json as _json
        import server
        app = SimpleNamespace(state=SimpleNamespace())
        b = CB.CallBoundary(clock=self.clock)
        with mock.patch.object(CB, "BOUNDARY", b):
            b.turn()
            r = asyncio.run(server.api_call_end(self.Req(app, host="203.0.113.5")))
            self.assertEqual(r.status_code, 403)
            self.assertTrue(b.in_progress_here())
            r = asyncio.run(server.api_call_end(self.Req(app)))
            self.assertEqual(r, {"ended": True, "start_pass": False})
            self.assertFalse(b.in_progress_here())
            app2 = SimpleNamespace(state=SimpleNamespace())
            self.assertEqual(asyncio.run(server.api_call_end(self.Req(app2))), {"ended": False, "start_pass": False})

    def test_answers_while_an_agent_switch_runs(self):
        """A2a: the call end runs in its own action group, so it answers within 3 s while a slow agent switch holds
        the "pipeline" group."""
        import asyncio
        import threading
        import time as _t
        import server
        app = SimpleNamespace(state=SimpleNamespace())
        b = CB.CallBoundary(clock=self.clock)

        async def scenario():
            switch = asyncio.ensure_future(server._actions(app).run("pipeline", lambda: _t.sleep(5)))
            await asyncio.sleep(0.1)
            t0 = _t.perf_counter()
            r = await server.api_call_end(self.Req(app))
            took = _t.perf_counter() - t0
            await switch
            return r, took
        with mock.patch.object(CB, "BOUNDARY", b):
            b.turn()
            r, took = asyncio.run(scenario())
        self.assertTrue(r["ended"])
        self.assertLess(took, 3.0)


@NEEDS_SPEECH
class PipelineSerialized(unittest.TestCase):
    def test_call_end_and_an_agent_switch_never_overlap(self):
        """S4 row 3: the pipeline's call-end reset, agent switch and clear history run one at a time."""
        import threading
        import time as _t
        import speech_pipeline_manager as SPM
        m = object.__new__(SPM.SpeechPipelineManager)
        inside, overlaps = [0], []

        def abort(**kw):
            inside[0] += 1
            overlaps.append(inside[0])
            _t.sleep(0.2)
            inside[0] -= 1
            raise RuntimeError("stop here")          # each method stops at its first step after the abort
        m.abort_generation = abort
        m.thinker = None
        calls = [lambda: SPM.SpeechPipelineManager.end_call(m),
                 lambda: SPM.SpeechPipelineManager._apply_persona(m, "ivy", "p"),
                 lambda: SPM.SpeechPipelineManager.reset(m)]

        def run(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        ts = [threading.Thread(target=run, args=(fn,)) for fn in calls]
        for t in ts:
            t.start()
        for t in ts:
            t.join(5)
        self.assertEqual(max(overlaps), 1)


class TickerResponsiveness(unittest.TestCase):
    def test_runner_boundary_lock_is_evaluated_off_the_event_loop(self):
        import asyncio
        import logging
        source = ast.parse((ROOT / 'server.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.AsyncFunctionDef) and n.name == '_call_ticker')
        proxy = SimpleNamespace(sleep=lambda _: asyncio.sleep(.001), to_thread=asyncio.to_thread,
                                CancelledError=asyncio.CancelledError)
        ns = {'asyncio': proxy, 'logger': logging.getLogger('test')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'server.py', 'exec'), ns)
        lock, entered, release, finished = threading.Lock(), threading.Event(), threading.Event(), threading.Event()
        lock.acquire()
        def in_progress():
            entered.set()
            with lock:
                return True
        def unlock():
            release.wait(1)
            lock.release()
        unlocker = threading.Thread(target=unlock)
        boundary = SimpleNamespace(tick=lambda: None, in_progress_here=in_progress, last_activity=123)
        runner = SimpleNamespace(tick=lambda active, last: finished.set())
        async def scenario():
            task = asyncio.create_task(ns['_call_ticker'](SimpleNamespace(state=SimpleNamespace(MemoryRunner=runner))))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, .5))
                # A direct in_progress_here() argument blocks the loop until the
                # one-second fallback unlocker fires, so this assertion fails.
                self.assertTrue(lock.locked(), 'the event loop waited on the boundary lock')
                await asyncio.sleep(.01)
                self.assertTrue(lock.locked())
                release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 1))
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        try:
            unlocker.start()
            with mock.patch.object(CB, 'BOUNDARY', boundary):
                asyncio.run(scenario())
        finally:
            release.set(); unlocker.join(2)


if __name__ == "__main__":
    unittest.main()
