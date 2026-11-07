"""Execute real pipeline worker/callback source with controlled CPU dependencies."""
import asyncio
import logging
import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest import TestCase
from test_response_decision import load_method
from response_decision import ResponseDecision, filter_response, finish_silent_generation

class Colors:
    @staticmethod
    def apply(text):
        return SimpleNamespace(magenta=text, blue=text, red=text, cyan=text, green=text, pink=text)


def event():
    return threading.Event()


def generation(wire):
    d = ResponseDecision()
    return SimpleNamespace(id=1, decision=d, llm_generator=filter_response(iter([wire]), d),
        quick_answer='', final_answer='', quick_answer_provided=False, response_held=False,
        llm_aborted=False, abortion_started=False, llm_finished=False,
        llm_finished_event=event(), tts_quick_finished_event=event(), tts_final_finished_event=event())

class WorkerTests(TestCase):
    def test_hold_then_speak_same_worker(self):
        import turn_trace
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', '_llm_inference_worker',
            {'time':time, 'Colors':Colors, 'finish_silent_generation':finish_silent_generation,
             '_turn_trace': turn_trace})
        m = SimpleNamespace(shutdown_event=event(), generator_ready_event=event(), stop_llm_request_event=event(),
            stop_llm_finished_event=event(), stop_everything_event=event(), llm_answer_ready_event=event(),
            stop_tts_quick_request_event=event(), stop_tts_final_request_event=event(),
            llm_generation_active=False, running_generation=None, no_think=False, preprocess_chunk=lambda c:c,
            text_context=SimpleNamespace(get_context=lambda text:(None,None)), on_partial_assistant_text=None)
        thread = threading.Thread(target=fn, args=(m,), daemon=True)
        thread.start()
        try:
            for wire, expected in [('[HOLD]', ''), ('[SPEAK to=user]\nSorry.', 'Sorry.')] * 100:
                g = generation(wire)
                m.running_generation = g
                m.llm_answer_ready_event.clear()
                m.generator_ready_event.set()
                self.assertTrue(g.llm_finished_event.wait(2), 'worker wedged')
                self.assertEqual(g.quick_answer, expected)
                self.assertEqual(m.llm_answer_ready_event.is_set(), bool(expected))
                self.assertEqual(g.response_held, not bool(expected))
        finally:
            m.shutdown_event.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_duplicate_partial_does_not_replace_active_generation(self):
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', 'process_prepare_generation', {'threading':threading})
        g = SimpleNamespace(text='Hello Ivy', abortion_started=False)
        m = SimpleNamespace(running_generation=g, generation_counter=1,
            check_abort=lambda *a,**k:False)
        # A duplicate must return before constructing a new generation or resetting events.
        fn(m, 'Hello Ivy')
        self.assertIs(m.running_generation, g)
        self.assertEqual(m.generation_counter, 1)

    def test_refined_utterance_keeps_speaking_generation(self):
        # Demo 10-04: 'Hi Wren, I'm Riley.' was already being spoken when the longer
        # transcript arrived; check_abort declined (same utterance) but a new Gen 5
        # replaced Gen 4, orphaning its audio -> empty turn, agent silent.
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', 'process_prepare_generation', {'threading':threading})
        g = SimpleNamespace(id=4, text="Hi, Wren. I'm Riley.", abortion_started=False)
        m = SimpleNamespace(running_generation=g, generation_counter=4,
            check_abort=lambda *a,**k:False)
        fn(m, "Hi, Wren. I'm Riley. We're trying to pick a game for tonight.")
        self.assertIs(m.running_generation, g)
        self.assertEqual(m.generation_counter, 4)
        self.assertEqual(g.text, "Hi, Wren. I'm Riley.")

    def test_abort_cancels_transport_before_waiting_even_in_final_worker(self):
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', 'process_abort_generation')
        for llm_active, final_active in [(True, False), (False, True)]:
            order=[]
            class Signal:
                def __init__(self): self.value=False
                def is_set(self): return self.value
                def set(self): self.value=True
                def clear(self): self.value=False
                def wait(self, **kw): order.append('wait'); return True
            cancelled=[]
            def cancel(request_id):
                cancelled.append(request_id);order.append('cancel')
            m=SimpleNamespace(abort_lock=threading.Lock(), running_generation=SimpleNamespace(id=1, llm_request_id='voice-request', abortion_started=False),
                llm_generation_active=llm_active, tts_quick_generation_active=False, tts_final_generation_active=final_active,
                llm=SimpleNamespace(cancel_generation=cancel), audio=SimpleNamespace())
            for name in ('abort_completed_event','abort_block_event','stop_everything_event','generator_ready_event',
                         'stop_llm_request_event','stop_llm_finished_event','llm_answer_ready_event','stop_tts_quick_request_event',
                         'stop_tts_quick_finished_event','stop_tts_final_request_event','stop_tts_final_finished_event'):
                setattr(m,name,Signal())
            fn(m)
            self.assertIn('cancel', order)
            self.assertEqual(cancelled,['voice-request'])
            self.assertLess(order.index('cancel'), order.index('wait'))

    def test_sender_retires_hold_without_history_or_audio(self):
        async def run():
            fn = load_method('server.py', '', 'send_tts_chunks')
        # Module-level function extraction is provided below.
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name=='send_tts_chunks')
        ns = dict(asyncio=asyncio, time=time, threading=threading, logger=logging.getLogger('test'), FastAPI=object,
                  Colors=Colors, Empty=__import__('queue').Empty, WebSocketDisconnect=type('Disconnect',(Exception,),{}))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'server.py', 'exec'), ns)
        async def scenario():
            gen = generation('[HOLD]')
            list(gen.llm_generator)
            finish_silent_generation(gen)
            gen.llm_finished=True
            gen.quick_answer_first_chunk_ready=False
            mgr=SimpleNamespace(running_generation=gen, is_valid_gen=lambda:True)
            app=SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=mgr, AudioInputProcessor=SimpleNamespace(interrupted=False)))
            done=asyncio.Event()
            cb=SimpleNamespace(interruption_time=0, tts_to_client=True, tts_client_playing=False,
                tts_chunk_sent=False, is_hot=False, synthesis_started=False,
                user_finished_turn=True, user_history_committed=False, reset_state=done.set)
            task=asyncio.create_task(ns['send_tts_chunks'](app, Queue(), cb))
            try:
                with self.assertRaises(asyncio.TimeoutError, msg='must not retire before history commit'):
                    await asyncio.wait_for(done.wait(), 0.1)
                cb.user_history_committed=True
                await asyncio.wait_for(done.wait(), 1)
                self.assertIsNone(mgr.running_generation)
            finally:
                task.cancel()
                await task
        asyncio.run(scenario())

    def test_compact_history_summarizes_oldest_keeps_recent(self):
        import threading
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', 'compact_history',
            {'HISTORY_SUMMARY_THRESHOLD': 24, 'HISTORY_KEEP_RECENT': 10})
        m = SimpleNamespace(summary_lock=threading.Lock(),
            history=[{"role": "user", "content": f"m{i}"} for i in range(30)],
            llm_generation_active=False, running_generation=None,
            _summarize_messages=lambda msgs: "prior launch discussion")
        self.assertTrue(fn(m))
        self.assertEqual(len(m.history), 11)  # 1 summary + 10 recent
        self.assertIn("prior launch discussion", m.history[0]["content"])
        self.assertEqual(m.history[1]["content"], "m20")
        self.assertEqual(m.history[-1]["content"], "m29")

    def test_compact_history_skips_when_generation_active(self):
        import threading
        fn = load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', 'compact_history',
            {'HISTORY_SUMMARY_THRESHOLD': 24, 'HISTORY_KEEP_RECENT': 10})
        m = SimpleNamespace(summary_lock=threading.Lock(),
            history=[{"role": "user", "content": f"m{i}"} for i in range(30)],
            llm_generation_active=True, running_generation=object(),
            _summarize_messages=lambda msgs: "should not run")
        self.assertFalse(fn(m))
        self.assertEqual(len(m.history), 30)  # untouched


class RefinementAddsClause(__import__("unittest").TestCase):
    """Demo take 8 (10-05): a draft on 'Okay, Wren. Settle it.' must not survive
    once the line grows the actual question; trivial refinements still keep it."""
    def test_adds_content(self):
        import re
        S = SimpleNamespace(_adds_content=load_method('speech_pipeline_manager.py',
            'SpeechPipelineManager', '_adds_content', {'re': re}))
        self.assertTrue(S._adds_content('[S1] Okay, Wren. Settle it.',
                        '[S1] Okay, Wren. Settle it. Co-op or competitive? There are four of us.'))
        self.assertFalse(S._adds_content('[S1] Hi Wren, I am Riley', '[S1] Hi Wren, I am Riley.'))
        self.assertFalse(S._adds_content('[S1] okay wren settle', '[S1] Okay, Wren. Settle it.'))


class SetupTailHoldsTurn(__import__("unittest").TestCase):
    """Demo take 9 (10-05): 'Okay, Wren. Settle it.' ended the turn; the question
    that followed was missed. Setup phrases hold the turn open longer."""
    def test_setup_tail(self):
        import ast, re
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / 'turndetect.py'
        nodes = [n for n in ast.parse(source.read_text(encoding='utf-8')).body
                 if (isinstance(n, ast.FunctionDef) and n.name == 'setup_tail') or
                    (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_SETUP_TAIL'
                                                      for t in n.targets))]
        ns = {'re': re}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        setup_tail = ns['setup_tail']
        for t in ('Okay, Wren. Settle it.', 'All right, Wren. Quick test.',
                  'Hey Wren, quick question', 'I have a question for you.'):
            self.assertTrue(setup_tail(t), t)
        for t in ('Settle it. Co-op or competitive?', 'Thanks Wren.', 'So what game?',
                  'Pick something short.', 'It settled it.'):
            self.assertFalse(setup_tail(t), t)


class ContinuationPromptTests(TestCase):
    """Real prepare/context/build_messages methods, CPU-only fake transport and LLM."""
    question = '[S1] Juniper what is 2 plus 3?'
    instruction = '[S1] Please keep your answer short.'

    def pipeline(self):
        import ast
        import re
        from pathlib import Path
        from types import MethodType
        from typing import Optional, List, Dict
        from conversation_log import ConversationLog
        from floor import ConversationFloor
        from test_conversation_log import Clock
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / 'speech_pipeline_manager.py').read_text(encoding='utf-8'))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('RunningGeneration', 'PipelineRequest')]
        ns = dict(Optional=Optional, time=time, uuid=__import__('uuid'), ResponseDecision=ResponseDecision, threading=threading, Queue=Queue)
        exec(compile(ast.Module(body=classes, type_ignores=[]), 'speech_pipeline_manager.py', 'exec'), ns)
        build = load_method('llm_module.py', 'LLM', 'build_messages',
                            dict(Optional=Optional, List=List, Dict=Dict, re=re))
        calls = []
        llm = SimpleNamespace(system_prompt_message={'role':'system', 'content':'test'}, no_think=False)
        def generate(**kw):
            calls.append(build(llm, **{k:v for k,v in kw.items() if k not in ('prefetch_tool','request_id')}, log=False))
            return iter(['[HOLD]'])
        llm.generate = generate
        clock = Clock()
        m = SimpleNamespace(running_generation=None, generation_counter=0, llm=llm,
            convo=ConversationLog(agent_name='Juniper', clock=clock), convo_enabled=True, history=[],
            floor=ConversationFloor(), agent=SimpleNamespace(profile=SimpleNamespace(names=('juniper',))),
            current_persona='test', threads=SimpleNamespace(active=None), people=None, cutoff=None,
            check_abort_lock=threading.Lock(), text_similarity=SimpleNamespace(calculate_similarity=lambda a,b:0))
        for name in ('llm_answer_ready_event', 'generator_ready_event', 'stop_llm_request_event',
                     'stop_llm_finished_event', 'stop_tts_quick_request_event', 'stop_tts_quick_finished_event',
                     'stop_tts_final_request_event', 'stop_tts_final_finished_event',
                     'abort_completed_event', 'abort_block_event'):
            setattr(m, name, event())
        trace = SimpleNamespace(prepared=lambda *a:None, mark=lambda *a:None)
        bindings = dict(re=re, time=time, threading=threading, RunningGeneration=ns['RunningGeneration'],
                        PipelineRequest=ns['PipelineRequest'], Empty=__import__('queue').Empty,
                        _turn_trace=trace, filter_response=filter_response, RETRY_TEMPERATURE=0.95)
        for name in ('process_prepare_generation', 'prepare_generation', '_request_processing_worker',
                     'check_abort', '_llm_context', '_llm_context_locked', 'note_audio_committed'):
            setattr(m, name, MethodType(load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', name, bindings), m))
        for name in ('_same_utterance', '_adds_content', '_trim_transcript'):
            setattr(m, name, load_method('speech_pipeline_manager.py', 'SpeechPipelineManager', name, bindings))
        m.abort_generation = lambda **kw:setattr(m, 'running_generation', None)
        m._room_context = lambda txt:''
        m._with_people = lambda room, *a:room
        m._finish_context = lambda h,r:(h,r)
        m._guard_lines = lambda txt:(set(), [])
        m._memory_context = lambda txt:('', [])
        m._attach_board = lambda *a:False
        m._live_on = lambda:False
        m._own_look_ok = lambda *a:False
        m.requests_queue, m.shutdown_event, m.previous_request = Queue(), event(), None
        m._request_type = ns['PipelineRequest']
        def enqueue(txt, **kw):
            from unittest.mock import patch
            with patch.dict(__import__('sys').modules, {'call_boundary':SimpleNamespace(activity=lambda *a:None)}):
                m.prepare_generation(txt, **kw)
        m.enqueue = enqueue
        for name in ('_check_bait', '_think_snapshot', '_look_prefetch', '_code_prefetch', '_room_prefetch',
                     '_unseen_note', '_recheck_note', '_game_note', '_depth_note', '_memory_cue', '_engage_note',
                     '_live_status_note', '_think_handover', '_clear_think_handover'):
            setattr(m, name, lambda *a:None)
        return m, clock, calls

    def server_callbacks(self, m):
        """Real final/interrupt/reset callbacks, only ancillary bookkeeping faked."""
        import ast
        from pathlib import Path
        from types import MethodType
        source = Path(__file__).resolve().parents[1] / 'server.py'
        cls = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                   if isinstance(n, ast.ClassDef) and n.name == 'TranscriptionCallbacks')
        names = ('send_final_assistant_answer', '_commit_final_assistant_answer',
                 'interrupt_agent', 'abort_generations', 'reset_state')
        nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
        reset_calls = []
        app = SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=m, CallBridge=None,
            AudioInputProcessor=SimpleNamespace(interrupted=False, abort_generation=lambda:reset_calls.append('reset'))))
        ns = dict(threading=threading, logger=logging.getLogger('test'), Colors=Colors,
                  app=app, _rec_event=lambda *a,**kw:None)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        cb = SimpleNamespace(app=app, assistant_answer='', final_assistant_answer='',
            final_assistant_answer_sent=False, message_queue=asyncio.Queue(),
            tts_to_client=True, tts_chunk_sent=True, tts_client_playing=False, interruption_time=0,
            is_hot=False, synthesis_started=False, user_history_committed=True, user_finished_turn=True)
        for name in names: setattr(cb, name, MethodType(ns[name], cb))
        m.note_reply_for_glance = lambda *a:None
        m.note_reply_for_promise = lambda *a:None
        m.trim_history = lambda:None
        m.reply_spoken = lambda *a:None
        m.board = lambda:SimpleNamespace(spoke=lambda *a:None)
        m.dynamics = SimpleNamespace(on_agent_turn=lambda **kw:None)
        m.is_valid_gen = lambda:m.running_generation is not None and not m.running_generation.abortion_started
        return cb, reset_calls

    def test_shared_forced_callback_and_interrupt_reject_resume_after_human_admission(self):
        from interrupts import CutOff
        from unittest.mock import patch
        for through_interrupt in (False, True):
            for heard in (None, 'Earlier words.'):
                with self.subTest(interrupt=through_interrupt, heard=heard):
                    m, _, _ = self.pipeline()
                    cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
                    m.cutoff = cut
                    m.process_prepare_generation(cut.cue(), resume_cut=cut)
                    g = m.running_generation
                    g.quick_answer = 'Earlier unfinished thought.'
                    g.decision.target = 'S1'
                    cb, _ = self.server_callbacks(m)
                    cb._cut_said = heard
                    m.enqueue('[S1] A new human request instead.')
                    with patch.dict(__import__('sys').modules, {'call_summary':SimpleNamespace(note=lambda *a,**kw:None)}):
                        if through_interrupt: cb.interrupt_agent('new human took the floor')
                        else: cb.send_final_assistant_answer(forced=True)
                    self.assertTrue(g.resume_cancelled)
                    self.assertFalse(g.final_answer_committed)
                    self.assertEqual(m.history, [])
                    self.assertFalse(any(e.role == 'assistant' for e in m.convo.ledger))
                    self.assertFalse(any(msg['type'] == 'final_assistant_answer' for msg in cb.message_queue._queue))

    def test_history_commit_is_generation_owned_and_dedup_does_not_suppress_successor(self):
        from unittest.mock import patch
        m, _, _ = self.pipeline()
        m.process_prepare_generation('[S1] The original human request.')
        old = m.running_generation
        old.quick_answer, old.decision.target = 'First answer.', 'S1'
        cb, _ = self.server_callbacks(m)
        with patch.dict(__import__('sys').modules, {'call_summary':SimpleNamespace(note=lambda *a,**kw:None)}):
            cb.send_final_assistant_answer(forced=True)
            cb.send_final_assistant_answer(forced=True)
            self.assertEqual(len(m.history), 1)
            m.enqueue('[S1] A different human request.')
            request = m.requests_queue.get_nowait()
            m.process_prepare_generation(request.data)
            successor = m.running_generation
            successor.quick_answer, successor.decision.target = 'Second answer.', 'S1'
            cb.send_final_assistant_answer(forced=True, generation=old)
            self.assertEqual(len(m.history), 1)
            cb.send_final_assistant_answer()
        self.assertEqual(len(m.history), 2)
        self.assertEqual(cb.message_queue.qsize(), 2)
        self.assertTrue(successor.final_answer_committed)
        # Connection text from an old generation is never a substitute for
        # an empty new generation's own buffers, including forced paths.
        m.process_prepare_generation('[S1] One more different request.')
        cb.assistant_answer = 'Unowned old partial.'
        cb.send_final_assistant_answer(forced=True)
        self.assertEqual(cb.message_queue.qsize(), 2)

    def test_sender_normal_and_hold_retirement_preserve_worker_allocated_successor(self):
        import ast
        from pathlib import Path
        from interrupts import CutOff
        from unittest.mock import patch
        source = Path(__file__).resolve().parents[1] / 'server.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'send_tts_chunks')
        for hold in (False, True):
            with self.subTest(hold=hold):
                ns = dict(asyncio=asyncio, time=time, threading=threading, logger=logging.getLogger('test'),
                    FastAPI=object, Colors=Colors, Empty=__import__('queue').Empty,
                    IDLE_POLL_S=0.005, TTS_FINAL_TIMEOUT=1, ENABLE_CALL_BRIDGE=False,
                    WebSocketDisconnect=type('Disconnect',(Exception,),{}))
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
                async def scenario():
                    m, _, _ = self.pipeline()
                    cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
                    m.cutoff = cut
                    m.process_prepare_generation(cut.cue(), resume_cut=cut)
                    original = m.running_generation
                    original.quick_answer = 'An earlier answer.'
                    original.decision.target = 'S1'
                    original.quick_answer_first_chunk_ready = True
                    original.quick_answer_provided = True
                    original.audio_final_finished = True
                    original.response_held = hold
                    original.llm_finished = True
                    cb, resets = self.server_callbacks(m)
                    done, successors, errors = asyncio.Event(), [], []
                    human = '[S1] A new human question after the old sender snapshot.'
                    def replace(*args):
                        def worker():
                            try:
                                m.enqueue(human)
                                request = m.requests_queue.get_nowait()
                                m.process_prepare_generation(request.data)
                                successors.append(m.running_generation)
                            except Exception as error:
                                errors.append(error)
                        thread = threading.Thread(target=worker)
                        thread.start(); thread.join(2)
                        if thread.is_alive(): errors.append(RuntimeError('replacement worker blocked'))
                        done.set()
                    m.threads.answered = replace
                    m.threads.retire_active = replace
                    task = asyncio.create_task(ns['send_tts_chunks'](cb.app, cb.message_queue, cb))
                    try:
                        await asyncio.wait_for(done.wait(), 2)
                        await asyncio.sleep(0.01)
                        self.assertEqual(errors, [])
                        self.assertEqual(len(successors), 1)
                        self.assertIs(m.running_generation, successors[0])
                        self.assertEqual(m.running_generation.text, human)
                        self.assertTrue(cb.tts_to_client)
                        self.assertTrue(cb.user_history_committed)
                        self.assertEqual(resets, [])
                    finally:
                        task.cancel(); await task
                with patch.dict(__import__('sys').modules, {'call_summary':SimpleNamespace(note=lambda *a,**kw:None)}):
                    asyncio.run(scenario())

    def test_resume_is_its_own_synthetic_request_without_human_continuation(self):
        from interrupts import CutOff
        m, _, calls = self.pipeline()
        human = '[S1] Answer the new question briefly.'
        m.convo.add_user('S1', human)
        cut = CutOff('Earlier words', 'a separate unfinished old thought', target='S1')
        m.cutoff = cut
        m.process_prepare_generation(cut.cue(), resume_cut=cut)
        self.assertEqual(self.users(calls)[-1]['content'], cut.cue())
        self.assertEqual(m.running_generation.text, cut.cue())
        self.assertIs(m.running_generation.resume_cut, cut)

    def test_start_resume_queues_exact_cutoff_identity(self):
        import ast
        from pathlib import Path
        from unittest.mock import patch
        from interrupts import CutOff
        source = Path(__file__).resolve().parents[1] / 'server.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'start_resume')
        ns = {'logger':logging.getLogger('test')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
        m, _, _ = self.pipeline()
        cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
        m.cutoff = cut
        callbacks = SimpleNamespace(reset_state=lambda:None)
        with patch.dict(__import__('sys').modules, {'call_boundary':SimpleNamespace(activity=lambda *a:None)}):
            self.assertTrue(ns['start_resume'](m, callbacks))
        request = m.requests_queue.get_nowait()
        self.assertIs(request.resume_cut, cut)
        self.assertEqual(request.data, cut.cue())
        self.assertTrue(cut.gap_tried)

    def test_queue_prefers_human_over_resume_in_both_orders(self):
        from interrupts import CutOff
        for resume_first in (False, True):
            with self.subTest(resume_first=resume_first):
                m, _, calls = self.pipeline()
                cut = CutOff('Earlier words', 'an unfinished thought from before', target='S1')
                m.cutoff = cut
                human = '[S1] Please answer my new question.'
                if resume_first: m.enqueue(cut.cue(), resume_cut=cut)
                m.enqueue(human)
                m.enqueue(human)  # duplicate/final ASR still invalidates the idle cutoff
                # A producer that already captured the old cutoff can race the
                # human enqueue. The public entry rejects it; also exercise a
                # stale item already handed to the queue by that producer.
                m.enqueue(cut.cue(), resume_cut=cut)
                m.requests_queue.put(m._request_type('prepare', cut.cue(), resume_cut=cut))
                m.abort_block_event.set()
                done = event()
                generate = m.llm.generate
                def record(**kw):
                    result = generate(**kw)
                    done.set()
                    return result
                m.llm.generate = record
                worker = threading.Thread(target=m._request_processing_worker, daemon=True)
                worker.start()
                try:
                    self.assertTrue(done.wait(2))
                finally:
                    m.shutdown_event.set(); worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(calls), 1)
                self.assertEqual(self.users(calls)[-1]['content'], human)
                self.assertIsNone(m.cutoff)

    def test_resume_dequeued_before_human_is_invalidated_while_worker_waits(self):
        from interrupts import CutOff
        m, _, calls = self.pipeline()
        cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
        m.cutoff = cut
        m.enqueue(cut.cue(), resume_cut=cut)
        entered, done = event(), event()
        class Gate(threading.Event):
            def wait(self, timeout=None):
                entered.set()
                return super().wait(timeout)
        m.abort_block_event = Gate()
        generate = m.llm.generate
        def record(**kw):
            result = generate(**kw); done.set(); return result
        m.llm.generate = record
        worker = threading.Thread(target=m._request_processing_worker, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            human = '[S1] A completely new human request.'
            m.enqueue(human)
            m.abort_block_event.set()
            self.assertTrue(done.wait(2))
        finally:
            m.shutdown_event.set(); m.abort_block_event.set(); worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.users(calls)[-1]['content'], human)

    def test_human_during_resume_context_build_prevents_old_llm_request(self):
        from interrupts import CutOff
        m, _, calls = self.pipeline()
        cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
        m.cutoff = cut
        human = '[S1] Answer this new request instead.'
        context = m._llm_context
        def interrupted_context(txt, **kw):
            result = context(txt, **kw)
            m.enqueue(human)
            return result
        m._llm_context = interrupted_context
        m.process_prepare_generation(cut.cue(), resume_cut=cut)
        self.assertEqual(calls, [])
        original = m.running_generation
        self.assertTrue(original.resume_cancelled)
        self.assertEqual(original.text, cut.cue())
        m._llm_context = context
        request = m.requests_queue.get_nowait()
        m.process_prepare_generation(request.data)
        self.assertEqual(self.users(calls)[-1]['content'], human)

    def users(self, calls):
        return [x for x in calls[-1] if x['role'] == 'user']

    def test_precommit_and_postcommit_deliver_identical_complete_turn_once(self):
        observed = []
        for committed in (False, True):
            m, clock, calls = self.pipeline()
            m.convo.add_user('S1', self.question)
            clock.tick(2.43)
            if committed:
                m.convo.add_user('S1', self.instruction)
            ledger_before = [(e.text, e.t_end) for e in m.convo.ledger]
            m.process_prepare_generation(self.instruction)
            self.assertEqual(m.running_generation.text, self.instruction)
            self.assertEqual(ledger_before, [(e.text, e.t_end) for e in m.convo.ledger])
            observed.append(self.users(calls))
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(observed[0], [{'role':'user', 'content':self.question + ' Please keep your answer short.'}])

    def test_duplicate_and_growing_partial_never_repeat_question(self):
        m, clock, calls = self.pipeline()
        m.convo.add_user('S1', self.question)
        clock.tick(2.43)
        partial = '[S1] Please keep'
        m.process_prepare_generation(partial)
        m.process_prepare_generation(partial)
        self.assertEqual(len(calls), 1)
        m.process_prepare_generation(self.instruction)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.users(calls)[-1]['content'], self.question + ' Please keep your answer short.')
        self.assertEqual(m.running_generation.text, self.instruction)
        # A cumulative ASR partial is also not appended to itself.
        m.running_generation = None
        m.process_prepare_generation(self.question + ' Please keep your answer short.')
        self.assertEqual(self.users(calls)[-1]['content'].count('what is 2 plus 3?'), 1)

    def test_other_unknown_expired_answered_and_audio_committed_stay_separate(self):
        for reason in ('other', 'unknown', 'expired', 'answered', 'foreign', 'audio'):
            with self.subTest(reason=reason):
                m, clock, calls = self.pipeline()
                m.convo.add_user('S1', self.question)
                clock.tick(2.43 if reason != 'expired' else 4.01)
                current = self.instruction
                if reason == 'other': current = current.replace('[S1]', '[S2]')
                if reason == 'unknown': current = current.replace('[S1]', '[user]')
                if reason == 'answered': m.convo.add_agent('S1', 'Five.')
                if reason == 'foreign': m.convo.add_user('S2', 'Wait for a moment.')
                if reason == 'audio':
                    m.process_prepare_generation(self.question)
                    m.running_generation.audio_committed = True
                m.process_prepare_generation(current)
                self.assertEqual(self.users(calls)[-1]['content'], current)
                self.assertIn(self.question, [x['content'] for x in self.users(calls)])

    def test_synthesized_but_unreleased_reply_allows_continuation(self):
        m, clock, calls = self.pipeline()
        m.convo.add_user('S1', self.question)
        m.process_prepare_generation(self.question)
        m.running_generation.tts_quick_started = True
        m.running_generation.quick_answer_first_chunk_ready = True
        clock.tick(2.43)
        m.process_prepare_generation(self.instruction)
        self.assertEqual(self.users(calls)[-1]['content'], self.question + ' Please keep your answer short.')

    def test_single_payload_change_reaches_new_llm_prompt_before_audio_commit(self):
        changes = [
            ('Juniper, stop. Instead, just say the number.', 'Juniper, stop. Instead, just say the number 7.'),
            ('Juniper, read the value 41.', 'Juniper, read the value 42.'),
            ('Juniper, describe the weather in', 'Juniper, describe the weather in Oslo.'),
            ('Juniper, should I close it?', 'Juniper, should I not close it?'),
            ('Juniper, I can go.', "Juniper, I can't go."),
        ]
        for before, after in changes:
            with self.subTest(before=before, after=after):
                m, _, calls = self.pipeline()
                # A semantic delta must override even a high similarity score.
                m.text_similarity.calculate_similarity = lambda a,b:1.0
                m.process_prepare_generation('[S1] ' + before)
                original = m.running_generation
                original.tts_quick_started = True
                original.quick_answer_first_chunk_ready = True
                m.process_prepare_generation('[S1] ' + after)
                self.assertEqual(len(calls), 2)
                self.assertIsNot(m.running_generation, original)
                self.assertEqual(original.text, '[S1] ' + before)
                self.assertEqual(m.running_generation.text, '[S1] ' + after)
                self.assertEqual(self.users(calls)[-1]['content'], '[S1] ' + after)

    def test_incidental_or_regressive_partials_keep_original_prompt_text(self):
        changes = [
            ('Juniper tell me the number 42.', 'Juniper tell me the number 42.'),
            ('Juniper tell me the number 42.', 'Juniper, tell me the number 42!'),
            ('Juniper tell me the number 42.', 'Juniper tell me the'),
            ('Juniper tell me the numb', 'Juniper tell me the number'),
            ('Juniper tell me', 'Juniper tell me the'),
            ('Juniper tell me the number 42.', 'Juniper tell me the number 42. [noise] um'),
        ]
        for before, after in changes:
            with self.subTest(before=before, after=after):
                m, _, calls = self.pipeline()
                m.process_prepare_generation('[S1] ' + before)
                original = m.running_generation
                m.process_prepare_generation('[S1] ' + after)
                self.assertIs(m.running_generation, original)
                self.assertEqual(len(calls), 1)
                self.assertEqual(original.text, '[S1] ' + before)
                self.assertEqual(self.users(calls)[-1]['content'], '[S1] ' + before)

    def test_same_utterance_after_audio_commit_keeps_owner_and_original_text(self):
        m, _, calls = self.pipeline()
        before = '[S1] Juniper tell me the number'
        m.process_prepare_generation(before)
        original = m.running_generation
        m.note_audio_committed(original)
        m.process_prepare_generation(before + ' 23.')
        self.assertIs(m.running_generation, original)
        self.assertEqual(len(calls), 1)
        self.assertEqual(original.text, before)

    def test_different_utterance_after_audio_commit_still_aborts_owner(self):
        m, _, calls = self.pipeline()
        m.process_prepare_generation('[S1] Juniper explain what stars are made of.')
        original = m.running_generation
        m.note_audio_committed(original)
        after = '[S1] Stop. Choose a city instead.'
        m.process_prepare_generation(after)
        self.assertIsNot(m.running_generation, original)
        self.assertEqual(self.users(calls)[-1]['content'], after)

    def test_sender_marks_audio_committed_only_after_transport_handoff(self):
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'server.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'send_tts_chunks')
        for native in (False, True):
            with self.subTest(native=native):
                ns = dict(asyncio=asyncio, time=time, logger=logging.getLogger('test'), FastAPI=object,
                    Colors=Colors, threading=threading, Empty=__import__('queue').Empty, IDLE_POLL_S=0.005,
                    TTS_FINAL_TIMEOUT=1, ENABLE_CALL_BRIDGE=native,
                    WebSocketDisconnect=type('Disconnect',(Exception,),{}))
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'server.py', 'exec'), ns)
                async def scenario():
                    sent = []
                    chunks = Queue()
                    chunks.put(b'fake-pcm')
                    g = SimpleNamespace(audio_committed=False, abortion_started=False, audio_quick_finished=False,
                        response_held=False, quick_answer_first_chunk_ready=False, audio_chunks=chunks,
                        quick_answer_provided=True, audio_final_finished=False, tts_quick_allowed_event=event())
                    mgr, clock, _ = self.pipeline()
                    mgr.running_generation = g
                    mgr.is_valid_gen = lambda:True
                    bridge = SimpleNamespace(play_tts=sent.append) if native else None
                    app = SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=mgr, CallBridge=bridge,
                        AudioInputProcessor=SimpleNamespace(interrupted=False),
                        Upsampler=SimpleNamespace(get_base64_chunk=lambda chunk:'encoded')))
                    cb = SimpleNamespace(interruption_time=0, tts_to_client=True, tts_client_playing=False,
                        tts_chunk_sent=True, is_hot=False, synthesis_started=False)
                    outgoing = asyncio.Queue()
                    task = asyncio.create_task(ns['send_tts_chunks'](app, outgoing, cb))
                    try:
                        await asyncio.sleep(0.02)
                        self.assertFalse(g.audio_committed)
                        self.assertIsNone(mgr.convo.last_audio_committed_at)
                        g.quick_answer_first_chunk_ready = True
                        async def wait_for_commit():
                            while not g.audio_committed:
                                await asyncio.sleep(0.001)
                        await asyncio.wait_for(wait_for_commit(), 1)
                        self.assertEqual(mgr.convo.last_audio_committed_at, clock())
                        self.assertEqual(sent, [b'fake-pcm'] if native else [])
                        self.assertEqual(outgoing.qsize(), 0 if native else 1)
                    finally:
                        task.cancel()
                        await task
                asyncio.run(scenario())

    def test_output_barrier_survives_abort_but_allows_a_new_unanswered_turn(self):
        for emitted in (False, True):
            for committed in (False, True):
                with self.subTest(emitted=emitted, committed=committed):
                    m, clock, calls = self.pipeline()
                    m.convo.add_user('S1', self.question)
                    m.process_prepare_generation(self.question)
                    clock.tick(0.5)
                    if emitted:
                        m.note_audio_committed(m.running_generation)
                    m.abort_generation()
                    clock.tick(1)
                    if committed:
                        m.convo.add_user('S1', self.instruction)
                    m.process_prepare_generation(self.instruction)
                    expected = self.instruction if emitted else self.question + ' Please keep your answer short.'
                    self.assertEqual(self.users(calls)[-1]['content'], expected)
                    if emitted:
                        # The later instruction starts a fresh unanswered turn, so a
                        # third fragment can join it without reviving the old question.
                        if not committed:
                            m.convo.add_user('S1', self.instruction)
                        m.abort_generation()
                        clock.tick(0.5)
                        m.process_prepare_generation('[S1] Just one word.')
                        self.assertEqual(self.users(calls)[-1]['content'], self.instruction + ' Just one word.')

    def test_new_human_blocks_resume_audio_and_history_commit_after_sender_snapshot(self):
        import ast
        from pathlib import Path
        from interrupts import CutOff
        source = Path(__file__).resolve().parents[1] / 'server.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'send_tts_chunks')
        for native in (False, True):
            for boundary in ('audio', 'history'):
                with self.subTest(native=native, boundary=boundary):
                    ns = dict(asyncio=asyncio, time=time, threading=threading, logger=logging.getLogger('test'),
                        FastAPI=object, Colors=Colors, Empty=__import__('queue').Empty,
                        IDLE_POLL_S=0.005, TTS_FINAL_TIMEOUT=1, ENABLE_CALL_BRIDGE=native,
                        WebSocketDisconnect=type('Disconnect',(Exception,),{}))
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
                    async def scenario():
                        m, _, calls = self.pipeline()
                        cut = CutOff('Earlier words', 'an unfinished old thought here', target='S1')
                        m.cutoff = cut
                        m.process_prepare_generation(cut.cue(), resume_cut=cut)
                        g = m.running_generation
                        human = '[S1] Please answer the new human request.'
                        admitted = asyncio.Event()
                        class RacingQueue(Queue):
                            def get_nowait(self):
                                # The sender already captured g and passed its
                                # first guard; admission wins before commit.
                                m.enqueue(human); admitted.set()
                                return super().get_nowait()
                        g.audio_chunks = RacingQueue()
                        if boundary == 'audio': g.audio_chunks.put(b'fake-pcm')
                        g.quick_answer_first_chunk_ready = True
                        g.quick_answer_provided = True
                        g.audio_final_finished = boundary == 'history'
                        sent, history = [], []
                        bridge = SimpleNamespace(play_tts=sent.append, end_reply=lambda:None,
                            playout_stats=lambda:{}) if native else None
                        m.is_valid_gen = lambda:True
                        app = SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=m, CallBridge=bridge,
                            AudioInputProcessor=SimpleNamespace(interrupted=False),
                            Upsampler=SimpleNamespace(get_base64_chunk=lambda chunk:'encoded')))
                        cb = SimpleNamespace(interruption_time=0, tts_to_client=True, tts_client_playing=False,
                            tts_chunk_sent=True, is_hot=False, synthesis_started=False,
                            send_final_assistant_answer=lambda:history.append('committed'))
                        outgoing = asyncio.Queue()
                        task = asyncio.create_task(ns['send_tts_chunks'](app, outgoing, cb))
                        try:
                            await asyncio.wait_for(admitted.wait(), 1)
                            await asyncio.sleep(0.01)
                            self.assertTrue(g.resume_cancelled)
                            self.assertFalse(g.audio_committed)
                            self.assertEqual(sent, [])
                            self.assertEqual(history, [])
                            self.assertEqual(outgoing.qsize(), 0)
                        finally:
                            task.cancel(); await task
                        self.assertEqual(g.text, cut.cue())
                        request = m.requests_queue.get_nowait()
                        m.process_prepare_generation(request.data)
                        self.assertEqual(self.users(calls)[-1]['content'], human)
                    asyncio.run(scenario())
