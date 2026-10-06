"""CPU-only regression tests; execute production methods without model startup."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from floor import ConversationFloor  # noqa: E402


def load_method(filename, classname, method, bindings=None):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    node = next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method)
    ns = {'logger': logging.getLogger('test'), **(bindings or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), ns)
    return ns[method]


class RoutingTests(unittest.TestCase):
    def test_followup_reaches_model_even_if_old_regex_holds(self):
        prepared = []
        mgr = SimpleNamespace(should_speak=lambda *a, **k: False, prepare_generation=prepared.append, dynamics=SimpleNamespace(agent_names=('ivy',)), floor=ConversationFloor(), running_generation=None)
        callback = SimpleNamespace(_live_speaker=lambda: None, _text_echo=lambda t: False, _monologue_hold=lambda t, s: False, app=SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=mgr, CallBridge=None)))
        fn = load_method('server.py', 'TranscriptionCallbacks', 'on_potential_sentence')
        fn(callback, "I'm not babe.")
        self.assertEqual(prepared, ["I'm not babe."])

    def test_explicit_silence_request_holds_without_llm(self):
        prepared = []
        mgr = SimpleNamespace(prepare_generation=prepared.append, dynamics=SimpleNamespace(agent_names=('ivy',)), floor=ConversationFloor(), running_generation=None)
        callback = SimpleNamespace(_live_speaker=lambda: None, _text_echo=lambda t: False, _monologue_hold=lambda t, s: False, app=SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=mgr, CallBridge=None)))
        fn = load_method('server.py', 'TranscriptionCallbacks', 'on_potential_sentence')
        for text in ("Ivy, stop talking for a moment.", "Hey Ivy, shut up.", "ivy be quiet"):
            fn(callback, text)
            self.assertEqual(prepared, [], f"silence request {text!r} reached the LLM")
        # Quiet mode persists: an unnamed line after a named silence request holds...
        fn(callback, "What's the weather like?")
        self.assertEqual(prepared, [])
        # ...until the agent is invited back by name, which reaches Model.
        fn(callback, "Ivy, what's the weather like?")
        self.assertEqual(prepared, ["Ivy, what's the weather like?"])
        # A fresh floor (no silence request) never blocks ordinary lines.
        mgr.floor = ConversationFloor()
        fn(callback, "What's the weather like?")
        self.assertEqual(prepared[-1], "What's the weather like?")

    def test_other_addressee_holds_and_retires_stale_generation(self):
        prepared, aborted = [], []
        mgr = SimpleNamespace(prepare_generation=prepared.append, dynamics=SimpleNamespace(agent_names=('ivy',)), floor=ConversationFloor(),
                              running_generation=object(),
                              abort_generation=lambda **kw: aborted.append(kw.get('reason')))
        callback = SimpleNamespace(_live_speaker=lambda: None, _text_echo=lambda t: False, _monologue_hold=lambda t, s: False, app=SimpleNamespace(state=SimpleNamespace(SpeechPipelineManager=mgr, CallBridge=None)))
        fn = load_method('server.py', 'TranscriptionCallbacks', 'on_potential_sentence')
        fn(callback, "Can you pass the salt, Jordan?")
        self.assertEqual(prepared, [])
        self.assertEqual(len(aborted), 1)
        mgr.running_generation = None
        fn(callback, "Ivy, can you pass the salt?")
        self.assertEqual(prepared, ["Ivy, can you pass the salt?"])


class PipelineTests(unittest.TestCase):
    def test_silent_generation_finishes_without_tts(self):
        import threading
        from response_decision import ResponseDecision, finish_silent_generation
        gen = SimpleNamespace(decision=ResponseDecision(action='HOLD'), quick_answer='', final_answer='',
                              tts_quick_finished_event=threading.Event(), tts_final_finished_event=threading.Event())
        self.assertTrue(finish_silent_generation(gen))
        self.assertTrue(gen.response_held)
        self.assertTrue(gen.audio_quick_finished and gen.audio_final_finished)
        self.assertTrue(gen.tts_quick_finished_event.is_set() and gen.tts_final_finished_event.is_set())

    def test_current_turn_is_not_lost_after_held_room_message(self):
        from llm_module import LLM
        captured = []
        llm = LLM(backend='ollama', model='local-model', no_think=True)
        llm._lazy_initialize_clients = lambda: True
        llm.ollama_session = object()
        llm.tools = [{'type': 'function'}]
        llm.tool_executor = lambda *args: ''
        def capture(url, messages, options, request_id, prefetch=None):
            captured.append(messages)
            yield '[HOLD]'
        llm._generate_ollama_agentic = capture
        history = [{'role': 'user', 'content': '[S2] The trailer is late.'}]
        list(llm.generate('Ivy, what did S2 say?', history=history, participation=True))
        self.assertIn('Ivy, what did S2 say?', captured[0][-1]['content'])
        self.assertEqual(history[0]['content'], '[S2] The trailer is late.')
        self.assertIn('PARTICIPATION PROTOCOL', captured[0][0]['content'])


class ProtocolTests(unittest.TestCase):
    def test_fragmented_speak_never_leaks_header(self):
        from response_decision import ResponseDecision, filter_response
        for split in range(len('[SPEAK to=S1]')):
            decision = ResponseDecision()
            wire = '[SPEAK to=S1]\nSorry, I won\'t call you that.'
            output = ''.join(filter_response(iter([wire[:split], wire[split:]]), decision))
            self.assertEqual(output, "Sorry, I won't call you that.")
            self.assertEqual((decision.action, decision.target), ('SPEAK', 'S1'))

    def test_hold_discards_all_body_and_closes_stream(self):
        from response_decision import ResponseDecision, filter_response
        closed = []
        def source():
            try:
                yield '[HO'
                yield 'LD]\nThis must never be spoken.'
                raise AssertionError('must not consume after HOLD')
            finally:
                closed.append(True)
        decision = ResponseDecision()
        self.assertEqual(list(filter_response(source(), decision)), [])
        self.assertEqual(decision.action, 'HOLD')
        self.assertEqual(closed, [True])

    def test_malformed_or_incomplete_headers_fail_closed(self):
        from response_decision import ResponseDecision, filter_response
        for wire in ('', '[SPE', 'Sure, hello.', '[SPEAK to=S1', 'x' * 100):
            d = ResponseDecision()
            self.assertEqual(list(filter_response(iter([wire]), d)), [])
            self.assertEqual(d.action, 'INVALID')


if __name__ == '__main__':
    unittest.main()


class CancelledDecisionTests(unittest.TestCase):
    def test_cancelled_empty_is_not_invalid(self):
        from response_decision import ResponseDecision, filter_response
        d = ResponseDecision(); d.cancelled = True
        self.assertEqual(''.join(filter_response(iter([]), d)), '')
        self.assertEqual(d.action, 'CANCELLED')

    def test_uncancelled_empty_is_still_invalid(self):
        from response_decision import ResponseDecision, filter_response
        d = ResponseDecision()
        ''.join(filter_response(iter([]), d))
        self.assertEqual(d.action, 'INVALID')
