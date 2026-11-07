"""Screen-look tool, CPU only: no real capture, no model, no server boot."""
import base64, io, sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import screen
from tests.test_llm_lifecycle import LifecycleTests, Response, packet

LOOK = {"function": {"name": "look_at_screen", "arguments": {"reason": "error"}}}


def fake_grab(w=3440, h=1440):
    return lambda: Image.new("RGB", (w, h), (40, 40, 40))


class ScreenToolTests(unittest.TestCase):
    def setUp(self):
        self.was = screen.status()["enabled"]
        self.saved = screen.LAST_PATH
        screen.LAST_PATH = Path(__file__).resolve().parent / "reports" / "_unit_last_screen.jpg"

    def tearDown(self):
        screen.set_enabled(self.was)
        screen.LAST_PATH.unlink(missing_ok=True)
        screen.LAST_PATH = self.saved

    def test_disabled_refuses_without_capturing(self):
        screen.set_enabled(False)
        called = []
        out = screen.capture("x", grab=lambda: called.append(1))
        self.assertEqual(called, [])
        self.assertNotIn("images", out)
        self.assertIn("switched off", out["text"])

    def test_text_only_model_disables_eyes_without_changing_owner_choice(self):
        from unittest import mock
        model = {'vision_supported': False, 'model': 'text-model', 'unavailable_reason': 'text-model is text-only.'}
        screen.set_enabled(True)
        with mock.patch.object(screen, 'model_vision', return_value=model):
            self.assertFalse(screen.status()['enabled'])
            self.assertTrue(screen.status()['requested_enabled'])
            grab = mock.Mock()
            result = screen.capture('look', grab=grab)
            grab.assert_not_called()
            self.assertNotIn('images', result)
            self.assertIn('text-only', result['text'])
        self.assertTrue(screen.status()['enabled'])

    def test_vision_metadata_and_model_changes_do_not_reuse_previous_capability(self):
        from unittest import mock
        self.assertTrue(screen._declared_vision({'capabilities': ['completion', 'vision']}))
        self.assertFalse(screen._declared_vision({'capabilities': ['completion', 'tools']}))
        self.assertTrue(screen._declared_vision({'architecture': {'input_modalities': ['text', 'image']}}))
        self.assertIsNone(screen._declared_vision({}))
        current = ('local', 'http://local', 'new-text-model')
        with mock.patch.object(screen, '_model_identity', return_value=(current, None)), \
                mock.patch.object(screen, '_vision', {'key': ('local', 'http://local', 'old-vision-model'),
                    'supported': True, 'reason': 'confirmed'}), mock.patch('requests.post') as http:
            state = screen.model_vision()
            self.assertIsNone(state['vision_supported'])
            self.assertEqual(state['model'], 'new-text-model')
            screen.status()
            http.assert_not_called()

    def test_explicit_provider_image_capability_needs_no_network(self):
        from unittest import mock
        key = ('remote', 'http://chosen', 'chosen-model', False)
        with mock.patch('requests.get') as get:
            self.assertFalse(screen._probe_vision(key, {'vision': False}))
            self.assertTrue(screen._probe_vision(key, {'vision': True}))
            get.assert_not_called()

    def test_active_local_fallback_does_not_inherit_remote_vision_support(self):
        from unittest import mock
        remote = SimpleNamespace(cfg={'url': 'http://remote', 'model': 'vision-model', 'vision': True},
                                 down_until=10**12)
        llm = SimpleNamespace(remote=remote, model='local-text', effective_ollama_url='http://local')
        with mock.patch.object(screen, '_model_provider', lambda: llm):
            key, cfg = screen._model_identity()
            self.assertEqual(key, ('local', 'http://local', 'local-text'))
            self.assertIsNone(cfg)
            remote.cfg['allow_local_fallback'] = False
            self.assertEqual(screen._model_identity()[0][0], 'remote')

    def test_ultrawide_stays_inside_token_budget(self):
        screen.set_enabled(True)
        out = screen.capture("error", grab=fake_grab())
        img = Image.open(io.BytesIO(base64.b64decode(out["images"][0])))
        w, h = img.size
        self.assertLessEqual(w * h, screen.MAX_PIXELS * 1.01)
        self.assertGreater(w, 1024)  # area budget keeps ultrawide text wide, not a sliver
        self.assertTrue(screen.LAST_PATH.exists())

    def test_capture_failure_is_spoken_not_raised(self):
        screen.set_enabled(True)
        def boom():
            raise OSError("no display")
        out = screen.capture("x", grab=boom)
        self.assertIn("couldn't get a look", out["text"])

    def test_screenshot_never_saved_under_public_static(self):
        self.assertNotIn("static", self.saved.parts)

    def test_names_the_window_in_front_and_keeps_notes_out(self):
        # live 11-02: a Discord call's two avatar tiles read as "two pixel-art characters on a
        # grassy meadow" (the game from her notes), and "what's the window's name?" got no answer
        from unittest import mock
        screen.set_enabled(True)
        front = 'Discord (the Discord call app; ...): "ChillZone | Active - Discord"'
        with mock.patch.object(screen, "front_window", return_value=front):
            out = screen.capture("look", grab=fake_grab())
        self.assertIn("The window in front is " + front, out["text"])
        self.assertIn("not a game or topic from your notes", out["text"])
        self.assertIn("people in the call, not game characters", screen._APP_KIND["discord"])


class ScreenToolLoopTests(LifecycleTests):
    def test_image_rides_on_tool_message_and_final_round_has_no_tools(self):
        first = Response([packet(tools=[LOOK], done=True)])
        second = Response([packet("[SPEAK to=user] Line 45.", done=True)])
        llm = self.make_llm([first, second], agentic=True)
        llm.tool_executor = lambda name, args: {"text": "shot", "images": ["QUJD"]}
        out = "".join(llm.generate("look at my screen", request_id="r", participation=True))
        self.assertEqual(out, "[SPEAK to=user] Line 45.")
        tool_msg = llm.ollama_session.payloads[1]["messages"][-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["images"], ["QUJD"])


class OwnerGuardTests(unittest.TestCase):
    def test_only_loopback_is_owner(self):
        import ast
        src = (Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_is_owner")
        ns = {}
        exec(compile(ast.Module([n for n in tree.body if isinstance(n, ast.Assign)
                                 and getattr(n.targets[0], "id", "") == "_LOOPBACK"] + [fn], []),
                     "server_guard", "exec"), ns)
        req = lambda h: SimpleNamespace(client=SimpleNamespace(host=h))
        self.assertTrue(ns["_is_owner"](req("127.0.0.1")))
        self.assertTrue(ns["_is_owner"](req("::1")))
        for remote in ("198.51.100.10", "203.0.113.20", ""):
            self.assertFalse(ns["_is_owner"](req(remote)), remote)
        for route in ('"/api/screen/last"', 'async def screen_set', 'async def agents_create',
                      'async def agents_delete', 'async def agents_draft'):
            i = src.index(route)
            self.assertIn("_is_owner(request)", src[i:i + 400], route)


if __name__ == "__main__":
    unittest.main()
