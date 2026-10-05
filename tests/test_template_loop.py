"""Template loops (live 09-30 11:43: 'What's the "smelly"?' on every line)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor import recent_templates, collapse_templates, template_key
from response_decision import _is_loop_filler


def a(t): return {'role': 'assistant', 'content': '[SPEAK to=user] ' + t}
def u(t): return {'role': 'user', 'content': t}


class TemplateLoopTests(unittest.TestCase):
    def test_live_loop_caught(self):
        h = [a('What\'s the "smelly"?'), u('x'), a('What\'s the "Oh"?')]
        self.assertTrue(_is_loop_filler('What\'s the "showered"?', recent_templates(h)))

    def test_single_use_not_a_loop(self):
        h = [a("What's the plan?"), a('Pizza, obviously.')]
        self.assertFalse(_is_loop_filler("What's the move tonight?", recent_templates(h)))

    def test_long_sentences_exempt(self):
        h = [a("What's the deal?"), a("What's the plan?")]
        self.assertFalse(_is_loop_filler(
            "What's the actual reason you keep buying a brand new mechanical keyboard every single month of the year?", recent_templates(h)))
        self.assertEqual(template_key('ok'), '')

    def test_history_keeps_only_latest(self):
        h = [a('What\'s the "a"?'), u('1'), a('What\'s the "b"?'), u('2'), a('Real line here, longer one.')]
        out = collapse_templates(h)
        said = [m['content'] for m in out if m['role'] == 'assistant']
        self.assertEqual(len(said), 2)
        self.assertIn('"b"', said[0])
        self.assertEqual(len([m for m in out if m['role'] == 'user']), 2)


if __name__ == '__main__':
    unittest.main()
