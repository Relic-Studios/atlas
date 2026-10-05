import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor import collapse_repeats

A = lambda t: {'role': 'assistant', 'content': f'[SPEAK to=S1] {t}'}
U = lambda t: {'role': 'user', 'content': t}

class CollapseRepeats(unittest.TestCase):
    def test_keeps_only_latest_copy(self):
        h = [U('a'), A("Yeah, what's good?"), U('b'), A("yeah what's good"), U('c'), A('Different.')]
        out = collapse_repeats(h)
        self.assertEqual([m['content'] for m in out],
                         ['a', 'b', "[SPEAK to=S1] yeah what's good", 'c', '[SPEAK to=S1] Different.'])

    def test_user_turns_untouched(self):
        h = [U('same'), U('same'), A('x')]
        self.assertEqual(collapse_repeats(h), h)

if __name__ == '__main__':
    unittest.main()
