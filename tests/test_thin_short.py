"""Model-facing history keeps only the latest few one-liners (live 10-01: Pip
'Pip see it.' / 'Pip hear you.' wall taught 4-word replies; thinning -> 8)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floor import thin_short_replies

A = lambda t: {"role": "assistant", "content": "[SPEAK to=S1] " + t}
U = lambda t: {"role": "user", "content": "[S1] " + t}


class ThinShortTests(unittest.TestCase):
    def test_keeps_latest_short_and_all_long(self):
        h = [U("hi"), A("Pip see it."), U("a"), A("Pip hear you."), U("b"),
             A("Pip watch."), U("c"), A("Pip know that one, the puppet show with the big hats."),
             A("Pip not sure."), A("Pip see.")]
        out = thin_short_replies(h, keep=3)
        spoken = [m["content"] for m in out if m["role"] == "assistant"]
        self.assertNotIn("[SPEAK to=S1] Pip see it.", spoken)
        self.assertNotIn("[SPEAK to=S1] Pip hear you.", spoken)
        self.assertIn("[SPEAK to=S1] Pip know that one, the puppet show with the big hats.", spoken)
        self.assertEqual(len([m for m in out if m["role"] == "user"]), 4)
        self.assertEqual(len(spoken), 4)

    def test_disabled(self):
        h = [A("ok."), A("sure."), A("yep.")]
        self.assertEqual(thin_short_replies(h, keep=-1), h)


if __name__ == "__main__":
    unittest.main()


class ClipOwnTests(unittest.TestCase):
    def test_off_by_default_and_keeps_users(self):
        import floor
        h = [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': '[SPEAK to=S1] one two three four five six'},
             {'role': 'user', 'content': 'b'}, {'role': 'assistant', 'content': '[SPEAK to=S1] seven eight nine ten eleven twelve'}]
        self.assertEqual(floor.clip_own_replies(h, -1), h)
        out = floor.clip_own_replies(h, 1)
        self.assertEqual([m['content'] for m in out if m['role'] == 'user'], ['a', 'b'])
        self.assertEqual(sum(m['role'] == 'assistant' for m in out), 1)
        self.assertIn('seven', out[-1]['content'])
