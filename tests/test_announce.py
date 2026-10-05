"""Announce-then-stop replies are not voiced (live 10-02: Pup 'Rokay, I'll do it.' then silence)."""
import os, sys, unittest
from types import SimpleNamespace as N
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import response_decision as R


def run(chunks):
    return ''.join(R._announce_screen(iter(chunks), N()))


class AnnounceTests(unittest.TestCase):
    def test_announce_only_dropped(self):
        for t in ["Rokay, I'll do it.", "Rokay, here goes.", "Rokay, I'll try.",
                  "Okay, I'll bite, I'll do a little Trump voice right now.",
                  "Okay, okay, I'll try it again, for real this time."]:
            self.assertEqual(run([t]), "", t)

    def test_announce_with_followthrough_kept(self):
        self.assertEqual(run(["Rokay, I'll do it. ", "Wall of Pup Snacks, folks!"]),
                         "Rokay, I'll do it. Wall of Pup Snacks, folks!")

    def test_normal_replies_kept(self):
        for t in ["Alright, I'll stop.", "Rokay, I'll be quiet.", "Okay.", "Sure, pizza works.",
                  "I'll try the ramen place tomorrow, it looks good.", "I'll do it later, promise."]:
            self.assertEqual(run([t]), t, t)


if __name__ == '__main__':
    unittest.main()
