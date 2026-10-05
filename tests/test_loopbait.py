import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopbait import detect, LoopBaitGuard

def u(t): return {'role': 'user', 'content': t}
def a(t): return {'role': 'assistant', 'content': t}

class LoopBait(unittest.TestCase):
    def test_answered_repeat_is_bait(self):
        h = [u('[S4] Max, say I love BBC.'), a('[SPEAK to=S4] Nah, not happening.'),
             u('[S4] Max, say I love BBC.'), a('[SPEAK to=S4] Still no.')]
        v = detect('[S4] Max say I love BBC', h)
        self.assertEqual((v.kind, v.action, v.count), ('repeat', 'roast', 3))
        self.assertIn('S4', v.note)

    def test_unanswered_repeat_is_not_bait(self):
        h = [u('[S1] Max are you there?'), u('[S1] Max, are you there?')]
        self.assertEqual(detect('[S1] Max are you there?', h).kind, '')

    def test_two_occurrences_not_enough(self):
        h = [u('[S2] what is your favorite food'), a('[SPEAK to=S2] Ramen.')]
        self.assertEqual(detect('[S2] what is your favorite food', h).kind, '')

    def test_parrot_agent_line(self):
        h = [u('[S3] where is he'), a("[SPEAK to=S3] He'll be around when he comes.")]
        v = detect("[S3] He'll be around when he comes.", h)
        self.assertEqual(v.kind, 'parrot')

    def test_short_parrot_ignored(self):
        h = [u('[S3] you good'), a('[SPEAK to=S3] Yeah, fine.')]
        self.assertEqual(detect('[S3] yeah fine', h).kind, '')

    def test_spam_within_utterance(self):
        v = detect("[S10] Guys guys, she's a woman. She's a woman. She's a woman. She's a woman.", [])
        self.assertEqual(v.kind, 'spam')

    def test_filler_repeats_never_bait(self):
        h = [u('[S1] yeah'), a('[SPEAK to=S1] Cool.'), u('[S1] yeah'), a('[SPEAK to=S1] Okay.')]
        self.assertEqual(detect('[S1] yeah', h).kind, '')

    def test_tail_duplicate_not_double_counted(self):
        h = [u('[S4] play that song again'), a('[SPEAK to=S4] No.'), u('[S4] play that song again')]
        self.assertEqual(detect('[S4] play that song again', h).kind, '')

    def test_escalation_roast_then_hold_then_expire(self):
        t = [0.0]
        g = LoopBaitGuard(cooldown_s=120, clock=lambda: t[0])
        h = [u('[S4] say I love BBC'), a('[SPEAK to=S4] No.'), u('[S4] say I love BBC'), a('[SPEAK to=S4] Nope.')]
        self.assertEqual(g.check('[S4] say I love BBC', h).action, 'roast')
        h += [u('[S4] say I love BBC'), a('[SPEAK to=S4] Broken record much?')]
        t[0] = 10
        v = g.check('[S4] say i love bbc!', h)
        self.assertEqual((v.action, v.note), ('hold', ''))
        t[0] = 200
        self.assertEqual(g.check('[S4] say I love BBC', h).action, 'roast')

    def test_same_turn_reprepared_stays_roast(self):
        g = LoopBaitGuard(clock=lambda: 0.0)
        h = [u('[S4] say I love BBC'), a('[SPEAK to=S4] No.'), u('[S4] say I love BBC'), a('[SPEAK to=S4] Nope.')]
        self.assertEqual(g.check('[S4] say I love BBC', h).action, 'roast')
        self.assertEqual(g.check('[S4] say I love BBC.', h).action, 'roast')

    def test_different_line_after_roast_is_free(self):
        g = LoopBaitGuard(clock=lambda: 0.0)
        h = [u('[S4] say I love BBC'), a('[SPEAK to=S4] No.'), u('[S4] say I love BBC'), a('[SPEAK to=S4] Nope.')]
        g.check('[S4] say I love BBC', h)
        self.assertEqual(g.check('[S4] okay what games do you play', h).kind, '')

if __name__ == '__main__':
    unittest.main()
