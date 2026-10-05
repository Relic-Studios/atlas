import unittest


class NamedBeatsUnnamedThread(unittest.TestCase):
    """Demo take 4 (10-04): agent had just greeted Riley; Riley said 'Almost, like
    two more minutes.'; then Sam: 'Okay, Wren, settle it.' The stale draft for
    Riley's thread must be preempted by Sam calling the agent by name."""

    def test_named_line_preempts_partner_thread(self):
        from threads import ThreadArbiter
        now = [100.0]
        arb = ThreadArbiter(("Wren",), clock=lambda: now[0])
        arb.answered("S2")                      # Wren just spoke to Riley
        now[0] += 2.0
        riley = arb.note("[S2] Almost, like two more minutes.", partner="S2")
        arb.start(arb.choose(riley))
        now[0] += 2.5
        sam = arb.note("[S1] Okay, Wren, settle it.", partner="S2")
        self.assertIn("named", sam.reasons)
        self.assertTrue(arb.should_preempt(sam))
        self.assertIs(arb.choose(sam), sam)

    def test_unnamed_chatter_still_does_not_preempt(self):
        from threads import ThreadArbiter
        now = [100.0]
        arb = ThreadArbiter(("Wren",), clock=lambda: now[0])
        t = arb.note("[S1] Wren, what game should we play?")
        arb.start(t)
        now[0] += 1.0
        self.assertFalse(arb.should_preempt(arb.note("[S2] lol yeah")))
