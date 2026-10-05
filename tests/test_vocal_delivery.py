"""Search results count as delivered only once they're actually voiced (live 10-01)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tasks as T
from untrusted import wrap

RES = wrap("Forbes Global 2000: Nvidia tops market cap at $4.5 trillion, followed by Microsoft and Apple; "
           "TSMC and Saudi Aramco round out the top five.")


def board(result=RES, query="richest companies in the world", gen=7):
    b = T.TaskBoard("x", path=None, searcher=None)
    t = b._new("search", query, "S10", gen=gen)
    t["status"], t["result"] = "done", " ".join(result.split())
    return b, t


class VocalDelivery(unittest.TestCase):
    def test_full_reply_with_facts_delivers(self):
        b, t = board()
        self.assertEqual(b.spoke(7, "Rokay! Nvidia's on top at like 4.5 trillion, then Microsoft and Apple."), [t["id"]])
        self.assertEqual(t["status"], "delivered")

    def test_stall_only_stays_owed(self):
        b, t = board()
        self.assertEqual(b.spoke(7, "Hold up, still digging on that one."), [])
        self.assertEqual(t["status"], "done")
        self.assertIsNone(t["gen"])               # free for the next gap
        self.assertIs(b.deliverable(), t)

    def test_interrupted_before_facts_stays_owed(self):
        b, t = board()
        self.assertEqual(b.spoke(7, "Ooh, okay, so I looked it up and"), [])
        self.assertEqual(t["status"], "done")

    def test_interrupted_with_no_heard_text_stays_owed(self):
        b, t = board()
        self.assertEqual(b.spoke(7, ""), [])

    def test_interrupted_after_facts_still_delivered(self):
        b, t = board()
        self.assertEqual(b.spoke(7, "Nvidia, by a mile, 4.5 trillion."), [t["id"]])

    def test_said_once_not_offered_again(self):
        b, t = board()
        b.spoke(7, "Nvidia at 4.5 trillion, then Microsoft.")
        self.assertIsNone(b.deliverable())

    def test_honest_empty_counts(self):
        b, t = board(result=wrap("weather.com Paris forecast links"), query="weather in Paris right now")
        self.assertEqual(b.spoke(7, "Ruh-roh, the search came up with links but no live numbers yet."), [t["id"]])

    def test_no_results_needs_saying_so(self):
        b, t = board(result="No results found.")
        self.assertEqual(b.spoke(7, "Yeah, totally."), [])
        b, t = board(result="No results found.")
        self.assertEqual(b.spoke(7, "Searched it, couldn't find anything on that."), [t["id"]])

    def test_other_turn_voicing_it_also_confirms(self):
        b, t = board(gen=3)
        self.assertEqual(b.spoke(9, "Oh, the richest companies thing: Nvidia, Microsoft, Apple."), [t["id"]])

    def test_unrelated_reply_does_not_confirm(self):
        b, t = board(gen=3)
        self.assertEqual(b.spoke(9, "I love Pup snacks more than anything."), [])
        self.assertEqual(t["status"], "done")

    def test_cut_then_resumed_turn_delivers_once(self):
        b, t = board()
        self.assertEqual(b.spoke(7, "Okay so I looked it up and the richest are"), [])   # cut off
        self.assertEqual(b.spoke(8, "Nvidia at 4.5 trillion, then Microsoft and Apple."), [t["id"]])
        self.assertIsNone(b.deliverable())                                              # no repeat cue

    def test_weak_unrelated_overlap_does_not_confirm(self):
        b, t = board(gen=3)
        self.assertEqual(b.spoke(9, "I'd rather have an Apple pie honestly."), [])

    def test_legacy_call_without_text(self):
        b, t = board()
        self.assertEqual(b.spoke(7), [t["id"]])


if __name__ == "__main__":
    unittest.main()
