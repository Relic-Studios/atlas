"""Live call 10-05: 'research something new' came back with old searches.

Replays the three causes found in that call's log:
  1. an exact-match reuse window (5 min) silently returned an OLD finished result,
  2. an old untold search kept being re-cued over the new question,
  3. the model appended a stale year ('... 2025' in 2026) to its query.
"""
import datetime, sys, time, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tasks as T
import websearch as W


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def board(clock, text=lambda q: f"results for {q}"):
    return T.TaskBoard("fae", None, text, clock=clock)


class FreshSearch(unittest.TestCase):
    def test_finished_search_not_reused_after_a_minute(self):
        c = Clock(); b = board(c)
        a = b.start_search("China fusion research progress", "S32", gen=1); b.wait(a, 2)
        c.t += T.REUSE_DONE_S + 5
        n = b.start_search("China fusion research progress", "S32", gen=2)
        self.assertIsNot(a, n)
        self.assertEqual(n["status"] if n["status"] != "done" else "running", "running")

    def test_running_search_still_deduped(self):
        b = T.TaskBoard("fae", None, lambda q: (time.sleep(0.2), "r")[1])
        a = b.start_search("x y", "S1", gen=1)
        self.assertIs(a, b.start_search("x  Y", "S1", gen=2))

    def test_new_search_supersedes_old_untold(self):
        c = Clock(); b = board(c)
        junk = b.start_search("Fae Legend of Zelda search internet", "S32", gen=1); b.wait(junk, 2)
        self.assertIs(b.deliverable(), junk)
        new = b.start_search("voice-to-voice full duplex models research", "S32", gen=2); b.wait(new, 2)
        self.assertEqual(junk["status"], "superseded")
        self.assertIs(b.deliverable(), new)          # the NEW answer is what's owed
        self.assertIn("don't bring it up", b.note())
        self.assertNotIn("results for Fae Legend", b.note())

    def test_other_askers_results_untouched(self):
        c = Clock(); b = board(c)
        a = b.start_search("bitcoin price", "S1", gen=1); b.wait(a, 2)
        n = b.start_search("weather", "S2", gen=2); b.wait(n, 2)
        self.assertEqual(a["status"], "done")

    def test_cued_only_once_and_expires(self):
        c = Clock(); b = board(c)
        t = b.start_search("q", "S1", gen=1); b.wait(t, 2)
        b.claim_delivery(t)
        self.assertIsNone(b.deliverable())
        t2 = b.start_search("q2", "S3", gen=2); b.wait(t2, 2)
        c.t += T.DELIVER_TTL_S + 1
        self.assertIsNone(b.deliverable())


class StaleYear(unittest.TestCase):
    D = datetime.date(2026, 10, 5)

    def f(self, q, said=""):
        return W.freshen_query(q, said, self.D)

    def test_model_appended_year_dropped(self):
        self.assertEqual(self.f("China fusion research progress 2025", "research Chinese fusion"),
                         "China fusion research progress")
        self.assertEqual(self.f("latest duplex voice models 2024 2025"), "latest duplex voice models")

    def test_years_people_said_or_history_kept(self):
        self.assertEqual(self.f("2024 election results", "what happened in the 2024 election"),
                         "2024 election results")
        self.assertEqual(self.f("World Cup 2018 winner"), "World Cup 2018 winner")
        self.assertEqual(self.f("GPU prices 2026"), "GPU prices 2026")
        self.assertEqual(self.f("2025"), "2025")   # never empty the query


if __name__ == "__main__":
    unittest.main()
