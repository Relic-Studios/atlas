"""Bet tracker plugin (owner 10-05): log who bet what, bring it up when due, never decide the winner."""
import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bets as B  # noqa: E402

NOW = dt.datetime(2026, 10, 5, 18, 0).timestamp()   # a Monday


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._old = B.STATE_PATH
        B.STATE_PATH = Path(self.tmp.name) / "bets.json"

    def tearDown(self):
        B.STATE_PATH = self._old
        self.tmp.cleanup()


class Intent(unittest.TestCase):
    def test_bet_with_stakes(self):
        self.assertEqual(B.intent("I bet you five bucks the Lakers win tonight"),
                         ("record_bet", {"claim": "the Lakers win tonight", "stakes": "five bucks",
                                         "settle_by": "tonight"}))
        self.assertEqual(B.intent("I bet you guys a pizza that it rains tomorrow")[1]["stakes"], "a pizza")
        self.assertEqual(B.intent("Fae, I bet you $20 that Sam finishes it by the end of the month")[1]["stakes"], "$20")

    def test_idioms_left_alone(self):
        for s in ("I bet you are tired", "I bet that was hard", "bet you could do it",
                  "you bet", "I'd bet on it honestly"):
            self.assertIsNone(B.intent(s), s)

    def test_list(self):
        for s in ("what are the open bets?", "who's winning the bets", "show me the bet scoreboard"):
            self.assertEqual(B.intent(s), ("list_bets", {}), s)


class Dates(unittest.TestCase):
    def day(self, w):
        return dt.datetime.fromtimestamp(B.settle_time(w, NOW)).date()

    def test_words(self):
        self.assertEqual(self.day("tonight"), dt.date(2026, 10, 5))
        self.assertEqual(self.day("tomorrow"), dt.date(2026, 10, 6))
        self.assertEqual(self.day("by Sunday"), dt.date(2026, 10, 11))
        self.assertEqual(self.day("Monday"), dt.date(2026, 10, 12))      # next Monday, not today
        self.assertEqual(self.day("end of the month"), dt.date(2026, 10, 31))
        self.assertEqual(self.day("in 2 weeks"), dt.date(2026, 10, 19))
        self.assertEqual(self.day("in three days"), dt.date(2026, 10, 8))
        self.assertEqual(self.day("2026-12-25"), dt.date(2026, 12, 25))
        self.assertEqual(self.day("Oct 20"), dt.date(2026, 10, 20))
        self.assertEqual(self.day("Jan 3"), dt.date(2027, 1, 3))          # past date -> next year
        self.assertEqual(self.day(""), dt.date(2026, 10, 12))             # default a week
        self.assertEqual(self.day("whenever"), dt.date(2026, 10, 12))


class Lifecycle(Base):
    def test_record_list_settle_scoreboard(self):
        out = B.record_bet("the Lakers win tonight", "Sam", "Riley", "five bucks", "tonight", clock=lambda: NOW)
        self.assertIn("#1", out)
        self.assertIn("Sam bets that the Lakers win tonight (against Riley), for five bucks", out)
        self.assertIn("#1", B.list_bets(clock=lambda: NOW))
        res = B.settle_bet(1, "lost", "Celtics by 6", clock=lambda: NOW)
        self.assertIn("Riley wins; Sam owes five bucks", res)
        self.assertIn("Riley 1-0", res)
        self.assertIn("Sam 0-1", res)
        self.assertEqual(B.list_bets(clock=lambda: NOW), "No open bets.")
        self.assertIn("LOST", B.list_bets(include_settled=True, clock=lambda: NOW))
        self.assertIn("already settled", B.settle_bet(1, "won", clock=lambda: NOW))

    def test_outcome_required_and_never_guessed(self):
        B.record_bet("it rains", "Sam", clock=lambda: NOW)
        self.assertIn("won, lost, push or void", B.settle_bet(1, "", clock=lambda: NOW))
        self.assertIn("won, lost, push or void", B.settle_bet(1, "probably", clock=lambda: NOW))
        self.assertIn("push", B.settle_bet(None, "tie", words="rains", clock=lambda: NOW))

    def test_duplicate_and_private(self):
        B.record_bet("GTA 6 slips again", "Sam", clock=lambda: NOW)
        self.assertIn("Already logged", B.record_bet("GTA 6 slips again", "Sam", clock=lambda: NOW))
        self.assertIn("Not logged", B.record_bet("call me at 555-867-5309 if I win", "Sam", clock=lambda: NOW))
        self.assertIn("what's the bet", B.record_bet("", "Sam", clock=lambda: NOW))

    def test_due_and_claim_once(self):
        B.record_bet("the Lakers win tonight", "Sam", settle_by="tonight", clock=lambda: NOW)
        self.assertIsNone(B.due(clock=lambda: NOW))                       # not yet
        later = NOW + 8 * 3600
        b = B.due(clock=lambda: later)
        self.assertEqual(b["id"], 1)
        cue = B.claim(1, clock=lambda: later)
        self.assertRegex(cue, r"\(BET CUE #1\)")
        self.assertIn("never invent the result", cue)
        self.assertEqual(B.claim(1, clock=lambda: later), "")              # only once
        self.assertIsNone(B.due(clock=lambda: later))
        self.assertTrue(B.cue_note(cue))

    def test_stale_bets_dropped(self):
        B.record_bet("x happens", "Sam", settle_by="tomorrow", clock=lambda: NOW)
        self.assertEqual(B.list_bets(clock=lambda: NOW + 40 * 86400), "No open bets.")


class Wiring(Base):
    def test_room_tools_execute_and_cue(self):
        import room_tools as R
        self.assertTrue({"record_bet", "list_bets", "settle_bet"} <= R.NAMES)
        out = R.execute("record_bet", {"claim": "it snows Friday", "stakes": "a coffee", "settle_by": "Friday"},
                        "S1", "Sam")
        self.assertIn("Sam bets that it snows Friday", out)
        self.assertIn("Scoreboard", R.execute("list_bets", {}, "S1", "Sam"))
        self.assertTrue(R.ROOM_CUE_RE.search("(BET CUE #4) x"))
        self.assertIn("BET", R.cue_note("(BET CUE #4) x"))
        self.assertEqual(R.intent("I bet you a beer the server crashes")[0], "record_bet")

    def test_plugin_switch_covers_tools(self):
        import plugins
        p = next(x for x in plugins.BUILTIN if x["id"] == "bets")
        self.assertEqual(set(p["tools"]), {"record_bet", "list_bets", "settle_bet"})


if __name__ == "__main__":
    unittest.main()


class SettleByWinner(Base):
    """Live sim 10-05: models got the bettor-side outcome wrong; settle by who won instead."""
    def setUp(self):
        super().setUp()
        B.record_bet("the Lakers win tonight", "Riley", "Sam", "five bucks", "tonight", clock=lambda: NOW)

    def test_winner_against_side(self):
        out = B.settle_bet(winner="Sam", clock=lambda: NOW)
        self.assertIn("Sam wins; Riley owes five bucks", out)

    def test_winner_bettor_side(self):
        self.assertIn("Riley wins; Sam owes five bucks", B.settle_bet(winner="riley", clock=lambda: NOW))

    def test_loser(self):
        self.assertIn("Sam wins", B.settle_bet(loser="Riley", clock=lambda: NOW))

    def test_unknown_name_asks(self):
        B.record_bet("it snows", "Jordan", "Max", clock=lambda: NOW)
        out = B.settle_bet(winner="Pat", clock=lambda: NOW)
        self.assertIn("Which bet?", out)

    def test_ambiguous_asks(self):
        B.record_bet("GTA slips", "Sam", "Riley", settle_by="next week", clock=lambda: NOW)
        self.assertIn("Which bet?", B.settle_bet(winner="Sam", clock=lambda: NOW))

    def test_room_tools_speaker_placeholder(self):
        import room_tools as R
        pf = R.intent("Fae, the Lakers lost by six, so I won the bet against Riley.")
        self.assertEqual(pf[0], "settle_bet")
        out = R.execute(pf[0], pf[1], "S1", "Sam")
        self.assertIn("Sam wins; Riley owes five bucks", out)

    def test_named_opponent_intent(self):
        self.assertEqual(B.intent("I bet Sam five bucks the Lakers win tonight")[1]["against"], "Sam")
        self.assertIsNone(B.intent("I win this bet easily"))
