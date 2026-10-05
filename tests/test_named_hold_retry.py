"""Named-HOLD retry (first_run sim 10-04: qwen3 8B held on 'Wren, you joining us?')."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_decision import directly_named, retry_named_hold  # noqa: E402

N = ("wren",)


class DirectlyNamed(unittest.TestCase):
    def test_vocative(self):
        for t in ["[S1] Wren, you joining us?", "[S1] okay Wren, settle it", "[S2] what do you think, Wren?",
                  "hey wren whats up"]:
            self.assertTrue(directly_named(t, N), t)

    def test_mentions_are_not_vocative(self):
        for t in ["[S1] I told my sister about Wren and she laughed at the whole thing",
                  "[S1] Sam said Wren was funny", "[S2] yeah it's in my bag", ""]:
            self.assertFalse(directly_named(t, N), t)


class Retry(unittest.TestCase):
    def run_it(self, first, enabled=True):
        calls = []
        def again():
            calls.append(1)
            return iter(["[SPEAK to=S1]", " Yeah, I'm in."])
        out = "".join(retry_named_hold(iter(first), again, enabled))
        return out, len(calls)

    def test_hold_split_across_chunks_retries(self):
        self.assertEqual(self.run_it(["[HO", "LD]"]), ("[SPEAK to=S1] Yeah, I'm in.", 1))

    def test_speak_passes_through_untouched(self):
        self.assertEqual(self.run_it(["[SPEAK ", "to=S1]", " Co-op."]), ("[SPEAK to=S1] Co-op.", 0))

    def test_disabled_keeps_hold(self):
        self.assertEqual(self.run_it(["[HOLD]"], enabled=False), ("[HOLD]", 0))

    def test_retry_only_once(self):
        out = "".join(retry_named_hold(iter(["[HOLD]"]), lambda: iter(["[HOLD]"]), True))
        self.assertEqual(out, "[HOLD]")

    def test_headerless_text_retries(self):
        # matrix 10-04: headerless text parses as INVALID (silence) -> retry instead
        self.assertEqual(self.run_it(["Sure thing."])[1], 1)


if __name__ == "__main__":
    unittest.main()


class CancelledTurnNoRetry(unittest.TestCase):
    """Demo take 6 (10-05): a newer partial aborted the turn, the first stream came back
    empty, and the retry spoke a stale 'Sure thing. What's the test?'."""

    def test_cancelled_empty_does_not_retry(self):
        calls = []
        out = "".join(retry_named_hold(iter([""]), lambda: calls.append(1) or iter(["[SPEAK to=S1] stale"]),
                                         True, cancelled=lambda: True))
        self.assertEqual(out, "")
        self.assertEqual(calls, [])

    def test_cancelled_hold_does_not_retry(self):
        out = "".join(retry_named_hold(iter(["[HOLD]"]), lambda: iter(["[SPEAK to=S1] stale"]),
                                         True, cancelled=lambda: True))
        self.assertNotIn("stale", out)

    def test_live_turn_still_retries(self):
        out = "".join(retry_named_hold(iter([""]), lambda: iter(["[SPEAK to=S1] hi"]),
                                         True, cancelled=lambda: False))
        self.assertIn("hi", out)
