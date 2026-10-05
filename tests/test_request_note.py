"""Request-following cue (live 10-01: 'No, I want you to roast Toshiro Sakura.' -> 'Alright.')."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capability as C

N = ("max", "ivy")


class RequestNoteTests(unittest.TestCase):
    def test_requests_fire(self):
        for t in ["No, I want you to roast Toshiro Sakura.", "Max, tell us a joke.", "can you give S2 a nickname",
                  "rate my pasta out of ten", "settle this: cats or dogs?", "yo Max pick a number", "Do it again."]:
            self.assertTrue(C.is_request(t, N), t)
            self.assertIn("actually do it", C.request_note(t, N))

    def test_non_requests_quiet(self):
        for t in ["Maya, tell me about your day", "I told him to roast the chicken", "he can't make it tonight",
                  "did you see the game", "Do you like shopping?", "do you remember", "lmao", "Max"]:
            self.assertFalse(C.is_request(t, N), t)
            self.assertEqual(C.request_note(t, N), "")


if __name__ == "__main__":
    unittest.main()
