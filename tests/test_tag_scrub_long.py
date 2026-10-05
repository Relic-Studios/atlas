"""Spoken S-labels with 3+ digits (live 10-02: Fae said 'S130', Ivy 'S104')."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import people
from response_decision import ResponseDecision, filter_response


class LongTagTests(unittest.TestCase):
    def setUp(self):
        self.b = people.NameBook()

    def test_unknown_long_tags_become_someone(self):
        for t in ["probably S130 messing around.", "Yeah, it's S130.", "S104 said he muted Fae"]:
            out = self.b.sanitize(t, "S5")
            self.assertNotRegex(out, r"\bS\d+\b", out)

    def test_product_names_kept(self):
        self.assertIn("S23", self.b.sanitize("my Galaxy S23 died", "S5"))
        self.assertIn("S2", self.b.sanitize("season S2 was better", "S5"))

    def test_split_across_chunks(self):
        d = ResponseDecision(); d.namebook = self.b
        src = iter(["[SPEAK to=S5] it's S1", "3", "0 again."])
        out = "".join(filter_response(src, d))
        self.assertNotRegex(out, r"S\d", out)


if __name__ == "__main__":
    unittest.main()
