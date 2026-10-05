"""Read-the-moment tree + scrap detector (owner 10-01: fewer rules, tree-like thinking)."""
import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import moment, capability as C


class MomentTests(unittest.TestCase):
    def test_on_by_default_and_toggle(self):
        os.environ.pop("ATLAS_MOMENT", None)
        self.assertIn("READ THE MOMENT", moment.note())
        os.environ["ATLAS_MOMENT"] = "0"
        try:
            self.assertEqual(moment.note(), "")
        finally:
            os.environ.pop("ATLAS_MOMENT", None)

    def test_note_is_constant(self):  # byte-identical -> remote prefix cache
        self.assertEqual(moment.note(), moment.note())


class ScrapTests(unittest.TestCase):
    H = [{"role": "user", "content": "[S1] I'm cooking pasta tonight"}]

    def test_unconnected_scrap(self):
        self.assertTrue(C.is_scrap("[S1] Emails.", self.H, ("Ivy",)))
        self.assertTrue(C.is_scrap("[S1] Pull the trigger.", self.H, ("Ivy",)))

    def test_not_scrap(self):
        self.assertFalse(C.is_scrap("[S1] The pasta is boiling.", self.H, ("Ivy",)))
        self.assertFalse(C.is_scrap("[S1] Ivy, emails.", self.H, ("Ivy",)))
        self.assertFalse(C.is_scrap("[S1] what time is it?", self.H, ("Ivy",)))
        self.assertFalse(C.is_scrap("[S1] yeah okay", self.H, ("Ivy",)))
        self.assertFalse(C.is_scrap("[S1] so I was thinking we should go to the store later", self.H, ()))
        self.assertEqual(C.scrap_note("[S1] The pasta is boiling.", self.H), "")


if __name__ == "__main__":
    unittest.main()
