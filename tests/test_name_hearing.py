"""Agent name misheard in a direct address (demo 10-04: Wren -> Ren/Ran)."""
import unittest
import name_hearing as N


DEV_NAMES = (__import__("pathlib").Path(__file__).resolve().parents[1] / "dev_pack").exists()  # public export renames dev agents
class Fix(unittest.TestCase):
    def tearDown(self):
        N.set_protect(None)

    def test_recovers_vocative(self):
        for t, want in [("Hey Ran, you there? It's Sam.", "Hey Wren, you there? It's Sam."),
                        ("Okay, Ren. Settle it.", "Okay, Wren. Settle it."),
                        ("So Ran, what now?", "So Wren, what now?")]:
            self.assertEqual(N.fix(t, ["Wren"]), want)

    @__import__("unittest").skipUnless(DEV_NAMES, "uses dev agent-name misspellings")
    def test_recovers_dev_agent(self):
        self.assertEqual(N.fix("Navy, can you hear me?", ["Fae"]), "Fae, can you hear me?")

    def test_leaves_ordinary_words_and_other_names(self):
        for t in ["I ran to the store.", "Ren is my cousin.", "Hey Sam, you there?",
                  "Okay Riley, did you finish?", "Okay, Ben, go.", "Max, you there?"]:
            self.assertEqual(N.fix(t, ["Wren", "Max"]), t)

    def test_known_human_name_is_protected(self):
        N.set_protect(lambda: ["Ren"])
        self.assertEqual(N.fix("Hey Ren, you there?", ["Wren"]), "Hey Ren, you there?")

    def test_no_names_is_noop(self):
        self.assertEqual(N.fix("Hey Ran, hi", []), "Hey Ran, hi")
        self.assertIsNone(N.fix(None, ["Wren"]))


if __name__ == "__main__":
    unittest.main()
