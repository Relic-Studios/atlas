"""heart.md metadata as 'key: value' lines / front matter (found live 09-30)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import heart_md as H

DOC = ("# Marge\nrole: deadpan · retired waitress\ninterests: bowling, true crime\n"
       "talkativeness: 0.3\n\n## Who they are\nForty years at a Dayton diner.\n\n## How they talk\nSlow, flat.\n")


class HeartMetaTests(unittest.TestCase):
    def test_preamble_key_values(self):
        d = H.parse(DOC)
        self.assertEqual(d["role"], "deadpan · retired waitress")
        self.assertEqual(d["interests"], ["bowling", "true crime"])
        self.assertAlmostEqual(d["talkativeness"], 0.3)
        self.assertIn("Dayton", d["fields"]["who"])

    def test_front_matter(self):
        d = H.parse("---\nname: Kix\ninterests: fighting games, ramen\n---\n## Who\nloud streamer")
        self.assertEqual(d["name"], "Kix")
        self.assertEqual(d["interests"], ["fighting games", "ramen"])

    def test_raw_prompt_untouched(self):
        d = H.parse("A raw prompt. role: not metadata")
        self.assertEqual(d["prompt"], "A raw prompt. role: not metadata")
        self.assertNotIn("role", d)

    def test_round_trip(self):
        d = H.parse(DOC)
        self.assertEqual(H.parse(H.dump(d["name"], d["fields"], d["interests"], d["role"], d["talkativeness"])), d)


if __name__ == "__main__":
    unittest.main()
