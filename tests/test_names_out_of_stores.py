"""Names stay out of lessons and predictions (design docs/designs/experience_memory.md, D8; acceptance A5b, write-time
cases): the call's names are scrubbed on write, nothing records who taught or asked, and old entries are cleaned once."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import call_boundary as CB  # noqa: E402
import self_tools as ST  # noqa: E402
import teach as T  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="atlas_names_"))
        self.b = CB.CallBoundary(write_files=False)
        self.b.add_name_source("test", lambda: ["Zarvik", "Will"])
        self.b.set_owner_name_source(lambda: "Relic")
        self.b.call_id = "test-call"                                 # names count only while a call is in progress
        self.p = mock.patch.object(CB, "BOUNDARY", self.b)
        self.p.start()

    def tearDown(self):
        self.p.stop()
        shutil.rmtree(self.root, ignore_errors=True)


class Lessons(Base):
    def test_no_teacher_and_no_names(self):
        out = T.learn("fae", "league scoring", "Zarvik says a win is 3 points and Relic keeps the table",
                      taught_by="Zarvik", root=self.root)
        self.assertNotIn("Zarvik", out)
        it = T._load("fae", self.root)[0]
        self.assertNotIn("taught_by", it)
        self.assertEqual(it["content"], "someone says a win is 3 points and the owner keeps the table")
        self.assertFalse(it["tagger_checked"])

    def test_rules_are_kept_and_directives_refused(self):
        self.assertIn("Learned", T.learn("fae", "scoring", "a win is 3 points, a draw is 1", root=self.root))
        self.assertIn("Not", T.learn("fae", "manners", "from now on you always say hi to newcomers",
                                     root=self.root))

    def test_legacy_lessons_lose_the_teacher(self):
        d = self.root / "fae"
        d.mkdir(parents=True)
        (d / "lessons.json").write_text(json.dumps([{"id": 1, "topic": "Zed's rule", "content": "Zed picks maps",
                                                     "taught_by": "Zed", "at": 1.0}]), encoding="utf-8")
        self.assertEqual(T.strip_legacy_names(self.root), 1)
        it = T._load("fae", self.root)[0]
        self.assertNotIn("taught_by", it)
        self.assertEqual((it["topic"], it["content"]), ("someone's rule", "someone picks maps"))
        self.assertEqual(T.strip_legacy_names(self.root), 0)


class Predictions(Base):
    def test_scrubbed_on_write_and_resolve(self):
        ST.make_prediction("fae", "Zarvik will rage quit before the boss", "after the raid", root=self.root)
        it = ST._load("fae", self.root)[0]
        self.assertEqual(it["claim"], "someone will rage quit before the boss")
        self.assertFalse(it["tagger_checked"])
        ST.check_predictions("fae", id=it["id"], outcome="right", note="Zarvik did rage quit", root=self.root)
        self.assertEqual(ST._load("fae", self.root)[0]["note"], "someone did rage quit")

    def test_existing_predictions_scrubbed_once(self):
        d = self.root / "fae"
        d.mkdir(parents=True)
        (d / "predictions.json").write_text(json.dumps([{"id": 1, "claim": "xX_sniper_99 wins tonight",
                                                         "status": "open", "made": 1.0}]), encoding="utf-8")
        self.assertEqual(ST.scrub_existing_predictions(self.root), 1)
        self.assertEqual(ST._load("fae", self.root)[0]["claim"], "someone wins tonight")
        self.assertEqual(ST.scrub_existing_predictions(self.root), 0)


if __name__ == "__main__":
    unittest.main()
