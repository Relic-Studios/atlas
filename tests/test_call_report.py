"""Post-call report (tools/call_report.py) on a synthetic recording."""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import call_report as R


def rec(rows):
    d = tempfile.mkdtemp(); p = os.path.join(d, "x.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"t": 1000 + i, **r}) + "\n")
    return p


class ReportTests(unittest.TestCase):
    def test_counts(self):
        rows = [{"k": "persona", "persona": "fae"}]
        for i in range(3):
            rows += [{"k": "user", "spk": "S1", "text": f"Fae, what's your favourite game number {i}?"},
                     {"k": "decision", "action": "SPEAK", "ms": 700},
                     {"k": "agent", "persona": "fae", "text": "Majora's Mask, easily."}]
        rows += [{"k": "user", "spk": "S2", "text": "pizza is here"},
                 {"k": "decision", "action": "SPEAK", "ms": 700},
                 {"k": "agent", "persona": "fae", "text": "Tell S2 hi."},
                 {"k": "user", "spk": "S2", "text": "the green dragon ate my lunch"},
                 {"k": "decision", "action": "SPEAK", "ms": 700},
                 {"k": "agent", "persona": "fae", "text": "The green dragon ate my lunch?"}]
        per, lat, samples, drops = R.analyse(R.load(rec(rows)))
        c = per["fae"]
        self.assertEqual(c["replies"], 5)
        self.assertEqual(c["spoken_tags"], 1)
        self.assertGreaterEqual(c["restates"], 1)
        self.assertGreaterEqual(c["self_repeats"], 1)  # "majora's mask" x3
        self.assertGreaterEqual(c["unprompted"] if "unprompted" in c else c["butt_ins"], 1)

    def test_sim_files_skipped(self):
        rows = [{"k": "persona", "persona": p} for p in ("ivy", "kai", "pip", "grim", "pup")] * 4
        self.assertFalse(R.is_real_call(R.load(rec(rows))))


if __name__ == "__main__":
    unittest.main()
