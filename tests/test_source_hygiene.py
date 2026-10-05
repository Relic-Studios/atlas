"""Source hygiene: shell-mangled escapes (\\b -> 0x08, \\n -> real newline in strings)
have broken regexes in this repo several times. Fail fast if one sneaks in."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GLOBS = ["*.py", "tests/*.py", "tools/*.py", "static/*.js", "desktop/*.js", "desktop/*.html", "static/*.html", "installer/*.iss", "*.ps1", "*.sh", "public_overlay/*.cmd"]


class SourceHygieneTests(unittest.TestCase):
    def test_no_control_bytes(self):
        bad = []
        for g in GLOBS:
            for p in ROOT.glob(g):
                b = p.read_bytes()
                for ch in (0, 7, 8, 11, 12):  # NUL, \a, \b, \v, \f
                    if bytes([ch]) in b:
                        bad.append(f"{p.relative_to(ROOT)}: byte {ch}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
