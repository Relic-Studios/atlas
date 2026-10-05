"""Every production module must resolve all its names (live 09-29: a deleted
_TALK_LESS_PATTERNS NameError silently skipped user-turn bookkeeping 135x)."""
import subprocess, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class UndefinedNameTests(unittest.TestCase):
    def test_no_undefined_names(self):
        try:
            import pyflakes  # noqa: F401
        except ImportError:
            self.skipTest("pyflakes not installed")
        files = [str(p) for p in ROOT.glob("*.py")]
        out = subprocess.run([sys.executable, "-m", "pyflakes", *files],
                             capture_output=True, text=True).stdout
        bad = [l for l in out.splitlines()
               if "undefined name" in l or "referenced before assignment" in l]
        self.assertEqual(bad, [])

if __name__ == "__main__":
    unittest.main()
