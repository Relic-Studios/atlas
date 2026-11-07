"""tools/pip_progress.py: pip's output passed on, plus INSTALLPROGRESS lines while it installs."""
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "pip_progress.py"


class PipProgress(unittest.TestCase):
    def run_fake(self, body, expected):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d, "fake_pip.py")
            fake.write_text(textwrap.dedent(body), encoding="utf-8")
            r = subprocess.run([sys.executable, str(TOOL), "--path", d, "--expected", str(expected), "--every", "0.1",
                                "--", sys.executable, str(fake), d], capture_output=True, text=True, timeout=120)
            return r.returncode, r.stdout.splitlines()

    def test_lines_pass_through_and_install_progress_follows_the_drive(self):
        rc, lines = self.run_fake("""
            import os, sys, time
            print("Collecting torch", flush=True)
            print("Progress 5 of 10", flush=True)
            print("Installing collected packages: torch", flush=True)
            time.sleep(0.4)
            with open(os.path.join(sys.argv[1], "big.bin"), "wb") as f:
                for _ in range(8):                       # 32 MB written while "installing"
                    f.write(b"\\0" * 4_000_000); f.flush(); os.fsync(f.fileno()); time.sleep(0.15)
            time.sleep(0.6)
            print("Successfully installed torch", flush=True)
            sys.exit(3)
        """, expected=200_000_000)
        self.assertEqual(rc, 3)                                                   # the command's exit code
        plain = [l for l in lines if not l.startswith("INSTALLPROGRESS ")]
        self.assertEqual(plain, ["Collecting torch", "Progress 5 of 10", "Installing collected packages: torch",
                                 "Successfully installed torch"])
        idx = lines.index("Installing collected packages: torch")
        prog = [int(l.split()[1]) for l in lines if l.startswith("INSTALLPROGRESS ")]
        self.assertTrue(prog, lines)
        self.assertTrue(all(lines.index(l) > idx for l in lines if l.startswith("INSTALLPROGRESS ")))
        self.assertEqual(prog, sorted(prog))                                      # never goes down
        self.assertGreaterEqual(prog[-1], 16_000_000)                             # most of the 32 MB seen
        self.assertTrue(all(l.split()[2] == "200000000" for l in lines if l.startswith("INSTALLPROGRESS ")))

    def test_capped_at_expected_and_silent_before_installing(self):
        rc, lines = self.run_fake("""
            import os, sys, time
            with open(os.path.join(sys.argv[1], "early.bin"), "wb") as f:   # before installing: not counted
                f.write(b"\\0" * 20_000_000); f.flush(); os.fsync(f.fileno())
            time.sleep(0.4)
            print("Installing collected packages: x", flush=True)
            with open(os.path.join(sys.argv[1], "late.bin"), "wb") as f:
                f.write(b"\\0" * 30_000_000); f.flush(); os.fsync(f.fileno())
            time.sleep(0.6)
        """, expected=5_000_000)
        self.assertEqual(rc, 0)
        prog = [int(l.split()[1]) for l in lines if l.startswith("INSTALLPROGRESS ")]
        self.assertTrue(prog)
        self.assertLessEqual(max(prog), 5_000_000)
        self.assertEqual(lines[0], "Installing collected packages: x")


if __name__ == "__main__":
    unittest.main()
