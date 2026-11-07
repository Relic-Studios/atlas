"""installer/gpu_check.ps1: one verdict line for Setup's GPU gate, against fake nvidia-smi outputs."""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "installer" / "gpu_check.ps1"
PS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"


@unittest.skipUnless(sys.platform == "win32" and PS.exists(), "Windows PowerShell only")
class GpuCheck(unittest.TestCase):
    def verdict(self, body, timeout=10, path=None):
        with tempfile.TemporaryDirectory() as d:
            fake = path or Path(d, "nvidia-smi.cmd")
            if path is None:
                fake.write_text("@echo off\r\n" + body + "\r\n", encoding="ascii")
            r = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                                "-NvidiaSmi", str(fake), "-TimeoutSec", str(timeout)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.strip()

    def test_supported_card(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce RTX 4090, 8.9, 591.86, 24564 MiB"), "OK NVIDIA GeForce RTX 4090")

    def test_gtx_10_series_is_too_old(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce GTX 1080, 6.1, 581.42, 8192 MiB"), "OLDGPU NVIDIA GeForce GTX 1080")

    def test_too_old_card_wins_over_old_driver(self):
        # a driver update would not help a GTX 1080, so it hears about the card first
        self.assertEqual(self.verdict("echo NVIDIA GeForce GTX 1080, 6.1, 516.94, 8192 MiB"), "OLDGPU NVIDIA GeForce GTX 1080")

    def test_driver_that_does_not_know_compute_cap(self):
        self.assertEqual(self.verdict('echo Field "compute_cap" is not a valid field to query. 1>&2\r\nexit /b 2'), "OLDDRIVER")

    def test_driver_below_cuda_12_8(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce RTX 3060, 8.6, 566.36, 12288 MiB"), "OLDDRIVER")

    def test_driver_floor_itself_is_ok(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce RTX 3060, 8.6, 570.65, 12288 MiB"), "OK NVIDIA GeForce RTX 3060")

    def test_six_gb_card_asks(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce GTX 1660, 7.5, 581.42, 6144 MiB"),
                         "LOWVRAM NVIDIA GeForce GTX 1660 6")

    def test_eight_gb_card_is_ok(self):
        self.assertEqual(self.verdict("echo NVIDIA GeForce RTX 4060, 8.9, 581.42, 8188 MiB"), "OK NVIDIA GeForce RTX 4060")

    def test_best_of_two_gpus(self):
        body = "echo NVIDIA GeForce GTX 1050, 6.1, 581.42, 4096 MiB\r\necho NVIDIA GeForce RTX 3070, 8.6, 581.42, 8192 MiB"
        self.assertEqual(self.verdict(body), "OK NVIDIA GeForce RTX 3070")

    def test_no_devices_found(self):
        self.assertEqual(self.verdict("echo No devices were found\r\nexit /b 6"), "NOGPU")

    def test_missing_nvidia_smi(self):
        self.assertEqual(self.verdict("", path=Path(tempfile.gettempdir(), "no_such_nvidia_smi.exe")), "NOGPU")

    def test_no_output_is_unknown(self):
        self.assertTrue(self.verdict("rem nothing").startswith("UNKNOWN "))

    def test_failed_run_is_unknown(self):
        v = self.verdict("echo NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\r\nexit /b 9")
        self.assertEqual(v, "UNKNOWN nvidia-smi exited with code 9")

    def test_garbled_output_is_unknown(self):
        self.assertTrue(self.verdict("echo something, else, entirely, here").startswith("UNKNOWN "))

    def test_hung_driver_times_out_as_unknown(self):
        t0 = time.time()
        self.assertEqual(self.verdict("ping -n 16 127.0.0.1 >nul", timeout=2), "UNKNOWN nvidia-smi did not answer within 2 s")
        self.assertLess(time.time() - t0, 12)

    def test_this_pc(self):
        # the real nvidia-smi: whatever this machine has, the script answers with a known verdict
        r = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT)],
                           capture_output=True, text=True, timeout=60)
        self.assertRegex(r.stdout.strip(), r"^(OK|NOGPU|UNKNOWN|OLDDRIVER|OLDGPU|LOWVRAM)\b")


if __name__ == "__main__":
    unittest.main()
