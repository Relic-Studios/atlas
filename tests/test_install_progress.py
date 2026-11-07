"""install_progress.ps1: the @@ lines the setup wizard reads (installer/atlas.iss; design: docs/designs/windows_setup.md)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "install_progress.ps1"
PS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"

PREAMBLE = f""". '{MOD}'
$global:T = [datetime]'2026-01-01'
$script:AtNow = {{ $global:T }}
function Tick([double]$s) {{ $global:T = $global:T.AddSeconds($s) }}
Enable-AtProgress
"""


@unittest.skipUnless(sys.platform == "win32" and PS.exists(), "Windows PowerShell only")
class InstallProgress(unittest.TestCase):
    def run_ps(self, body):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "t.ps1")
            f.write_text(PREAMBLE + body, encoding="utf-8-sig")
            r = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(f)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            return [l for l in r.stdout.splitlines() if l.strip()]

    def overalls(self, lines):
        return [int(l.split()[2].split("|")[0]) for l in lines if l.startswith("@@ P ")]

    def test_pytorch_download_progress(self):
        out = self.run_ps("""
Start-AtStep 1; Complete-AtStep; Tick 1
Start-AtStep 2
$exp = 2.9GB
foreach ($l in @('Collecting torch==2.10.0',
                 'Downloading https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl (2900.0 MB)',
                 'Progress 0 of 3113851290', 'Progress 1556925645 of 3113851290', 'Progress 3113851290 of 3113851290',
                 'Installing collected packages: torch',
                 'INSTALLPROGRESS 2350000000 4700000000', 'INSTALLPROGRESS 4700000000 4700000000',
                 'Successfully installed torch-2.10.0+cu128')) { Tick 1; Read-AtPipLine $l $exp 'PyTorch' 0.42 0.58 }
Complete-AtStep
""")
        self.assertIn("@@ STEP 2", out)
        p = [l for l in out if l.startswith("@@ P ")]
        self.assertIn("@@ P 10||", p)                                 # step 1 done: 1 of 100 -> 10; no time left yet
        # halfway (1.45 GiB) through the download: 10 + 430 * 0.42 * 0.5 = 100
        self.assertTrue(any(l.startswith("@@ P 100|1.5 GB of 2.9 GB") for l in p), p)
        self.assertIn("@@ P 190|Installing|", p)                       # download done: 10 + 430 * 0.42
        self.assertIn("@@ P 315|Installing|", p)                       # half installed: 1000 * (1 + 43 * 0.71) / 100
        self.assertEqual(len(p), len(set(p)), p)                       # no duplicate lines
        self.assertEqual(self.overalls(out)[-1], 440)                  # (1 + 43) / 100

    def test_silent_install_is_not_called_downloading(self):
        # pip prints nothing for minutes while it installs; a line arriving then must not turn the detail into
        # "Downloading" (that fallback is for a download whose progress lines no longer parse)
        out = self.run_ps("""
Start-AtStep 3; Tick 1
Read-AtPipLine 'Installing collected packages: fastapi, torch' 330MB 'packages' 0.07 0.85
Tick 200
Read-AtPipLine 'WARNING: something pip says late' 330MB 'packages' 0.07 0.85
Tick 70
Read-AtPipLine 'some line that is not progress' 330MB 'packages' 0.07 0.85
Tick 1; Set-AtFraction 0.5
""")
        details = [l.split("|")[1] for l in out if l.startswith("@@ P ")]
        self.assertIn("Installing", details, out)
        self.assertNotIn("Downloading", details, out)
        self.assertEqual(details[-1], "Installing", out)

    def test_overall_never_decreases(self):
        out = self.run_ps("""
Start-AtStep 3; Set-AtFraction 0.6 'x'; Tick 1; Set-AtFraction 0.2 'y'; Tick 1; Set-AtFraction 0.7 'z'
""")
        o = self.overalls(out)
        self.assertEqual(o, sorted(o))

    def test_updates_are_throttled_to_twice_a_second(self):
        out = self.run_ps("""
Start-AtStep 2
for ($i = 1; $i -le 10; $i++) { Tick 0.1; Set-AtFraction ($i / 20) ('n' + $i) }
""")
        p = [l for l in out if l.startswith("@@ P ")]
        self.assertLessEqual(len(p), 4, p)        # the forced one at the step start + at most 2 in 1 s

    def test_model_progress_line(self):
        out = self.run_ps("""
Start-AtStep 4; Tick 1
Read-AtModelLine 'MODELPROGRESS 2040109466 4080218932 Speech and voice models'
""")
        self.assertTrue(any("Speech and voice models - 1.9 GB of 3.8 GB" in l for l in out), out)

    def test_ascii_only(self):
        out = self.run_ps("Write-AtLine 'Downloading\u2026 1.2 GB \u00b7 2.9 GB'")
        self.assertEqual(out, ["Downloading? 1.2 GB ? 2.9 GB"])

    def test_failure_codes(self):
        cases = {
            "OSError: [Errno 28] No space left on device": "NODISK",
            "ERROR: Could not install packages due to an OSError: [WinError 32] The process cannot access the file because it is being used by another process": "LOCKED",
            "SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed'))": "TLS",
            "ERROR: HTTP error 503 while getting https://pypi.org/simple/fastapi/": "SERVER",
            # pip gave up on a 503: it also says "Max retries exceeded", and SERVER wins over NET
            "Max retries exceeded with url: /simple/torch/ (Caused by ResponseError('too many 503 error responses'))": "SERVER",
            "WARNING: Retrying (Retry(total=4)) after connection broken by 'ProxyError('Cannot connect to proxy.')'": "NET",
            "Max retries exceeded with url: /simple/torch/": "NET",
            # a dropped secure connection is NET, not TLS
            "ssl.SSLEOFError: EOF occurred in violation of protocol (_ssl.c:2427)": "NET",
            "WinError 53 something": "FAILED",
            "Something else entirely": "FAILED",
        }
        body = "\n".join(f"Get-AtFailureCode @('{t.replace(chr(39), chr(39) * 2)}')" for t in cases)
        out = self.run_ps(body)
        self.assertEqual(out, list(cases.values()))

    def minutes(self, lines):
        return [l.split("|")[2] for l in lines if l.startswith("@@ P ")]

    def test_time_left_corrects_after_sustained_speed_changes(self):
        out = self.run_ps("""
Start-AtStep 2
for ($i = 1; $i -le 180; $i++) {
    Tick 1
    # Slow preparation, sustained acceleration, then a sustained slowdown.
    $f = if ($i -le 60) { $i / 6000 } elseif ($i -le 120) { 0.01 + ($i - 60) / 150 } else { 0.41 + ($i - 120) / 1500 }
    Set-AtFraction $f 'x'
    if ($i -in @(60,120,180)) { Write-Output ("CHECK " + $i + " " + $script:AtMinutes) }
}
""")
        m = self.minutes(out)
        shown = [int(v) for v in m if v]
        self.assertTrue(shown, m)
        checks = {int(l.split()[1]): int(l.split()[2]) for l in out if l.startswith('CHECK ')}
        self.assertGreater(checks[60], 100, checks)
        self.assertLess(checks[120], 10, checks)                     # old clamp leaves hundreds of minutes
        self.assertGreater(checks[180], checks[120] * 4, checks)

    def test_time_left_is_empty_during_the_first_minute(self):
        out = self.run_ps("""
Start-AtStep 2
for ($i = 1; $i -le 59; $i++) { Tick 1; Set-AtFraction ($i / 100) 'x' }
""")
        self.assertTrue(all(v == '' for v in self.minutes(out)), out)

    def test_time_left_updates_at_most_every_five_seconds(self):
        out = self.run_ps("""
Start-AtStep 2
Tick 60; Set-AtFraction 0.1 'x'
for ($i = 1; $i -le 4; $i++) { Tick 1; Set-AtFraction (0.1 + $i / 10) 'x' }
""")
        shown = [v for v in self.minutes(out) if v]
        self.assertGreaterEqual(len(shown), 2)
        self.assertEqual(len(set(shown)), 1, shown)

    def test_capped_install_progress_explains_the_quiet_finishing_phase(self):
        out = self.run_ps("""
Start-AtStep 3; Tick 1
Read-AtPipLine 'INSTALLPROGRESS 1000000000 1000000000' 330MB 'packages' 0.07 0.85
Tick 200; Read-AtPipLine 'WARNING: ordinary pip output' 330MB 'packages' 0.07 0.85
Tick 1; Set-AtFraction 0.92
""")
        p = [l for l in out if l.startswith('@@ P ')]
        self.assertIn('Finishing (can take several minutes)', p[-1])
        self.assertEqual(self.overalls(out)[-1], 725)                 # not success or next step
        self.assertEqual(self.overalls(out), sorted(self.overalls(out)))
        self.assertNotIn('@@ DONE', out)

    def test_disabled_prints_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d, "t.ps1")
            f.write_text(f". '{MOD}'\nStart-AtStep 2\nSet-AtFraction 0.5 'x'\nWrite-AtFail 'NET' 'm'\n", encoding="utf-8-sig")
            r = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(f)],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
