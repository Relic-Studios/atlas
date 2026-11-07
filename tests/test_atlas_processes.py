"""installer/atlas_processes.ps1: the running-ATLAS rule shared by the uninstaller and the setup program."""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "installer" / "atlas_processes.ps1"
WINDIR = Path(os.environ.get("WINDIR", r"C:\Windows"))
PS = WINDIR / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
NO_WINDOW = 0x08000000


@unittest.skipUnless(sys.platform == "win32" and PS.exists(), "Windows only")
class AtlasProcesses(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="atlas_proc_"))
        self.procs = []
        self.external = Path(tempfile.mkdtemp(prefix="atlas_proc_external_"))
        self.fixture = self.external / "AtlasProcessFixture.exe"
        # A copied Windows system binary can exit early on a hosted image because its
        # localization/support files are missing. This fixture has no network dependency.
        source = 'public class AtlasProcessFixture { public static void Main() { System.Threading.Thread.Sleep(60000); } }'
        command = "Add-Type -TypeDefinition '" + source + "' -OutputAssembly '" + str(self.fixture).replace("'", "''") + "' -OutputType ConsoleApplication"
        subprocess.run([str(PS), "-NoProfile", "-Command", command], check=True,
                       capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.external, ignore_errors=True)

    def start(self, argv):
        p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
        self.procs.append(p)
        return p

    def run_script(self, *extra):
        r = subprocess.run([str(PS), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                            "-Folder", str(self.dir), *extra], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return {int(l.split()[0]): l.split()[1] for l in r.stdout.splitlines() if l.strip()}

    def test_lists_then_stops_programs_from_the_folder(self):
        exe = self.dir / "sub" / "AtlasProcessFixture.exe"
        exe.parent.mkdir()
        shutil.copy2(self.fixture, exe)
        inside = self.start([str(exe)])
        by_cmdline = self.start([sys.executable, "-c", "import time; time.sleep(60)", str(self.dir / "server.py")])
        outside = self.start([str(self.fixture)])
        uninstaller = self.dir / "unins000.exe"
        shutil.copy2(self.fixture, uninstaller)
        uninstalling = self.start([str(uninstaller)])
        time.sleep(1.0)
        found = self.run_script()
        self.assertIn(inside.pid, found)
        self.assertIn(by_cmdline.pid, found)          # a python*.exe whose command line names the folder
        self.assertNotIn(outside.pid, found)
        self.assertNotIn(os.getpid(), found)
        self.assertNotIn(uninstalling.pid, found)
        left = self.run_script("-Stop")
        self.assertEqual(left, {})
        self.assertIsNotNone(inside.wait(timeout=10))
        self.assertIsNotNone(by_cmdline.wait(timeout=10))
        self.assertIsNone(outside.poll())             # left alone
        self.assertIsNone(uninstalling.poll())         # uninstall must not kill its own launcher

    def test_nothing_running_prints_nothing(self):
        self.assertEqual(self.run_script(), {})
        self.assertEqual(self.run_script("-Stop"), {})


if __name__ == "__main__":
    unittest.main()
