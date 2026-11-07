"""Build the one-file Windows installer: fresh public export -> leak check -> staging -> Inno Setup.

    python tools/build_installer.py [--version 0.7.0] [--iscc PATH] [--ref REF] [--test] [--stage-only]

Output: dist/ATLAS-Setup-<version>.exe (with --test: ATLAS-Setup-<version>-test.exe, which adds the test-only
command-line parameters of installer/atlas.iss). Never writes inside the dev tree except dist/.

Staging adds what lets users install without Python or Node.js:
  runtime/python/        a standalone CPython 3.12 (python-build-standalone; pinned URL and SHA-256, cached outside dist/)
  desktop/node_modules/  the desktop shell's Electron, from `npm ci` with the exported lockfile
--stage-only prepares the staging folder, prints its path and stops (for running install.ps1 by hand)."""
import argparse
import hashlib
import io
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISCC_GUESSES = [r"E:\AstraRelease-buildtools\InnoSetup\ISCC.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"]
# Each release checks python-build-standalone for a newer 3.12.x build and bumps both lines (security fixes).
RUNTIME_URL = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
               "cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz")
RUNTIME_SHA256 = "590dfcbe3a34c07301e6332e13e2188b98655cd072b08afa45a7998dc4c22bc5"
CACHE = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()) / "atlas_build_cache"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_runtime() -> Path:
    """The pinned runtime archive, from the cache when its SHA-256 matches; the build stops on a mismatch."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / RUNTIME_URL.rsplit("/", 1)[1].replace("%2B", "+")
    if dest.exists() and sha256(dest) == RUNTIME_SHA256:
        return dest
    part = dest.with_name(dest.name + ".part")
    print("downloading", RUNTIME_URL)
    req = urllib.request.Request(RUNTIME_URL, headers={"User-Agent": "atlas-build"})
    try:  # certifi's roots when installed: some PCs' Windows store lacks GitHub's intermediate for Python's ssl
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(part)
    if got != RUNTIME_SHA256:
        part.unlink()
        sys.exit(f"runtime archive SHA-256 mismatch: got {got}, pinned {RUNTIME_SHA256}")
    part.replace(dest)
    return dest


def stage_runtime(out: Path) -> None:
    with tarfile.open(fetch_runtime()) as t:
        t.extractall(out / "runtime", filter="data")          # the archive's top folder is python/
    exe = out / "runtime" / "python" / "python.exe"
    if not exe.exists():
        sys.exit(f"runtime staging failed: {exe} missing")
    for pdb in (out / "runtime").rglob("*.pdb"):            # debug symbols, about 83 MB: not needed to run
        pdb.unlink()
    subprocess.run([str(exe), "-I", "-c", "import venv, ensurepip, ssl, sqlite3, ctypes"], check=True)


def stage_desktop(out: Path) -> None:
    d = out / "desktop"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        sys.exit("Node.js (npm and node) is needed to stage the desktop shell")
    subprocess.run([npm, "ci", "--no-audit", "--no-fund", "--loglevel=error"], cwd=d, check=True)
    subprocess.run([node, os.path.join("node_modules", "electron", "install.js")], cwd=d, check=True)
    exe = d / "node_modules" / "electron" / "dist" / "electron.exe"
    if not exe.exists():
        sys.exit(f"desktop staging failed: {exe} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.7.0")
    ap.add_argument("--iscc", default=next((p for p in ISCC_GUESSES if os.path.exists(p)), None))
    ap.add_argument("--ref", default="HEAD", help="git ref to package (public repo: a release tag)")
    ap.add_argument("--test", action="store_true", help="test build: adds the test-only Setup parameters")
    ap.add_argument("--stage-only", action="store_true", help="prepare the staging folder and stop")
    ap.add_argument("--quick", action="store_true", help="light compression, for local iteration only")
    ap.add_argument("--reuse-staging", action="store_true", help="compile from the last staging folder as it is")
    a = ap.parse_args()
    if not a.stage_only and (not a.iscc or not os.path.exists(a.iscc)):
        sys.exit("ISCC.exe not found; pass --iscc")
    out = Path(tempfile.gettempdir()) / "atlas_installer_src"
    if a.reuse_staging and (out / "runtime" / "python" / "python.exe").exists():
        return compile_setup(a, out)
    shutil.rmtree(out, ignore_errors=True)
    exporter = os.path.join(ROOT, "tools", "export_public.py")
    if os.path.exists(exporter):
        # dev tree: build from a fresh public export (leak-checked)
        subprocess.run([sys.executable, exporter, "--out", str(out), "--check"], check=True)
    else:
        # public repo: package exactly the committed ref, never the working folder
        if subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip() and a.ref == "HEAD":
            print("note: uncommitted changes are NOT included; packaging", a.ref)
        out.mkdir(parents=True)
        tar = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", a.ref],
                             capture_output=True, check=True).stdout
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(out, filter="data")
    stage_runtime(out)
    stage_desktop(out)
    if a.stage_only:
        print(out)
        return
    compile_setup(a, out)


def compile_setup(a, out: Path) -> None:
    dist = os.path.join(ROOT, "dist")
    os.makedirs(dist, exist_ok=True)
    cmd = [a.iscc, f"/DSrcDir={out}", f"/DAppVersion={a.version}", f"/O{dist}"]
    if a.test:
        cmd.append("/DTEST")
    if a.quick:
        cmd.append("/DQUICK")
    subprocess.run(cmd + [str(out / "installer" / "atlas.iss")], check=True)
    print(os.path.join(dist, f"ATLAS-Setup-{a.version}{'-test' if a.test else ''}.exe"))


if __name__ == "__main__":
    main()
