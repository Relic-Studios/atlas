"""Build the one-file Windows installer: fresh public export -> leak check -> Inno Setup.
Usage: python tools/build_installer.py [--version 0.2.1] [--iscc PATH]
Output: dist/ATLAS-Setup-<version>.exe. Never writes inside the dev tree except dist/."""
import argparse, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISCC_GUESSES = [r"E:\AstraRelease-buildtools\InnoSetup\ISCC.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.2.3")
    ap.add_argument("--iscc", default=next((p for p in ISCC_GUESSES if os.path.exists(p)), None))
    ap.add_argument("--ref", default="HEAD", help="git ref to package (public repo: a release tag)")
    a = ap.parse_args()
    if not a.iscc or not os.path.exists(a.iscc):
        sys.exit("ISCC.exe not found; pass --iscc")
    out = os.path.join(tempfile.gettempdir(), "atlas_installer_src")
    shutil.rmtree(out, ignore_errors=True)
    exporter = os.path.join(ROOT, "tools", "export_public.py")
    if os.path.exists(exporter):
        # dev tree: build from a fresh public export (leak-checked)
        subprocess.run([sys.executable, exporter, "--out", out, "--check"], check=True)
    else:
        # public repo: package exactly the committed ref, never the working folder
        if subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip() and a.ref == "HEAD":
            print("note: uncommitted changes are NOT included; packaging", a.ref)
        os.makedirs(out)
        tar = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar", a.ref],
                             capture_output=True, check=True).stdout
        import io, tarfile
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(out, filter="data")
    dist = os.path.join(ROOT, "dist"); os.makedirs(dist, exist_ok=True)
    subprocess.run([a.iscc, f"/DSrcDir={out}", f"/DAppVersion={a.version}",
                    f"/O{dist}", os.path.join(ROOT, "installer", "atlas.iss")], check=True)
    print(os.path.join(dist, f"ATLAS-Setup-{a.version}.exe"))


if __name__ == "__main__":
    main()
