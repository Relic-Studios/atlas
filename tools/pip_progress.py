"""Runs a command (pip) and passes its output on line by line. Once the command prints "Installing collected packages",
it also prints "INSTALLPROGRESS <written> <expected>" about twice a second: how many bytes the drive holding --path has
lost since then (its free space going down), capped at --expected and never going down. pip prints nothing while it
unpacks and installs, which takes minutes for PyTorch; with these lines the setup program's bar keeps moving.

    python tools/pip_progress.py --path <folder on the target drive> --expected <bytes> -- <command ...>

Exit code: the command's. Used by install.ps1 -Progress (design: section 4, "Install progress")."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--expected", type=int, required=True)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command after --")
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    lock = threading.Lock()
    state = {"start": None, "shown": 0}
    stop = threading.Event()

    def emit(line: str) -> None:
        with lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def watch() -> None:
        while not stop.wait(a.every):
            if state["start"] is None:
                continue
            try:
                grown = state["start"] - shutil.disk_usage(a.path).free
            except OSError:
                continue
            value = max(state["shown"], min(a.expected, grown))
            if value != state["shown"]:
                state["shown"] = value
                emit(f"INSTALLPROGRESS {value} {a.expected}")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        emit(line)
        if state["start"] is None and "Installing collected packages" in line:
            state["start"] = shutil.disk_usage(a.path).free
    rc = proc.wait()
    stop.set()
    watcher.join(timeout=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
