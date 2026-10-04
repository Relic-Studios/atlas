"""Record a live ATLAS demo: real pipeline, scripted human voices, UI + audio -> MP4.

  python tools/demo/run_demo.py --scene <dir with scene.json> --persona fae \
      [--port 8012] [--out <dir>] [--caption "..."]

Runs from a throwaway COPY of this tree (fresh memory, no call recording), on its
own port, so the dev build's agents, memories, recordings and logs are untouched.
Needs the GPU (Whisper + TTS): never run while a real call is live.
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = {".git", "recordings", "logs", "corpus", "training", "_bak", "dist", "memory_db",
           "__pycache__", "node_modules", "agent_state", "tests"}
ELECTRON = ROOT / "desktop" / "node_modules" / "electron" / "dist" / "electron.exe"


def copy_tree(dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(ROOT, dst, ignore=lambda d, names: [n for n in names if n in EXCLUDE
                                                       or n.endswith((".log", ".out"))])


def wait_http(url, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


async def set_persona(port, persona):
    import websockets
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": "set_persona", "persona": persona}))
        t0 = time.time()
        while time.time() - t0 < 15:
            msg = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if msg.get("type") == "personas":
                return msg
    return None


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--out")
    ap.add_argument("--caption", default="")
    ap.add_argument("--warm_s", type=float, default=20.0, help="settle time after boot")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out or ROOT / "logs" / "demo" / f"{a.persona}_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "Temp" / "atlas_demo_run"
    print("copying tree ->", run, flush=True)
    copy_tree(run)

    env = dict(os.environ, PYTHONPATH="", PYTHONIOENCODING="utf-8", ATLAS_PORT=str(a.port),
               ATLAS_HOST="127.0.0.1", ATLAS_DEMO_SCENE=str(Path(a.scene, "scene.json").resolve()),
               ATLAS_DEMO_OUT=str(out), ATLAS_MEMORY_DIR=str(run / "agent_state" / "memory"),
               ATLAS_RECORD="0")
    log = open(out / "server.log", "w", encoding="utf-8")
    srv = subprocess.Popen([sys.executable, "-u", "server.py"], cwd=run, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    cap = None
    try:
        print("booting backend on", a.port, flush=True)
        if not wait_http(f"http://127.0.0.1:{a.port}/"):
            raise SystemExit("backend did not come up; see " + str(out / "server.log"))
        msg = asyncio.run(set_persona(a.port, a.persona))
        print("persona:", (msg or {}).get("current"), flush=True)
        time.sleep(a.warm_s)
        frames = out / "frames"
        cap = subprocess.Popen([str(ELECTRON), str(ROOT / "tools" / "demo" / "capture.js"),
                                f"http://127.0.0.1:{a.port}/", str(frames), str(out / "DONE")],
                               stdout=open(out / "capture.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(6)                      # UI loads, websocket connects, first paint
        (out / "GO").write_text("go")
        print("scene running...", flush=True)
        t0 = time.time()
        while not (out / "DONE").exists():
            if srv.poll() is not None:
                raise SystemExit("backend exited mid-scene; see server.log")
            if time.time() - t0 > 900:
                raise SystemExit("scene did not finish in 15 min")
            time.sleep(1)
        cap.wait(timeout=30)
    finally:
        if cap and cap.poll() is None:
            kill_tree(cap.pid)
        kill_tree(srv.pid)
        log.close()
    mp4 = out / f"atlas_demo_{a.persona}.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools" / "demo" / "assemble.py"), str(out),
                    str(out / "frames"), str(mp4)] + (["--caption", a.caption] if a.caption else []),
                   check=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "demo" / "timeline.py"), str(out)], check=False)
    print("DONE", mp4)


if __name__ == "__main__":
    main()
