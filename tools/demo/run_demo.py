"""Record a live ATLAS demo: real pipeline, scripted human voices, UI + audio -> MP4.

  python tools/demo/run_demo.py --scene <dir with scene.json> --persona fae \
      [--port 8012] [--out <dir>] [--caption "..."]

Runs from a throwaway COPY of this tree (fresh memory, no call recording), on its
own port, so the dev build's agents, memories, recordings and logs are untouched.
Needs the GPU (Whisper + TTS): never run while a real call is live.

Public marketing footage (no dev agents on screen, original public-domain-voiced agent):
  python tools/demo/run_demo.py --src ../../atlas --model qwen3:14b
      --agent tools/demo/agents/wren.json --scene <rendered wren scene dir>
"""
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXCLUDE = {".git", "recordings", "logs", "corpus", "training", "_bak", "dist", "memory_db",
           "__pycache__", "node_modules", "agent_state", "tests"}
ELECTRON = ROOT / "desktop" / "node_modules" / "electron" / "dist" / "electron.exe"


def copy_tree(dst: Path, src: Path = ROOT):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in EXCLUDE
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


def _post(port, path, body=None, raw=None, timeout=240):
    data = raw if raw is not None else json.dumps(body).encode()
    ctype = "application/octet-stream" if raw is not None else "application/json"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method="POST",
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def create_agent(port, spec_path: Path) -> str:
    """Make the demo agent the way a user would: import its voice clip, then create it
    from plain fields. Returns the new agent's id."""
    from urllib.parse import quote
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    wav = spec_path.parent / spec["voice_wav"]
    tr = wav.with_suffix(".txt").read_text(encoding="utf-8") if wav.with_suffix(".txt").exists() else ""
    st, r = _post(port, f"/api/voices/import?label={quote(spec['voice_label'])}&filename={quote(wav.name)}"
                        f"&transcript={quote(tr)}", raw=wav.read_bytes())
    if st != 200 or not r.get("key"):
        raise SystemExit(f"voice import failed: {st} {r}")
    st, r = _post(port, "/api/agents", {"name": spec["name"], "voice": r["key"], "fields": spec["fields"]})
    if st != 200:
        raise SystemExit(f"agent create failed: {st} {r}")
    return (r.get("agent") or {}).get("id") or spec["name"].lower()


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--persona", help="existing agent id (dev build only)")
    ap.add_argument("--agent", help="agent spec json: created via the real import/create APIs")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--src", help="tree to run, e.g. the public repo (default: this dev tree)")
    ap.add_argument("--model", help="local Ollama model; writes a finished-setup user/settings.json")
    ap.add_argument("--out")
    ap.add_argument("--caption", default="")
    ap.add_argument("--warm_s", type=float, default=20.0, help="settle time after boot")
    a = ap.parse_args()
    if not (a.persona or a.agent):
        ap.error("give --persona or --agent")
    tag = a.persona or Path(a.agent).stem

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out or ROOT / "logs" / "demo" / f"{tag}_{stamp}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    run = Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "Temp" / "atlas_demo_run"
    print("copying tree ->", run, flush=True)
    copy_tree(run, Path(a.src).resolve() if a.src else ROOT)
    if a.model:
        (run / "user").mkdir(exist_ok=True)
        (run / "user" / "settings.json").write_text(json.dumps({"setup_complete": True, "llm": {
            "mode": "local", "local": {"url": "http://127.0.0.1:11434", "model": a.model}},
            "audio": {"input": None, "output": None}}), encoding="utf-8")

    env = dict(os.environ, PYTHONPATH="", PYTHONIOENCODING="utf-8", ATLAS_PORT=str(a.port),
               ATLAS_HOST="127.0.0.1", ATLAS_DEMO_SCENE=str((Path(a.scene) if a.scene.endswith(".json") else Path(a.scene, "scene.json")).resolve()),
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
        persona = a.persona or create_agent(a.port, Path(a.agent).resolve())
        msg = asyncio.run(set_persona(a.port, persona))
        print("persona:", (msg or {}).get("current"), flush=True)
        time.sleep(a.warm_s)
        frames = out / "frames"
        # Electron 41 exits 127 when a URL argument is combined with other
        # arguments, so the capture settings travel in the environment.
        cenv = dict(os.environ, CAP_URL=f"http://127.0.0.1:{a.port}/",
                    CAP_DIR=str(frames), CAP_DONE=str(out / "DONE"))
        cap = subprocess.Popen([str(ELECTRON), str(ROOT / "tools" / "demo" / "capture.js")], env=cenv,
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
    mp4 = out / f"atlas_demo_{tag}.mp4"
    subprocess.run([sys.executable, str(ROOT / "tools" / "demo" / "assemble.py"), str(out),
                    str(out / "frames"), str(mp4)] + (["--caption", a.caption] if a.caption else []),
                   check=True)
    print("DONE", mp4)


if __name__ == "__main__":
    main()
