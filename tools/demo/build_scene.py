"""Render a demo scene's human lines with Piper (public-domain / CC0 voices, CPU).

  <piper-venv-python> tools/demo/build_scene.py tools/demo/game_night.json --agent Fae --out <dir>
Writes <dir>/scene.json (paths relative) + <dir>/NN_<who>.wav. Feed it to ATLAS with
ATLAS_DEMO_SCENE=<dir>/scene.json.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("scene"); ap.add_argument("--agent", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--voices", default=os.environ.get("PIPER_VOICES", ""))
a = ap.parse_args()
sc = json.loads(Path(a.scene).read_text(encoding="utf-8"))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
vd = Path(a.voices)
for i, ln in enumerate(sc["lines"]):
    ln["text"] = ln["text"].replace("{N}", a.agent)
    wav = f"{i:02d}_{ln['who']}.wav"
    model = vd / (sc["voices"][ln["who"]] + ".onnx")
    subprocess.run([sys.executable, "-m", "piper", "-m", str(model), "-f", str(out / wav)],
                   input=ln["text"].encode("utf-8"), check=True, capture_output=True)
    ln["wav"] = wav
sc["agent"] = a.agent
(out / "scene.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
print("scene:", out / "scene.json", len(sc["lines"]), "lines")
