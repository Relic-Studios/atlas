"""Mux a demo run into an MP4: UI frames (capture.js) + master.wav (demo_call.py).

  python tools/demo/assemble.py <demo_out_dir> <frames_dir> <out.mp4> [--caption TEXT]

Frames carry wall-clock ms in their names; DONE carries the audio's wall_t0, so
picture and sound line up to one frame (33 ms).
"""
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("demo"); ap.add_argument("frames"); ap.add_argument("out")
ap.add_argument("--caption", default="")
a = ap.parse_args()

demo, frames = Path(a.demo), Path(a.frames)
done = json.loads((demo / "DONE").read_text(encoding="utf-8"))
t0, dur = done["wall_t0"] * 1000.0, done["seconds"]
fs = sorted(frames.glob("*.jpg"), key=lambda p: int(p.stem))
ts = [int(p.stem) for p in fs]
if not fs:
    raise SystemExit("no frames")
# Frame shown at audio time t = latest frame captured at or before t0 + t.
lines, fps, i = [], 30, 0
n_out = int(dur * fps)
for k in range(n_out):
    wall = t0 + k * 1000.0 / fps
    while i + 1 < len(ts) and ts[i + 1] <= wall:
        i += 1
    lines.append(f"file '{fs[i].as_posix()}'\nduration {1 / fps:.6f}\n")
lst = demo / "frames.txt"
lst.write_text("".join(lines) + f"file '{fs[i].as_posix()}'\n", encoding="utf-8")

vf = "scale=1600:-2,format=yuv420p"
if a.caption:
    cap = a.caption.replace(":", r"\:").replace("'", r"\'")
    vf += (",drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':text='" + cap +
           "':fontcolor=white@0.85:fontsize=22:x=24:y=h-44:box=1:boxcolor=black@0.45:boxborderw=8")
cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
       "-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(demo / "master.wav"),
       "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-crf", "20", "-preset", "medium",
       "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out]
subprocess.run(cmd, check=True)
print("wrote", a.out, f"{dur:.1f}s", len(fs), "captured frames")
