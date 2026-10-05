"""Auto-clip a demo run into vertical short-form candidates (STRATEGY.md section 4, steps 4-5).

  python tools/demo/clip.py <demo_out_dir> <assembled.mp4> [--out DIR] [--max N]

Inputs come from a real run: events.jsonl (demo_call.py), DONE, master.wav and the
16:9 MP4 from assemble.py. Every candidate is built around one agent turn and
opens on the hook (the human line that set it up, if short, else the agent line).

Each candidate goes through the rule pre-filter; nothing is auto-posted. The
output folder gets clip_XX.mp4 for survivors plus shortlist.json with the scores
and the reasons any candidate was dropped, for a human pick.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Rule pre-filter (STRATEGY.md section 5). Tunable, deliberately simple.
MAX_CLIP_S = 45.0
MIN_CLIP_S = 6.0
HOOK_MAX_S = 2.0        # the hook must be audible within this
DEAD_AIR_S = 1.5        # longest allowed silence inside a clip
SETUP_MAX_S = 3.5       # a human setup line longer than this is cut, clip opens on the agent
PRE_ROLL_S = 0.25
POST_ROLL_S = 0.9
SILENCE_RMS = 0.006


@dataclass
class Turn:
    who: str
    text: str
    start: float
    end: float
    agent: bool = False


@dataclass
class Candidate:
    start: float
    end: float
    hook: str
    turns: list = field(default_factory=list)
    score: float = 0.0
    keep: bool = True
    reasons: list = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.end - self.start


def load_turns(events: list[dict]) -> tuple[list[Turn], set]:
    """Human lines from line_start/line_end; agent lines from agent_start/agent_end,
    labelled with the text from the nearest preceding said_agent event."""
    turns, open_line, said, timeouts = [], {}, [], set()
    agent_t0 = None
    for e in events:
        k, t = e.get("kind"), float(e.get("t", 0.0))
        if k == "line_start":
            open_line[e["line"]] = Turn(e.get("who") or "?", e.get("text") or "", t, t)
        elif k == "line_end" and e.get("line") in open_line:
            tr = open_line.pop(e["line"])
            tr.end = t
            turns.append(tr)
        elif k == "said_agent":
            said.append([t, e.get("text") or "", e.get("persona") or "agent", False])
        elif k == "agent_start":
            agent_t0 = t
        elif k == "agent_end" and agent_t0 is not None:
            text, who = "", "agent"
            # Text is finalised while the reply streams, so said_agent can land up to a
            # couple of seconds AFTER agent_start (take 11: +1.5s). Prefer the earliest
            # unused text logged during this reply; fall back to the latest one before it.
            during = [s for s in said if not s[3] and agent_t0 - 0.5 <= s[0] <= t]
            before = [s for s in said if not s[3] and s[0] < agent_t0 - 0.5]
            pick = during[0] if during else (before[-1] if before else None)
            if pick is not None:
                pick[3] = True
                text, who = pick[1], pick[2]
            turns.append(Turn(who, text, agent_t0, t, agent=True))
            agent_t0 = None
        elif k == "reply_timeout":
            timeouts.add(t)
    turns.sort(key=lambda x: x.start)
    return turns, timeouts


def longest_silence(rms: list[float], hop_s: float, a: float, b: float) -> float:
    i0, i1 = max(0, int(a / hop_s)), min(len(rms), int(b / hop_s) + 1)
    best = run = 0
    for v in rms[i0:i1]:
        run = run + 1 if v < SILENCE_RMS else 0
        best = max(best, run)
    return best * hop_s


def plan(turns: list[Turn], timeouts: set, rms: list[float], hop_s: float) -> list[Candidate]:
    out = []
    for idx, tr in enumerate(turns):
        if not tr.agent or not tr.text.strip():
            continue
        prev = turns[idx - 1] if idx and not turns[idx - 1].agent else None
        if prev and prev.end - prev.start <= SETUP_MAX_S and tr.start - prev.end < 2.5:
            start, hook = prev.start - PRE_ROLL_S, prev.text
        else:
            start, hook = tr.start - PRE_ROLL_S, tr.text
        start = max(0.0, start)
        end = tr.end + POST_ROLL_S
        # Extend through the following back-and-forth while it stays tight.
        j = idx + 1
        while j < len(turns) and turns[j].start - end < 1.2 and turns[j].end + POST_ROLL_S - start <= MAX_CLIP_S:
            end = turns[j].end + POST_ROLL_S
            j += 1
        c = Candidate(start, end, hook, [asdict(x) for x in turns if x.end > start and x.start < end])
        score_candidate(c, tr, timeouts, rms, hop_s)
        out.append(c)
    # Drop overlapping duplicates: keep the higher score.
    out.sort(key=lambda c: -c.score)
    kept: list[Candidate] = []
    for c in out:
        if any(min(c.end, k.end) - max(c.start, k.start) > 0.5 * c.length for k in kept if k.keep):
            c.keep, c.reasons = False, c.reasons + ["overlaps a better clip"]
        kept.append(c)
    return sorted(kept, key=lambda c: c.start)


def score_candidate(c: Candidate, agent_turn: Turn, timeouts: set, rms, hop_s) -> None:
    first_sound = next((t["start"] for t in c.turns if t["end"] > c.start), c.end)
    hook_at = max(0.0, first_sound - c.start)
    gap = longest_silence(rms, hop_s, c.start + 0.3, c.end - POST_ROLL_S) if rms else 0.0
    words = len(agent_turn.text.split())
    checks = [
        (c.length <= MAX_CLIP_S, f"too long ({c.length:.1f}s)"),
        (c.length >= MIN_CLIP_S, f"too short ({c.length:.1f}s)"),
        (hook_at <= HOOK_MAX_S, f"hook at {hook_at:.1f}s"),
        (gap <= DEAD_AIR_S, f"dead air {gap:.1f}s"),
        (not any(c.start <= t <= c.end for t in timeouts), "agent missed a reply in this window"),
        (3 <= words <= 60, f"agent line has {words} words"),
    ]
    c.reasons = [why for ok, why in checks if not ok]
    c.keep = not c.reasons
    # Tie-break score among survivors: early hook, tight pacing, short punchy agent line,
    # several turns (a real exchange, not a monologue).
    n_turns = len(c.turns)
    c.score = round((HOOK_MAX_S - min(hook_at, HOOK_MAX_S)) + (DEAD_AIR_S - min(gap, DEAD_AIR_S))
                    + min(n_turns, 5) * 0.4 + (1.0 if 6 <= words <= 30 else 0.0)
                    - max(0.0, c.length - 30.0) * 0.05, 3)


def _ts(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def ass_captions(c: Candidate) -> str:
    """Burned-in captions from the exact script and the agent's real text."""
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
            "Bold, Alignment, MarginL, MarginR, MarginV, BorderStyle, Outline, Shadow\n"
            "Style: H,Segoe UI,54,&H00FFFFFF,&H00000000,&H80000000,1,2,70,70,520,1,4,0\n"
            "Style: A,Segoe UI,58,&H00FFE14D,&H00000000,&H80000000,1,2,70,70,520,1,4,0\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    rows = []
    for t in c.turns:
        txt = (t["text"] or "").replace("\n", " ").replace("{", "(").replace("}", ")")
        if not txt:
            continue
        who = t["who"].capitalize() if t["agent"] else t["who"]
        rows.append(f"Dialogue: 0,{_ts(t['start'] - c.start)},{_ts(t['end'] - c.start + 0.2)},"
                    f"{'A' if t['agent'] else 'H'},{{\\b1}}{who}:{{\\b0}} {txt}")
    return head + "\n".join(rows) + "\n"


def render(c: Candidate, mp4: Path, out: Path, n: int) -> Path:
    ass = out / f"clip_{n:02d}.ass"
    ass.write_text(ass_captions(c), encoding="utf-8")
    sub = ass.as_posix().replace(":", r"\:")
    # 9:16: blurred full-bleed background + the 16:9 UI centred, captions below it.
    vf = ("[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
          "boxblur=24:2,eq=brightness=-0.15[bg];[b]scale=1080:-2[fg];"
          f"[bg][fg]overlay=0:(H-h)/2-180,subtitles='{sub}'")
    dst = out / f"clip_{n:02d}.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{c.start:.2f}", "-t", f"{c.length:.2f}", "-i", str(mp4),
                    "-filter_complex", vf, "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)], check=True)
    return dst


def rms_track(wav: Path, hop_s: float = 0.05) -> list[float]:
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    h = max(1, int(sr * hop_s))
    n = len(x) // h
    return [float(v) for v in np.sqrt((x[:n * h].reshape(n, h) ** 2).mean(axis=1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo"); ap.add_argument("mp4")
    ap.add_argument("--out", default=None); ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    demo = Path(a.demo)
    out = Path(a.out or demo / "clips"); out.mkdir(parents=True, exist_ok=True)
    events = [json.loads(l) for l in (demo / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    turns, timeouts = load_turns(events)
    rms = rms_track(demo / "master.wav")
    cands = plan(turns, timeouts, rms, 0.05)
    good = sorted([c for c in cands if c.keep], key=lambda c: -c.score)[:a.max]
    files = {}
    if not a.plan_only:
        for n, c in enumerate(good, 1):
            files[id(c)] = render(c, Path(a.mp4), out, n).name
    (out / "shortlist.json").write_text(json.dumps([
        dict(file=files.get(id(c)), start=round(c.start, 2), length=round(c.length, 2), hook=c.hook,
             score=c.score, keep=c.keep, reasons=c.reasons) for c in cands], indent=1), encoding="utf-8")
    print(f"{len(cands)} candidates, {sum(c.keep for c in cands)} pass the filter, rendered {len(files)} -> {out}")
    for c in cands:
        print(f"  {'KEEP' if c.keep else 'drop'} {c.start:6.1f}s {c.length:5.1f}s score={c.score:5.2f}  "
              f"{c.hook[:60]!r} {'; '.join(c.reasons)}")


if __name__ == "__main__":
    main()
