"""Highlight reel plugin (owner 10-05): find the best moments of a finished call
and cut captioned vertical clips of them.

Works from the structured call recording (recordings/<date>/<HHMMSS>.jsonl), so it
needs nothing extra recorded. A "moment" is: the human setup (up to two lines just
before), the agent's reply, and the room's reaction (up to two lines just after).

Scoring is plain and inspectable (no model call):
  + laughter / big reactions in the reaction lines ("lol", "haha", "no way", "dude")
  + the agent's line is punchy (4-30 words)          - it's a lecture (60+ words)
  + someone addressed the agent by name in the setup
  - the agent's line was dropped/timed out, or the moment is all filler

Consent: moments are listed privately for the owner (it's their own call log),
but a clip is only rendered after the owner sets "Everyone in my calls agreed to
appear". Clips are captions on a plain brand card (no screen capture). Call audio
is not used: the recorder only keeps the input side, so the agent's voice would be
missing and the clip would misrepresent the moment.

Output: recordings/<date>/<HHMMSS>.highlights.json and recordings/<date>/highlights/.
Off by default. Runs only when no call is live.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
LIVE_WINDOW_S = 180.0
SETUP_GAP_S = 12.0      # a human line this close before the agent counts as setup
REACT_GAP_S = 9.0       # a human line this close after the agent counts as reaction
MAX_MOMENTS = 12

_lock = threading.RLock()

_LAUGH = re.compile(r"\b(?:lol+|lmao+|lmfao|rofl|haha+|hahaha\w*|hehe+|ha ha)\b", re.I)
_BIG = re.compile(r"\b(?:no way|oh my god|omg|that's (?:hilarious|insane|crazy|amazing|so good|so funny)|"
                  r"wait what|holy (?:shit|crap)|let'?s go+|i'?m dying|nooo+|good one|nice one|"
                  r"savage|facts|exactly)\b", re.I)
# insults, slurs, abuse: a moment with any of these is never offered as a clip
_HARSH = re.compile(r"(?:\b(?:bitch\w*|stupid|idiot|retard\w*|shut (?:the fuck )?up|fuck (?:you|off)|"
                    r"simp|kys|kill yourself|whore|slut|f[a@]gg?\w*|n[i1]gg\w*|chigg\w*|n-word)\b)", re.I)
_FILLER = re.compile(r"^(?:yeah|yes|no|ok(?:ay)?|mhm|uh+|um+|hmm+|right|sure|cool|nice)[.!?]*$", re.I)
_PRIVATE = re.compile(r"(?:\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|[\w.+-]+@[\w-]+\.[\w.]+|"
                      r"\b\d{1,5}\s+\w+\s+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr)\b)", re.I)


# ---------------------------------------------------------------- reading
def _recordings() -> List[str]:
    fs = glob.glob(str(ROOT / "recordings" / "*" / "*.jsonl"))
    return sorted(fs, key=os.path.getmtime)


def call_live(clock: Callable[[], float] = time.time) -> bool:
    fs = _recordings()
    return bool(fs) and clock() - max(os.path.getmtime(f) for f in fs) < LIVE_WINDOW_S


def _rows(path: str) -> list:
    out = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def turns(rows: list) -> list:
    """[{'t', 'agent': bool, 'who', 'text', 'dropped': bool}] with recorded names applied."""
    names, persona, out = {}, "agent", []
    for r in rows:
        k = r.get("k") or r.get("kind")
        if k == "name" and r.get("spk") and r.get("name"):
            names[r["spk"]] = r["name"]
        elif k == "persona":
            persona = r.get("persona") or persona
        elif k == "user" and (r.get("text") or "").strip():
            out.append({"t": float(r.get("t", 0)), "agent": False, "spk": r.get("spk") or "S1",
                        "who": None, "text": r["text"].strip(), "dropped": False})
        elif k == "agent" and (r.get("text") or "").strip():
            out.append({"t": float(r.get("t", 0)), "agent": True, "spk": None,
                        "who": (r.get("persona") or persona).title(), "text": r["text"].strip(),
                        "dropped": False})
        elif k == "drop" and out and out[-1]["agent"]:
            out[-1]["dropped"] = True
    # names learned later in the call still apply to earlier lines
    for x in out:
        if not x["agent"]:
            x["who"] = names.get(x["spk"]) or _guest(x["spk"])
    return out


def _guest(spk: str) -> str:
    if (spk or "").lower() in ("user", "you"):
        return "You"
    m = re.search(r"\d+", spk or "")
    return f"Guest {int(m.group()) + 1}" if m else "Guest"


def _words(s: str) -> int:
    return len(re.findall(r"[\w']+", s or ""))


# ---------------------------------------------------------------- scoring
def score(setup: list, agent: dict, react: list) -> tuple:
    s, why = 0.0, []
    rtxt = " ".join(x["text"] for x in react if _words(x["text"]) <= 12)  # reactions are short
    laughs = len(_LAUGH.findall(rtxt))
    bigs = len(_BIG.findall(rtxt))
    if laughs:
        s += 3.0 * min(laughs, 2); why.append("laughter")
    if bigs:
        s += 2.0 * min(bigs, 2); why.append("big reaction")
    n = _words(agent["text"])
    if 4 <= n <= 30:
        s += 1.0; why.append("punchy")
    elif n >= 60:
        s -= 2.0; why.append("long")
    name = agent["who"].lower()
    if any(re.search(rf"\b{re.escape(name)}\b", x["text"], re.I) for x in setup):
        s += 1.0; why.append("called by name")
    if agent.get("dropped"):
        s -= 5.0; why.append("dropped")
    if react and all(_FILLER.match(x["text"]) for x in react) and not laughs:
        s -= 1.0
    if any(_PRIVATE.search(x["text"]) for x in setup + [agent] + react):
        s = -99.0; why.append("private detail")
    if any(_HARSH.search(x["text"]) for x in setup + [agent] + react):
        s = -99.0; why.append("harsh language")
    return s, why


def moments(rows: list, limit: int = MAX_MOMENTS) -> list:
    tt = turns(rows)
    found = []
    for i, a in enumerate(tt):
        if not a["agent"]:
            continue
        setup = [x for x in tt[max(0, i - 2):i] if not x["agent"] and a["t"] - x["t"] <= SETUP_GAP_S]
        react = []
        for x in tt[i + 1:i + 3]:
            if x["agent"] or x["t"] - a["t"] > REACT_GAP_S:
                break
            react.append(x)
        if not setup:
            continue
        s, why = score(setup, a, react)
        if s < 2.0:
            continue
        found.append({"t": a["t"], "score": round(s, 1), "why": why,
                      "lines": [{"who": x["who"], "agent": x["agent"], "text": x["text"]}
                                for x in setup + [a] + react]})
    found.sort(key=lambda m: -m["score"])
    picked = []
    for m in found:  # no two moments within 20 s of each other
        if all(abs(m["t"] - p["t"]) > 20 for p in picked):
            picked.append(m)
        if len(picked) >= limit:
            break
    picked.sort(key=lambda m: m["t"])
    for m in picked:
        m["id"] = uuid.uuid4().hex[:8]
    return picked


# ---------------------------------------------------------------- storage
def _store(path: str) -> Path:
    return Path(path).with_suffix(".highlights.json")


def _load(path: str) -> dict:
    try:
        return json.loads(_store(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: str, data: dict) -> None:
    p = _store(path)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _latest_done(clock=time.time, force: bool = False) -> Optional[str]:
    for f in reversed(_recordings()):
        if (force or clock() - os.path.getmtime(f) >= LIVE_WINDOW_S) and turns(_rows(f)):
            return f
    return None


def find(path: Optional[str] = None, force: bool = False, clock=time.time, limit: Optional[int] = None) -> dict:
    """Rank the moments of one call (default: the latest finished one) and keep them."""
    if not force and call_live(clock):
        return {"error": "A call looks live (recording updated in the last 3 minutes). Try after the call."}
    path = path or _latest_done(clock, force)
    if not path:
        return {"error": "No finished call recording found yet."}
    if limit is None:
        try:
            import plugins
            limit = int((plugins.settings_of("highlights") or {}).get("count", 6))
        except Exception:  # noqa: BLE001
            limit = 6
    ms = moments(_rows(path), limit)
    with _lock:
        _save(path, {"call": path, "made": clock(), "moments": ms})
    return {"call": path, "moments": len(ms)}


def _current() -> tuple:
    for f in reversed(_recordings()):
        d = _load(f)
        if d.get("moments") is not None:
            return f, d
    return None, {}


def feed() -> list:
    path, d = _current()
    if not path:
        return []
    stamp = Path(path).stem
    day = Path(path).parent.name
    out = []
    for m in d.get("moments", []):
        clip = m.get("clip")
        acts = ([{"act": "open", "label": "Show file"}] if clip and Path(clip).exists()
                else [{"act": "clip", "label": "Make clip"}]) + [{"act": "delete", "label": "Remove"}]
        agent_line = next((x["text"] for x in m["lines"] if x["agent"]), "")
        out.append({"title": agent_line[:160], "id": m["id"], "agent": "",
                    "meta": f"{day} {stamp[:2]}:{stamp[2:4]} · {time.strftime('%H:%M:%S', time.localtime(m['t']))}"
                            f" · {', '.join(m['why'])}" + (" · clip ready" if clip else ""),
                    "items": [{"text": f"{x['who']}: {x['text']}"} for x in m["lines"]],
                    "actions": acts})
    return out


def consent_ok() -> bool:
    try:
        import plugins
        return (plugins.settings_of("highlights") or {}).get("consent") == "agreed"
    except Exception:  # noqa: BLE001
        return False


def action(agent: str, mid: str, act: str) -> bool:
    path, d = _current()
    if not path:
        return False
    with _lock:
        ms = d.get("moments", [])
        m = next((x for x in ms if x["id"] == mid), None)
        if m is None:
            return False
        if act == "delete":
            d["moments"] = [x for x in ms if x["id"] != mid]
            _save(path, d)
            return True
        if act == "clip":
            if not consent_ok():
                raise ValueError("First confirm in the settings above that everyone in your calls "
                                 "agreed to appear in clips.")
            m["clip"] = str(render(m, Path(path).parent / "highlights", f"{Path(path).stem}_{mid}"))
            _save(path, d)
            return True
        if act == "open":
            clip = m.get("clip")
            if clip and Path(clip).exists() and os.name == "nt":
                subprocess.Popen(["explorer", "/select,", str(Path(clip))])
            return bool(clip)
    raise ValueError("unknown action")


# ---------------------------------------------------------------- rendering
def _ts(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def line_seconds(text: str) -> float:
    return max(1.8, min(9.0, 0.9 + 0.32 * _words(text)))


def ass(m: dict) -> tuple:
    head = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 0\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, "
            "Bold, Alignment, MarginL, MarginR, MarginV, BorderStyle, Outline, Shadow\n"
            "Style: H,Segoe UI,60,&H00F2F2F2,&H00000000,&H00000000,0,5,90,90,0,1,0,0\n"
            "Style: A,Segoe UI,64,&H00E8D45A,&H00000000,&H00000000,1,5,90,90,0,1,0,0\n"
            "Style: W,Segoe UI,40,&H8CFFFFFF,&H00000000,&H00000000,1,2,0,0,120,1,0,0\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Text\n")
    rows, t = [], 0.4
    for x in m["lines"]:
        d = line_seconds(x["text"])
        txt = x["text"].replace("\n", " ").replace("{", "(").replace("}", ")")
        rows.append(f"Dialogue: 0,{_ts(t)},{_ts(t + d)},{'A' if x['agent'] else 'H'},"
                    f"{{\\fad(180,120)}}{{\\b1}}{x['who']}{{\\b0}}\\N{txt}")
        t += d + 0.25
    total = t + 0.6
    rows.append(f"Dialogue: 0,{_ts(0)},{_ts(total)},W,ATLAS · unedited moment from a real call")
    return head + "\n".join(rows) + "\n", total


def render(m: dict, out_dir: Path, name: str) -> Path:
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg isn't installed, so clips can't be made.")
    out_dir.mkdir(parents=True, exist_ok=True)
    text, total = ass(m)
    sub = out_dir / f"{name}.ass"
    sub.write_text(text, encoding="utf-8")
    dst = out_dir / f"{name}.mp4"
    s = sub.as_posix().replace(":", r"\:")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"color=c=0x0B0D10:s=1080x1920:d={total:.2f}:r=30",
           "-vf", f"subtitles='{s}'", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0 or not dst.exists():
        raise ValueError("ffmpeg failed: " + (r.stderr or "")[-200:])
    return dst
