"""Where reply time goes, from the turn trace (turn_trace.py writes logs/turns.jsonl).

  python tools/turn_report.py                 # everything in logs/turns.jsonl
  python tools/turn_report.py --since 2h      # only the last two hours (also 30m, 1d)
  python tools/turn_report.py --slow 3        # list turns whose reply took longer than 3 s

For replies to a person, `wait` is their turn end -> her first audio. Each slow reply is split into: start (turn end
-> generation prepared; negative = started early on a partial), llm (prepared -> first token, includes the decision and
any tools), audio (first token -> first audio). Holds, aborts and errors are counted separately."""
import argparse
import json
import re
import statistics as st
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "logs" / "turns.jsonl"


def since_s(text: str) -> float:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", text.strip())
    if not m:
        raise SystemExit(f"bad --since {text!r} (use 30m, 2h, 1d)")
    return float(m[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m[2]]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def load(path: Path, since=None):
    rows = []
    if not path.exists():
        return rows
    cut = time.time() - since if since else 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("t", 0) >= cut:
            rows.append(r)
    return rows


def report(rows, slow_s=3.0, out=sys.stdout):
    replies = [r for r in rows if not r.get("cue") and r.get("outcome") == "spoke" and r.get("wait") is not None]
    print(f"{len(rows)} generations: " + ", ".join(f"{n} {k}" for k, n in Counter(r.get("outcome", "open")
                                                                                     for r in rows).most_common()),
          file=out)
    if not replies:
        print("no spoken replies with a measured wait yet", file=out)
        return
    w = [r["wait"] for r in replies]
    print(f"replies to people: {len(w)}; wait median {st.median(w):.2f} s, p90 {pct(w, .9):.2f} s, "
          f"max {max(w):.2f} s; over {slow_s:g} s: {sum(x > slow_s for x in w)}", file=out)
    slow = [r for r in replies if r["wait"] > slow_s]
    if not slow:
        return
    parts = Counter()
    for r in slow:
        start = r.get("start", 0.0)
        llm = (r.get("first_token") or start) - max(start, 0.0)
        audio = r["first_audio"] - (r.get("first_token") or r["first_audio"])
        biggest = max((("late start", max(start, 0.0)), ("llm", llm), ("audio", audio)), key=lambda x: x[1])[0]
        if biggest == "llm" and r.get("tools"):
            biggest = "llm + tools (" + ", ".join(sorted({t[0] for t in r["tools"]})) + ")"
        parts[biggest] += 1
    print("slow replies, by the biggest part of the wait:", file=out)
    for k, n in parts.most_common():
        print(f"  {n:4d}  {k}", file=out)
    print("slowest:", file=out)
    for r in sorted(slow, key=lambda r: -r["wait"])[:10]:
        print(f"  {r['wait']:5.1f} s  start {r.get('start', 0):+.1f}  token {r.get('first_token', float('nan')):5.1f}  "
              f"audio {r['first_audio']:5.1f}  {r.get('decision') or ''} {r.get('header_ms') or ''}ms  "
              f"tools={[t[0] for t in r.get('tools', [])]}  {r['text'][:50]!r}", file=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=str(PATH))
    ap.add_argument("--since", default=None)
    ap.add_argument("--slow", type=float, default=3.0)
    a = ap.parse_args()
    report(load(Path(a.file), since_s(a.since) if a.since else None), a.slow)


if __name__ == "__main__":
    main()
