"""Post-call report from a recordings/ JSONL file. CPU only, no model calls.

Usage:
  python tools/call_report.py                 # latest real call
  python tools/call_report.py recordings/2026-10-03/073948.jsonl
  python tools/call_report.py --all-today

Per persona it counts the things that kept coming up in live calls:
  restates      reply mostly re-says the last line heard (echo_reply.is_parrot)
  self_repeats  reply opens the same way as one of its last 6 (template loops)
  spoken_tags   an internal S-label (S12, S130) made it into a voiced reply
  butt_ins      spoke on a line that didn't name it and wasn't continuing a
                thread with that speaker (rough proxy; read the samples)
  missed        a line named the agent and it never answered within 8s
  latency       decision ms p50 / p95
  drops         what the speak-time screens removed, by reason
Writes a markdown report next to the recording (<name>.report.md).
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echo_reply import is_parrot  # noqa: E402

TAG = re.compile(r"\b[Ss]\d{1,4}\b")
WORD = re.compile(r"[a-z']+")
NAMES = {
    "max": ("max",), "ivy": ("ivy",), "kai": ("kai",),
    "pip": ("pip",), "grim": ("grim", "smeagol", "sméagol"),
    "pup": ("pup", "scoob"), "fae": ("fae", "navvy", "navy"),
}


def load(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def is_real_call(rows):
    users = sum(r.get("k") == "user" for r in rows)
    personas = sum(r.get("k") == "persona" for r in rows)
    return users >= 5 and personas <= max(3, users // 4)


def opener(text):
    w = WORD.findall(text.lower())
    return " ".join(w[:2]) if len(w) >= 2 else ""


def pct(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))]


def analyse(rows):
    persona = None
    per = collections.defaultdict(lambda: collections.Counter())
    lat = collections.defaultdict(list)
    samples = collections.defaultdict(lambda: collections.defaultdict(list))
    drops = collections.defaultdict(collections.Counter)
    recent_users = []          # (t, spk, text)
    openers = collections.defaultdict(list)
    pending_named = []         # (t, persona, text)
    partner = {}               # persona -> (spk, t) of last exchange
    decided_on = None
    for r in rows:
        k, t = r.get("k"), r.get("t", 0)
        if k == "persona":
            persona = r.get("persona")
            continue
        if not persona:
            continue
        if k == "user":
            text = r.get("text", "")
            recent_users.append((t, r.get("spk", "user"), text))
            recent_users = recent_users[-6:]
            per[persona]["user_lines"] += 1
            low = text.lower()
            if any(re.search(rf"\b{n}\b", low) for n in NAMES.get(persona, (persona,))):
                pending_named.append((t, persona, text))
        elif k == "decision":
            if r.get("ms") and r["ms"] > 1:
                lat[persona].append(r["ms"])
            per[persona]["decisions"] += 1
            if r.get("action") == "HOLD":
                per[persona]["holds"] += 1
            elif recent_users:
                # the reply that follows answers the line this decision was made on
                decided_on = recent_users[-1]
        elif k == "drop":
            drops[persona][r.get("why", "?")] += 1
        elif k == "agent":
            p = r.get("persona") or persona
            text = r.get("text", "").strip()
            if not text:
                continue
            c = per[p]
            c["replies"] += 1
            last = decided_on or (recent_users[-1] if recent_users else (0, "", ""))
            decided_on = None
            heard = " ".join(u[2] for u in recent_users[-2:])
            if TAG.search(text):
                c["spoken_tags"] += 1
                samples[p]["spoken_tags"].append(text)
            first = re.split(r"(?<=[.!?])\s+", text)[0]
            if heard and is_parrot(first, heard):
                c["restates"] += 1
                samples[p]["restates"].append(f"{last[2][:70]!r} -> {text[:90]!r}")
            op = opener(text)
            if op and openers[p][-6:].count(op) >= 2:
                c["self_repeats"] += 1
                samples[p]["self_repeats"].append(text[:100])
            openers[p].append(op)
            named = any(pp == p and abs(pt - last[0]) < 0.01 or (pp == p and 0 <= t - pt < 8)
                        for pt, pp, _ in pending_named)
            low = last[2].lower()
            named = named or any(re.search(rf"\b{n}\b", low) for n in NAMES.get(p, (p,)))
            spk = last[1]
            ptn = partner.get(p)
            continuing = ptn and ptn[0] == spk and t - ptn[1] < 45
            if not named and not continuing and last[0] and t - last[0] < 15:
                c["butt_ins"] += 1
                samples[p]["butt_ins"].append(f"{last[2][:70]!r} -> {text[:80]!r}")
            if named:
                pending_named = [x for x in pending_named if not (x[1] == p and t - x[0] < 8)]
            partner[p] = (spk, t)
        # expire named lines that never got an answer
        keep = []
        for pt, pp, txt in pending_named:
            if t - pt >= 8:
                per[pp]["missed"] += 1
                samples[pp]["missed"].append(txt[:90])
            else:
                keep.append((pt, pp, txt))
        pending_named = keep
    return per, lat, samples, drops


def render(path, per, lat, samples, drops):
    out = [f"# Call report: {path}", ""]
    out.append("| persona | lines heard | replies | reply rate | restates | self-repeats | spoken S-tags | unprompted* | missed | p50 ms | p95 ms |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for p, c in sorted(per.items(), key=lambda x: -x[1]["replies"]):
        if not c["replies"] and not c["user_lines"]:
            continue
        L = lat.get(p, [])
        rate = c['replies'] / c['user_lines'] if c['user_lines'] else 0
        out.append(f"| {p} | {c['user_lines']} | {c['replies']} | {rate:.0%} | {c['restates']} | {c['self_repeats']} | "
                   f"{c['spoken_tags']} | {c['butt_ins']} | {c['missed']} | "
                   f"{statistics.median(L) if L else 0:.0f} | {pct(L, .95):.0f} |")
    out.append("")
    out.append("*unprompted = replied to a line that did not name it and was not a continuing thread with that speaker. Includes fine replies to 'you' questions; read the samples.")
    for p in per:
        if drops.get(p):
            out.append(f"\n**{p} screen drops:** " + ", ".join(f"{k} {v}" for k, v in drops[p].most_common()))
        for kind in ("spoken_tags", "restates", "self_repeats", "missed", "butt_ins"):
            xs = samples[p].get(kind)
            if xs:
                out.append(f"\n**{p} {kind}** ({len(xs)}):")
                out += [f"- {x}" for x in xs[:6]]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all-today", action="store_true")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "recordings" / "*" / "*.jsonl")), key=os.path.getmtime)
    if a.path:
        files = [a.path]
    elif a.all_today:
        day = max(Path(f).parent.name for f in files) if files else ""
        files = [f for f in files if Path(f).parent.name == day]
    else:
        files = [f for f in reversed(files) if is_real_call(load(f))][:1]
    for f in files:
        rows = load(f)
        if not is_real_call(rows):
            continue
        rep = render(f, *analyse(rows))
        Path(f).with_suffix(".report.md").write_text(rep, encoding="utf-8")
        print(rep)


if __name__ == "__main__":
    main()
