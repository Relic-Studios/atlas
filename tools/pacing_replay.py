"""Replay recorded calls through the pacing gate, counterfactually.

For every recorded agent reply we find the human line it answered and ask the
CURRENT turn gate what it would do. Replies the gate now holds are removed from
the talk accounting, so later decisions see the quieter agent (the share drops
and the gate relaxes again) -- otherwise the replay would over-count holds.

Reports, per call: agent word share before/after, replies kept/held by reason,
and every held reply whose trigger line was a question or used "you" (so a
person can read them and judge whether the agent should have answered).

    python tools/pacing_replay.py recordings/2026-10-06/001420.jsonl [...]
    python tools/pacing_replay.py --last 3 --json tests/reports/pacing_replay.json
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.disable(logging.CRITICAL)

from floor import ConversationFloor, names_agent  # noqa: E402
from pacing import count_words  # noqa: E402


class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


def _profile(persona: str):
    try:
        from agent_runtime import build_profile
        return build_profile(persona, persona.capitalize(), None)
    except Exception:  # noqa: BLE001
        class P:
            talkativeness = 0.35
            names = (persona.capitalize(),)

            def interest_hits(self, _):
                return []
        return P()


def replay(path: str, all_held: bool = False) -> dict:
    ev = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ev = [e for e in ev if (e.get("k") or e.get("kind")) in ("user", "agent", "decision", "persona")]
    clock = _Clock()
    floor = ConversationFloor(clock=clock)
    # owner controls (quiet/mute/background) are live state, not part of a replay
    import owner_controls
    owner_controls.gate = lambda *a, **k: None
    owner_controls.background_on = lambda *a, **k: False
    persona = next((e.get("persona") for e in ev if e.get("persona")), "agent") or "agent"
    floor.profile = _profile(persona)
    names = tuple(getattr(floor.profile, "names", (persona.capitalize(),)))
    words_before = {"agent": 0, "human": 0}
    words_after = {"agent": 0, "human": 0}
    kept = held = 0
    reasons: dict = {}
    review = []
    named_held = []
    last_user = None
    gate_for_last = None
    target = None
    for e in ev:
        clock.t = float(e["t"])
        k = e.get("k") or e.get("kind")
        if k == "persona" and e.get("persona") and e["persona"] != persona:
            persona = e["persona"]
            floor.profile = _profile(persona)
            names = tuple(getattr(floor.profile, "names", (persona.capitalize(),)))
        elif k == "user":
            spk = e.get("spk") or "user"
            if spk == "self":
                continue
            text = e.get("text") or ""
            labeled = f"[{spk}] {text}"
            gate_for_last = floor.turn_gate(labeled, spk, names)
            last_user = (spk, text, labeled)
            floor.on_user_turn(spk)
            floor.pacing_turn(spk, text)
            w = count_words(text)
            words_before["human"] += w
            words_after["human"] += w
        elif k == "decision":
            target = e.get("target")
        elif k == "agent":
            text = e.get("text") or ""
            w = count_words(text)
            words_before["agent"] += w
            g = gate_for_last
            if g and g.startswith("pacing") and last_user:
                held += 1
                reasons[g.split(" (")[0]] = reasons.get(g.split(" (")[0], 0) + 1
                spk, utext, labeled = last_user
                if names_agent(labeled, names):
                    named_held.append(utext)
                if all_held or "?" in utext or re.search(r"\b(you|your)\b", utext, re.I):
                    review.append({"spk": spk, "line": utext[:140], "reply": text[:140], "gate": g})
                gate_for_last = None   # one reply per trigger line
                continue
            kept += 1
            words_after["agent"] += w
            floor.on_agent_spoke(target, text)
            gate_for_last = None

    def share(d):
        tot = d["agent"] + d["human"]
        return round(d["agent"] / tot, 3) if tot else 0.0

    return {"file": os.path.relpath(path, ROOT).replace(os.sep, "/"),
            "persona": persona, "replies": kept + held, "kept": kept, "held": held,
            "held_by": reasons, "agent_share_before": share(words_before),
            "agent_share_after": share(words_after),
            "named_held": named_held, "review": review}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--last", type=int, default=0, help="the N most recent recordings")
    ap.add_argument("--json", default="")
    ap.add_argument("--show", type=int, default=12, help="held question lines to print per call")
    ap.add_argument("--all-held", action="store_true", help="review every held reply, not only questions")
    ap.add_argument("--sample", type=int, default=0, help="print a random sample of N reviewed lines")
    a = ap.parse_args()
    files = list(a.files)
    if a.last:
        allf = sorted(glob.glob(str(ROOT / "recordings" / "*" / "*.jsonl")), key=os.path.getmtime)
        files += allf[-a.last:]
    out = []
    for f in files:
        r = replay(f, a.all_held)
        if a.sample:
            import random
            random.seed(7)
            r["review"] = random.sample(r["review"], min(a.sample, len(r["review"])))
        out.append(r)
        print(f"== {r['file']} ({r['persona']}): replies {r['replies']} -> kept {r['kept']}, "
              f"held {r['held']} {r['held_by']}; agent share {r['agent_share_before']:.0%} -> "
              f"{r['agent_share_after']:.0%}; NAMED HELD: {len(r['named_held'])}")
        for x in r["review"][: a.show]:
            print(f"   ? [{x['spk']}] {x['line']}\n       -> {x['reply']}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 1 if any(r["named_held"] for r in out) else 0


if __name__ == "__main__":
    sys.exit(main())
