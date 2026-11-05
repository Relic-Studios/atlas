"""Rate Live screen's tips and tune how picky they are (docs/designs/live_screen.md, section 7, G4).

Every tip Atlas prepared is in private/live_screen_tips.jsonl (spoken or not, with the reason it was dropped). This
asks you to rate the ones that reached the worth check: useful (u), annoying (a), neutral (n), skip (Enter), quit (q).
Ratings go to private/live_screen_tip_ratings.jsonl. It then suggests the lowest worth threshold at which at least
70% of the tips above it were useful (G4), to set as ATLAS_TIP_WORTH.

  python tools/rate_tips.py            # rate new tips, then show the suggestion
  python tools/rate_tips.py --report   # just the numbers
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "private" / "live_screen_tips.jsonl"
RATINGS = ROOT / "private" / "live_screen_tip_ratings.jsonl"


def rows(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def candidates():
    """Tips that got as far as the worth check (ready, spoken, or 'not worth it'), newest last."""
    seen, out = set(), []
    for r in rows(LOG):
        if r.get("tip") and r.get("worth") is not None and r.get("outcome") in ("ready", "not worth it"):
            key = (round(r["t"]), r["tip"])
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def suggest(rated):
    """Lowest threshold where >= 70% of rated tips at or above it were useful (and at least 5 of them)."""
    best = None
    for thr in [x / 20 for x in range(4, 20)]:
        above = [r for r in rated if r["worth"] >= thr]
        if len(above) < 5:
            continue
        useful = sum(1 for r in above if r["rating"] == "useful") / len(above)
        if useful >= 0.7:
            best = (thr, useful, len(above))
            break
    return best


def report():
    ratings = {(round(r["t"]), r["tip"]): r["rating"] for r in rows(RATINGS)}
    rated = [dict(c, rating=ratings[(round(c["t"]), c["tip"])]) for c in candidates()
             if (round(c["t"]), c["tip"]) in ratings]
    spoken = [r for r in rows(LOG) if r.get("outcome") == "spoken"]
    reactions = [r for r in rows(LOG) if r.get("outcome") == "reaction"]
    print(f"{len(spoken)} tips spoken, {len(rated)} rated "
          f"({sum(1 for r in rated if r['rating'] == 'useful')} useful, "
          f"{sum(1 for r in rated if r['rating'] == 'annoying')} annoying); reactions in the call: "
          f"{sum(1 for r in reactions if r['reaction'] == 'positive')} positive, "
          f"{sum(1 for r in reactions if r['reaction'] == 'negative')} negative")
    s = suggest(rated)
    print(f"suggested ATLAS_TIP_WORTH={s[0]} ({round(s[1] * 100)}% useful over {s[2]} tips)" if s else
          "not enough ratings yet for a suggestion (need 5 tips above a threshold)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        done = {(round(r["t"]), r["tip"]) for r in rows(RATINGS)}
        todo = [c for c in candidates() if (round(c["t"]), c["tip"]) not in done]
        print(f"{len(todo)} tips to rate")
        RATINGS.parent.mkdir(parents=True, exist_ok=True)
        with open(RATINGS, "a", encoding="utf-8") as out:
            for c in todo:
                print(f"\n[{c['game']}, {c['kind']}] {c['observation'][:120]}\n  tip: {c['tip']}  "
                      f"(worth {c['worth']:.2f}, {c['outcome']})")
                try:
                    ans = input("  useful / annoying / neutral / skip / quit [u/a/n/Enter/q]: ").strip().lower()
                except EOFError:
                    break
                if ans == "q":
                    break
                rating = {"u": "useful", "a": "annoying", "n": "neutral"}.get(ans[:1])
                if rating:
                    out.write(json.dumps({"t": c["t"], "tip": c["tip"], "rating": rating}) + "\n")
    report()


if __name__ == "__main__":
    sys.exit(main())
