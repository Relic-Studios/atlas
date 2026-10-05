"""The four-step loop of using ATLAS, as shown in the dashboard's guide strip.

    Set up  ->  Make an agent  ->  Start a conversation  ->  Review

Pure reads of state that already exists (settings, agent registry, recordings,
memory candidates); no model calls, safe to poll. GET /api/journey (owner only)
returns the steps, which one is current and a one-line hint for it.

Review is the step that makes agents better over time (approve memories, read
the call report, clear memory), so after the first full loop the strip keeps
pointing back at it whenever there is something new to review.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REC_DIR = ROOT / "recordings"
STEPS = ("setup", "agent", "call", "review")
LABELS = {"setup": "Set up", "agent": "Make an agent", "call": "Start a conversation", "review": "Review"}


def _real_calls(rec_dir: Path = REC_DIR, limit: int = 400) -> tuple[int, float]:
    """(number of recorded calls with at least one human line, mtime of the latest)."""
    n, latest = 0, 0.0
    files = sorted(rec_dir.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i > 200:
                        break
                    if '"user"' in line and json.loads(line).get("k") == "user":
                        n += 1
                        latest = max(latest, f.stat().st_mtime)
                        break
        except (OSError, ValueError):
            continue
    return n, latest


def _user_agents() -> list[str]:
    """Agents the user can actually talk to (not the setup placeholder)."""
    try:
        import agent_registry as R
        return [a for a in R.agent_ids() if not R.agent(a).get("placeholder")]
    except Exception:  # noqa: BLE001
        return []


def _pending_candidates(agents: list[str]) -> int:
    try:
        import call_memory as CM
        return sum(1 for a in agents for i in CM.load(a) if i.get("status") == "candidate")
    except Exception:  # noqa: BLE001
        return 0


def state(settings: dict | None = None, rec_dir: Path = REC_DIR) -> dict:
    import user_settings as US
    s = US.load() if settings is None else settings
    public = US.is_public()
    agents = _user_agents()
    calls, last_call = _real_calls(rec_dir)
    reviewed_at = float(s.get("reviewed_at", 0) or 0)
    pending = _pending_candidates(agents)
    unreviewed = last_call > reviewed_at or pending > 0

    done = {
        "setup": not (public and not s.get("setup_complete")),
        "agent": bool(agents),
        "call": calls > 0,
        "review": calls > 0 and not unreviewed,
    }
    current = next((k for k in STEPS if not done[k]), "call")
    hints = {
        "setup": "Pick a model (local or cloud), your audio devices and a voice.",
        "agent": "Click + in the Agent dock: give it a voice and describe who it is.",
        "call": "Bring it into a group conversation (a voice call or the room you're in) and say its name. It answers when it's spoken to.",
        "review": (f"{pending} memory suggestion(s) waiting. Right-click an agent to review or clear its memory."
                   if pending else "Your last conversation is ready to look over: what it said and what it remembered."),
    }
    if all(done.values()):
        hints["call"] = "All set. Every conversation teaches your agents a little more."
    return {
        "steps": [{"id": k, "label": LABELS[k], "done": done[k]} for k in STEPS],
        "current": current,
        "hint": hints[current],
        "calls": calls,
        "pending_memories": pending,
        "hidden": bool(s.get("journey_hidden")),
        # Owner 10-04: the card is first-run help only. Once set up, with an agent
        # and a first conversation, it never shows again: no progress recap, no
        # call count, no review nag.
        "show": not s.get("journey_hidden") and not (done["setup"] and done["agent"] and done["call"]),
        "public": public,
    }


def mark_reviewed(now: float | None = None) -> None:
    import time
    import user_settings as US
    US.update({"reviewed_at": float(now if now is not None else time.time())})


def set_hidden(hidden: bool) -> None:
    import user_settings as US
    US.update({"journey_hidden": bool(hidden)})
