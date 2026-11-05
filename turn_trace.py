"""Per-turn timing trace (owner 11-03: "it's lagging ... sometimes it freezes"; the server log couldn't tell a reply
waiting for a gap from a stuck request). One JSON line per reply attempt in logs/turns.jsonl:

  gen, text, cue          the generation and what started it (a cue is the agent's own turn)
  end                     wall time the user's turn ended (absent for cues)
  start, llm, first_token, first_audio, llm_done
                          seconds after `end` (or after `start` for cues) for each step
  wait                    end -> first audio: what the room hears as the reply's delay
  decision, header_ms     the speak/hold decision and how long it took (remote_llm.TURN_LOG)
  tools                   [[name, seconds after start], ...]
  outcome                 spoke | hold | aborted | error, abort: the abort reason

Never raises into the pipeline. ATLAS_TURN_TRACE=0 turns it off. tools/turn_report.py summarises the file."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PATH = Path(os.environ.get("ATLAS_TURN_TRACE_PATH", Path(__file__).resolve().parent / "logs" / "turns.jsonl"))
ENABLED = os.environ.get("ATLAS_TURN_TRACE", "1") != "0"
END_LINK_S = 20.0     # a generation prepared this soon after a turn end answers that turn
SETTLE_S = 15.0       # written once the LLM is done for this long (its first audio comes after the LLM)
MAX_AGE_S = 180.0     # written regardless after this long

_lock = threading.Lock()
_recs: dict = {}
_last_end = 0.0


def _now() -> float:
    return time.time()


def turn_end(gen_id=None) -> None:
    """The user's turn ended (server on_before_final). A running hot-started generation answers it."""
    global _last_end
    if not ENABLED:
        return
    try:
        now = _now()
        with _lock:
            _last_end = now
            rec = _recs.get(gen_id)
            # A generation still running when a turn ends was started on this turn's partial line (a hot start),
            # even if prepared() linked it to the previous turn's end (live 11-04: "[S6] What?" prepared 0.4 s before
            # its own turn ended was logged as a 12 s wait from the turn before).
            if rec is not None and not rec.get("cue") and ("end" not in rec or rec["start"] > rec["end"]):
                rec["end"] = now
    except Exception:  # noqa: BLE001
        pass


def prepared(gen_id, text: str) -> None:
    if not ENABLED:
        return
    try:
        now = _now()
        t = text or ""
        cue = " CUE" in t[:60] or "(GLANCE CUE)" in t or t.lstrip().startswith("(")
        rec = {"gen": gen_id, "text": t[:80], "cue": cue, "start": now, "tools": []}
        with _lock:
            if not cue and _last_end and now - _last_end <= END_LINK_S:
                rec["end"] = _last_end
            _recs[gen_id] = rec
        _flush(now)
    except Exception:  # noqa: BLE001
        pass


def mark(gen_id, step: str, **data) -> None:
    """First time this step happened for the generation (later calls only add data)."""
    if not ENABLED:
        return
    try:
        now = _now()
        with _lock:
            rec = _recs.get(gen_id)
            if rec is None:
                return
            rec.setdefault(step, now)
            rec.update({k: v for k, v in data.items() if v is not None})
    except Exception:  # noqa: BLE001
        pass


def tool(gen_id, name: str) -> None:
    if not ENABLED:
        return
    try:
        with _lock:
            rec = _recs.get(gen_id)
            if rec is not None:
                rec["tools"].append([str(name), _now()])
    except Exception:  # noqa: BLE001
        pass


def done(gen_id, outcome: str, **data) -> None:
    """The LLM side of the generation finished (spoke text, hold, aborted, error)."""
    if not ENABLED:
        return
    try:
        with _lock:
            rec = _recs.get(gen_id)
            if rec is None:
                return
            rec.setdefault("llm_done", _now())
            rec.setdefault("outcome", outcome)
            rec.update({k: v for k, v in data.items() if v is not None})
    except Exception:  # noqa: BLE001
        pass


def last_decision() -> dict:
    """The latest speak/hold decision and its time (remote_llm.TURN_LOG), as mark()/done() keywords."""
    try:
        import remote_llm as _rl
        e = _rl.TURN_LOG[-1] if _rl.TURN_LOG else None
        return {"decision": e.get("decision"), "header_ms": e.get("ms")} if e else {}
    except Exception:  # noqa: BLE001
        return {}


def row(rec: dict) -> dict:
    """The JSON line for a record: step times as seconds after the turn end (or the start, for cues)."""
    base = rec.get("end") or rec["start"]
    out = {"t": round(base, 2), "gen": rec["gen"], "text": rec["text"], "cue": rec["cue"]}
    for step in ("start", "llm", "first_token", "first_audio", "llm_done"):
        if rec.get(step) is not None:
            out[step] = round(rec[step] - base, 3)
    if rec.get("end") and rec.get("first_audio"):
        out["wait"] = round(rec["first_audio"] - rec["end"], 3)
    out["tools"] = [[n, round(t - rec["start"], 3)] for n, t in rec.get("tools", [])]
    for k in ("decision", "header_ms", "outcome", "abort", "error"):
        if rec.get(k) is not None:
            out[k] = rec[k]
    return out


def _flush(now: float, everything: bool = False) -> None:
    with _lock:
        ready = [g for g, r in _recs.items()
                 if everything or now - r["start"] > MAX_AGE_S
                 or (r.get("llm_done") and now - r["llm_done"] > SETTLE_S)]
        rows = [row(_recs.pop(g)) for g in ready]
    if not rows:
        return
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PATH, "a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    except OSError as e:
        logger.debug("turn trace write failed: %s", e)


def flush_all() -> None:
    """Write everything still open (shutdown)."""
    if ENABLED:
        try:
            _flush(_now(), everything=True)
        except Exception:  # noqa: BLE001
            pass
