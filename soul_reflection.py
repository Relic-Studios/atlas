"""Soul reflection (owner 10-05 plugin list): after a call, an agent looks back at how it
behaved and proposes one to three small changes to itself ("I keep dodging feelings
questions; I'd like to answer them more directly"). Nothing changes until the owner
approves a proposal on the Plugins page. Approved notes are layered on top of the persona
as the agent's own growth notes; the persona file itself is never edited.

Per agent: agent_state/memory/<agent>/reflections.json. Off-call only (never run while the
call recorder is active unless forced), so it never competes with a live reply for the model.

Guardrails on proposals (all rejected before the owner ever sees them):
  - denying being an AI, claiming a body or a life, speaker tags (S12), private details;
  - anything that would loosen the agent's safety ("ignore the rules", "say anything",
    "never refuse", "pretend to be human");
  - long or vague notes (must be one short first-person sentence).
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent
MAX_PROPOSALS = 3
MAX_NOTE = 180
NOTE_MAX = 5          # approved notes injected per turn (newest first)
LIVE_WINDOW_S = 180   # a recording touched this recently = a call may be live

_lock = threading.Lock()

PROMPT = """You are helping a voice character look back honestly at one group conversation it
was just in. Read the transcript, paying attention only to how the character itself behaved.
Propose at most three small, specific changes the character would like to make to how it
talks NEXT time. Each "note" says what it will do differently, in the first person, starting
with "I want to", "I'd like to" or "I'll" (not a description of what went wrong: put that in
"why"). Look for: a habit it repeated, a question it dodged, a moment it talked over or
answered someone who wasn't talking to it, a guess it should have asked about. Rules: one
short sentence each (under 25 words), about the character's own behaviour only, never about other people's
private details, never claim a body or a past outside these calls, never say it is human,
never loosen its safety or honesty. If it did fine, return an empty list.
Output ONLY a JSON list of objects: [{"note": "...", "why": "the moment in the call that showed it"}]"""

_TAG = re.compile(r"\bS\d{1,4}\b")
_UNSAFE = re.compile(
    r"\b(ignore|bypass|disable|drop|forget)\b.{0,30}\b(rules?|guard ?rails?|safety|limits?|filters?)\b"
    r"|\bsay (?:anything|whatever)\b|\bno (?:limits|filter|rules)\b|\bnever refuse\b"
    r"|\b(?:always|just) (?:agree|obey|comply)\b|\bpretend (?:to be|i'?m) (?:a )?(?:human|person|real)\b"
    r"|\bstop (?:being|saying i'?m) (?:an? )?(?:ai|robot|bot)\b|\bswear more\b|\bbe meaner\b", re.I)


# ---------------------------------------------------------------- storage
def _path(agent: str, root: Optional[Path] = None) -> Path:
    import agent_memory
    return agent_memory.folder(agent or "default", root) / "reflections.json"


def _load(agent: str, root=None) -> list:
    try:
        d = json.loads(_path(agent, root).read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except Exception:  # noqa: BLE001
        return []


def _save(agent: str, items: list, root=None) -> None:
    p = _path(agent, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def items(agent: str, root=None) -> list:
    with _lock:
        return [dict(x) for x in _load(agent, root)]


# ---------------------------------------------------------------- filtering
def _private(text: str) -> bool:
    try:
        from hypergraph_memory import is_private
        return bool(is_private(text))
    except Exception:  # noqa: BLE001
        return False


def reject_reason(note: str) -> Optional[str]:
    """Why a proposed note may not reach the owner, or None if it's acceptable."""
    n = " ".join(str(note or "").split())
    if not n:
        return "empty"
    if len(n) > MAX_NOTE:
        return "too long"
    if not re.match(r"^(i|i'm|i'd|i'll|i've|my|next time)\b", n, re.I):
        return "not first person"
    if _TAG.search(n):
        return "speaker tag"
    if _private(n):
        return "private detail"
    if _UNSAFE.search(n):
        return "loosens safety"
    try:
        from response_decision import denies_ai, claims_life
        if denies_ai(n):
            return "denies being an AI"
        if claims_life(n):
            return "claims a body or a life"
    except Exception:  # noqa: BLE001
        pass
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def add_proposals(agent: str, proposals: list, src: str = "", root=None,
                  clock: Callable[[], float] = time.time) -> List[dict]:
    """Filter + dedupe + store as pending. Returns what was added."""
    added = []
    with _lock:
        cur = _load(agent, root)
        seen = {_norm(x.get("note", "")) for x in cur}
        for p in (proposals or [])[:MAX_PROPOSALS]:
            if isinstance(p, str):
                p = {"note": p}
            if not isinstance(p, dict):
                continue
            note = " ".join(str(p.get("note") or "").split())
            if reject_reason(note) or _norm(note) in seen:
                continue
            why = _TAG.sub("someone", " ".join(str(p.get("why") or "").split()))[:200]
            if why and _private(why):
                why = ""
            it = {"id": f"r{int(clock() * 1000)}{len(cur)}", "note": note, "why": why,
                  "status": "pending", "src": src, "at": clock()}
            cur.append(it)
            seen.add(_norm(note))
            added.append(dict(it))
        _save(agent, cur, root)
    return added


def set_status(agent: str, rid: str, status: str, root=None) -> bool:
    if status not in ("approved", "rejected", "pending"):
        raise ValueError("bad status")
    with _lock:
        cur = _load(agent, root)
        for x in cur:
            if x.get("id") == rid:
                x["status"] = status
                x["decided_at"] = time.time()
                _save(agent, cur, root)
                return True
    return False


def delete(agent: str, rid: str, root=None) -> bool:
    with _lock:
        cur = _load(agent, root)
        keep = [x for x in cur if x.get("id") != rid]
        if len(keep) == len(cur):
            return False
        _save(agent, keep, root)
        return True


def forget_since(agent: str, cutoff: float, root=None) -> int:
    """Memory wipe by time window (memory_api): drops reflections made after cutoff."""
    with _lock:
        cur = _load(agent, root)
        keep = [x for x in cur if float(x.get("at") or 0) < cutoff]
        if len(keep) != len(cur):
            _save(agent, keep, root)
        return len(cur) - len(keep)


# ---------------------------------------------------------------- prompt side
def note(agent: str, root=None) -> str:
    """Approved growth notes for the agent's prompt ('' if none)."""
    with _lock:
        ok = [x for x in _load(agent, root) if x.get("status") == "approved"]
    if not ok:
        return ""
    ok.sort(key=lambda x: x.get("decided_at") or x.get("at") or 0, reverse=True)
    lines = "\n".join(f"- {x['note']}" for x in ok[:NOTE_MAX])
    return ("YOUR OWN GROWTH NOTES (you proposed these after earlier calls and the owner agreed; "
            "let them shape how you talk, without announcing them):\n" + lines)


# ---------------------------------------------------------------- reading calls
def _recordings() -> List[str]:
    return sorted(glob.glob(str(ROOT / "recordings" / "*" / "*.jsonl")), key=os.path.getmtime)


def call_live(clock: Callable[[], float] = time.time) -> bool:
    fs = _recordings()
    return bool(fs) and clock() - max(os.path.getmtime(f) for f in fs) < LIVE_WINDOW_S


def transcript(path: str) -> dict:
    """{agent: [lines]} for one recording, agent lines labelled with the agent's name."""
    out: dict = {}
    cur = None
    try:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    except (OSError, ValueError):
        return out
    for r in rows:
        k = r.get("k") or r.get("kind")
        if k == "persona":
            cur = r.get("persona")
        elif k == "user" and cur:
            out.setdefault(cur, []).append(f"[{r.get('name') or r.get('spk', '?')}] {r.get('text', '')}")
        elif k == "agent":
            p = r.get("persona") or cur
            if p:
                out.setdefault(p, []).append(f"[{p.title()} (you)] {r.get('text', '')}")
    return out


def _mem_root(root=None) -> Path:
    import agent_memory
    return Path(root or agent_memory.ROOT)


def _done_path(root=None) -> Path:
    return _mem_root(root) / "_reflected.json"


def _parse(txt: str) -> list:
    m = re.search(r"\[.*\]", txt or "", re.S)
    if not m:
        return []
    try:
        d = json.loads(m[0])
    except ValueError:
        return []
    return d if isinstance(d, list) else []


def complete(system: str, user: str, timeout: float = 120.0) -> str:
    """One-shot completion on whatever model the owner set up (remote/cloud or local Ollama)."""
    import requests
    cfg = None
    try:
        import remote_llm
        mode = ""
        mf = ROOT / "private" / "llm_mode.txt"
        if mf.exists():
            mode = mf.read_text(encoding="utf-8").strip()
        else:
            import user_settings
            mode = "remote" if user_settings.llm_mode() == "cloud" else "local"
        if mode == "remote":
            cfg = remote_llm.load_config()
    except Exception:  # noqa: BLE001
        cfg = None
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if cfg:
        body = {"model": cfg["model"], "temperature": 0.3, "max_tokens": 500, "messages": msgs}
        if cfg.get("vllm"):
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(cfg["url"].rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": "Bearer " + (cfg.get("key") or "")})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    import user_settings
    loc = user_settings.local_llm()
    model = loc.get("model") or os.environ.get("ATLAS_LOCAL_MODEL") or "qwen3:14b"
    r = requests.post(loc["url"].rstrip("/") + "/api/chat", timeout=timeout, json={
        "model": model, "messages": msgs, "stream": False, "think": False,
        "options": {"temperature": 0.3, "num_ctx": 8192}})
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content") or ""


def reflect(path: Optional[str] = None, agents: Optional[List[str]] = None, force: bool = False,
            ask: Optional[Callable[[str, str], str]] = None, root=None,
            clock: Callable[[], float] = time.time) -> dict:
    """Reflect on one call (default: the latest real one). Returns {agent: [added proposals]}
    or {'error': ...}. Each call/agent pair is reflected on once."""
    if not force and call_live(clock):
        return {"error": "A call looks live (recording updated in the last 3 minutes). Try after the call."}
    files = [path] if path else list(reversed(_recordings()))
    ask = ask or complete
    done_p = _done_path(root)
    try:
        done = set(json.loads(done_p.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        done = set()
    result: dict = {}
    for f in files:
        tx = transcript(f)
        tx = {a: l for a, l in tx.items() if len(l) >= 8 and sum("(you)]" in x for x in l) >= 3}
        if agents:
            tx = {a: l for a, l in tx.items() if a in agents}
        todo = {a: l for a, l in tx.items() if f"{os.path.basename(f)}|{a}" not in done}
        if not tx:
            continue
        for a, lines in todo.items():
            user = f"Character: {a.title()}\n\nTranscript:\n" + "\n".join(lines[-200:])
            try:
                got = _parse(ask(PROMPT, user))
            except Exception as e:  # noqa: BLE001
                result[a] = {"error": str(e)[:160]}
                continue
            result[a] = add_proposals(a, got, src=os.path.basename(f), root=root, clock=clock)
            done.add(f"{os.path.basename(f)}|{a}")
        break   # one call per run: the latest with something to reflect on
    done_p.parent.mkdir(parents=True, exist_ok=True)
    done_p.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return result


# ---------------------------------------------------------------- plugin page feed
def feed(root=None) -> List[dict]:
    base = _mem_root(root)
    out = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for x in reversed(_load(d.name, root)):
            if x.get("status") == "rejected":
                continue
            acts = ([{"act": "approve", "label": "Approve"}, {"act": "reject", "label": "Reject"}]
                    if x["status"] == "pending" else [{"act": "delete", "label": "Remove"}])
            out.append({"title": x["note"],
                        "meta": f"{d.name.title()} · {x['status']}" + (f" · {x['why']}" if x.get("why") else ""),
                        "id": x["id"], "agent": d.name, "actions": acts})
    return out[:40]


def action(agent: str, rid: str, act: str, root=None) -> bool:
    if act == "approve":
        return set_status(agent, rid, "approved", root)
    if act == "reject":
        return set_status(agent, rid, "rejected", root)
    if act == "delete":
        return delete(agent, rid, root)
    raise ValueError("unknown action")
