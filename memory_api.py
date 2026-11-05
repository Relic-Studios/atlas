"""Owner-only memory viewer API (docs/designs/experience_memory.md, D7): see, confirm and delete what the agents
remember, and run the experience memory.

  GET  /api/memory                      agents + counts
  GET  /api/memory/{aid}                one agent: episodes, lore entries and lessons waiting for review
  POST /api/memory/{aid}/episodes       {"id", "action": "delete"}
  POST /api/memory/{aid}/review         {"kind": "lore"|"lesson", "id", "action": "approve"|"delete"}
  POST /api/memory/{aid}/forget_all     {"what": "everything"}: same as wipe "all"
  POST /api/memory/{aid}/wipe           {"window": "1h"|"12h"|"1d"|"1w"|"all"}  this agent only
  POST /api/memory/{aid}/open           open the folder in the file manager
  GET  /api/memory_state                owner facts, switches, the evaluation report, waiting and failed calls, tagger
  POST /api/memory_state/settings       {"key": "use_memories"|"paused"|"cloud_allowed", "value": bool}
  POST /api/memory_state/facts          {"id", "action": "confirm"|"delete"}  or {"action": "clear"}
  POST /api/memory_state/allow          {"names": [...]}  game and product names redaction keeps
  POST /api/memory_state/process        run the post-call pass now (refused while a call is in progress)
  GET  /api/memory_state/log            the pass log (no line text)
  POST /api/memory_state/tagger         download the name tagger
  POST /api/memory_state/old_data       delete the retired memories' files

Only reachable from this computer (same rule as the other owner controls).
"""
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import agent_memory as AM

logger = logging.getLogger(__name__)
router = APIRouter()
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _owner(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in _LOOPBACK or host.startswith("127.")


def _deny():
    return JSONResponse({"error": "memory can only be viewed or changed from this computer"}, status_code=403)


def _known_agents() -> dict:
    out = {}
    try:
        import agent_registry as R
        for a in R.agent_ids():
            out[a] = R.display_name(a)
    except Exception as e:  # noqa: BLE001
        logger.debug("registry unavailable: %s", e)
    for a in AM.agents_on_disk():
        out.setdefault(a, a.title())
    return out


def _valid(aid: str) -> bool:
    try:
        AM.folder(aid)
        return True
    except ValueError:
        return False


@router.get("/api/memory")
async def memory_index(request: Request):
    if not _owner(request):
        return _deny()

    def work():
        import experience_memory as EM
        rows = [{"id": aid, "name": name, "folder": str(AM.folder(aid)), "episodes": len(EM.episodes(aid))}
                for aid, name in _known_agents().items()]
        return {"agents": rows}
    return await asyncio.to_thread(work)


def _review_items(aid: str) -> list:
    out = []
    try:
        import lore
        out += [{"kind": "lore", "id": str(i.get("id")), "text": f"{i.get('title', '')}: {i.get('story', '')}"}
                for i in lore._load(aid) if i.get("status") == "review"]
    except Exception as e:  # noqa: BLE001
        logger.debug("lore review list failed: %s", e)
    try:
        import teach
        out += [{"kind": "lesson", "id": str(i.get("id")), "text": f"{i.get('topic', '')}: {i.get('content', '')}"}
                for i in teach._load(aid) if i.get("status") == "review"]
    except Exception as e:  # noqa: BLE001
        logger.debug("lesson review list failed: %s", e)
    return out


@router.get("/api/memory/{aid}")
async def memory_agent(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)

    def work():
        import experience_memory as EM
        eps = [{"id": e.get("id"), "date": e.get("date"), "summary": " ".join(e.get("summary") or []),
                "items": len(EM.episode_items(e))} for e in EM.episodes(aid)]
        return {"id": aid, "folder": str(AM.folder(aid)), "episodes": eps, "review": _review_items(aid)}
    return await asyncio.to_thread(work)


@router.post("/api/memory/{aid}/episodes")
async def memory_episode(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    body = await request.json()
    if body.get("action") != "delete":
        return JSONResponse({"error": "action must be delete"}, status_code=400)
    import experience_memory as EM
    return {"ok": await asyncio.to_thread(EM.delete_episode, aid, str(body.get("id", "")))}


@router.post("/api/memory/{aid}/review")
async def memory_review(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    body = await request.json()
    kind, iid, act = body.get("kind"), str(body.get("id", "")), body.get("action")
    if kind not in ("lore", "lesson") or act not in ("approve", "delete"):
        return JSONResponse({"error": "kind must be lore or lesson; action approve or delete"}, status_code=400)

    def work():
        mod = __import__("lore" if kind == "lore" else "teach")
        with mod._lock:
            items = mod._load(aid)
            for i, it in enumerate(items):
                if str(it.get("id")) == iid and it.get("status") == "review":
                    if act == "delete":
                        items.pop(i)
                    else:
                        it["status"] = "approved" if kind == "lore" else "ok"
                        if kind == "lesson":
                            it.pop("status", None)
                    mod._save(aid, items)
                    return True
        return False
    return {"ok": await asyncio.to_thread(work)}


@router.post("/api/memory/{aid}/forget_all")
async def memory_forget_all(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    body = await request.json()
    if body.get("what", "everything") != "everything":
        return JSONResponse({"error": "what must be everything"}, status_code=400)
    return await asyncio.to_thread(wipe, aid, "all")


WINDOWS = {"1h": 3600, "12h": 12 * 3600, "1d": 86400, "1w": 7 * 86400, "all": 0}


def wipe(aid: str, window: str, svc=None, root=None, now: float = None) -> dict:
    """Clear what ONE agent remembers from the last `window` (or everything): its lessons, lore and reflection
    proposals. Other agents are never touched. (`svc` is accepted for old callers and ignored.)"""
    import time as _t
    secs = WINDOWS[window]
    cutoff = 0.0 if secs == 0 else (now or _t.time()) - secs
    try:
        import teach
        lessons = teach.forget_since(aid, cutoff, root)
    except Exception as e:  # noqa: BLE001 - lessons are optional
        logger.warning("lesson wipe failed: %s", e)
        lessons = 0
    try:
        import lore
        lore.forget_since(aid, cutoff, root)
    except Exception as e:  # noqa: BLE001
        logger.warning("lore wipe failed: %s", e)
    try:
        import soul_reflection
        soul_reflection.forget_since(aid, cutoff, root)
    except Exception as e:  # noqa: BLE001
        logger.warning("reflection wipe failed: %s", e)
    episodes = 0
    if root is None:
        try:
            import experience_memory
            episodes = experience_memory.wipe_episodes(aid, cutoff, now)
        except Exception as e:  # noqa: BLE001
            logger.warning("episode wipe failed: %s", e)
    logger.info("memory wipe %s window=%s: %d episodes, %d lessons", aid, window, episodes, lessons)
    return {"ok": True, "agent": aid, "window": window, "items": episodes, "episodes": episodes, "learned": 0,
            "lessons": lessons}


@router.post("/api/memory/{aid}/wipe")
async def memory_wipe(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid) or aid not in _known_agents():
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    body = await request.json()
    window = str(body.get("window", ""))
    if window not in WINDOWS:
        return JSONResponse({"error": "window must be one of " + ", ".join(WINDOWS)}, status_code=400)
    return await asyncio.to_thread(wipe, aid, window)


@router.post("/api/memory/{aid}/open")
async def memory_open(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    d = AM.ensure(aid)
    try:
        if sys.platform == "win32":
            os.startfile(str(d))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(d)])
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e), "folder": str(d)}, status_code=500)
    return {"ok": True, "folder": str(d)}


# ---------------------------------------------------------------- guide strip (journey.py)
@router.get("/api/journey")
async def journey_get(request: Request):
    if not _owner(request):
        return _deny()
    import journey
    return await asyncio.to_thread(journey.state)


@router.post("/api/journey")
async def journey_post(request: Request):
    if not _owner(request):
        return _deny()
    import journey
    body = await request.json()
    act = body.get("action")
    if act == "reviewed":
        journey.mark_reviewed()
    elif act in ("hide", "show"):
        journey.set_hidden(act == "hide")
    else:
        return JSONResponse({"error": "action must be reviewed, hide or show"}, status_code=400)
    return await asyncio.to_thread(journey.state)


# ---------------------------------------------------------------- the shared memory: facts, switches, the pass (D7)
_TAGGER = {"downloading": False, "error": None}
SHIPPED_REPORT = Path(__file__).resolve().parent / "memory_eval_report.json"   # public builds (tools/export_public.py)


def _report() -> dict:
    """The latest D9 report, local first, else the one shipped with a public build; valid only for this code."""
    import experience_memory as EM
    local = EM.root() / "_eval_report.json"
    shipped = SHIPPED_REPORT
    path = local if local.exists() else shipped
    d = EM.read_json(path, None)
    if not isinstance(d, dict):
        return {"exists": False, "passed": False}
    code_ok = d.get("code") == EM.code_hash()
    return {"exists": True, "passed": bool(d.get("all_zero")) and code_ok, "code_ok": code_ok,
            "date": d.get("date"), "shipped": path == shipped, "gates": d.get("gates", {}),
            "tracked": d.get("tracked", {})}


def _calls_waiting() -> tuple:
    import call_boundary as CB
    import memory_writer as MW
    import time as _t
    waiting, failed = [], []
    d = CB.pending_dir()
    cur = (CB.read_current() or {}).get("call")
    for p in sorted(d.glob("*.jsonl")) if d.exists() else []:
        if p.stem == cur:
            continue
        age_h = round((_t.time() - p.stat().st_mtime) / 3600, 1)
        (failed if MW._fails(p) >= MW.MAX_FAILS else waiting).append({"call": p.stem, "age_h": age_h})
    return waiting, failed


@router.get("/api/memory_state")
async def memory_state(request: Request):
    if not _owner(request):
        return _deny()

    def work():
        import call_boundary as CB
        import experience_memory as EM
        import memory_redact as MR
        waiting, failed = _calls_waiting()
        runner = getattr(request.app.state, "MemoryRunner", None)
        facts = [{k: f.get(k) for k in ("id", "date", "source", "confirmed", "fact")} for f in EM.owner_facts()]
        return {"settings": EM.settings(), "report": _report(), "facts": facts, "waiting": waiting,
                "failed": failed, "call_in_progress": CB.in_progress(),
                "tagger": {"present": MR.tagger_files_present(), **_TAGGER},
                "allow": _allow(), "runner": {"running": bool(runner and runner.running()),
                                              "reason": getattr(runner, "last_reason", None)},
                "old_data": len(_old_data_files())}
    return await asyncio.to_thread(work)


@router.post("/api/memory_state/settings")
async def memory_settings(request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    key, value = body.get("key"), bool(body.get("value"))
    if key not in ("use_memories", "paused", "cloud_allowed"):
        return JSONResponse({"error": "key must be use_memories, paused or cloud_allowed"}, status_code=400)
    if key == "use_memories" and value and not (await asyncio.to_thread(_report))["passed"]:
        return JSONResponse({"error": "Use memories waits for an evaluation report with every gate at zero"},
                            status_code=409)
    import experience_memory as EM
    return {"settings": await asyncio.to_thread(EM.set_setting, key, value)}


@router.post("/api/memory_state/facts")
async def memory_facts(request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    act, fid = body.get("action"), str(body.get("id", ""))
    import experience_memory as EM
    if act == "confirm":
        return {"ok": await asyncio.to_thread(EM.confirm_fact, fid)}
    if act == "delete":
        return {"ok": await asyncio.to_thread(EM.delete_fact, fid)}
    if act == "clear":
        return {"ok": True, "cleared": await asyncio.to_thread(EM.clear_owner_facts)}
    return JSONResponse({"error": "action must be confirm, delete or clear"}, status_code=400)


def _allow() -> list:
    import experience_memory as EM
    try:
        return [w.strip() for w in (EM.root() / "_allow_names.txt").read_text(encoding="utf-8").splitlines()
                if w.strip()]
    except OSError:
        return []


@router.post("/api/memory_state/allow")
async def memory_allow(request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    names = body.get("names")
    if not isinstance(names, list):
        return JSONResponse({"error": "names must be a list"}, status_code=400)
    clean = []
    for n in names:
        n = " ".join(str(n or "").split())[:40]
        if n and n not in clean:
            clean.append(n)
    import experience_memory as EM

    def work():
        with EM.store_lock():
            p = EM.root() / "_allow_names.txt"
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_text("\n".join(clean[:500]) + "\n", encoding="utf-8")
            os.replace(tmp, p)
        return clean[:500]
    return {"allow": await asyncio.to_thread(work)}


@router.post("/api/memory_state/process")
async def memory_process(request: Request):
    if not _owner(request):
        return _deny()
    import call_boundary as CB
    if CB.in_progress():
        return JSONResponse({"error": "A call is in progress; memories are made after it."}, status_code=409)
    runner = getattr(request.app.state, "MemoryRunner", None)
    if runner is None:
        return JSONResponse({"error": "the memory pass isn't available"}, status_code=409)
    res = await asyncio.to_thread(runner.start_now)
    if res in ("started", "running"):
        return {"ok": True, "status": res}
    return JSONResponse({"error": {"paused": "Memory is paused.", "cloud_not_allowed":
                                   "Memory processing on the cloud provider is switched off.",
                                   "tagger_missing": "The name tagger isn't installed yet.",
                                   "nothing": "No calls are waiting."}.get(res, str(res))}, status_code=409)


@router.get("/api/memory_state/log")
async def memory_log(request: Request):
    if not _owner(request):
        return _deny()
    import experience_memory as EM
    p = EM.root() / "_pass_log.txt"
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-200:]
    except OSError:
        lines = []
    return {"lines": lines}


@router.post("/api/memory_state/tagger")
async def memory_tagger(request: Request):
    if not _owner(request):
        return _deny()
    import memory_redact as MR
    if MR.tagger_files_present():
        return {"ok": True, "present": True}
    if _TAGGER["downloading"]:
        return {"ok": True, "downloading": True}

    def work():
        _TAGGER.update(downloading=True, error=None)
        try:
            from huggingface_hub import snapshot_download
            for repo, files in MR.TAGGER_FILES:          # only what the tagger loads (~590 MB)
                snapshot_download(repo, allow_patterns=list(files))
        except Exception as e:  # noqa: BLE001
            _TAGGER["error"] = type(e).__name__
        finally:
            _TAGGER["downloading"] = False
    import threading
    threading.Thread(target=work, name="TaggerDownload", daemon=True).start()
    return {"ok": True, "downloading": True}


def _old_data_files() -> list:
    """The retired memories' files (design S1), for the owner's one Delete old memory data button."""
    import re as _re
    import experience_memory as EM
    from pathlib import Path as _P
    root = EM.root()
    out = []
    for base in [root] + ([d for d in (root / "_deleted").iterdir() if d.is_dir()] if (root / "_deleted").exists()
                          else []):
        for d in base.iterdir() if base.exists() else []:
            if d.is_dir() and (base != root or not d.name.startswith("_")):
                for f in ("graph.json", "emb.npy", "items.json", "review.html"):
                    if (d / f).exists():
                        out.append(d / f)
    for f in (root / "_proposed.json", root.parent / "hypergraph"):
        if f.exists():
            out.append(f)
    mdb = _P(__file__).resolve().parent / "memory_db"
    if mdb.exists():
        for f in mdb.iterdir():
            if f.name == "chroma.sqlite3" or (f.is_dir() and _re.fullmatch(r"[0-9a-f-]{36}", f.name)) or \
                    f.name == "voiceprints.json" or _re.fullmatch(r"voiceprints\.bak-.*\.json", f.name):
                out.append(f)
    return out


@router.post("/api/memory_state/old_data")
async def memory_old_data(request: Request):
    if not _owner(request):
        return _deny()

    def work():
        import shutil
        n = 0
        for f in _old_data_files():
            try:
                shutil.rmtree(f) if f.is_dir() else f.unlink()
                n += 1
            except OSError as e:
                logger.warning("old memory file not deleted: %s", e)
        return n
    return {"ok": True, "deleted": await asyncio.to_thread(work)}
