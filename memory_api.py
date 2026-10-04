"""Owner-only memory panel API: see, keep and forget what each agent remembers.

Every agent has a folder agent_state/memory/<agent>/ (agent_memory.py) with
  items.json   approved + candidate memories (call_memory.py; approved ones reach the prompt)
  graph.json   learned memories (hypergraph_memory.py; Hebbian strengthening + decay)

  GET  /api/memory                      agents + counts
  GET  /api/memory/{aid}                approved, candidates, learned
  POST /api/memory/{aid}/item           {"id", "action": approve|reject|forget}
  POST /api/memory/{aid}/learned        {"id", "action": keep|forget}
  POST /api/memory/{aid}/forget_all     {"what": "learned"|"everything"}
  POST /api/memory/{aid}/wipe           {"window": "1h"|"12h"|"1d"|"1w"|"all"}  this agent only
  POST /api/memory/{aid}/open           open the folder in the file manager

Only reachable from this computer (same rule as the other owner controls).
"""
import asyncio
import logging
import os
import subprocess
import sys

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import agent_memory as AM
import call_memory as CM

logger = logging.getLogger(__name__)
router = APIRouter()
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _owner(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in _LOOPBACK or host.startswith("127.")


def _deny():
    return JSONResponse({"error": "memory can only be viewed or changed from this computer"}, status_code=403)


def _svc(request: Request):
    mgr = getattr(request.app.state, "SpeechPipelineManager", None)
    return getattr(mgr, "hgmem", None)


def _graph(request: Request, aid: str):
    """Live graph if the pipeline is up, otherwise read straight from the folder."""
    svc = _svc(request)
    if svc is not None:
        return svc.graph(aid)
    import hypergraph_memory as H
    return H.HyperMemory(aid, embedder=None)


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


def _resync(request: Request, aid: str):
    svc = _svc(request)
    if svc is not None:
        svc.sync_approved(aid, [i["text"] for i in CM.approved(aid)])


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
        rows = []
        for aid, name in _known_agents().items():
            items = CM.load(aid)
            try:
                learned = len(_graph(request, aid).edges)
            except Exception:  # noqa: BLE001
                learned = 0
            rows.append({"id": aid, "name": name, "folder": str(AM.folder(aid)),
                         "approved": sum(1 for i in items if i.get("status") == "approved"),
                         "candidates": sum(1 for i in items if i.get("status") == "candidate"),
                         "learned": learned})
        return {"agents": rows}
    return await asyncio.to_thread(work)


@router.get("/api/memory/{aid}")
async def memory_agent(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    if not _valid(aid):
        return JSONResponse({"error": "unknown agent"}, status_code=404)

    def work():
        items = CM.load(aid)
        return {"id": aid, "folder": str(AM.ensure(aid)),
                "approved": [i for i in items if i.get("status") == "approved"],
                "candidates": [i for i in items if i.get("status") == "candidate"],
                "learned": [r for r in _graph(request, aid).listing(300) if r["kind"] != "approved"]}
    return await asyncio.to_thread(work)


@router.post("/api/memory/{aid}/item")
async def memory_item(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    iid, act = str(body.get("id", "")), body.get("action")
    if act == "approve":
        n = CM.set_status(aid, [iid], "approved")
    elif act == "reject":
        n = CM.set_status(aid, [iid], "rejected")
    elif act == "forget":
        n = CM.forget(aid, [iid])
    else:
        return JSONResponse({"error": "action must be approve, reject or forget"}, status_code=400)
    _resync(request, aid)          # rejected/forgotten approved memories leave the graph too
    return {"ok": bool(n)}


@router.post("/api/memory/{aid}/learned")
async def memory_learned(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    eid, act = str(body.get("id", "")), body.get("action")
    g = _graph(request, aid)
    row = next((r for r in g.listing(10 ** 6) if r["id"] == eid), None)
    if row is None:
        return JSONResponse({"error": "no such memory"}, status_code=404)
    if act == "forget":
        n = g.forget([eid])
        g.save()
        return {"ok": bool(n)}
    if act == "keep":                 # the owner approves it: it now reaches every prompt
        g.set_pinned(eid, True)
        g.save()
        CM.add_approved(aid, row["text"])
        return {"ok": True}
    return JSONResponse({"error": "action must be keep or forget"}, status_code=400)


@router.post("/api/memory/{aid}/forget_all")
async def memory_forget_all(aid: str, request: Request):
    if not _owner(request):
        return _deny()
    body = await request.json()
    what = body.get("what", "learned")
    if what == "everything":
        return await asyncio.to_thread(wipe, aid, "all", _svc(request))
    g = _graph(request, aid)
    ids = [r["id"] for r in g.listing(10 ** 6) if what == "everything" or r["kind"] != "approved"]
    n = g.forget(ids)
    g.save()
    if what == "everything":
        CM.save(aid, [])
        _resync(request, aid)
    return {"ok": True, "forgot": n}


WINDOWS = {"1h": 3600, "12h": 12 * 3600, "1d": 86400, "1w": 7 * 86400, "all": 0}


def wipe(aid: str, window: str, svc=None, root=None, now: float = None) -> dict:
    """Clear what ONE agent remembers from the last `window` (or everything). Both stores:
    items.json (approved/candidates) and the learned graph. Other agents are never touched."""
    import time as _t
    import hypergraph_memory as H
    secs = WINDOWS[window]
    cutoff = 0.0 if secs == 0 else (now or _t.time()) - secs
    items = CM.forget_since(aid, cutoff, root)
    if svc is not None:
        learned = svc.wipe(aid, cutoff)
        svc.sync_approved(aid, [i["text"] for i in CM.approved(aid, root)])
    else:   # pipeline down: edit the files; approved pins are re-synced at next start
        g = H.HyperMemory(aid, root=root, embedder=None)
        learned = g.forget_since(cutoff)
        g.save()
    logger.info("memory wipe %s window=%s: %d items, %d learned", aid, window, items, learned)
    return {"ok": True, "agent": aid, "window": window, "items": items, "learned": learned}


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
    svc = _svc(request)
    return await asyncio.to_thread(wipe, aid, window, svc)


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
