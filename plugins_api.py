"""Owner-only Plugins API (plugins.py). Same locality rule as the memory panel:
only reachable from this computer.

  GET  /api/plugins                    every plugin: enabled, tools, settings schema, values
  POST /api/plugins/{pid}/enabled      {"enabled": bool}
  POST /api/plugins/{pid}/settings     {"values": {...}}  secrets are write-only
  POST /api/plugins/{pid}/test         {"field"?: str, "value"?: str}  test a key or the connection
  GET  /api/plugins/marketplace        catalog (built-ins + coming soon)
"""
import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import plugins as P

logger = logging.getLogger(__name__)
router = APIRouter()
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _owner(request: Request) -> bool:
    host = getattr(getattr(request, "client", None), "host", "") or ""
    return host in _LOOPBACK or host.startswith("127.")


def _deny():
    return JSONResponse({"error": "plugins can only be changed from this computer"}, status_code=403)


def _missing(pid):
    return JSONResponse({"error": f"no plugin called {pid!r}"}, status_code=404)


async def _body(request: Request) -> dict:
    try:
        d = await request.json()
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


@router.get("/api/plugins")
async def list_plugins(request: Request):
    if not _owner(request):
        return _deny()
    return {"plugins": await asyncio.to_thread(P.listing)}


@router.get("/api/plugins/marketplace")
async def market(request: Request):
    if not _owner(request):
        return _deny()
    return {"plugins": P.marketplace()}


@router.get("/api/plugins/{pid}/feed")
async def feed(pid: str, request: Request):
    """Read-only activity list for plugins that keep one (Quiet fact-check)."""
    if not _owner(request):
        return _deny()
    if pid == "fact_check":
        import factcheck
        return {"items": factcheck.feed()}
    return {"items": []}


@router.post("/api/plugins/{pid}/enabled")
async def set_enabled(pid: str, request: Request):
    if not _owner(request):
        return _deny()
    d = await _body(request)
    if not isinstance(d.get("enabled"), bool):
        return JSONResponse({"error": "send {\"enabled\": true|false}"}, status_code=400)
    try:
        return {"plugin": await asyncio.to_thread(P.set_enabled, pid, d["enabled"])}
    except KeyError:
        return _missing(pid)


@router.post("/api/plugins/{pid}/settings")
async def save(pid: str, request: Request):
    if not _owner(request):
        return _deny()
    d = await _body(request)
    vals = d.get("values")
    if not isinstance(vals, dict):
        return JSONResponse({"error": "send {\"values\": {...}}"}, status_code=400)
    try:
        r = await asyncio.to_thread(P.save_settings, pid, vals)
    except KeyError:
        return _missing(pid)
    return r if r.get("ok") else JSONResponse(r, status_code=400)


@router.post("/api/plugins/{pid}/test")
async def test(pid: str, request: Request):
    if not _owner(request):
        return _deny()
    d = await _body(request)
    try:
        return await asyncio.to_thread(P.test, pid, str(d.get("field") or ""), str(d.get("value") or ""))
    except KeyError:
        return _missing(pid)
