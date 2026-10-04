"""One memory folder per agent: agent_state/memory/<agent>/

    items.json   owner-approved / candidate memories (call_memory.py)
    graph.json   learned hypergraph memory (hypergraph_memory.py)
    emb.npy      its embeddings
    review.html  off-call review page (tools/memory_review.py, dev only)

A folder is created the moment an agent exists (startup for every registered
agent, and when the + creator saves a new one). Deleting an agent ARCHIVES its
folder to _deleted/<agent>_<time>/ so a new agent with the same name never
inherits someone else's memories. Names starting with "_" are reserved.
"""
import json
import logging
import os
import re
import shutil
import threading
import uuid
import time
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("ATLAS_MEMORY_DIR") or
            Path(__file__).resolve().parent / "agent_state" / "memory")
ITEMS = "items.json"
_lock = threading.Lock()
_migrated: set = set()


def slug(agent: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", (agent or "").strip().lower()).strip("-")
    return s or "unknown"


def folder(agent: str, root: Path = None) -> Path:
    s = slug(agent)
    if s.startswith("_"):
        raise ValueError(f"reserved memory name: {agent!r}")
    return Path(root or ROOT) / s


def migrate(root: Path = None) -> list:
    """Old flat layout -> per-agent folders. Idempotent; returns what moved."""
    root = Path(root or ROOT)
    key = str(root.resolve()) if root.exists() else str(root)
    if key in _migrated:
        return []
    moved = []
    with _lock:
        if root.exists():
            for f in list(root.glob("*.json")):
                if f.name.startswith("_"):
                    continue                      # _proposed.json etc. stay at the root
                dst = root / f.stem / ITEMS
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, dst)
                moved.append(f"{f.name} -> {f.stem}/{ITEMS}")
            for f in list(root.glob("review_*.html")):
                a = f.stem[len("review_"):]
                dst = root / a / "review.html"
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(f, dst)
                moved.append(f"{f.name} -> {a}/review.html")
            old_hg = root.parent / "hypergraph"       # first hypergraph layout
            if old_hg.is_dir():
                for d in old_hg.iterdir():
                    if d.is_dir():
                        dst = root / d.name
                        dst.mkdir(parents=True, exist_ok=True)
                        for f in d.iterdir():
                            if not (dst / f.name).exists():
                                os.replace(f, dst / f.name)
                                moved.append(f"hypergraph/{d.name}/{f.name} -> {d.name}/")
        _migrated.add(key)
    for m in moved:
        logger.info("memory layout: %s", m)
    return moved


IDENT = ".id"


def ident(agent: str, root: Path = None, create: bool = True) -> str:
    """The folder's identity. A new folder (new agent, or the same name re-created after a
    delete) gets a new id, so anything still holding the old id can tell it is stale."""
    d = folder(agent, root)
    f = d / IDENT
    try:
        v = f.read_text(encoding="utf-8").strip()
        if v:
            return v
    except OSError:
        pass
    if not create:
        return ""
    with _lock:
        d.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            f.write_text(uuid.uuid4().hex, encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


def ensure(agent: str, root: Path = None) -> Path:
    """Create the agent's folder (and an empty items.json) if missing."""
    migrate(root)
    d = folder(agent, root)
    d.mkdir(parents=True, exist_ok=True)
    ident(agent, root)
    f = d / ITEMS
    if not f.exists():
        f.write_text(json.dumps({"items": []}), encoding="utf-8")
    return d


def ensure_all(agents, root: Path = None) -> list:
    out = []
    for a in agents:
        try:
            out.append(ensure(a, root))
        except ValueError:
            pass
    return out


def archive(agent: str, root: Path = None) -> Path | None:
    """Move an agent's memory out of reach (kept for the owner, never loaded)."""
    migrate(root)
    d = folder(agent, root)
    if not d.exists():
        return None
    dst = Path(root or ROOT) / "_deleted" / f"{d.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        shutil.move(str(d), str(dst))
    return dst


def agents_on_disk(root: Path = None) -> list:
    migrate(root)
    root = Path(root or ROOT)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))
