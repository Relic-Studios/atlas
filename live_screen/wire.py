"""Messages between Atlas and the capture process (docs/designs/live_screen.md, item 8): 1 kind byte, a 4-byte
little-endian length, then the payload (the framing nemotron_client.py uses on its socket), over pipes.

Kinds: Q request (JSON {"id", "op", ...}), A answer (4-byte header length, JSON header {"id", "ok", "error",
"images": [{"source", "age_s", "w", "h", "len", "role"}], ...}, then the JPEG bytes in header order), E event (JSON).
"""
from __future__ import annotations

import json
import struct

REQUEST, ANSWER, EVENT = b"Q", b"A", b"E"
MAX_PAYLOAD = 64 * 1024 * 1024


def read_exact(f, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("pipe closed")
        buf += chunk
    return buf


def read_msg(f) -> tuple[bytes, bytes]:
    kind = read_exact(f, 1)
    (n,) = struct.unpack("<I", read_exact(f, 4))
    if n > MAX_PAYLOAD:
        raise ValueError(f"message too large ({n} bytes)")
    return kind, read_exact(f, n)


def write_msg(f, lock, kind: bytes, payload: bytes) -> None:
    with lock:
        f.write(kind + struct.pack("<I", len(payload)) + payload)
        f.flush()


def pack_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def pack_answer(header: dict, jpegs: list) -> bytes:
    images = header.get("images") or []
    for meta, jpg in zip(images, jpegs):
        meta["len"] = len(jpg)
    h = pack_json(header)
    return struct.pack("<I", len(h)) + h + b"".join(jpegs)


def unpack_answer(payload: bytes) -> tuple[dict, list]:
    (hn,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4:4 + hn].decode("utf-8"))
    jpegs, pos = [], 4 + hn
    for meta in header.get("images") or []:
        jpegs.append(payload[pos:pos + meta["len"]])
        pos += meta["len"]
    return header, jpegs
