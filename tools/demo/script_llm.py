"""Scripted stand-in for the language model, for demo recordings that must not load a real model.

  python tools/demo/script_llm.py --scene <rendered scene.json> [--port 11500] [--model scripted] [--log llm.jsonl]

Speaks Ollama's HTTP API (what ATLAS's local mode calls). Every chat request is answered from the scene:
if the LAST thing ATLAS heard matches a scripted human line that carries a "reply", the answer is
"[SPEAK to=<that speaker's label>] <reply>"; anything else gets "[HOLD]". So the words are scripted, while
hearing, speaker labels, when to answer and the voice are still ATLAS's own. The first words come after a
model-like delay (--ttft), then stream word by word, so the timing on screen is not faster than a real model.

Scene lines: {"who": "Sam", "text": "...", "reply": "..."}  ({N} is already replaced by make_voices / build_scene).
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORD = re.compile(r"[a-z0-9']+")
LABEL = re.compile(r"\[([^\]\s=]{1,40})\]")
MIN_COVER = 0.75          # share of a cue's words that must be in the last stretch of what ATLAS heard


def words(s: str) -> list[str]:
    return [w for w in (t.strip("'") for t in WORD.findall(s.lower().replace("’", "'"))) if w]


class Script:
    def __init__(self, scene: Path):
        sc = json.loads(scene.read_text(encoding="utf-8"))
        self.lines = sc["lines"]
        self.cues = [(i, ln["text"], ln["reply"]) for i, ln in enumerate(sc["lines"]) if ln.get("reply")]

    def respond(self, messages: list) -> tuple[str, list | None, dict]:
        """(text, tool_calls, debug) for a chat request. A cue with "tool" first asks ATLAS to run that tool; once
        its result is in the conversation, the reply is the first "reply_by_result" entry whose pattern is in the
        result (else "reply"), so what Max says follows what the tool really returned."""
        heard = last_user(messages)
        text, dbg = self.answer(heard)
        cue = dbg.get("cue")
        ln = self.lines[cue] if cue is not None else {}
        if not ln.get("tool"):
            return text, None, dbg
        k = max(i for i, m in enumerate(messages) if m.get("role") == "user")
        results = " ".join(str(m.get("content") or "") for m in messages[k + 1:] if m.get("role") == "tool")
        if not results:
            return "", [{"function": {"name": ln["tool"]["name"], "arguments": ln["tool"].get("arguments", {})}}], \
                {**dbg, "tool": ln["tool"]["name"]}
        reply = ln["reply"]
        for pat, r in ln.get("reply_by_result", []):
            if re.search(pat, results, re.I):
                reply = r
                break
        return f"[SPEAK to={dbg['to']}] {reply}", None, {**dbg, "tool_result": results[:120]}

    def answer(self, heard: str) -> tuple[str, dict]:
        """(reply text, debug) for the last user message."""
        # ATLAS appends coaching notes on their own lines ("(This line is to you: ... Reply to S2 with ...)"):
        # only the heard speech counts.
        heard = "\n".join(ln for ln in heard.splitlines() if not ln.strip().startswith("("))
        hw = words(heard)
        best = None
        for i, text, reply in self.cues:
            cw = words(text)
            tail = set(hw[-(len(cw) + 8):])
            cover = sum(w in tail for w in cw) / max(1, len(cw))
            # "early_words": N lets a cue answer once its first N words are heard, like a real model greeting
            # "Hey Max, it's Sam." before "You there?" (ATLAS asks on that partial and, after a hold, may not
            # ask again for the rest of the line).
            n = int(self.lines[i].get("early_words") or 0)
            if n and cover < MIN_COVER and all(w in tail for w in cw[:n]) and hw[-1:] and hw[-1] in cw[:n]:
                cover = MIN_COVER
            if cover >= MIN_COVER and (best is None or cover > best[0] or (cover == best[0] and i > best[1])):
                best = (cover, i, text, reply)
        if best is None:
            return "[HOLD]", {"cue": None}
        cover, i, text, reply = best
        # The speaker label ATLAS gave the matched line: the last [X] before its first words.
        first = words(text)[0]
        pos = heard.lower().rfind(first)
        labels = LABEL.findall(heard[:pos] if pos > 0 else heard)
        to = labels[-1] if labels else "user"
        return f"[SPEAK to={to}] {reply}", {"cue": i, "cover": round(cover, 2), "to": to}


def last_user(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return ""


def make_handler(script: Script, model: str, ttft: float, word_s: float, log: Path | None):
    lock = threading.Lock()

    def note(rec: dict):
        if log:
            with lock, log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):      # quiet console
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        def do_GET(self):
            p = self.path.split("?")[0]
            info = {"name": model, "model": model, "size": 1, "digest": "scripted",
                    "details": {"family": "scripted", "parameter_size": "0", "quantization_level": "none"}}
            if p == "/":
                b = b"Ollama is running"
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif p == "/api/version":
                self._json({"version": "0.0.0-scripted"})
            elif p in ("/api/tags", "/api/ps"):
                self._json({"models": [info]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            p = self.path.split("?")[0]
            body = self._body()
            if p == "/api/show":
                return self._json({"capabilities": ["completion", "tools"], "details": {"family": "scripted"},
                                   "model_info": {}, "template": ""})
            if p in ("/api/embed", "/api/embeddings"):
                return self._json({"embeddings": [[0.0] * 8], "embedding": [0.0] * 8})
            if p == "/api/generate":            # prewarm / keep-alive pings
                return self._json({"model": model, "response": "", "done": True, "done_reason": "stop"})
            if p != "/api/chat":
                return self._json({"error": "not found"}, 404)
            msgs = body.get("messages") or []
            heard = last_user(msgs)
            system = " ".join(m.get("content") or "" for m in msgs
                              if m.get("role") == "system" and isinstance(m.get("content"), str))
            offered = {(t.get("function") or {}).get("name") for t in body.get("tools") or []}
            text, calls, dbg = script.respond(msgs)
            if calls and calls[0]["function"]["name"] not in offered:
                dbg["tool_not_offered"] = calls[0]["function"]["name"]
                text, calls = "[HOLD]", None      # never call a tool ATLAS didn't offer
            if "[HOLD]" not in system and text == "[HOLD]":
                text = ""                       # not a reply-protocol call: say nothing
            note({"t": time.time(), "heard_tail": heard[-300:], "answer": text or (calls and f"TOOL {calls[0]['function']['name']}"),
                  **dbg, "stream": body.get("stream", True), "tools": sorted(n for n in offered if n)[:6]})
            now = lambda: datetime.now(timezone.utc).isoformat()
            if not body.get("stream", True):
                time.sleep(ttft)
                msg = {"role": "assistant", "content": text, **({"tool_calls": calls} if calls else {})}
                return self._json({"model": model, "created_at": now(), "done": True, "done_reason": "stop",
                                   "message": msg})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                b = (json.dumps(obj) + "\n").encode()
                self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()

            try:
                time.sleep(ttft)
                if calls:
                    send({"model": model, "created_at": now(), "done": False,
                          "message": {"role": "assistant", "content": "", "tool_calls": calls}})
                    send({"model": model, "created_at": now(), "done": True, "done_reason": "stop",
                          "message": {"role": "assistant", "content": ""}})
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                parts = re.findall(r"\S+\s*", text) or [""]
                for k, part in enumerate(parts):
                    send({"model": model, "created_at": now(), "done": False,
                          "message": {"role": "assistant", "content": part}})
                    if k + 1 < len(parts):
                        time.sleep(word_s)
                send({"model": model, "created_at": now(), "done": True, "done_reason": "stop",
                      "message": {"role": "assistant", "content": ""}, "eval_count": len(parts)})
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                note({"t": time.time(), "cancelled": True, **dbg})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--model", default="scripted")
    ap.add_argument("--ttft", type=float, default=0.45, help="seconds before the first word (a real model's pace)")
    ap.add_argument("--word_s", type=float, default=0.03)
    ap.add_argument("--log")
    a = ap.parse_args()
    script = Script(Path(a.scene))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port),
                              make_handler(script, a.model, a.ttft, a.word_s, Path(a.log) if a.log else None))
    print(f"scripted model '{a.model}' on 127.0.0.1:{a.port}, {len(script.cues)} cues", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
