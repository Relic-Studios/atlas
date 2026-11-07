"""Declaration-only stdio MCP bridge for Claude Code.

It never executes tools. Claude's tool_use message is handed back to Atlas,
which ends this process tree before its existing executor performs the action.
"""
from __future__ import annotations

import json
import sys


def serve(path):
    with open(path, encoding="utf-8") as stream:
        tools = json.load(stream)
    names = {tool["name"] for tool in tools}
    while True:
        raw=sys.stdin.buffer.readline(1024*1024+1)
        if not raw:
            return
        if len(raw) > 1024 * 1024:
            return
        try:
            request = json.loads(raw)
            method, rid = request.get("method"), request.get("id")
            if rid is None:
                continue
            if method == "initialize":
                result = {"protocolVersion": request.get("params", {}).get("protocolVersion", "2024-11-05"),
                          "capabilities": {"tools": {}}, "serverInfo": {"name": "atlas-tool-declarations", "version": "1.0.0"}}
            elif method == "tools/list":
                result = {"tools": tools}
            elif method == "ping":
                result = {}
            elif method == "tools/call" and request.get("params", {}).get("name") in names:
                # No reply: the parent captures the completed tool_use block,
                # terminates this CLI/bridge generation, then lets Atlas act.
                continue
            else:
                print(json.dumps({"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"Unsupported operation"}}),flush=True)
                continue
            print(json.dumps({"jsonrpc":"2.0","id":rid,"result":result}),flush=True)
        except (ValueError, TypeError, KeyError):
            return


if __name__ == "__main__":
    serve(sys.argv[1])
