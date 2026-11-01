"""Keep "I'll look into it": if the agent promises a lookup but never searched, search for real.

Live 11-01 (Fae): "Do you think you could search up some bodies for yourself?" -> "Okay, I'll
look into it." with no search, then "Did you end up looking it up?" -> "I did, right after you
asked" (nothing had run). The reply's own promise starts a background search on the task board,
so it's true, and the board brings the result back at the next gap (server.start_delivery).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

PROMISE_RE = re.compile(
    r"\b(?:i'?ll|i\s+will|let\s+me|lemme|i'?m\s+gonna|i'?m\s+going\s+to|gonna)\s+(?:go\s+|quickly\s+|just\s+)?"
    r"(?:look\s+(?:it|that|this|them|those)\s+up|look\s+up|look\s+into|search|google|research|dig\s+(?:into|around|up)"
    r"|find\s+out|check\s+(?:online|on\s+that|into))\b", re.I)
_CUE_RE = re.compile(r"\((?:TASK|GLANCE|ROOM)\s+CUE\b", re.I)

QUERY_SYSTEM = ("Write the web search query for what this person asked a voice assistant to look up. "
                "Use their words, at most 8 words, no quotes, no explanation. If they asked several things, "
                "use the one they wanted looked up.")


def promised(reply: str) -> bool:
    return bool(PROMISE_RE.search(reply or ""))


def _fallback_query(line: str) -> str:
    body = re.sub(r"^\s*\[S\d+\]\s*", "", line or "")
    m = re.search(r"\b(?:search(?:\s+up)?|google|look\s+(?:up|into)|research|find\s+out(?:\s+about)?)\s+(.+?)(?:[?.!]|$)",
                  body, re.I)
    q = m.group(1) if m else body
    q = re.sub(r"\b(?:for\s+(?:yourself|me|us)|please|real\s+quick|for\s+a\s+sec)\b", " ", q, flags=re.I)
    return " ".join(q.split())[:120]


def search_query(line: str, timeout: float = 2.0) -> str:
    """A short query from the live model; their own words if it can't answer."""
    try:
        import glance_judge
        s, cfg = glance_judge._client()
        if s is not None:
            body = {"model": cfg["model"], "temperature": 0, "max_tokens": 24,
                    "messages": [{"role": "system", "content": QUERY_SYSTEM},
                                 {"role": "user", "content": re.sub(r"^\s*\[S\d+\]\s*", "", line or "")}],
                    "chat_template_kwargs": {"enable_thinking": False}}
            r = s.post(cfg["url"] + "/v1/chat/completions", json=body, timeout=(1.0, timeout))
            q = (r.json()["choices"][0]["message"].get("content") or "").strip().strip('"').splitlines()[0]
            if 2 <= len(q) <= 120:
                return q
    except Exception as e:  # noqa: BLE001
        logger.debug("query writer failed: %s", e)
    return _fallback_query(line)


_FACT_Q = re.compile(r"\b(?:what|who|when|where|which|how\s+(?:much|many|long|far|old|big|tall)|is\s+there|"
                     r"are\s+there)\b[^?]*\?", re.I)


def should_keep(asked: str, reply: str) -> bool:
    """A promise worth keeping: its reply promises a lookup, and the line it answered asked
    for one (search words) or asked a factual question ("you good?" doesn't count)."""
    if not asked or _CUE_RE.search(asked) or not promised(reply):
        return False
    import backchannels
    body = re.sub(r"^\s*\[S\d+\]\s*", "", asked)
    return backchannels.wants_search(body) or bool(_FACT_Q.search(body))
