"""Quiet screen context: what's on screen, in a line, for the agent's next few turns.

Owner 11-01: "she needs to be able to look and have context without always saying stuff and
doesn't need to look every time". When the room reacts to something on screen ("is that a
guy?", "check this out") the agent glances in the background (same rate limit as its other
glances), the vision model writes one plain line about what's there, and that line rides in
the room context for CONTEXT_TTL_S. It never makes the agent speak; it only means that if it
does speak, it isn't blind and doesn't invent what's on screen.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

CONTEXT_TTL_S = 90.0
TIMEOUT_S = 12.0
PROMPT = ("This is a screenshot of someone's computer during a Discord voice call. In one or two short, plain "
          "sentences, say what is on it that people in the call could be reacting to: what's being shown or "
          "played, and who is on camera and what they look like. Facts only, no guesses about what anyone is "
          "thinking.")


def caption(b64: str, timeout: float = TIMEOUT_S) -> str:
    """One plain line about the screenshot from the live (vision) model; '' if it can't."""
    try:
        import glance_judge
        s, cfg = glance_judge._client()
        if s is None:
            return ""
        body = {"model": cfg["model"], "temperature": 0, "max_tokens": 90,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.perf_counter()
        r = s.post(cfg["url"] + "/v1/chat/completions", json=body, timeout=(2.0, timeout))
        text = " ".join((r.json()["choices"][0]["message"].get("content") or "").split())
        logger.info("👁️ screen context in %.0f ms: %s", (time.perf_counter() - t0) * 1000, text[:160])
        return text[:400]
    except Exception as e:  # noqa: BLE001
        logger.debug("screen caption failed: %s", e)
        return ""


def note(ctx: dict | None, now: float | None = None) -> str:
    """Room-context line for a fresh caption ('' when there is none or it's stale)."""
    if not ctx or not ctx.get("text"):
        return ""
    age = (time.time() if now is None else now) - ctx["at"]
    if age > CONTEXT_TTL_S:
        return ""
    return (f"ON THE SCREEN (you glanced ~{max(1, int(age))}s ago; background only: don't mention looking, "
            f"and only bring it up if it fits what's said): {ctx['text']}")
