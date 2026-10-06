"""Agent panel (owner 10-06, OK'd): two agents share one call.

Exactly one agent speaks per turn. This module only decides WHICH one; the
pipeline swaps the active persona (voice, prompt, runtime) before drafting.

Routing, in order:
  1. A panel agent named in the line ("Fae, ...", "what do you think, Max?")
     takes it. If both are named, the one named first.
  2. A human answering the agent that spoke last keeps that agent (partner).
  3. Otherwise the lead agent (the first member) handles it.

Hand-off: when the speaking agent's own reply names the other panel agent with
a question ("Fae, what do you reckon?"), the other agent gets ONE turn to
answer at the next pause. Agent-to-agent turns are capped per human turn so two
agents can never talk to each other in a loop.

Pure and clock-injected; no audio or model calls.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable, Dict, List, Optional

MAX_MEMBERS = 2
HANDOFF_CAP = 1          # agent->agent turns allowed per human turn
PARTNER_TTL_S = 40.0     # a human reply within this keeps the last agent
HANDOFF_TTL_S = 20.0     # an unclaimed hand-off expires


def _name_re(names) -> re.Pattern:
    alts = "|".join(sorted({re.escape(n) for n in names if n}, key=len, reverse=True))
    return re.compile(r"(?<![\w'])(?:" + alts + r")(?![\w'])", re.I) if alts else re.compile(r"(?!x)x")


class Panel:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self.members: List[str] = []           # agent ids, members[0] = lead
        self.names: Dict[str, tuple] = {}      # id -> names/aliases it answers to
        self.last_agent: Optional[str] = None
        self.last_agent_at = 0.0
        self.handoffs_this_turn = 0
        self.pending: Optional[dict] = None    # {"to": id, "from": id, "line": str, "at": t}

    # ------------------------------------------------------------ setup
    @property
    def active(self) -> bool:
        return len(self.members) >= 2

    def set_members(self, ids: List[str], names: Dict[str, tuple]) -> None:
        ids = [i for i in dict.fromkeys(i for i in ids if i)][:MAX_MEMBERS]
        with self._lock:
            self.members = ids
            self.names = {i: tuple(n.lower() for n in (names.get(i) or (i,)) if n) for i in ids}
            self.pending = None
            if self.last_agent not in ids:
                self.last_agent = None

    def clear(self) -> None:
        self.set_members([], {})

    # ------------------------------------------------------------ routing
    def _named(self, text: str) -> List[str]:
        hits = []
        for aid in self.members:
            m = _name_re(self.names.get(aid, ())).search(text or "")
            if m:
                hits.append((m.start(), aid))
        return [a for _, a in sorted(hits)]

    def route(self, text: str) -> Optional[str]:
        """Which member handles this human line? None when no panel is set."""
        if not self.active:
            return None
        named = self._named(text)
        if named:
            return named[0]
        if self.last_agent and self.clock() - self.last_agent_at <= PARTNER_TTL_S:
            return self.last_agent
        return self.members[0]

    def on_human_turn(self) -> None:
        with self._lock:
            self.handoffs_this_turn = 0
            # A human spoke: an unclaimed hand-off is stale now (they moved on).
            self.pending = None

    def on_agent_spoke(self, agent: str, text: str) -> Optional[str]:
        """Record who spoke; returns the member the reply handed off to (or None)."""
        now = self.clock()
        with self._lock:
            self.last_agent, self.last_agent_at = agent, now
            if not self.active or agent not in self.members:
                return None
            if self.handoffs_this_turn >= HANDOFF_CAP:
                return None
            others = [a for a in self._named(text) if a != agent]
            if not others or "?" not in (text or ""):
                return None
            to = others[0]
            self.handoffs_this_turn += 1
            self.pending = {"to": to, "from": agent, "line": (text or "").strip()[:300], "at": now}
            return to

    def take_handoff(self) -> Optional[dict]:
        with self._lock:
            p = self.pending
            if not p:
                return None
            self.pending = None
            if self.clock() - p["at"] > HANDOFF_TTL_S:
                return None
            return p

    def peek_handoff(self) -> Optional[dict]:
        p = self.pending
        if p and self.clock() - p["at"] <= HANDOFF_TTL_S:
            return dict(p)
        return None

    def note(self, me: str, display: Callable[[str], str] = lambda x: x) -> str:
        """One line for the speaking agent's context: who else is on the panel."""
        if not self.active or me not in self.members:
            return ""
        others = ", ".join(display(a) for a in self.members if a != me)
        return (f"PANEL: you share this call with {others}, another AI agent. One of you speaks "
                f"per turn. Don't answer for {others} or repeat what they just said; if a "
                f"question is better for them, you may hand it over by name. Their lines show "
                f"up in the conversation under their name.")

    def snapshot(self) -> dict:
        return {"members": list(self.members), "active": self.active,
                "last_agent": self.last_agent, "pending": self.peek_handoff()}


PANEL = Panel()

HANDOFF_CUE_RE = re.compile(r"\(PANEL CUE\)")


def cue_text(p: dict, display: Callable[[str], str] = lambda x: x) -> str:
    who = display(p["from"])
    return (f"(PANEL CUE) This is NOT a new line from a person: it's your cue. {who}, the "
            f"other agent on this call, just turned to you and said: \"{p['line']}\" Answer "
            f"{who} in one or two short lines, in your own words. Don't [HOLD].")


def cue_note(text: str) -> str:
    if HANDOFF_CUE_RE.search(text or ""):
        return ("PANEL: the other agent just handed you this turn. Answer them briefly, in "
                "character. Do not [HOLD] and do not hand it straight back.")
    return ""
