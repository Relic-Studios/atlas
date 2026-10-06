"""One runtime per agent. Everything an agent knows lives here, nowhere else.

Before this module, per-persona state was ~8 loose fields on SpeechPipelineManager
(history, lookups, floor, threads, bait, convo, dynamics names, prompt) reset one by
one in set_persona. Missing one = persona bleed. Now:

  AgentProfile  -- static identity: id, display name, interests, talkativeness.
  AgentRuntime  -- live state owned by ONE agent (conversation, lookups, floor...).
  AgentRegistry -- id -> runtime; switching = activating a different runtime.

Shared across agents on purpose (same humans, same machine): NameBook (people),
the task board store (already keyed per persona in tasks.Boards), the LLM, audio.

Stale-state rule: an agent that has been inactive longer than STALE_S comes back
with a fresh conversation (the room moved on without it). Floor/threads/bait are
per-session behaviour counters and always reset on activation.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

STALE_S = float(os.environ.get("ATLAS_AGENT_STALE_S", "600"))

# Profiles (interests + talkativeness) come from agent_registry, so the passivity
# rule ("speak unprompted only about what you love") has real content for every
# agent. talkativeness: 0 = only when addressed, 1 = jumps into everything.
import agent_registry as _registry


class _Profiles:
    def get(self, pid, default=None):
        a = _registry.agent(pid)
        return _registry.profile(pid) if a else (default if default is not None else {})


BUILTIN_PROFILES = _Profiles()


def _clamp01(x, default=0.35) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class AgentProfile:
    id: str
    name: str
    interests: tuple[str, ...] = ()
    talkativeness: float = 0.35

    @property
    def names(self) -> tuple[str, ...]:
        """Lower-case names the agent answers to (id and display name)."""
        out = [self.name.lower()]
        if self.id.lower() not in out:
            out.append(self.id.lower())
        return tuple(out)

    def interest_hits(self, text: str) -> list[str]:
        low = (text or "").lower()
        hits = []
        for it in self.interests:
            w = it.lower()
            # plural/possessive tolerant word match ("ramen", "rings", "hobbitses")
            if re.search(rf"\b{re.escape(w)}(?:e?s|'s)?\b", low):
                hits.append(it)
        return hits

    def floor_share_limit(self) -> float:
        """Share of recent turns above which the agent is told it's hogging the air.
        talkativeness 0 -> 30%, 0.35 -> ~42%, 1 -> 65%."""
        return 0.30 + 0.35 * self.talkativeness


def build_profile(pid: str, name: str, meta: Optional[dict] = None) -> AgentProfile:
    meta = meta or {}
    base = BUILTIN_PROFILES.get(pid, {})
    interests = meta.get("interests") or base.get("interests")
    if not interests:
        # Custom agents without explicit interests: use their role tags
        # ("crass · quick · banter") as a weak interest hint.
        interests = tuple(t.strip() for t in re.split(r"[·,/|]", meta.get("role", "")) if t.strip())
    if isinstance(interests, str):
        interests = tuple(t.strip() for t in re.split(r"[,;]", interests) if t.strip())
    talk = meta.get("talkativeness", base.get("talkativeness", 0.35))
    try:
        import owner_controls as _owner
        ov = _owner.talkativeness_override(pid)
        if ov is not None:
            talk = ov       # owner's dashboard slider wins
    except Exception:  # noqa: BLE001
        pass
    return AgentProfile(id=pid, name=name, interests=tuple(interests)[:16],
                        talkativeness=_clamp01(talk))


@dataclass
class AgentRuntime:
    profile: AgentProfile
    system_prompt: str
    history: list = field(default_factory=list)
    lookups: list = field(default_factory=list)
    floor: object = None
    threads: object = None
    bait: object = None
    convo: object = None
    cutoff: object = None      # interrupts.CutOff: what she didn't get to say
    last_active: float = 0.0

    @property
    def id(self) -> str:
        return self.profile.id

    def reset_conversation(self) -> None:
        self.history = []
        self.lookups = []
        self.cutoff = None
        if self.convo is not None:
            self.convo.reset()
            self.convo.agent_name = self.profile.name

    def reset_session(self) -> None:
        for part in (self.floor, self.threads, self.bait):
            if part is not None:
                part.reset()


def _default_parts(profile: AgentProfile):
    from floor import ConversationFloor
    from threads import ThreadArbiter
    from loopbait import LoopBaitGuard
    from conversation_log import ConversationLog
    floor = ConversationFloor()
    floor.profile = profile   # interests feed the passive gate
    return dict(floor=floor, threads=ThreadArbiter(profile.names),
                bait=LoopBaitGuard(), convo=ConversationLog(agent_name=profile.name))


class AgentRegistry:
    """id -> AgentRuntime. Runtimes are created lazily, never shared."""

    def __init__(self, prompt_for: Callable[[str], str], name_for: Callable[[str], str],
                 meta_for: Callable[[str], dict] = lambda pid: {},
                 clock=time.monotonic, parts=_default_parts):
        self.prompt_for = prompt_for
        self.name_for = name_for
        self.meta_for = meta_for
        self.clock = clock
        self.parts = parts
        self.runtimes: dict[str, AgentRuntime] = {}
        self.active: Optional[AgentRuntime] = None

    def get(self, pid: str) -> AgentRuntime:
        rt = self.runtimes.get(pid)
        if rt is None:
            prof = build_profile(pid, self.name_for(pid), self.meta_for(pid))
            rt = AgentRuntime(profile=prof, system_prompt=self.prompt_for(pid),
                              **self.parts(prof))
            self.runtimes[pid] = rt
        return rt

    def activate(self, pid: str, fresh_session: bool = True) -> AgentRuntime:
        """Make pid the speaking agent. The previous agent's state is left intact
        (its own memory), the new one resumes its own -- or starts fresh if stale."""
        now = self.clock()
        prev = self.active
        if prev is not None:
            prev.last_active = now
        rt = self.get(pid)
        # Refresh identity: prompt/profile may have been edited or re-registered.
        rt.system_prompt = self.prompt_for(pid)
        self.refresh_profile(pid)
        if rt.last_active and now - rt.last_active > STALE_S:
            rt.reset_conversation()
        if fresh_session:
            rt.reset_session()
        rt.last_active = now
        self.active = rt
        return rt

    def refresh_profile(self, pid: str) -> None:
        """Rebuild a runtime's profile (owner slider, edited meta). The floor keeps
        a reference to the profile, so it must be repointed too."""
        rt = self.runtimes.get(pid)
        if rt is None:
            return
        rt.profile = build_profile(pid, self.name_for(pid), self.meta_for(pid))
        if rt.floor is not None:
            rt.floor.profile = rt.profile
        if rt.threads is not None:
            rt.threads.names = rt.profile.names

    def forget(self, pid: str) -> None:
        if self.active is not None and self.active.id == pid:
            return
        self.runtimes.pop(pid, None)
