"""Diarization-backed conversation log with a hard prompt budget.

Replaces "last 16 messages, whatever their size" with:

* a bounded LEDGER of every committed turn (speaker label from the diarizer,
  timestamps, agent target) -- never sent whole to the model;
* a TOKEN-BUDGETED view rendered newest-first for each generation, so a
  200-word Discord ramble costs at most `max_entry_words` and the prompt size
  is flat no matter how long the call runs;
* a per-speaker ROSTER that survives eviction: who is in the call, who is
  active, who has gone quiet, and each speaker's last line once it has scrolled
  out of the verbatim window -- so the model keeps a picture of the room without
  an LLM summary call competing for the GPU slot;
* fragment MERGING (fast VAD splits one thought into several finals from the
  same voice) and REACTION folding ("lol"/"haha" after the agent spoke becomes
  "S2 laughed at your line" instead of a history message).

Pure Python, no model, deterministic -> fully simulatable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import re
import time

_WORD = re.compile(r"\S+")
_LABEL = re.compile(r"^\s*\[(S\d+|user|self)\]\s*")
_LAUGH = re.compile(r"\b(lol+|lmao+|lmfao|rofl|(ha){2,}h?|he(he)+|dead|💀|😂)\b", re.I)


def est_tokens(text: str) -> int:
    """Cheap, conservative token estimate (~1.3 tok/word + punctuation)."""
    return int(len(_WORD.findall(text or "")) * 1.35) + 4


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def clip_words(text: str, n: int, keep: str = "tail") -> str:
    w = (text or "").split()
    if len(w) <= n:
        return text.strip()
    return ("… " + " ".join(w[-n:])) if keep == "tail" else (" ".join(w[:n]) + " …")


@dataclass
class Entry:
    role: str                    # "user" | "assistant"
    speaker: str | None          # "S1".. for users; target for assistant
    parts: list[str]
    t_start: float
    t_end: float

    @property
    def text(self) -> str:
        return " ".join(self.parts)


@dataclass
class Speaker:
    label: str
    first_seen: float
    last_seen: float
    turns: int = 0
    words: int = 0
    addressed_agent: int = 0
    last_line: str = ""
    reactions: int = 0


@dataclass
class ConversationLog:
    agent_name: str = "Agent"
    budget_tokens: int = 500         # verbatim window budget (system prompt is separate)
    max_entry_words: int = 35        # a ramble is clipped to its most recent words
    max_agent_words: int = 40
    merge_window_s: float = 4.0      # same voice, no agent line between -> same thought
    reaction_window_s: float = 12.0  # "lol" this soon after the agent spoke = reaction
    ledger_cap: int = 400
    roster_cap: int = 8
    min_entries: int = 2
    clock: object = time.monotonic
    ledger: deque = field(default_factory=deque)
    roster: dict = field(default_factory=dict)
    last_agent_at: float | None = None
    last_agent_target: str | None = None
    last_reaction: str = ""

    # ------------------------------------------------------------- mutation
    def reset(self) -> None:
        self.ledger.clear()
        self.roster.clear()
        self.last_agent_at = self.last_agent_target = None
        self.last_reaction = ""

    def add_user(self, speaker: str | None, text: str) -> str:
        """Commit a final user transcript. Returns 'added' | 'merged' | 'reaction' | 'ignored'."""
        body = _LABEL.sub("", text or "").strip()
        if not body or speaker == "self":
            return "ignored"
        now = self.clock()
        spk = speaker if speaker and speaker != "user" else None
        from floor import filler_only
        if filler_only(body):
            if spk:
                self._touch(spk, now)
            if self.last_agent_at is not None and now - self.last_agent_at <= self.reaction_window_s:
                who = spk or "someone"
                verb = "laughed at" if _LAUGH.search(body) else "reacted to"
                self.last_reaction = f"{who} {verb} your last line"
                if spk:
                    self.roster[spk].reactions += 1
                return "reaction"
            return "ignored"
        rec = self._touch(spk, now) if spk else None
        if rec:
            rec.turns += 1
            rec.words += len(body.split())
            if re.search(rf"\b{re.escape(self.agent_name.lower())}\b", body.lower()):
                rec.addressed_agent += 1
        last = self.ledger[-1] if self.ledger else None
        if (last and last.role == "user" and last.speaker == spk
                and now - last.t_end <= self.merge_window_s):
            last.parts.append(body)
            last.t_end = now
            if rec:
                rec.last_line = last.text
            return "merged"
        self.ledger.append(Entry("user", spk, [body], now, now))
        if rec:
            rec.last_line = body
        self._cap()
        return "added"

    def add_agent(self, target: str | None, text: str) -> None:
        body = (text or "").strip()
        if not body:
            return
        now = self.clock()
        tgt = target or "user"
        self.ledger.append(Entry("assistant", tgt, [body], now, now))
        self.last_agent_at, self.last_agent_target = now, tgt
        self.last_reaction = ""
        self._cap()

    def _touch(self, spk: str, now: float) -> Speaker:
        rec = self.roster.get(spk)
        if rec is None:
            if len(self.roster) >= self.roster_cap:   # evict the longest-silent voice
                del self.roster[min(self.roster.values(), key=lambda r: r.last_seen).label]
            rec = self.roster[spk] = Speaker(spk, now, now)
        rec.last_seen = now
        return rec

    def _cap(self) -> None:
        while len(self.ledger) > self.ledger_cap:
            self.ledger.popleft()

    # ------------------------------------------------------------ rendering
    def _render(self, e: Entry, parts: list[str] | None = None) -> dict:
        text = " ".join(parts) if parts is not None else e.text
        if e.role == "assistant":
            return {"role": "assistant",
                    "content": f"[SPEAK to={e.speaker}] {clip_words(text, self.max_agent_words, 'head')}"}
        body = clip_words(text, self.max_entry_words)
        return {"role": "user", "content": f"[{e.speaker}] {body}" if e.speaker else body}

    def messages(self, current: str | None = None) -> tuple[list[dict], set]:
        """Budgeted verbatim window, oldest->newest, plus the set of speakers in view.

        If the newest ledger fragment IS the in-flight turn (committed before the
        generation started), it is left out; the LLM layer appends the current
        turn itself, so it would otherwise appear twice.
        """
        cur = _norm(_LABEL.sub("", current or ""))
        entries = list(self.ledger)
        out: list[dict] = []
        seen: set = set()
        used = 0
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            parts = None
            if i == len(entries) - 1 and cur and e.role == "user" and _norm(e.parts[-1]) == cur:
                if len(e.parts) == 1:
                    continue
                parts = e.parts[:-1]
            msg = self._render(e, parts)
            cost = est_tokens(msg["content"])
            if used + cost > self.budget_tokens and len(out) >= self.min_entries:
                break
            out.append(msg)
            used += cost
            if e.role == "user" and e.speaker:
                seen.add(e.speaker)
        out.reverse()
        return out, seen

    def state_note(self, current: str | None = None, in_view: set | None = None) -> str:
        """Compact picture of the room. Bounded by roster_cap lines."""
        now = self.clock()
        if not self.roster and self.last_agent_at is None:
            return ""
        in_view = in_view or set()
        lines = []
        people = sorted(self.roster.values(), key=lambda r: -r.last_seen)
        active = [r for r in people if now - r.last_seen <= 120]
        if people:
            lines.append(f"CALL STATE (voices tracked by diarization; S-labels are distinct "
                         f"people): {len(active)} active in the last 2 min.")
        for r in people:
            ago = now - r.last_seen
            when = ("speaking now" if r.label == current else
                    f"{int(ago)}s ago" if ago < 120 else f"quiet {int(ago // 60)}m")
            bits = [f"{r.label}: {when}", f"{r.turns} turns"]
            if r.addressed_agent:
                bits.append(f"called you {r.addressed_agent}x")
            if r.reactions:
                bits.append(f"reacted to you {r.reactions}x")
            line = "- " + ", ".join(bits)
            if r.label not in in_view and r.label != current and r.last_line:
                # scrolled out of the verbatim window: keep their last words
                line += f'; earlier said: "{clip_words(r.last_line, 16)}"'
            lines.append(line)
        if self.last_agent_at is not None:
            lines.append(f"You last spoke to {self.last_agent_target} "
                         f"{int(now - self.last_agent_at)}s ago.")
        if self.last_reaction:
            lines.append(self.last_reaction + ".")
        return "\n".join(lines)

    def stats(self) -> dict:
        msgs, _ = self.messages()
        return {"ledger": len(self.ledger), "speakers": len(self.roster),
                "window_msgs": len(msgs), "window_tokens": sum(est_tokens(m["content"]) for m in msgs),
                "state_tokens": est_tokens(self.state_note())}
