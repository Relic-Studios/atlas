"""Thread arbiter: pick WHICH open thread the agent answers, not just the newest.

In a busy call every speaker's line used to abort the in-flight generation and
start a new one for the most recent line (live log: 335 generations started,
173 aborted, 110 answered). The agent chased every new voice, answered side
chatter ("Oh?", "What zone?"), and dropped the person who had actually asked it
something.

The arbiter keeps the latest open line per speaker, scores how much each one is
*for the agent*, and decides:
  * should a new line preempt the thread already being generated?  (hysteresis:
    another speaker must clearly outrank it, so chatter stops causing churn)
  * which open thread to answer when a turn fires (an older line that named the
    agent beats a newer side remark)
  * after answering, is there still a thread aimed at the agent to follow up on?

Pure + deterministic (no model, no audio) so the sims and the server share it.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import time

_LABEL = re.compile(r"^\s*\[(S\d+)\]\s*")
_SECOND_PERSON = re.compile(r"\b(you|your|you're|youre|u|ur)\b", re.IGNORECASE)

# Score weights (additive). Tuned in tests/test_threads.py scenarios.
W_NAMED = 3.0        # line uses the agent's name / "hey bot"
W_PARTNER = 1.5      # from the person the agent is mid-conversation with
W_REPLY = 1.0        # arrives right after the agent addressed this speaker
W_QUESTION = 0.7     # a question or second-person line not aimed at someone else
W_ROAST = 1.0        # roasting the agent (deserves a comeback)
W_OTHER = -3.0       # vocatively addressed to another named person
W_FILLER = -2.0      # laughter / backchannel only
DECAY_PER_S = 0.08   # older lines matter less
PREEMPT_MARGIN = 0.75
FOLLOW_UP_MIN = 3.0  # only follow up on lines clearly aimed at the agent
TTL_S = 20.0


def speaker_of(text: str) -> str:
    m = _LABEL.match(text or "")
    return m[1] if m else "user"


@dataclass
class Thread:
    speaker: str
    text: str
    at: float
    base: float
    reasons: tuple[str, ...] = ()

    def score(self, now: float) -> float:
        return self.base - DECAY_PER_S * max(0.0, now - self.at)


class ThreadArbiter:
    def __init__(self, names: tuple[str, ...] = (), clock=time.monotonic,
                 ttl_s: float = TTL_S):
        self.names = tuple(names)
        self.clock = clock
        self.ttl_s = ttl_s
        self.open: dict[str, Thread] = {}
        self.active: Thread | None = None
        self.last_target: str | None = None
        self.last_spoke_at = 0.0

    # ------------------------------------------------------------- scoring
    def score_text(self, text: str, speaker: str, partner: str | None) -> tuple[float, tuple]:
        from floor import names_agent, filler_only, detect_roast
        from conversation_dynamics import detect_other_addressee
        s, why = 1.0, []
        body = _LABEL.sub("", text or "")
        named = names_agent(text, self.names)
        if named:
            s += W_NAMED; why.append("named")
        if partner and speaker == partner:
            s += W_PARTNER; why.append("partner")
        if (self.last_target and speaker == self.last_target
                and self.clock() - self.last_spoke_at <= 12.0):
            s += W_REPLY; why.append("reply")
        other = None if named else detect_other_addressee(text, self.names)
        if other:
            s += W_OTHER; why.append(f"to:{other}")
        elif "?" in body or _SECOND_PERSON.search(body):
            s += W_QUESTION; why.append("question/you")
        if detect_roast(text, self.names):
            s += W_ROAST; why.append("roast")
        if filler_only(text):
            s += W_FILLER; why.append("filler")
        return s, tuple(why)

    # ------------------------------------------------------------ bookkeeping
    def _prune(self) -> None:
        now = self.clock()
        for spk in [k for k, t in self.open.items() if now - t.at > self.ttl_s]:
            del self.open[spk]

    def note(self, text: str, partner: str | None = None) -> Thread:
        """Record the latest line of a speaker as their open thread."""
        spk = speaker_of(text)
        base, why = self.score_text(text, spk, partner)
        prev = self.open.get(spk)
        # A refinement of an open line keeps its original timestamp (no decay
        # reset on every partial) but never loses a "named" boost it had.
        at = prev.at if prev and _extends(prev.text, text) else self.clock()
        if prev and _extends(prev.text, text) and prev.base > base:
            base, why = prev.base, prev.reasons
        t = Thread(spk, text, at, base, why)
        self.open[spk] = t
        self._prune()
        return t

    def start(self, thread: Thread) -> None:
        self.active = thread

    def should_preempt(self, new: Thread) -> bool:
        """Does `new` justify aborting the thread being generated right now?"""
        a = self.active
        if a is None or new.speaker == a.speaker:
            return True  # same speaker: the normal refinement/abort logic decides
        now = self.clock()
        return new.score(now) > a.score(now) + PREEMPT_MARGIN

    def best(self) -> Thread | None:
        self._prune()
        if not self.open:
            return None
        now = self.clock()
        return max(self.open.values(), key=lambda t: (t.score(now), t.at))

    def choose(self, latest: Thread) -> Thread:
        """Thread to answer when a turn fires on `latest`."""
        b = self.best()
        if b is None or b is latest:
            return latest
        now = self.clock()
        return b if b.score(now) > latest.score(now) + PREEMPT_MARGIN else latest

    def answered(self, target: str | None) -> None:
        """The agent finished speaking to `target` (or held)."""
        if self.active is not None:
            self.open.pop(self.active.speaker, None)
        if target and target not in ("self",):
            self.open.pop(target, None)
            self.last_target, self.last_spoke_at = target, self.clock()
        self.active = None

    def retire_active(self) -> None:
        """The active thread got a HOLD / was vetoed: don't re-answer it."""
        if self.active is not None:
            self.open.pop(self.active.speaker, None)
        self.active = None

    def follow_up(self) -> Thread | None:
        """An open line still clearly aimed at the agent after it spoke."""
        b = self.best()
        if b and b.score(self.clock()) >= FOLLOW_UP_MIN:
            return b
        return None

    def reset(self) -> None:
        self.open, self.active = {}, None
        self.last_target, self.last_spoke_at = None, 0.0

    def context_note(self, chosen: Thread) -> str:
        """Tell the model which thread it is answering when it isn't the newest."""
        newer = [t for t in self.open.values()
                 if t.speaker != chosen.speaker and t.at > chosen.at]
        if not newer:
            return ""
        who = ", ".join(sorted({t.speaker for t in newer}))
        return (f"THREAD: you are answering {chosen.speaker}'s line to you. "
                f"Later lines from {who} were side chatter -- don't answer those.")


def _extends(old: str, new: str) -> bool:
    a = set(_LABEL.sub("", old).lower().split())
    b = set(_LABEL.sub("", new).lower().split())
    return bool(a and b) and len(a & b) >= 0.6 * min(len(a), len(b))
