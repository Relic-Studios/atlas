"""Speak / stay-silent decision for Max, ported from Ivy's stance.py.

The Ivy system separates *being in* the conversation from *speaking in* it:
the agent is a ratified hearer by default — it has standing to speak any time,
but isn't obligated to. This module implements the deterministic appetite gate
that decides which side of that line each turn falls on.

Design constraint: ZERO added latency. Every signal here is
already live at the turn boundary — name mention in the transcript, silence
requests in the transcript, recent turn balance from history, and speaker
count from diarization. compute_appetite is pure arithmetic (microseconds),
never an LLM call, and never blocks the input path. The LLM still starts
generating during the silence window as before; this gate only decides
whether to actually fire / speak.

Signals (additive, faithful to Ivy's tuned constants):
    base ................................. 0.60
    direct address ("ivy"/"max") ..... +0.30
    address to someone else .............. -0.10
    ambient / nobody ..................... -0.40
    silence request ("shut up") .......... -0.70 (direct address overrides)
    talk-less request ("talk less") ...... threshold +0.10
    floor imbalance (agent over-talked) .. -0.05 per excess turn (cap -0.20)
    multi-party restraint ................ HARD BLOCK (threshold 999)
    stance bias (settled) ................ -0.20
    echo (person ignores agent) .......... threshold +/- up to 0.10

Fires when appetite > threshold (default 0.58).
"""
from __future__ import annotations

import functools
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Detectors (pure text → signal). Faithful ports of Ivy's patterns.
# ---------------------------------------------------------------------------

_SILENCE_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bshut\s*up\b", re.IGNORECASE), 30.0),
    (re.compile(r"\bshut\s+the\s+(?:fuck|hell)\s+up\b", re.IGNORECASE), 60.0),
    (re.compile(r"\bbe\s+quiet(?:er)?\b", re.IGNORECASE), 30.0),
    (re.compile(r"\bquiet\s+down\b", re.IGNORECASE), 30.0),
    (re.compile(r"\b(?:please\s+)?stop\s+talking\b", re.IGNORECASE), 45.0),
    (re.compile(r"\b(?:please\s+)?don['’]?t\s+(?:respond|reply|talk)\b", re.IGNORECASE), 30.0),
    # Only as a complete clause: "let me finish." / "let me talk, bro" -- NOT
    # "let me speak him up" / "let me speak in my American accent" (live false mutes).
    (re.compile(r"\blet\s+me\s+(?:think|finish|talk|speak)(?:\s+(?:bro|man|dude|please|real\s+quick))?\s*(?:[.,!?]|$)",
                re.IGNORECASE), 30.0),
    (re.compile(r"\bgive\s+me\s+a\s+(?:sec|second|moment|minute)\b", re.IGNORECASE), 30.0),
    (re.compile(r"\bhold\s+(?:on|up)\b", re.IGNORECASE), 20.0),
    (re.compile(r"\b(?:just\s+)?listen\s+(?:for\s+)?a\s+(?:sec|moment|minute)\b", re.IGNORECASE), 30.0),
    (re.compile(r"\bpause\s+for\s+a\s+(?:sec|moment|minute)\b", re.IGNORECASE), 30.0),
    # Longer "sit this out" requests: held until the agent is named again (floor.py quiet mode).
    (re.compile(r"\b(?:stay|keep|be)\s+(?:quiet|silent)\b", re.IGNORECASE), 180.0),
    (re.compile(r"\bmute\s+(?:yourself|urself)\b", re.IGNORECASE), 180.0),
    # Other ways people say "shut up" (same class, not phrase-specific fixes).
    (re.compile(r"\b(?:zip\s+it|pipe\s+down|hush|give\s+it\s+a\s+rest|put\s+a\s+sock\s+in\s+it)\b", re.IGNORECASE), 30.0),
    (re.compile(r"\bnobody\s+asked\s+(?:you|u)\b", re.IGNORECASE), 30.0),
    # "speak when you're spoken to" (live 10-02 Fae): quiet until named.
    (re.compile(r"\b(?:(?:only\s+)?(?:speak|talk)\s+(?:only\s+)?when\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to|(?:don(?:\'|’)?t\s+)?(?:speak|talk)\s+(?:until|unless|till)\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to|(?:stay\s+quiet|wait)\s+(?:until|till)\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to)\b", re.IGNORECASE), 180.0),
]

# Words allowed BEFORE a silence command in its own clause. Anything else means the
# command is embedded or reported: "I told my brother to shut up", "my mom says be quiet".
_COMMAND_LEAD = set("""
please pls plz just can could would will you u ya yo hey ok okay bro bruh dude man guys
now ai bot so oh and like i need want to seriously real god damn wait lol for
""".split())
# Literal (non-command) readings right after the match: "hold on to your seat",
# "keep quiet about the party".
_LITERAL_AFTER = re.compile(r"^\s*(?:to|about|at|when|while\s+(?:they|he|she|we))\b", re.IGNORECASE)


_PRONOUNS = {"my", "his", "her", "their", "our", "she", "he", "they", "we", "your"}


def _is_command(text: str, m: re.Match) -> bool:
    """Is this silence phrase an imperative at the start of its clause (optionally
    after a vocative / softener), rather than embedded or reported speech?"""
    before = re.split(r"[.!?;:]", text[max(0, m.start() - 80):m.start()])[-1]
    before = re.split(r",|\band\b", before)[-1]
    toks = re.findall(r"[A-Za-z']+", before)
    low = {t.lower() for t in toks}
    # "or I just keep quiet and wait" is about themselves, not an order (sim 11-01);
    # "I said shut up" / "I need you to be quiet" still are.
    if low & {"i", "we"} and not low & {"you", "u", "ya", "said", "told", "say"}:
        return False
    for k, tok in enumerate(toks):
        low = tok.lower()
        if low in _COMMAND_LEAD:
            continue
        if low in _PRONOUNS:
            return False
        # one vocative is fine: "ivy be quiet", "Max shut up", "bro shut up"
        if k == 0 or tok[:1].isupper():
            continue
        return False
    return not _LITERAL_AFTER.match(text[m.end():m.end() + 40])


def detect_silence_request(text: str) -> Optional[float]:
    """Return silence window seconds if the text asks the agent to be quiet."""
    if not text:
        return None
    hits = []
    for pat, d in _SILENCE_PATTERNS:
        for k, m in enumerate(pat.finditer(text)):
            if k >= 6:          # spam ("shut up shut up ...") needs no more evidence
                break
            if _is_command(text, m):
                hits.append(d)
                break
    return max(hits) if hits else None


_STRONG_SILENCE = re.compile(
    r"\bshut\s*(?:the\s+(?:fuck|hell)\s+)?up\b|\bstop\s+talking\b|\b(?:be|stay|keep)\s+(?:quiet|silent)\b"
    r"|\bmute\s+(?:yourself|urself)\b|\bdon['’]?t\s+(?:respond|reply|talk)\b|\bquiet\s+down\b"
    r"|\b(?:(?:only\s+)?(?:speak|talk)\s+(?:only\s+)?when\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to|(?:don(?:\'|’)?t\s+)?(?:speak|talk)\s+(?:until|unless|till)\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to|(?:stay\s+quiet|wait)\s+(?:until|till)\s+(?:you(?:\s*(?:\'|’)?re|\s+are|\s+were|\s+get)?\s+)?(?:spoken|talked)\s+to)\b",
    re.IGNORECASE)


def strong_silence_request(text: str) -> bool:
    """A real 'go quiet' command (arms quiet mode). "hold up", "let me think" and
    "give me a sec" pause one turn but never mute the agent for minutes."""
    return bool(text) and any(_is_command(text, m) for m in _STRONG_SILENCE.finditer(text))


_QUESTION_LEAD = re.compile(
    r"^\W*(?:(?:so|like|um|uh|wait|okay|ok|and|but|bro|dude|man)\W+)*"
    r"(?:what|who|why|how|when|where|which|are|is|was|were|do|does|did|can|could|would|will|should|have|has)\b"
    r"\W+\w+", re.I)


_QUESTION_MID = re.compile(
    r"(?:[,;:]\s*|\b(?:like|so|but|and|then|okay|ok|now|or)\s+)"
    r"(?:(?:what|who|why|how|when|where|which)(?:'s|'d|'re|'ll)?\s+\w+|(?:do|does|did|are|is|can|could|would|will|"
    r"should)\s+(?:you|u|ya|i|we|it|they|that|this)\b)"
    # ...or a request or a callback to what the agent said: "hold on, go ahead, roast the Prof" /
    # "hold up, you said earlier Orion..." / "hold up, say that trash line again"
    r"|\b(?:you\s+(?:just\s+)?(?:said|told)|say\s+(?:it|that|something)|tell\s+\w+|give\s+(?:us|me)|go\s+ahead|"
    r"go\s+on|roast|pick\s+(?:one|a\s+side)|choose|explain|guess|settle|decide|answer)\b", re.I)


def _pause_before_question(sentence: str, rest: str) -> bool:
    """'Wait, hold up, so are you like... a real bot?' leads into a question: a
    discourse marker, not 'be quiet' (sim 11-01: held as a silence request). Only
    the soft pauses (hold on/up, give me a sec, let me think); a real 'shut up' or
    'stay quiet' is a command whatever follows it."""
    if strong_silence_request(sentence):
        return False
    first = None
    for pat, _ in _SILENCE_PATTERNS:
        m = pat.search(rest)
        if m and (first is None or m.end() < first):
            first = m.end()
    after = rest[first:] if first is not None else ""
    # fix1 sim: "hold up dude, go back to the start, ... like what even is that" (no '?')
    return "?" in after or bool(_QUESTION_LEAD.match(after) or _QUESTION_MID.search(after))


_TALK_LESS_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\btalk\s+less\b", re.IGNORECASE), 60.0),
    (re.compile(r"\byou(?:'re|\s+are)?\s+talking\s+too\s+much\b", re.IGNORECASE), 90.0),
    (re.compile(r"\b(?:calm|chill|slow)\s+down\b", re.IGNORECASE), 45.0),
    (re.compile(r"\bnot\s+so\s+(?:much|fast|chatty)\b", re.IGNORECASE), 45.0),
    (re.compile(r"\bless\s+(?:chatty|talkative)\b", re.IGNORECASE), 60.0),
    (re.compile(r"\btoo\s+(?:chatty|talkative|much)\b", re.IGNORECASE), 60.0),
    (re.compile(r"\bquiet\s+(?:down|a\s+bit)\b", re.IGNORECASE), 30.0),
    (re.compile(r"\btone\s+it\s+down\b", re.IGNORECASE), 45.0),
    (re.compile(r"\bramble|rambling\b", re.IGNORECASE), 45.0),
]


def detect_talk_less_request(text: str) -> Optional[float]:
    """Return talk-less window seconds (soft mute) if requested."""
    if not text:
        return None
    hits = [d for pat, d in _TALK_LESS_PATTERNS if pat.search(text)]
    return max(hits) if hits else None


# Generic address aliases — every persona responds to these regardless of name.
# "hey ai", "hey bot", "hey assistant", "robot", etc. The address prefix
# ("hey"/"hi"/"yo"/"ok"/...) is REQUIRED — bare nouns like "chatbot"/"robot"
# mid-sentence are normal speech and must not trigger ("that chatbot was weird",
# "the robot arm moved"). Bare-name addressing ("ivy"/"max") is handled by
# the persona name above, so the generic path only needs prefixed address.
_GENERIC_ALIAS_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:hey|hi|hello|yo|ok|okay|sup|excuse\s+me)\s+(?:there\s+)?"
        r"(?:ai|a\.i\.|assistant|bot|robot|machine|chatbot|voice\s*ai|gpt)\b",
        re.IGNORECASE,
    ),
]


def is_direct_address(text: str, names: tuple[str, ...], generic: bool = True) -> bool:
    """True if the user is directly engaging the agent by name (or generic alias).

    A silence request in the SAME utterance dominates the name mention
    (e.g. "ivy shut up" is a request to stop, not to engage).
    """
    if not text:
        return False
    if detect_silence_request(text) is not None:
        return False
    for name in names:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            return True
    if generic:
        for pat in _GENERIC_ALIAS_PATTERNS:
            if pat.search(text):
                return True
    return False


def detect_user_correction(text: str) -> bool:
    """True if the user is correcting / pushing back on the agent's framing."""
    if not text:
        return False
    patterns = [
        r"\bI\s+(?:wasn'?t|was\s+not)\s+(?:talking|saying|asking)\b",
        r"\bthat'?s\s+not\s+what\s+I\b",
        r"\bwhere\s+(?:did|are)\s+you\s+(?:get|getting)\s+that\b",
        r"\byou'?re\s+not\s+(?:even\s+)?listening\b",
        r"\byou'?re\s+making\s+(?:that|this|stuff|it)\s+up\b",
        r"\bI\s+(?:didn'?t|did\s+not)\s+say\b",
        r"\bwhat\s+are\s+you\s+(?:talking\s+about|on\s+about)\b",
        r"\bstop\s+making\s+(?:things|stuff|that)\s+up\b",
        r"\byou\s+(?:just\s+)?made\s+(?:it|that|this)\s+up\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Addressee classification — deterministic replacement for the Thinker.
# ---------------------------------------------------------------------------
# The Ivy pipeline used a Thinker (an LLM) to tag each turn's addressee as
# "ivy" / "other" / "ambient", plus a should_reply flag. We dropped the LLM
# (VRAM), so compute_appetite was getting addressee="" every turn and never
# applied the ambient penalty — the agent replied to *everything*, including
# people talking ABOUT it ("she's not responding") or TO each other.
#
# classify_addressee() fills that hole with pure regex + word-position
# heuristics. Microseconds, no model, no latency. It's cruder than an LLM
# (can't catch sarcasm or implicit address) but gets the common cases right:
#   * vocative name / "hey ai"  -> "self"   (speak)
#   * second person ("you/your")-> "self"   (likely addressed)
#   * third person ("she/her"/bare name) -> "ambient" (hold)
#
# Distinguishes vocative ("hey ivy", "ivy, ...") from referential
# ("did ivy say that") by word position + punctuation.

# Third-person pronouns (referential — talking ABOUT someone, not to the agent).
_REFERENTIAL_RE = re.compile(
    r"\b(?:she|her|he|him|they|them|it'?s)\b", re.IGNORECASE
)
# Second-person pronouns (addressed — likely talking TO the agent).
_SECOND_PERSON_RE = re.compile(
    r"\b(?:you|your|you'?re|you'?ll|you'?ve|ya|u)\b", re.IGNORECASE
)


def classify_addressee(text: str, names: tuple[str, ...]) -> str:
    """Return 'self', 'other', or 'ambient' for a user turn.

    Deterministic replacement for the Thinker's addressee field. Called from
    should_speak() when no Thinker plan supplies an addressee.
    """
    if not text:
        return "ambient"

    # 1. Silence request in the same utterance dominates everything (hold).
    if detect_silence_request(text) is not None:
        return "ambient"

    lowered = text.lower()

    # 2. Vocative name — the agent's own name used as direct address.
    for name in names:
        nm = name.lower()
        # name at start followed by punctuation/whitespace+verb (vocative)
        if re.search(rf"^{re.escape(nm)}\b\s*[,!:.?]", lowered):
            return "self"
        # "hey ivy" / "yo ivy" / "excuse me ivy"
        if re.search(rf"\b(?:hey|hi|hello|yo|ok|okay|sup|excuse\s+me)\s+(?:there\s+)?{re.escape(nm)}\b", lowered):
            return "self"

    # 3. Generic alias ("hey ai", "ok robot") — address-prefixed by construction.
    #    Names are deliberately NOT checked here (is_direct_address would match
    #    bare "ivy" anywhere, erasing the vocative/referential distinction).
    for pat in _GENERIC_ALIAS_PATTERNS:
        if pat.search(text):
            return "self"

    # 4. Referential — third-person pronoun about the agent (or anyone).
    #    e.g. "she's not responding", "ivy said that", "it's fine".
    #    Bare name NOT in vocative position => referential, not address.
    if _REFERENTIAL_RE.search(text):
        return "ambient"
    for name in names:
        # name appears, but not as a vocative (checked above) => referential.
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            return "ambient"

    # 5. Second person => likely addressed to the agent (in 1:1, "you" = agent).
    if _SECOND_PERSON_RE.search(text):
        return "self"

    # 6. No clear signal => ambient (don't jump into an ambiguous turn).
    return "ambient"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpeakDecision:
    should_speak: bool
    appetite: float
    threshold: float
    reasons: list[str] = field(default_factory=list)

    def reason_str(self) -> str:
        return " ".join(self.reasons) if self.reasons else "(no signals)"


DEFAULT_APPETITE_THRESHOLD = 0.58


# ---------------------------------------------------------------------------
# Participation tracker (Gini coefficient) — faithful port of Ivy's.
# ---------------------------------------------------------------------------

class ParticipationTracker:
    """Speaking-time distribution over a rolling window → social modifier."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._utterances: deque[tuple[float, str, float]] = deque()
        self._agent_id = "agent"

    def log_utterance(self, speaker_id: str, duration_s: float, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.perf_counter()
        self._utterances.append((ts, speaker_id, duration_s))
        self._prune()

    def _prune(self) -> None:
        cutoff = time.perf_counter() - self._window
        while self._utterances and self._utterances[0][0] < cutoff:
            self._utterances.popleft()

    @staticmethod
    def _gini(values) -> float:
        if not values or sum(values) <= 0:
            return 0.0
        arr = sorted(values)
        n = len(arr)
        idx = range(1, n + 1)
        return float(sum((2 * i - n - 1) * v for i, v in zip(idx, arr)) / (n * sum(arr)))

    def get_social_modifier(self, active_user_ids: set[str], agent_id: str = "agent") -> float:
        self._agent_id = agent_id
        self._prune()
        if not self._utterances:
            return 1.0
        # warmup: don't suppress in the first 120s of a call
        elapsed = time.perf_counter() - self._utterances[0][0]
        if elapsed < 120.0:
            return 1.0
        durations: dict[str, float] = {uid: 0.0 for uid in active_user_ids}
        durations.setdefault(self._agent_id, 0.0)
        for _, spk, dur in self._utterances:
            durations[spk] = durations.get(spk, 0.0) + dur
        total = sum(durations.values())
        if total <= 0:
            return 1.0
        gini = self._gini(durations.values())
        agent_share = durations.get(self._agent_id, 0.0) / total
        n = max(len(active_user_ids), 1)
        fair = 1.0 / n
        mod = 1.0
        if gini < 0.35:
            mod *= 1.15
        if agent_share > fair:
            mod *= __import__("math").exp(-2.0 * (agent_share - fair))
        human_shares = [d / total for uid, d in durations.items() if uid != self._agent_id]
        if human_shares and max(human_shares) > 0.60:
            mod *= 0.3
        if n >= 4:
            mod *= 0.85
        if n >= 6:
            mod *= 0.85
        return max(0.05, min(mod, 1.5))


# ---------------------------------------------------------------------------
# ConversationDynamics — the gate, wired to Max's state.
# ---------------------------------------------------------------------------

class ConversationDynamics:
    """Per-call speak/not-speak decision, keyed off Max's live state.

    Two phases, deliberately separated:
      * `should_speak(...)` — PURE, no mutation. Called at the potential-
        sentence boundary (fires multiple times per turn) so it must be a
        read-only function of current state. Microseconds, no LLM.
      * `on_user_turn(...)` / `on_agent_turn(...)` — mutate state ONCE per
        resolved turn (silence windows, participation, echo). Called from
        on_before_final (USER TURN END), never from the hot path.
    """

    def __init__(self, agent_names: tuple[str, ...] = ()):
        self.agent_names = agent_names
        self.participation = ParticipationTracker()
        self._silence_until: float = 0.0
        self._talk_less_until: float = 0.0
        # per-speaker echo rate (P(respond | agent addressed them)), EMA
        self._echo_rate = 0.5
        self._echo_samples = 0

    def on_user_turn(self, text: str, speaker_id: str = "user", duration_s: float = 2.0) -> None:
        """Update silence/talk-less windows and participation — call once per turn."""
        now = time.time()
        if detect_silence_request(text) is not None:
            self._silence_until = now + detect_silence_request(text)
        if detect_talk_less_request(text) is not None:
            self._talk_less_until = now + detect_talk_less_request(text)
        self.participation.log_utterance(speaker_id, duration_s)

    def on_agent_turn(self, duration_s: float = 2.0) -> None:
        self.participation.log_utterance("agent", duration_s)

    def should_speak(
        self,
        text: str,
        *,
        speaker_id: str = "user",
        recent_history: Optional[list[dict]] = None,
        diarizer_participants: Optional[list[str]] = None,
        addressee: str = "",  # from a Thinker/plan if present, else derived
    ) -> SpeakDecision:
        """Decide whether the agent should speak this turn. PURE — no mutation.

        Pure arithmetic on live signals; no LLM, no blocking. Call this at the
        turn boundary BEFORE firing the LLM — if it returns should_speak=False,
        skip generation entirely (which is *faster*, not slower, since we save
        the whole LLM round-trip).

        recent_history: list of {"role": "user"|"assistant", "content": str}
        diarizer_participants: active speaker labels (["S1","S2"]) or None.
        """
        now = time.time()

        base = 0.60
        threshold = DEFAULT_APPETITE_THRESHOLD
        reasons: list[str] = [f"base={base:+.2f}"]

        # Direct address — deterministic classifier replaces the Thinker's
        # addressee tag. No LLM: classify_addressee() is pure regex.
        if not addressee:
            addressee = classify_addressee(text, self.agent_names)
        if addressee == "self":
            base += 0.30
            reasons.append("addr=agent(+0.30)")
        elif addressee == "other":
            base -= 0.10
            reasons.append("addr=other(-0.10)")
        elif addressee == "ambient":
            base -= 0.40
            reasons.append("addr=ambient(-0.40)")

        # Silence window (dominant when active) — direct address overrides
        if now < self._silence_until:
            if addressee == "self":
                reasons.append(f"silence=ACTIVE({self._silence_until - now:.0f}s) BUT addressed")
            else:
                base -= 0.70
                reasons.append(f"silence=ACTIVE({self._silence_until - now:.0f}s)(-0.70)")

        # Talk-less window (soft mute)
        if now < self._talk_less_until:
            threshold += 0.10
            reasons.append(f"talk_less=ACTIVE(thresh+0.10)")

        # Floor balance from history
        if recent_history:
            agent_t = sum(1 for m in recent_history if m.get("role") == "assistant")
            user_t = sum(1 for m in recent_history if m.get("role") == "user")
            excess = agent_t - user_t
            if excess >= 2:
                penalty = -0.05 * min(excess, 4)
                base += penalty
                reasons.append(f"floor=A{agent_t}/U{user_t}({penalty:+.2f})")

        # Multi-party restraint: 2+ humans talking to each other, not the agent
        if diarizer_participants and len(diarizer_participants) >= 2:
            if addressee != "self":
                return SpeakDecision(
                    should_speak=False, appetite=base, threshold=999.0,
                    reasons=reasons + ["multi_party(HARD BLOCK)"],
                )
            reasons.append("multi_party(BUT addressed)")

        # Echo: people who ignore the agent get a higher threshold
        if self._echo_samples >= 5:
            echo_delta = max(-0.05, min(0.10, (0.5 - self._echo_rate) * 0.20))
            threshold += echo_delta
            threshold = max(threshold, 0.30)
            if abs(echo_delta) > 0.02:
                reasons.append(f"echo={self._echo_rate:.2f}(thresh{echo_delta:+.2f})")

        # Participation modifier (Gini)
        mod = self.participation.get_social_modifier(set(diarizer_participants or [speaker_id]))
        if mod != 1.0:
            threshold *= mod
            reasons.append(f"participation(x{mod:.2f})")

        base = max(0.0, min(1.0, base))
        should = base > threshold
        return SpeakDecision(should_speak=should, appetite=base, threshold=threshold, reasons=reasons)

    # ── feedback hooks (call after a turn resolves) ──────────────────────
    def note_user_responded_to_agent(self, responded: bool) -> None:
        """Update echo EMA: did the addressed person respond back to us?"""
        alpha = 0.10
        self._echo_rate = alpha * (1.0 if responded else 0.0) + (1 - alpha) * self._echo_rate
        self._echo_samples += 1


# ---------------------------------------------------------------------------
# Pre-LLM veto: turns unambiguously addressed to ANOTHER named person.
# ---------------------------------------------------------------------------
# Sim evidence (held-out decision sim): the persona's "always engage"
# pull makes the LLM answer "Can you pass the salt, Jordan?" / "Oliver, could
# you" for the other person. A proper-noun vocative that is not the agent is a
# structural signal, so it is decided here (microseconds, no generation) the
# same way explicit silence requests are. Anything ambiguous falls through to
# the LLM. Whisper misspellings of the agent's own name ("Tomas", "Tom") count
# as the agent, so a direct address is never vetoed.

# Capitalized sentence-openers that are NOT names ("Honestly, ...", "Look, ...").
_NOT_NAMES = frozenset("""
okay ok alright right yeah yes yep yup no nope nah well so now then look listen
wait hey hi hello yo oh ah um uh hmm man dude bro bruh guys babe honey buddy
sorry please thanks thank anyway anyways also and but or like see sure fine
cool great nice wow damn god jesus lord seriously literally basically actually
honestly anyways besides plus first second finally again still though however
meanwhile otherwise sometimes today tomorrow yesterday tonight here there
what why how when where who which whatever everyone everybody someone somebody
anyone anybody nobody sir maam mom mum dad chat boys girls folks people team
lol lmao lmfao rofl haha hahaha ngl fr tbh imo idk bet yall bruv mate fam
aw aww awww mm mmh mhm mmm hm ugh oof welp yikes whoa woah dang ooh ohh ahh eh huh yay
ja nein jo nee si oui non da nyet tak nie evet hayir digga alter bruder bra wallah yalla
ha hah heh hehe rad sick dope lit sheesh wild crazy aight ight facts true word sweet awesome
legit deadass gg ez jeez geez gosh boom bingo touche sus cap
ey oye pues parce vale wey orale cara mano habibi habibti inshallah mashallah bhai yaar arre lah
cuz cuh dawg homie ay ayy ayo aye all
mira pero jaja jajaja jeje jejeje kkk kkkk rsrs bueno claro dale vamos chevere cabron chingon guey hermano
amigo compa tio tia pana mijo mija weon
""".split())
class _Nicknames:
    """Live view over agent_registry: agent id or display name -> nicknames."""
    def get(self, n, default=()):
        import agent_registry as _r
        for aid in _r.agent_ids():
            if n in (aid.lower(), _r.display_name(aid).lower()):
                return _r.nicknames(aid) or default
        return default


_AGENT_NICKNAMES = _Nicknames()
_SPEAKER_LABEL_RE = re.compile(r"^\s*\[S\d+\]\s*")
_LEAD_VOCATIVE_RE = re.compile(r"^\s*(S\d+|[A-Z][a-z]{1,15})\s*,\s+\S")
_TAIL_VOCATIVE_RE = re.compile(r",\s*(S\d+|[A-Z][a-z]{1,15})\s*[?.!]*\s*$")
# Comma-less lead vocative as Whisper often writes it: "Maya did you bring the
# charger". Requires an aux/"you" right after the name AND a 2nd-person word.
_LEAD_BARE_VOCATIVE_RE = re.compile(
    r"^\s*(S\d+|[A-Z][a-z]{1,15})\s+(?:(?:you|u|ya)\b|"
    r"(?:did|do|can|could|would|will|are|were|have)\b(?=.*\b(?:you|your|u|ya)\b))")
# ("Dana you still have my bottle?" has one 'you', and it IS the cue: first_run sim B 10-04.)


def _is_agent_name(word: str, names: tuple[str, ...]) -> bool:
    return _is_agent_name_cached(word.lower(), tuple(names))


@functools.lru_cache(maxsize=64)
def role_words(names: tuple) -> frozenset:
    """How people address the agent when they don't use its name, from its pack 'what'
    ("a ninja from the Hidden Leaf" -> ninja; "a green dinosaur" -> dinosaur, dino): the head
    noun of each phrase. fix2 sim: "Yeah, ninja, you're the local" was held as "addressed to
    ninja". Address positions only (floor.names_agent): as a plain noun it isn't its name."""
    out = set()
    try:
        import agent_registry as _r
        for aid in _r.agent_ids():
            if aid.lower() not in names and _r.display_name(aid).lower() not in names:
                continue
            what = re.split(r"\bfrom\b|\(", _r.identity(aid).get("what") or "")[0]   # not "...from the Hidden Leaf"
            for seg in what.split(","):
                ws = re.findall(r"[A-Za-z]+", seg)
                w = ws[-1].lower() if ws else ""
                if len(w) >= 4 and w not in names and w not in _NOT_NAMES and not _is_common_word(w):
                    out.add(w)
                    if len(w) >= 7:
                        out.add(w[:4])          # dinosaur -> dino
    except Exception:  # noqa: BLE001
        pass
    return frozenset(out)


@functools.lru_cache(maxsize=4096)
def _is_agent_name_cached(w: str, names: tuple) -> bool:
    from difflib import SequenceMatcher
    for name in names:
        n = name.lower()
        if w == n or w in _AGENT_NICKNAMES.get(n, ()):
            return True
        if SequenceMatcher(None, w, n).ratio() >= 0.72:  # tomas, thoma, lunna
            return True
    return False


try:  # general "is this a common English word" test; replaces growing word lists
    from wordfreq import zipf_frequency as _zipf
except Exception:  # noqa: BLE001
    _zipf = None
# Common function words / verbs sit >= 5.1 ("are" 6.7, "okay" 5.1, "fuck" 5.4);
# given names sit below ("mark" 5.0, "alex" 4.6, "maya" 3.9). The one loss
# ("Will") just falls through to the model, which is the safe direction.
_COMMON_WORD_ZIPF = 5.1


# Fallback when wordfreq is unavailable (public test run 10-05: without it,
# "Did you see what she did?" read as "addressed to Did" and was vetoed).
# Sentence-initial words that are never a vocative name; failing open here
# silences real questions, so keep this list conservative but present.
_COMMON_FALLBACK = frozenset("""
a an the and but or so if then yes yeah yep no nope ok okay hey hi hello yo
do does did done can could will would shall should may might must is are was
were be been am have has had get got go going went come came let lets let's
what who whom whose why when where which how how's what's who's where's
i me my we us our you your he she it they them this that these those there
here now just really actually like well oh ah um uh hmm wait look see
please thanks thank sorry maybe also anyway though still even only all any
some every each both not never always don't doesn't didn't can't won't isn't
""".split())


@functools.lru_cache(maxsize=4096)
def _is_common_word(w: str) -> bool:
    if _zipf:
        return _zipf(w, "en") >= _COMMON_WORD_ZIPF
    return w in _COMMON_FALLBACK


def _adverb(w: str) -> bool:
    """'honestly', 'basically' -- not 'Wally', 'Kelly', 'Emily' (fix1 sim: "what do you think,
    Wally" never counted as address because every -ly word was ruled out)."""
    if not w.endswith("ly"):
        return False
    if not _zipf:
        return True
    if _is_common_word(w):
        return True
    stems = [w[:-2], w[:-4] if w.endswith("ally") else "", w[:-3] + "e" if w.endswith("bly") else "",
             w[:-3] + "y" if w.endswith("ily") else ""]
    return any(s and len(s) >= 4 and _zipf(s, "en") >= 3.5 for s in stems)


def _looks_like_other_name(word: str, names: tuple[str, ...]) -> bool:
    if re.fullmatch(r"S\d+", word):
        return True
    w = word.lower()
    # "Flipped, my ass, man, you said..." (sim 11-01): a past tense or -ing word
    # leading a line is a reaction, not a name; real names the room uses are
    # trusted before this check (_alias_ok).
    return (w not in _NOT_NAMES and not _adverb(w)
            and not (len(w) > 4 and w.endswith(("ed", "ing")))
            and not _is_common_word(w)
            and not _is_agent_name(word, names))


# Rooms switch language (live 10-31: a German side chat held as "addressed to Ja" x7).
# A vocative candidate is judged in the language the rest of the turn is in, so
# "Ja, das ist sehr." isn't a name while "Kim, kannst du helfen?" still is.
_TURN_LANGS = ("en", "de", "es", "fr", "pt", "it", "nl", "tr", "pl")


def _turn_language(body: str) -> Optional[str]:
    """The turn's language when it clearly isn't English, else None."""
    if not _zipf:
        return None
    words = re.findall(r"[^\W\d_]+", body.lower())[1:]   # skip the vocative slot
    if len(words) < 3:
        return None
    mean = {lang: sum(_zipf(w, lang) for w in words) / len(words) for lang in _TURN_LANGS}
    best = max(mean, key=lambda k: mean[k])
    return best if best != "en" and mean[best] - mean["en"] >= 0.75 else None


def _common_in(word: str, lang: Optional[str]) -> bool:
    return bool(lang) and _zipf is not None and _zipf(word.lower(), lang) >= _COMMON_WORD_ZIPF


if _zipf:  # load the frequency tables at import, not on the first live turn
    _is_common_word("the")
    for _lang in _TURN_LANGS:
        _zipf("the", _lang)

_SECOND_PERSON_CUE = re.compile(r"\b(?:you|your|you're|u|ya|yours)\b|\?", re.I)


# ---- aliases / gamer tags (live 10-06: "Hey Perma, I think you were muted
# before. Did you, are you back?" got "Yeah, I'm back." from Max). People in
# these rooms go by handles, not given names, and Whisper writes them after a
# greeting, sometimes lowercase. Names the room has actually used (learned
# intros, heard vocatives) are trusted even when they are common words
# ("Ghost", "Will", "Toast").
_ROOM_NAMES: dict = {}          # lower-case name -> last time seen
_ROOM_NAME_TTL = 6 * 3600.0


def note_room_name(name: str, when: Optional[float] = None) -> None:
    """Record a human name/alias used in this room (people.NameBook feeds it)."""
    import time as _t
    n = (name or "").strip().lower()
    if 2 <= len(n) <= 24 and n not in _NOT_NAMES:
        new = n not in _ROOM_NAMES
        _ROOM_NAMES[n] = _t.time() if when is None else when
        if new:
            try:   # the pending file keeps the call's names (docs/designs/experience_memory.md, D2)
                import call_boundary as _cbn
                _cbn.name_event("room", (name or "").strip())
            except Exception:  # noqa: BLE001
                pass


def clear_room_names() -> None:
    """Call end (docs/designs/experience_memory.md, S4): names used in this call do not carry into the next one."""
    _ROOM_NAMES.clear()


def room_names() -> list:
    """Names the room has used in this call (call_boundary.call_names reads them)."""
    return list(_ROOM_NAMES)


try:
    import call_boundary as _cb
    _cb.register("conversation_dynamics.clear_room_names", clear_room_names, order=6)
    _cb.BOUNDARY.add_name_source("room_names", room_names)
except Exception:  # noqa: BLE001
    pass


def _is_room_name(word: str) -> bool:
    import time as _t
    at = _ROOM_NAMES.get((word or "").lower())
    return at is not None and _t.time() - at < _ROOM_NAME_TTL


_GREETING_LEAD_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|yo|oi|ay+|ok|okay|alright|aight|so|wait|oh\s+my\s+(?:god|gosh)|omg|oh|and|but|well|"
    r"listen|look|bro|dude|man|dale|vale|órale|orale|pues)\b[,!.]?\s+)+", re.I)
# Name after a greeting, any case: "hey perma, ..." / "Yo Kraken did you ..." / "yo Kayy what song"
# (offcall sim 11-02: comma-less question to a friend, answered by the agent).
_GREETED_NAME_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9]{1,15})(?:(?P<p>\s*[,!?]|\s*$)|\s+(?:you|u|ya|your|you're)\b|"
    r"\s+(?:did|do|can|could|would|will|are|were|have)\s+(?:you|u|ya)\b|"
    r"\s+(?:what|what's|whats|who|who's|where|where's|why|how|how's|when|which)\b)", re.I)
# Lines for people who aren't in the call (offcall sim 11-02): a streamer talking to their chat
# ("vale chat, chat, calm down", "gracias for the follow, Lucas", "chat says I'm bad") and someone
# answering family at home ("mom I'm on a call"). Only checked when the agent isn't named.
_STREAM_CHAT_RE = re.compile(
    r"\b(?:thanks|thank\s+you|thx|ty|gracias|merci|danke)\s+(?:so\s+much\s+)?for\s+(?:the\s+)?(?:follow|sub|subs|"
    r"subscription|raid|bits|prime|gifted|gift|dono|donation|host)\b"
    r"|(?:^|[,.!?;:]\s*|\b(?:okay|ok|vale|yo|hey|alright|aight|no|yes|yeah|nah|bro|wait|pues|ay)\s+)chat\b(?!\s*gpt)",
    re.I)
_HOUSEHOLD_RE = re.compile(
    r"(?:^|[,.!?;:]\s*|\b(?:hey|hi|hello|yo|ay|oh)\s+)(mom|mum|mommy|mama|dad|daddy|papa|grandma|nana|abuela|abuelo)\b"
    r"(?=\s*(?:[,!.?]|$|i'm\b|im\b|i'll\b|i\s|wait\b|hold\b|one\s+sec|give\b|what\b|yes\b|yeah\b|no\b|okay\b|ok\b|"
    r"stop\b|leave\b|please\b|coming\b))", re.I)
# The agent called "AI" in address position wins over everything else in the line: "chat wants
# to know, what do you think, AI, ..." (gaming sim 11-02: held as "addressed to chat").
_AI_VOCATIVE_RE = re.compile(r"(?:^|[,.!?;:]\s*|\b(?:hey|hi|yo|okay|ok|alright|aight|so|oi|oye)\s+)"
                             r"(?:ai|a\.i\.|bot|robot)\s*(?:[,?!.]|$)", re.I)
# Someone answering a phone or a person at home: "I'm on a call", "estoy en una llamada".
_ON_A_CALL_RE = re.compile(r"\b(?:i'?m|i\s+am)\s+(?:on|in)\s+a\s+call\b|\bestoy\s+en\s+una\s+llamada\b"
                           r"|\b(?:i'?ll|i\s+go|let\s+me)\s+call\s+you\s+back\b", re.I)
# Mid-sentence vocative set off by commas, aimed at someone:
# "...and now I've heard, Perma, you were talking about your brain."
_MID_VOCATIVE_RE = re.compile(
    r",\s*([A-Z][A-Za-z0-9]{1,15})\s*,\s*(?:you|your|you're|did|do|are|were|can|"
    r"could|would|will|have|what|what's|who|why|how|when|where|which|is|was|should|tell)\b")


# Words people use for the agent itself ("Yo, AI!", "hey Atlas, ...").
_AGENT_GENERIC = frozenset({"ai", "atlas", "bot", "robot", "computer", "assistant"})


def _alias_ok(word: str, names: tuple[str, ...]) -> bool:
    if _is_agent_name(word, names) or word.lower() in _AGENT_GENERIC:
        return False
    if _is_room_name(word):
        return True
    return _looks_like_other_name(word, names)


_ROOM_VOC_LEAD = re.compile(
    r"\b(?:listen|see|look|yo|hey|hi|bye|later|night|goodnight|cya|thanks|thank\s+you|but|nah|man|bro|dude|"
    r"right|okay|ok|so|like|you|now|come\s+on|real|honestly|anyway|anyways|well|alright|aight|oh)\s*$", re.I)
_ROOM_VOC_AFTER = re.compile(r"\s*(?:[,?!]|\.(?:\s|$)|$)")
_GREET_JUST_BEFORE = re.compile(r"(?:^|[,.!?;]\s*|\s)(?:yo|hey|hi|hello|oi|ayo)\s+$", re.I)
_FAREWELL_NAME_RE = re.compile(r"^\W*(?:bye|later|night|goodnight|cya)\s+([a-z][a-z0-9_]{2,15})\b")
_FAREWELL_LEAD = re.compile(r"\b(?:bye|later|night|goodnight|cya)\s*$", re.I)
FAREWELL_START = re.compile(r"^\W*(?:bye|later|night|goodnight|cya|see\s+ya|see\s+you)\b", re.I)
_LOWER_MID_VOC_RE = re.compile(r",\s*([a-z][a-z0-9_]{2,15})\s*,\s*(?:you(?!\s+(?:know|see|feel|get|mean)\b)|your|you're|u)\b")
LOWER_NAME_ZIPF = 4.6       # a lower-case word this rare in a vocative slot is a handle, not a noun


def _room_vocative(body: str, names: tuple[str, ...]) -> Optional[str]:
    """A name set off as address anywhere in the line (sim 11-01: 40 of 98 replies
    cut into side talk that named the other person like this):
      * a name this room already uses: "Pues listen, Flash, dogs are loyal" /
        "You hear that, Flash?" / "Bye Peach,"
      * an unknown word capitalised mid-sentence between commas (speech recognition
        capitalises names, not nouns): "You see, Jefe, even the ninja grasps it"
        -- not apposition like "Lisbon, Porto, and Faro"
      * a rare lower-case word in a farewell or before 'you': "bye peach, ..." /
        "but tell me, beast, you scream too"."""
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{1,23}(?!['’])", body):
        w = m.group(0)
        if w.lower() in _NOT_NAMES or w.lower() in _AGENT_GENERIC or _is_agent_name(w, names):
            continue
        before = body[:m.start()]
        if _is_room_name(w):
            lead = (not before.strip() or re.search(r"[,.!?;]\s*$", before) is not None
                    or _ROOM_VOC_LEAD.search(before) is not None)
            # "yo Maria wine at midnight, that's a vibe" (no comma came through): a greeting
            # right before a name the room uses is address on its own
            if _GREET_JUST_BEFORE.search(before):
                return w
            if lead and _ROOM_VOC_AFTER.match(body[m.end():]):
                if _FAREWELL_LEAD.search(before) and _SECOND_PERSON_CUE.search(_after_sentence(body, m.end())):
                    continue          # "later Peach. so like, ... what's the first track": the rest isn't Peach's
                return w
            continue
        mid = re.search(r"\w[^\w.!?]*,\s*$", before) is not None     # after a comma, not a sentence start
        if (mid and w[0].isupper() and w[1:].islower() and _ROOM_VOC_AFTER.match(body[m.end():])
                and _looks_like_other_name(w, names) and (not _zipf or _zipf(w.lower(), "en") < LOWER_NAME_ZIPF)):
            prev = re.findall(r"([A-Za-z']+)\s*,\s*$", before)
            if not (prev and prev[0][:1].isupper() and prev[0].lower() not in _NOT_NAMES):
                return w
    for rx in (_FAREWELL_NAME_RE, _LOWER_MID_VOC_RE):
        m = rx.search(body)
        if m and m.group(1) not in _NOT_NAMES and m.group(1) not in _AGENT_GENERIC \
                and _looks_like_other_name(m.group(1), names) \
                and (not _zipf or _zipf(m.group(1), "en") < LOWER_NAME_ZIPF):
            # "later Peach. so like, ... what's the first track": the goodbye closes Peach's
            # exchange and the rest is for whoever is left (fix1 replay)
            if rx is _FAREWELL_NAME_RE and _SECOND_PERSON_CUE.search(_after_sentence(body, m.end())):
                continue
            return m.group(1)
    return None


def _after_sentence(body: str, pos: int) -> str:
    """Text after the sentence that contains `pos`."""
    m = re.search(r"[.!?]\s", body[pos:])
    return body[pos + m.end():] if m else ""


_OWN_NAME_RE = re.compile(r"\b(?:my\s+name(?:'s|\s+is)|name's|call\s+me|i'?m|i\s+am)\s+([A-Za-z][\w'-]{1,23})",
                          re.I)


def detect_other_addressee(text: str, names: tuple[str, ...]) -> Optional[str]:
    """Return the other person's name if the turn is vocatively addressed to
    someone who is not the agent, else None. Never fires if the agent is also
    named anywhere in the turn ("Alex, be quiet. Max, keep explaining."), or
    for the speaker's own name ("Bob, parce, my name is Bob, I told you", sim 11-01)."""
    who = _detect_other_addressee(text, names)
    if who and who.lower() in {m.group(1).lower() for m in _OWN_NAME_RE.finditer(text or "")}:
        return None
    if who and who.lower() in role_words(tuple(n.lower() for n in names)):
        return None               # "hey ninja, ..." to Kai is the agent itself
    if who and _not_really_address(_SPEAKER_LABEL_RE.sub("", text or ""), who):
        return None
    return who


_REACTION_ONLY = re.compile(r"\s*,\s*(?:ok|okay|exactly|right|yeah|yes|yep|no|nope|true|sure|nice|wow|cool|correct|"
                            r"same|facts|damn|lol|indeed)\s*(?:[.!?]|$)", re.I)


def _not_really_address(body: str, who: str) -> bool:
    """A capitalised word before a comma that isn't someone being spoken to (live 11-02: each of these
    held the line, and the speaker's next lines were held as "talking to someone else" -- a guest
    asking Ivy about GPUs got no answer for a minute): a repeated word ("Admin, admin, I'm the
    admin", "Betra, betra"), a list ("Syria, Lebanon, Palestine and Jordan"), or a word the room has
    never used as a name followed by only a reaction ("Deadlock, exactly.", "Quinn, ok." for Qwen)."""
    if who.lower() in ("chat", "phone") or _HOUSEHOLD_RE.search(who + ","):
        return False
    w = re.escape(who)
    # doubled AND used as a plain noun or spelled out: "Admin, admin, I'm the admin" / "Betra, betra,
    # ja. P-E-T-R-E" -- but "Bob, Bob, run to me" is address (gaming sims: 33 such lines)
    noun_use = re.search(r"\b(?:the|a|an|my|your|our|this|that)\s+" + w + r"\b", body, re.I)
    spelled = re.search(r"\b[A-Z](?:-[A-Z]){2,}\b", body)
    for m in re.finditer(r"\b" + w + r"\b", body):
        after = body[m.end():]
        if (noun_use or spelled) and re.match(r"\s*,?\s*" + w + r"\b", after, re.I):
            return True
        if re.match(r"\s*,\s*[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+|\s+and\s+[A-Z][a-z]+)", after):
            return True
        if not _is_room_name(who) and _REACTION_ONLY.match(after):
            return True
    return False


def _detect_other_addressee(text: str, names: tuple[str, ...]) -> Optional[str]:
    if not text:
        return None
    body = _SPEAKER_LABEL_RE.sub("", text)
    lang = _turn_language(body)
    for token in re.findall(r"[A-Za-z]+", body):
        if _is_agent_name(token, names) and (token[0].isupper()
                                             or token.lower() in tuple(n.lower() for n in names)):
            return None
    if any(p.search(body) for p in _GENERIC_ALIAS_PATTERNS) or _AI_VOCATIVE_RE.search(body):
        return None
    if _STREAM_CHAT_RE.search(body):
        return "chat"
    hm = _HOUSEHOLD_RE.search(body)
    if hm:
        return hm.group(1)
    if _ON_A_CALL_RE.search(body):
        return "phone"
    # Greeting + handle, per sentence ("I'm here. Hey Perma, you back?").
    for sent in re.split(r"(?<=[.!?])\s+", body):
        g = _GREETING_LEAD_RE.match(sent)
        if not g:
            continue
        m = _GREETED_NAME_RE.match(sent[g.end():])
        if not m or not (_alias_ok(m.group(1), names) and (_is_room_name(m.group(1)) or not _common_in(m.group(1), lang))):
            continue
        w = m.group(1)
        # A bare exclamation ("Well, voila!", "Yo, weee!") only counts as a
        # name when it is capitalised.
        rest = sent[g.end() + m.end():].strip()
        if m.group("p") is not None and not rest and w[0].islower() and not _is_room_name(w):
            continue
        # lower-case after a greeting is often just a noun ("so pizza, you in?");
        # only rare words (handles like "perma") or names the room used count.
        # Without wordfreq we can't tell a handle from a noun, so only names the
        # room already used count (errs toward answering, never toward silence).
        if w.islower() and not _is_room_name(w) and (not _zipf or _zipf(w.lower(), "en") >= 3.2):
            continue
        return w
    rv = _room_vocative(body, names)
    if rv:
        return rv
    m = _MID_VOCATIVE_RE.search(body)
    if m and (_alias_ok(m.group(1), names) and (_is_room_name(m.group(1)) or not _common_in(m.group(1), lang))):
        # "Paris, France, you should go" is apposition, not address: skip when
        # the word before the first comma is itself capitalised mid-sentence.
        prev = re.findall(r"([A-Za-z']+)\s*$", body[:m.start()])
        start = body[:m.start()].rstrip()
        sent_start = not start or start[-1] in ".!?" or len(start.split()) <= 1
        if not (prev and prev[0][0].isupper() and not sent_start):
            return m.group(1)
    for rx in (_LEAD_VOCATIVE_RE, _TAIL_VOCATIVE_RE, _LEAD_BARE_VOCATIVE_RE):
        m = rx.search(body)
        if not m or not (_alias_ok(m.group(1), names) and (_is_room_name(m.group(1)) or not _common_in(m.group(1), lang))):
            continue
        # A comma-less "Will you ..." stays a question to the room even when a
        # friend is called Will: room-name trust only applies to rare words here.
        if rx is _LEAD_BARE_VOCATIVE_RE and not _looks_like_other_name(m.group(1), names):
            continue
        # A trailing capitalised word ("Yeah, Munich.") is only address when the
        # line is actually aimed at someone: a question or a 2nd-person word.
        if rx is _TAIL_VOCATIVE_RE and not _SECOND_PERSON_CUE.search(body[:m.start()]):
            continue
        return m.group(1)
    return None


# Reported speech: "Ben said Max tell us a joke" quotes an invocation made
# by someone else (STT drops the quote marks). Third-person subject + reporting
# verb BEFORE the first agent-name mention => content, not address.
# "I said Max..." (repeating one's own call) and "Max, Ben said..." are
# NOT matched and still reach the LLM.
_REPORTED_RE = re.compile(
    r"\b(?:(?!i\b|we\b)[a-z]+)\s+(?:said|says|told\s+\w+|asked|was\s+like|goes|went)\b",
    re.IGNORECASE)


def reported_invocation(text: str, names: tuple[str, ...]) -> bool:
    """True when EVERY mention of the agent's name sits inside a reported
    clause ("Ben said Max ..."). One direct mention anywhere => False."""
    body = _SPEAKER_LABEL_RE.sub("", text or "")
    mentions = [m for m in re.finditer(r"[A-Za-z']+", body)
                if _is_agent_name(m.group(), names)]
    if not mentions:
        return False
    for m in mentions:
        # only look back within the mention's own sentence
        sent_start = max(body.rfind(c, 0, m.start()) for c in ".!?,;") + 1
        before = body[sent_start:m.start()]
        r = None
        for r in _REPORTED_RE.finditer(before):
            pass
        if r is None:
            return False
        subj = r.group().split()[0].lower()
        if subj in {"you", "u"} or _is_agent_name(subj, names):
            return False
    return True


def pre_llm_veto(text: str, names: tuple[str, ...]) -> Optional[str]:
    """Single production gate run before any generation. Returns a reason
    string when the turn must be HOLD without calling the LLM, else None."""
    body = _SPEAKER_LABEL_RE.sub("", text or "")
    # Quoted speech is content, not a command: 'translate "please stop talking"'.
    body = re.sub(r'"[^"]*"|“[^”]*”', '""', body)
    # A silence request only counts when its own sentence isn't addressed to
    # someone else: "Alex, be quiet. Max, keep explaining." must reach the LLM.
    if reported_invocation(text, names):
        return "reported speech"
    sentences = re.split(r"(?<=[.!?])\s+", body)
    for k, sentence in enumerate(sentences):
        if (detect_silence_request(sentence) is not None
                and not detect_other_addressee(sentence, names)
                and not _pause_before_question(sentence, " ".join(sentences[k:]))):
            return "explicit silence request"
    other = detect_other_addressee(text, names)
    if other:
        return f"addressed to {other}"
    return None
