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
    r"^\s*(S\d+|[A-Z][a-z]{1,15})\s+(?:did|do|can|could|would|will|are|were|have|you|u)\b"
    r"(?=.*\b(?:you|your|u|ya)\b)")


def _is_agent_name(word: str, names: tuple[str, ...]) -> bool:
    return _is_agent_name_cached(word.lower(), tuple(names))


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


@functools.lru_cache(maxsize=4096)
def _is_common_word(w: str) -> bool:
    return bool(_zipf) and _zipf(w, "en") >= _COMMON_WORD_ZIPF


def _looks_like_other_name(word: str, names: tuple[str, ...]) -> bool:
    if re.fullmatch(r"S\d+", word):
        return True
    w = word.lower()
    return (w not in _NOT_NAMES and not w.endswith("ly")
            and not _is_common_word(w)
            and not _is_agent_name(word, names))


if _zipf:  # load the frequency table at import, not on the first live turn
    _is_common_word("the")

_SECOND_PERSON_CUE = re.compile(r"\b(?:you|your|you're|u|ya|yours)\b|\?", re.I)


def detect_other_addressee(text: str, names: tuple[str, ...]) -> Optional[str]:
    """Return the other person's name if the turn is vocatively addressed to
    someone who is not the agent, else None. Never fires if the agent is also
    named anywhere in the turn ("Alex, be quiet. Max, keep explaining.")."""
    if not text:
        return None
    body = _SPEAKER_LABEL_RE.sub("", text)
    for token in re.findall(r"[A-Za-z]+", body):
        if _is_agent_name(token, names) and token[0].isupper():
            return None
    if any(p.search(body) for p in _GENERIC_ALIAS_PATTERNS):
        return None
    for rx in (_LEAD_VOCATIVE_RE, _TAIL_VOCATIVE_RE, _LEAD_BARE_VOCATIVE_RE):
        m = rx.search(body)
        if not m or not _looks_like_other_name(m.group(1), names):
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
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        if (detect_silence_request(sentence) is not None
                and not detect_other_addressee(sentence, names)):
            return "explicit silence request"
    other = detect_other_addressee(text, names)
    if other:
        return f"addressed to {other}"
    return None
