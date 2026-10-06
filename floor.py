"""Deterministic turn management for multi-party group conversations.

Everything here is pure text/time logic -- microseconds, no model -- and sits
in front of / alongside the Model SPEAK/HOLD decision:

* filler_only()          -> laughter/filler ("lol", "haha", "mhm") never costs an LLM call
* incomplete_fragment()  -> fast-VAD cut mid-clause ("so I was thinking maybe") waits
* real_interruption()    -> barge-in only for real words, not backchannels
* ConversationFloor      -> who Max is currently talking with; a short per-turn
                            room note keeps him on that thread and off side chatter
* detect_roast()/vibe    -> banter steering, kept OUT of the persona files
"""
from __future__ import annotations

import os
import re
import time

_WORD = re.compile(r"[a-z0-9']+")

# Pure non-lexical reactions. Deliberately excludes yes/yeah/ok/no: those can be
# real answers to Max's question and must still reach the LLM.
FILLER = {
    "lol", "lmao", "lmfao", "rofl", "haha", "hahaha", "hahahaha", "ha", "hah", "heh",
    "hehe", "mhm", "mm", "mmm", "hmm", "hm", "uh", "um", "uhh", "umm", "ah", "oh",
    "ooh", "huh", "pfft", "psh", "bruh", "sheesh", "damn", "dang", "wow", "whoa",
}
# Backchannels: what people say WHILE someone else talks. Never a barge-in.
BACKCHANNEL = FILLER | {
    "yeah", "yea", "yep", "yup", "ya", "ok", "okay", "right", "true", "nice", "sure",
    "facts", "real", "fr", "bro", "dude", "man", "exactly", "totally", "no", "nah",
    "wait", "what", "really", "cool", "gotcha", "word",
}
_DANGLING = {
    "and", "but", "or", "so", "because", "cause", "cuz", "like", "the", "a", "an",
    "to", "of", "with", "if", "then", "that", "which", "who", "when", "my", "your",
    "his", "her", "their", "our", "is", "was", "were", "are", "um", "uh", "just",
    "maybe", "gonna", "wanna", "i", "we",
    "she", "he", "they", "could", "would", "should", "can", "will", "might", "gotta",
    # NOT "you"/"on"/"in"/"about"/"for"/"from": ordinary sentence endings
    # ("thank you", "come on", "what's that for").
}

_DANGLING |= {"probably", "definitely", "also", "really", "still"}
_WH = {"what", "who", "whom", "which", "where", "how", "whatever"}
_PREPS = {"with", "like", "about", "of", "for", "at", "on", "in", "from", "to", "by", "than"}
_STRANDABLE = _PREPS | {"up"}
_CLAUSE_OPENERS = {"if", "when", "because", "cause", "cuz", "and", "but", "so", "unless",
                   "while", "before", "after", "until", "than", "whether"}


def words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def strip_label(text: str) -> str:
    return re.sub(r"^\s*\[(?:S\d+|user|self)\]\s*", "", text or "")


def names_agent(text: str, names: tuple[str, ...]) -> bool:
    """The agent's name (or a nickname / near-miss STT spelling, or "hey bot")
    appears outside quotes."""
    from conversation_dynamics import _GENERIC_ALIAS_PATTERNS, _is_agent_name
    body = re.sub(r'"[^"]*"|“[^”]*”', '', strip_label(text))
    if any(p.search(body) for p in _GENERIC_ALIAS_PATTERNS):
        return True
    return any(_is_agent_name(t, names) for t in re.findall(r"[A-Za-z]{3,}", body))


# Whole-utterance Whisper hallucinations (noise/silence/laughs), live 09-29 17:54:
# "You." -> agent answered "No, but Moth has opinions about it."
STT_HALLUCINATIONS = {
    "you", "laughter", "laughs", "laughing", "music", "applause", "silence",
    "thank you", "thanks for watching", "bye", "inaudible", "noise",
}
# Pure reactions: the room reacting, not a turn handed to the agent
# (live 10-02: "Oh my god." -> "Yeah, it's worse than you think.").
REACTIONS = {
    "oh my god", "oh my gosh", "my god", "oh god", "oh lord", "oh my lord",
    "holy shit", "holy crap", "holy fuck", "jesus", "jesus christ", "omg",
    "oh wow", "wow okay", "yay", "oh no", "oh shit", "bro", "dude", "yo",
    "lord", "goddamn", "goddammit", "no way", "oh wow okay",
}


def stale_draft(draft: str, final: str) -> bool:
    """True if a reply drafted from an early fragment would answer the wrong
    thing now that the whole turn is known (live 10-02: drafted on "Bro.",
    voiced "Bro, I'm Max" over "Bro, I thought you meant your pickup game
    as in picking up girls..."; 82/401 replies that day answered a fragment)."""
    dw = words(strip_label(draft or ""))
    fw = words(strip_label(final or ""))
    if not dw or not fw or len(fw) < len(dw) + 3:
        return False
    fc = [w for w in fw if w not in FILLER]
    if not fc:
        return False
    covered = sum(1 for w in fc if w in set(dw))
    return covered / len(fc) < 0.6


def filler_only(text: str) -> bool:
    body = strip_label(text).strip()
    key = ' '.join(re.sub(r"[^a-z' ]+", ' ', body.lower()).split())
    if key in STT_HALLUCINATIONS and not body.endswith("?"):
        return True
    if key in REACTIONS and not body.endswith("?"):
        return True
    if body.endswith("?"):
        return False  # "huh?" / "what?" asks the agent to repeat -- not filler
    w = words(body)
    if not w:
        return True
    # "hahahahaha" / "lolol" style elongations
    return all(t in FILLER or re.fullmatch(r"(ha)+h?|(lo)+l|a+h+|h+m+|m+", t) for t in w)


def incomplete_fragment(text: str) -> bool:
    """Ends mid-clause: a fast VAD cut someone off while they drew breath."""
    body = strip_label(text).strip()
    if not body or body.endswith(("?", "!", ".")) and not body.endswith("..."):
        return False
    if body.endswith(("...", ",", "-", "--")):
        return True
    w = words(body)
    if len(w) < 2:
        return False
    last, prev = w[-1], w[-2]
    # Stranded preposition in a wh-clause is a complete English sentence:
    # "what are you up to", "who are you talking to", "what I was thinking of".
    if last in _STRANDABLE and any(x in _WH for x in w[:-1]):
        return False
    # Demonstrative "that" as an object: "down with that", "like that", "about that".
    if last == "that" and prev in _PREPS:
        return False
    # "if you" / "but if you" / "when you": subject with no verb yet.
    if last == "you" and prev in _CLAUSE_OPENERS:
        return True
    return last in _DANGLING


def real_interruption(text: str, names: tuple[str, ...] = ()) -> bool:
    """Should speech heard WHILE the agent talks stop the agent?"""
    body = strip_label(text)
    w = words(body)
    if not w:
        return False
    if any(n.lower() in w for n in names):
        return True                                   # addressed by name
    from conversation_dynamics import detect_silence_request, detect_other_addressee
    if detect_silence_request(body) is not None and not detect_other_addressee(body, names):
        return True                                   # "shut up", "stop", "hold on" (not "Jake, shut up")
    content = [t for t in w if t not in BACKCHANNEL]
    return len(w) >= 3 and len(content) >= 2


class ConversationFloor:
    """Tracks the agent's current conversation partner in a group call."""

    def __init__(self, partner_ttl_s: float = 45.0, quiet_ttl_s: float = 120.0,
                 clock=time.monotonic):
        self.partner_ttl_s = partner_ttl_s
        self.quiet_ttl_s = quiet_ttl_s
        self.clock = clock
        self.partner: str | None = None
        self.partner_at = 0.0
        self.recent: dict[str, float] = {}
        self.quiet_until = 0.0
        self.self_quiet = False          # quiet chosen by the agent (step_back)
        self.parting_until = 0.0         # grace after quiet / close: farewells don't reopen
        self.last_user_at = 0.0          # previous user turn (any speaker)
        self.turn_log: list[tuple[str, float]] = []   # ('u'|'a', t), bounded
        self.profile = None              # AgentProfile (interests), set by runtime
        self.last_agent_at = 0.0         # agent's last spoken reply
        self.side = None                 # (asker, addressee_label|None, t): human->human question
        self.asked = None                # (target, t): the agent's last reply ended in a question
        self.direct = None               # background mode: [speaker, follow-ups left, last t]
        self.turned_away = None          # (speaker, t): they just turned to another human
        self.engaged = None              # live conversation: {"who", "since", "last", "agent_at"}
        self.mono = None                 # floor-holder: {"spk", "start", "last", "words", "turns"}
        self.mono_pending = None         # (speaker, text, t): reply held until the monologue ends
        from pacing import Pacing
        self.pacing = Pacing(clock=clock)  # rolling talk share (owner 10-06: talk less)
        self.last_gate = None            # last turn_gate verdict (logged per turn)

    def reset(self) -> None:
        self.partner, self.partner_at, self.recent = None, 0.0, {}
        self.quiet_until = 0.0
        self.self_quiet = False
        self.parting_until = 0.0
        self.last_user_at = 0.0
        self.turn_log = []
        self.last_agent_at = 0.0
        self.side = None
        self.asked = None
        self.direct = None
        self.engaged = None
        self.turned_away = None
        self.pacing.reset()
        self.last_gate = None

    def _log_turn(self, who: str) -> None:
        self.turn_log.append((who, self.clock()))
        del self.turn_log[:-24]

    def talk_share(self, last_n: int = 10) -> tuple[int, int]:
        """(agent turns, total turns) over the last `last_n` turns."""
        recent = self.turn_log[-last_n:]
        return sum(1 for w, _ in recent if w == 'a'), len(recent)

    def quiet_gap(self) -> float:
        """Seconds of silence before the current line (0 if nobody spoke yet)."""
        return self.clock() - self.last_user_at if self.last_user_at else 0.0

    def step_back(self, minutes: float = 2.0) -> float:
        """The agent chose to go quiet (tool). Its name still wakes it."""
        minutes = max(0.5, min(5.0, float(minutes or 2.0)))
        self.quiet_until = self.clock() + minutes * 60.0
        self.self_quiet = True
        self._start_parting(self.quiet_until)
        return minutes

    # Quiet mode: "Max, stay quiet while we plan" must persist across the
    # following turns (in sim he answered the very next planning line). It ends
    # when he is named again or after quiet_ttl_s.
    def set_quiet(self) -> None:
        self.quiet_until = self.clock() + self.quiet_ttl_s
        self.self_quiet = False
        self._start_parting(self.quiet_until)

    # Parting grace (owner 10-07: "silence requests and choices need a grace
    # period so one final goodbye doesn't open her gates up again"). After a
    # silence request, step_back, or a closed exchange, a line that NAMES the
    # agent but only says goodbye / thanks / shh ("bye Fae", "night Fae, love
    # you", "thanks Max") is held and keeps the gate closed. A real question
    # or request (or the bare name) still wakes it.
    PARTING_GRACE_S = 60.0
    PARTING_AFTER_CLOSE_S = 90.0

    def _start_parting(self, until: float) -> None:
        self.parting_until = max(self.parting_until, until + self.PARTING_GRACE_S)

    def in_parting_grace(self) -> bool:
        return self.clock() < self.parting_until

    def is_quiet(self) -> bool:
        import owner_controls as _owner
        return (self.clock() < self.quiet_until or _owner.STATE["mute"]
                or _owner.quiet_active(self.clock))

    def release_quiet(self) -> None:
        self.quiet_until = 0.0
        self.self_quiet = False

    def turn_gate(self, text: str, speaker: str | None, names: tuple[str, ...]) -> str | None:
        """The single pre-LLM gate (server AND sims). Returns a HOLD reason or None."""
        verdict = self._turn_gate(text, speaker, names)
        self.last_gate = verdict or "pass"
        return verdict

    def _turn_gate(self, text: str, speaker: str | None, names: tuple[str, ...]) -> str | None:
        from conversation_dynamics import pre_llm_veto
        import owner_controls as _owner
        owner = _owner.gate(speaker, names_agent(text, names), self.clock)
        if owner:
            return owner
        if ack_only(text) and not names_agent(text, names) and not self._answers_agent(speaker):
            return "acknowledgement"
        veto = pre_llm_veto(text, names)
        if veto == "explicit silence request":
            # Only a silence request aimed at the agent starts quiet mode: its
            # name is in the line, or its current partner said it. "bro shut up"
            # between friends must not mute it for two minutes.
            # Quiet mode needs a STRONG command ("shut up", "stay quiet"), not a
            # discourse marker ("oh wait, hold up"); from the partner it must be a
            # short direct line -- "shut the fuck up, man" buried in a ramble is
            # banter between friends, not an order (live: muted Ivy for minutes).
            from conversation_dynamics import strong_silence_request
            if strong_silence_request(strip_label(text)) and (
                    names_agent(text, names)
                    or (speaker and speaker == self.active_partner()
                        and (len(words(text)) <= 8 or _command_leads(text)))):
                self.set_quiet()
            return veto
        if self.in_parting_grace() and parting_only(text, names):
            self.parting_until = max(self.parting_until, self.clock() + 45.0)
            return "parting (grace)"
        if self.is_quiet():
            if names_agent(text, names):
                self.release_quiet()
            else:
                return "quiet mode"
        bg = self.background_gate(text, speaker, names, veto)
        if bg is not None:
            return bg or None
        mono = self.monologue_gate(text, speaker, names)
        if mono:
            return mono
        eng = self.engagement_gate(text, speaker, names, veto)
        if isinstance(eng, str):
            return eng
        if veto:
            if veto.startswith("addressed to") and speaker:
                self.side = (speaker, None, self.clock())
            return veto
        if eng:
            return None      # the person it's talking with: no passive/pacing holds
        passive = self.passive_gate(text, speaker, names)
        if passive:
            return passive
        return self.pacing_gate(text, speaker, names)

    # --- monologues (owner 10-07: "people keep complaining Fae is interrupting
    # them when they are monologuing for a loooong time", all agents) -------------
    # A long monologue reaches us as many finals: every breath pause ends a "turn"
    # and the agent answered mid-thought ("Yeah, that's annoying.", replies to
    # "What was meant to happen is that"). Two recorded calls: 27 agent replies
    # landed inside 3+-turn, 40+-word runs; in 9 the speaker kept going.
    # While one person holds the floor, their lines are held unless they name the
    # agent or make an explicit request; the last held line is answered (through
    # the normal gate) once they actually stop -- MONO_TAIL_S of silence, see
    # server.start_monologue_tail. Turn detection also waits a little longer.
    MONO_MIN_WORDS = int(os.environ.get("ATLAS_MONO_WORDS", "35"))
    MONO_MIN_TURNS = 2
    MONO_GAP_S = 8.0          # a pause longer than this ends the run
    MONO_BREAK_WORDS = 4      # another person saying this much takes the floor
    MONO_EXTRA_WAIT_S = 0.8   # extra end-of-turn silence while a run is live
    MONO_PENDING_TTL_S = 30.0

    def _note_monologue(self, speaker: str, text: str, now: float) -> None:
        n = len(words(strip_label(text or "")))
        m = self.mono
        if m and m["spk"] == speaker and now - m["last"] <= self.MONO_GAP_S:
            m["last"], m["words"], m["turns"] = now, m["words"] + n, m["turns"] + 1
        elif m and m["spk"] != speaker and n < self.MONO_BREAK_WORDS and now - m["last"] <= self.MONO_GAP_S:
            pass                      # a listener's "yeah" / "mm" doesn't take the floor
        else:
            self.mono = {"spk": speaker, "start": now, "last": now, "words": n, "turns": 1}
            if self.mono_pending and self.mono_pending[0] != speaker:
                self.mono_pending = None   # someone else took the floor: moment passed
        p = self.mono_pending
        if p and p[0] == speaker and text:
            self.mono_pending = (speaker, text, now)   # answer their LAST (final) line

    def monologue_active(self, speaker: str | None = None) -> bool:
        m = self.mono
        if not m or self.clock() - m["last"] > self.MONO_GAP_S:
            return False
        if m["turns"] < self.MONO_MIN_TURNS or m["words"] < self.MONO_MIN_WORDS:
            return False
        return speaker is None or speaker == m["spk"]

    def monologue_extra_wait(self) -> float:
        return self.MONO_EXTRA_WAIT_S if self.monologue_active() else 0.0

    def monologue_gate(self, text: str, speaker: str | None, names: tuple[str, ...]) -> str | None:
        if not speaker or not self.monologue_active(speaker):
            return None
        body = strip_label(text)
        if names_agent(text, names) or self._REQUEST.search(body):
            return None
        self.mono_pending = (speaker, text, self.clock())
        return f"monologue (letting {speaker} finish)"

    def take_monologue_pending(self) -> tuple[str, str] | None:
        """The held line, once the speaker has stopped; ends the run so the normal
        gate decides whether to answer it."""
        p, self.mono_pending = self.mono_pending, None
        if not p or self.clock() - p[2] > self.MONO_PENDING_TTL_S:
            return None
        self.mono = None
        return p[0], p[1]

    # --- engagement (owner 10-06: "he goes silent after the first mention of his
    # name and can't maintain a conversation ... it needs to be a stable
    # conversation until dropped") -----------------------------------------------
    # Saying the agent's name opens a conversation with that person. Until it is
    # DROPPED, their follow-ups reach the model without the name, past the
    # passive and pacing holds. Dropped when: they turn to someone else (vocative
    # or side exchange), they close it ("ok thanks", "never mind"), someone else
    # calls the agent (focus moves to them), or nobody in it has spoken for
    # ENGAGE_IDLE_S. Diarizer drift: a directed line from a new label right after
    # the agent answered, before the engaged person said anything else, is taken
    # as that person (one mixed stream splits a voice across labels).
    ENGAGE_IDLE_S = 45.0
    ENGAGE_DRIFT_S = 12.0
    _CLOSE = re.compile(r"^\s*(?:never\s*mind|nvm|forget\s+it|that'?s\s+(?:all|it)|"
                        r"all\s+good|cool\s+thanks|ok(?:ay)?\s+thanks?|thanks?(?:\s+you)?|"
                        r"that'?s\s+enough|we'?re\s+good|got\s+it)\b[\s.!,]*"
                        r"(?:\w+[\s.!]*)?$", re.I)

    def engaged_with(self) -> str | None:
        e = self.engaged
        if not e:
            return None
        if self.clock() - max(e["last"], e["agent_at"]) > self.ENGAGE_IDLE_S:
            self.engaged = None
            return None
        return e["who"]

    def _directed(self, body: str) -> bool:
        return bool(self._ASK.search(body) or self._REQUEST.search(body)
                    or self._TO_YOU.search(body))

    _LEAVING = re.compile(r"\b(?:gonna|going\s+to|gotta|need\s+to|have\s+to)\s+(?:go|head\s+out|"
                          r"run|dip|bounce|grab|get\s+(?:food|some|a\s+drink))\b|\b(?:brb|afk|be\s+right\s+back|"
                          r"gtg|g2g|i'?m\s+out|peace\s+out|later\s+guys)\b", re.I)
    TURNED_AWAY_S = 20.0

    def engagement_gate(self, text: str, speaker: str | None,
                        names: tuple[str, ...], veto: str | None):
        """True = this line continues the agent's live conversation (skip the
        passive/pacing holds). A string = a HOLD reason (they closed it or are
        talking to someone else). False = no opinion (normal gating)."""
        now = self.clock()
        if not speaker or speaker == "self":
            return False
        if names_agent(text, names):
            e = self.engaged
            if e and e["who"] == speaker:
                e["last"] = now
            else:
                self.engaged = {"who": speaker, "since": now, "last": now, "agent_at": 0.0}
            return True
        ta = getattr(self, "turned_away", None)
        if ta and ta[0] == speaker and now - ta[1] <= self.TURNED_AWAY_S:
            # "Hey Riley, did you watch it?" -> "it was so good dude": still Riley's
            self.turned_away = (speaker, now)
            return "talking to someone else"
        who = self.engaged_with()
        if not who:
            return False
        e = self.engaged
        body = strip_label(text)
        if speaker != who:
            drift = (e["agent_at"] > e["last"]                       # it answered; they haven't spoken since
                     and now - e["agent_at"] <= self.ENGAGE_DRIFT_S
                     and self._directed(body)
                     and not (veto and veto.startswith("addressed to")))
            if drift:
                e["who"], e["last"] = speaker, now
                return True
            # Live sim 10-06: mid-conversation with Sam, Riley's "bro this lobby
            # is taking forever" got "Yeah, it's a bit slow." A remark from someone
            # else that asks nothing and isn't aimed at the agent is not its turn.
            if not self._directed(body):
                return "side remark (in conversation with someone else)"
            return False
        # it IS the engaged person
        if veto and veto.startswith("addressed to"):
            self.engaged = None                 # turned to someone else
            self.turned_away = (speaker, now)
            if self.partner == speaker:
                self.partner = None             # ...so the old thread is over too
            return False
        sd = self.side
        if (sd and sd[0] != speaker and sd[1] in (speaker, None)
                and sd[2] > max(e["last"], e["agent_at"])
                and now - sd[2] <= self.SIDE_TTL_S):
            # someone just asked THEM something ("Sam did you bring the controller"):
            # this answer belongs to that exchange; the conversation stays open
            e["last"] = now
            return False
        if self._CLOSE.match(body) or _thanks_only(text, names) or self._LEAVING.search(body):
            self.engaged = None                 # "ok thanks" / "never mind" / "gonna go make tea"
            if self.partner == speaker:
                self.partner = None
            self._start_parting(now + self.PARTING_AFTER_CLOSE_S - self.PARTING_GRACE_S)
            return "conversation closed"
        e["last"] = now
        if filler_only(body) or ack_only(text):
            return False                        # "yeah" keeps it alive but isn't a turn
        return True

    # --- pacing (owner 10-06: "our agent needs to talk less") ------------------
    # Live 10-06: 31% of all words in a 7-person room, ~7% of turns after its name.
    # Once over its fair share the agent stops volunteering: soft tier keeps only
    # its active partner and real questions to the room/agent; hard tier keeps
    # only a question from the person it is already talking with. Its name, or
    # an answer to its own question, always gets through.
    _ASK = re.compile(r"\?|^\s*(?:what|who|whom|whose|why|how|when|where|which|can|could|"
                      r"would|will|do|does|did|is|are|was|were|should|have|has|any)\b", re.I)
    _TO_YOU = re.compile(r"\b(?:you|your|you're|ya|u)\b", re.I)
    _REQUEST = re.compile(r"\b(?:give\s+(?:us|me)|tell\s+(?:us|me)|talk\s+about|explain|"
                          r"go\s+on|keep\s+going|more\s+about|what\s+about|"
                          r"(?:from|take\s+it\s+from)\s+.{0,30}point\s+of\s+view)\b", re.I)
    THREAD_WINDOW_S = 25.0

    def talkativeness(self) -> float:
        try:
            return float(getattr(self.profile, "talkativeness", 0.35))
        except Exception:  # noqa: BLE001
            return 0.35

    def pacing_gate(self, text: str, speaker: str | None,
                    names: tuple[str, ...]) -> str | None:
        now = self.clock()
        if names_agent(text, names):
            return None
        if self._answers_agent(speaker):
            return None
        tier = self.pacing.tier(self.talkativeness())
        if tier == "ok":
            return None
        body = strip_label(text)
        asks = bool(self._ASK.search(body))
        # Live 10-06 (pyramids): pacing must cut volunteering, never a live thread.
        # A question/request right after the agent spoke (from anyone) continues
        # the conversation; the person it's engaged with never reaches this gate
        # (engagement_gate lets their lines through first).
        req = asks or bool(self._REQUEST.search(body))
        if req and (now - self.last_agent_at) <= self.THREAD_WINDOW_S:
            return None
        partner = bool(speaker) and speaker == self.active_partner()
        to_you = asks and bool(self._TO_YOU.search(body))
        if tier == "soft":
            # over budget: stop volunteering on statements, but anything that
            # could be a request (its partner, a 'you' question, a question to
            # the room) still reaches the model, which decides if it's for it.
            if partner or to_you or (asks and self._ROOM_Q.search(body)):
                return None
        elif (partner and asks) or to_you:
            # well over budget: only questions from its partner or put to 'you'
            return None
        snap = self.pacing.snapshot(self.talkativeness())
        return (f"pacing {tier} (agent {round(snap['agent_share'] * 100)}% "
                f"vs target {round(snap['target_share'] * 100)}%)")

    def pacing_turn(self, speaker: str | None, text: str, feats: dict | None = None) -> dict:
        """Account a finished human turn; returns the per-turn pacing record."""
        if speaker == "self":
            return {}
        rec = self.pacing.note_human(speaker, text, feats)
        snap = self.pacing.snapshot(self.talkativeness())
        rec.update({"agent_share": snap["agent_share"], "target_share": snap["target_share"],
                    "pressure": snap["pressure"], "humans": snap["humans"],
                    "gate": self.last_gate})
        return rec

    def pacing_note(self, name_of=lambda s: None) -> str:
        return self.pacing.note(name_of, self.talkativeness())

    # --- background mode (owner 10-06) ----------------------------------------
    # The agent is a quiet helper: it answers only when spoken to. Saying its name
    # opens a direct exchange with THAT person; their next few lines (no name
    # needed) stay with the agent while the exchange is warm, then it drops back.
    DIRECT_FOLLOWUPS = 3
    DIRECT_WINDOW_S = 30.0

    def background_active(self) -> bool:
        import owner_controls as _owner
        pid = getattr(self.profile, "id", None)
        return _owner.background_on(pid)

    def background_gate(self, text, speaker, names, veto):
        """None = background mode is off (normal gating). "" = let this line
        through (bypassing the passive heuristics). Otherwise a HOLD reason."""
        if not self.background_active():
            self.direct = None
            return None
        now = self.clock()
        if names_agent(text, names):
            self.direct = [speaker, self.DIRECT_FOLLOWUPS, now]
            return veto or ""
        d = self.direct
        if d and speaker and d[0] != speaker and not filler_only(text):
            # Someone else jumped in ("lol Riley you always ask that"): the room has
            # the floor again, so the next line needs the name (live sim 10-06).
            self.direct = d = None
        if d and speaker and d[0] == speaker and d[1] > 0 and now - d[2] <= self.DIRECT_WINDOW_S:
            if veto and veto.startswith("addressed to"):
                self.direct = None          # they turned to someone else
                return veto
            if ack_only(text) or _thanks_only(text, names) or status_only(text, names):
                self.direct = None          # "ok thanks" closes the exchange
                self._start_parting(now + self.PARTING_AFTER_CLOSE_S - self.PARTING_GRACE_S)
                return "background mode (exchange closed)"
            d[1] -= 1
            d[2] = now
            return veto or ""
        if d and (now - d[2] > self.DIRECT_WINDOW_S or d[1] <= 0):
            self.direct = None
        return "background mode"

    # --- passivity: structural holds the model reliably ignored in sims ------
    SIDE_TTL_S = 15.0
    AGENT_RECENT_S = 30.0
    _ROOM_Q = re.compile(r"\b(?:any\s?one|any\s?body|some\s?one|some\s?body|"
                         r"you\s+guys|y'?all|everyone|everybody|chat)\b", re.I)
    _LABEL_VOC = re.compile(r"^\s*(?:\[(?:S\d+|user)\]\s*)?(S\d+)\b")

    def passive_gate(self, text: str, speaker: str | None,
                     names: tuple[str, ...]) -> str | None:
        """HOLD reason for lines that are plainly not the agent's, else None.

        Never fires: 1-on-1 rooms, unknown speaker, agent named/aliased, the
        agent's active partner, right after the agent spoke, or an interest hit.
        """
        now = self.clock()
        body = strip_label(text)
        # a human vocatively opening on another speaker's label ("S1 did you...")
        m = self._LABEL_VOC.match(text or "")
        if speaker and m and m.group(1) != speaker and not names_agent(text, names):
            self.side = (speaker, m.group(1), now)
            return f"addressed to {m.group(1)}"
        if not speaker or speaker in ("user", "self") or names_agent(text, names):
            return None
        # Humans turned to each other AFTER the agent last spoke ("Sam did you bring
        # the controller" -> "yeah it's in my bag"): their exchange is theirs even if
        # the agent talked a moment ago, or this speaker is its partner (first_run sim 10-04: 14B narrated it).
        if (self.side and now - self.side[2] <= self.SIDE_TTL_S
                and self.side[2] >= self.last_agent_at):  # same clock tick = after (Windows ~15ms)
            asker, target, _ = self.side
            if speaker != asker and (target is None or speaker == target):
                self.side = (asker, speaker, now)
                return "side exchange"
        if speaker == self.active_partner():
            return None
        if now - self.last_agent_at <= self.AGENT_RECENT_S:
            return None   # people react to what the agent just said
        others = [s for s, t in self.recent.items() if s != speaker and now - t <= 60.0]
        if not others:
            return None   # 1-on-1: a bare 'you' question may well be for the agent
        hits = self.profile.interest_hits(body) if self.profile is not None else []
        # 1) the answer to a question another human just asked someone else
        if self.side and now - self.side[2] <= self.SIDE_TTL_S:
            asker, target, _ = self.side
            if speaker != asker and (target is None or speaker == target):
                self.side = (asker, speaker, now)   # exchange continues
                return "side exchange"
            if speaker == asker and target and not hits:
                self.side = (asker, target, now)
                return "side exchange"
        # 2) unaddressed muttering in a group that isn't the agent's thing
        if not hits and not self._ROOM_Q.search(body):
            return "not addressed (group)"
        return None

    def on_user_turn(self, speaker: str | None, text: str = "") -> None:
        if not speaker or speaker in ("self", "user"):
            return
        now = self.clock()
        try:
            self._note_monologue(speaker, text, now)
        except Exception:  # noqa: BLE001
            pass
        self.recent[speaker] = now
        self.last_user_at = now
        self._log_turn('u')
        if speaker == self.partner:
            self.partner_at = now

    def on_agent_spoke(self, target: str | None, text: str = "") -> None:
        self._log_turn('a')
        # A reaction ("Yeah, that's rough.") doesn't end someone's monologue -- if they
        # keep going, it's still theirs (live: replies landed mid-run and the run
        # reset, so the next breath pause was answered too). A question hands them
        # a new turn: that's dialogue, not a monologue.
        self.mono_pending = None
        if (text or "").rstrip().endswith("?"):
            self.mono = None
        try:
            self.pacing.note_agent(text)
        except Exception:  # noqa: BLE001
            pass
        self.last_agent_at = self.clock()
        self.asked = (target, self.last_agent_at) if (text or "").rstrip().endswith("?") else None
        if self.direct and target == self.direct[0]:
            self.direct[2] = self.last_agent_at   # the window runs from her last reply
        if self.engaged and target == self.engaged["who"]:
            self.engaged["agent_at"] = self.last_agent_at
        if target and target not in ("user", "self"):
            self.partner, self.partner_at = target, self.clock()

    def _answers_agent(self, speaker: str | None, window_s: float = 20.0) -> bool:
        """The agent just asked a question and this line could be the answer ('yeah')."""
        if not self.asked:
            return False
        target, t = self.asked
        if self.clock() - t > window_s:
            return False
        return not target or target in ("user", "self") or target == speaker

    def active_partner(self) -> str | None:
        if self.partner and self.clock() - self.partner_at <= self.partner_ttl_s:
            return self.partner
        return None

    def room_size(self, window_s: float = 60.0) -> int:
        now = self.clock()
        return sum(1 for t in self.recent.values() if now - t <= window_s)

    def room_note(self, current: str | None, text: str = "",
                  names: tuple[str, ...] = (), profile=None) -> str:
        partner = self.active_partner()
        if not current or current == "user":
            return ""
        eng = self.engaged_with()
        if eng and current == eng and not names_agent(text, names):
            return (f"ROOM: {current} called you by name a moment ago and you are in a "
                    f"conversation with them. This line is {current} continuing it -- "
                    "they don't need to say your name again. Answer them.")
        if not partner:
            others = [s for s, t in self.recent.items()
                      if s != current and self.clock() - t <= 60.0]
            if names_agent(text, names):
                return ""
            crowd = (" Several people are talking; their 'you' means each other."
                     if others else "")
            hits = profile.interest_hits(text) if profile is not None else []
            loves = ""
            if profile is not None and profile.interests:
                loves = " Things you genuinely care about: " + ", ".join(profile.interests[:10]) + "."
            if hits:
                loves += (f" This line touches {', '.join(hits[:3])} -- one short, in-character "
                          "remark is fair game if it adds something; otherwise still [HOLD].")
            gap = self.quiet_gap()
            quiet = (f" The room was quiet for ~{int(gap // 60)} min before this."
                     if gap >= 90 else " The room was quiet before this." if gap >= 20 else "")
            return ("ROOM: nobody is in a conversation with you right now and "
                    f"{current} did not say your name.{quiet}{crowd} Be PASSIVE: people mutter, "
                    "react to their game or screen, talk to someone off-mic, or read "
                    "things out loud -- a stray line, even with 'you' in it, is usually NOT "
                    "for you. Default [HOLD]. Speak only if it's plainly a question to you, "
                    "or it's squarely about something you genuinely love and one short "
                    "remark would be welcome. Gripes about lag, internet, downloads or "
                    "stepping away (brb) are not requests: let them pass, don't offer "
                    "fixes, and never claim the same problem yourself." + loves)
        if current == partner:
            return (f"ROOM: you are mid-conversation with {partner} and this is {partner} "
                    "talking. Keep the thread going.")
        return (f"ROOM: you are mid-conversation with {partner}; {current} is someone else. "
                f"If {current} is talking to another person or chatting on the side, [HOLD]. "
                f"Speak to {current} only if they address you or react to what you said.")


# ---------------------------------------------------------------- banter layer
VIBE = (
    "VIBE: casual group conversation with friends. You like these people; talk like one of "
    "them. Not every line needs to be a joke."
)
_INSULT = re.compile(
    r"\b(dumb|stupid|trash|garbage|mid|cringe|ugly|bozo|clown|npc|loser|idiot|moron|"
    r"weird|boring|broke|bald|fat|lame|annoying|useless|washed|dogshit|shit|sucks?|"
    r"stfu|ratio|nerd|virgin|goofy|dumbass|dipshit|jackass|robot voice|microwave)\b",
    re.IGNORECASE,
)


def detect_roast(text: str, names: tuple[str, ...] = ()) -> bool:
    """A jab aimed at the agent: second person or its name + an insult."""
    body = strip_label(text)
    if not _INSULT.search(body):
        return False
    low = body.lower()
    aimed = re.search(r"\b(you|you're|youre|ur|your|u)\b", low) or any(
        re.search(rf"\b{re.escape(n.lower())}\b", low) for n in names)
    return bool(aimed)


ROAST_NOTE = ("They're teasing you. Play along like a friend would: a light comeback "
              "if one comes to you, or laugh it off. You don't have to win.")


_BAIT = re.compile(
    r"\b(masturbat\w*|goon\w*|jerk\w* off|beat (?:your|my|ur) meat|horny|sex\w*|"
    r"dick|cock|pussy|bbc|gay|straight|black|white|female|male|girl|boy|virgin|"
    r"racist|elon|chatgpt|openai|created you|made you|are you (?:real|an? ai|a bot|human)|"
    r"rage ?bait\w*|lost the plot|your voice|say (?:i|that|it)\b)",
    re.IGNORECASE,
)
_AIMED_AT_AI = re.compile(r"\b(you|you're|youre|ur|your|u|ai|bot|robot)\b", re.IGNORECASE)


def detect_bait(text: str, names: tuple[str, ...] = ()) -> bool:
    """Troll bait aimed at the agent: provocative/identity/'say X' lines.

    Live pattern: these got the safe dodge "What's good?" instead of a reply.
    """
    body = strip_label(text)
    # "say X" requests: bait only if X is crude or degrading (owner 10-04: "say I love
    # BBC" is bait, "say I love you" is a sweet ask the agent can just answer warmly).
    try:
        from loopbait import dictation, _DICTATE
        if _DICTATE.search(body):
            if dictation(body):
                return True
            body = _DICTATE.sub(' ', body)
    except Exception:
        pass
    if not _BAIT.search(body):
        return False
    low = body.lower()
    return bool(_AIMED_AT_AI.search(body) or any(
        re.search(rf"\b{re.escape(n.lower())}\b", low) for n in names))


BAIT_NOTE = ("They're baiting you to see what you'll do. Don't dodge with a greeting "
             "or 'what's good' -- actually answer it in character: clown them, flip it "
             "back on them, or play along, in one line.")


def steering_note(text: str, current: str | None, floor: ConversationFloor,
                  names: tuple[str, ...], vibe: bool = True, profile=None) -> str:
    """Per-turn guidance appended to the routing protocol (kept short)."""
    parts = []
    from tasks import CUE_RE
    try:
        from room_tools import cue_note as _room_cue
        rc = _room_cue(text)
    except Exception:  # noqa: BLE001
        rc = ""
    if rc:
        return rc + ("\n" + VIBE if vibe else "")
    if CUE_RE.search(text or ""):
        # Our own "your search finished" cue: not a user line, no passivity HOLD.
        parts.append("DELIVERY: this turn is your own cue, not someone speaking. Tell the "
                     "asker what you found in one or two lines, in character, like it just "
                     "came through -- or [HOLD] if the room has clearly moved on.")
        if vibe:
            parts.append(VIBE)
        return "\n".join(parts)
    bg = False
    try:
        bg = floor.background_active()
    except Exception:  # noqa: BLE001
        pass
    d = floor.direct if bg else None
    if bg and d and current and d[0] == current and not names_agent(text, names):
        parts.append(f"BACKGROUND MODE: {current} is still talking to you (they called you a "
                     "moment ago), so this line is for you: answer it briefly and helpfully.")
    else:
        note = floor.room_note(current, text, names, profile=profile)
        if note:
            parts.append(note)
    if bg:
        parts.append("You're a quiet background helper right now: answer what you were asked, "
                     "short and useful. No small talk, no follow-up questions unless you need one.")
    said, total = floor.talk_share()
    limit = profile.floor_share_limit() if profile is not None else 0.5
    if total >= 6 and said >= limit * total and not names_agent(text, names):
        parts.append(f"FLOOR: you've spoken in {said} of the last {total} turns. You're "
                     "taking a lot of the air: let others carry it -- [HOLD] more, and "
                     "call step_back if you're not needed for a bit.")
    if vibe:
        parts.append(VIBE)
        from moment import enabled as _moment_on
        if detect_roast(text, names) and not _moment_on():
            parts.append(ROAST_NOTE)
    return "\n".join(parts)


# ------------------------------------------------------------ anti-repetition
def _spoken(msg: dict) -> str:
    body = msg.get('content', '')
    if body.startswith('['):
        end = body.find(']')
        if end != -1:
            body = body[end + 1:]
    return body.strip()


def anti_repeat_note(history: list, last_n: int = 4) -> str:
    """Name the agent's recent openers so it doesn't loop on them.

    Built from the history actually sent to the model; empty when there is
    nothing repetitive to warn about.
    """
    said = [_spoken(m) for m in history if m.get('role') == 'assistant']
    said = [s for s in said if s][-last_n:]
    if len(said) < 2:
        return ""
    openers = []
    for s in said:
        o = ' '.join(s.split()[:2]).rstrip(',.!?')
        if o and o.lower() not in {x.lower() for x in openers}:
            openers.append(o)
    quoted = ', '.join(f'"{o}"' for o in openers)
    return (f"Your recent replies opened with {quoted}. Don't reuse those openings "
            "or repeat a previous line; answer what was actually just said.")


def collapse_repeats(history: list) -> list:
    """Drop earlier copies of an assistant line the agent has already repeated.

    A model that sees its own catchphrase N times in context copies it again
    (self-reinforcing rut). Keep only the most recent copy of each repeated
    spoken line; user turns are untouched.
    """
    def key(m):
        return ' '.join(re.sub(r"[^a-z0-9' ]+", ' ', _spoken(m).lower()).split())
    last = {}
    for i, m in enumerate(history):
        if m.get('role') == 'assistant':
            last[key(m)] = i
    return [m for i, m in enumerate(history)
            if m.get('role') != 'assistant' or last.get(key(m)) == i]


_SENT_END = ('.', '!', '?', '"', "'")


def model_safe_reply(text: str, min_words: int = 3) -> str:
    """What of a spoken reply may enter the model's history.

    A reply cut off by barge-in ("Yeah, punk's a whole spectrum,") saved as a
    complete line teaches the model to emit stubs; a few of them and it copies
    the pattern ("Yeah, I" / "Yeah, what's") on every turn. Keep only complete
    sentences; drop what is left if it's too short to be a real line.
    """
    t = ' '.join((text or '').split())
    if not t:
        return ''
    if t.endswith(_SENT_END):
        return t  # a finished line of any length ("Nope.") is a real reply
    cut = max(t.rfind(c) for c in '.!?')
    t = t[:cut + 1] if cut >= 0 else ''
    return t if len(t.split()) >= min_words else ''


def drop_stub_replies(history: list) -> list:
    """Remove assistant lines that are fragments (safety net for old context)."""
    out = []
    for m in history:
        if m.get('role') == 'assistant' and not model_safe_reply(_spoken(m)):
            continue
        out.append(m)
    return out


# ------------------------------------------------------------ stock-filler loops
# Content-free acknowledgements the model falls back on when it has nothing to
# say ("Yeah, alright." x26 in one live call). None are banned outright: a
# filler is only suppressed when the agent ALREADY said it recently (a loop).
STOCK_FILLERS = frozenset("""
yeah|yeah yeah|yeah alright|alright|alright then|yeah okay|okay|ok|sounds good|yeah sounds good
yeah no|no|yes|yep|yup|nope|cool|yeah cool|fair|fair enough|yeah fair|true|yeah true|for sure
bet|word|same|what|huh|hmm|mhm|i see|got it|nice|right|exactly|totally|yeah totally|lol|haha
yeah right|yeah exactly|yeah for sure|oh okay|oh yeah|gotcha|makes sense|yeah that makes sense
""".replace("\n", "|").strip("|").split("|"))

def _pack_alt(key: str) -> str:
    try:
        import agent_registry as _reg
        alt = _reg.opener_alt(key)
        return ("|" + alt) if alt else ""
    except Exception:  # noqa: BLE001
        return ""


_FILLER_OPENER_RE = re.compile(r"^(?:(?:yeah|yea|yep|alright|r?okay|ok|so|well|oh" + _pack_alt("reply_openers") + r")\b[,.!]?\s+)+", re.I)


def filler_key(text: str) -> str:
    return ' '.join(re.sub(r"[^a-z' ]+", ' ', (text or '').lower()).split())


def is_stock_filler(sentence: str) -> bool:
    return filler_key(sentence) in STOCK_FILLERS


_OPENER = re.compile(r"^(?:(?:r?okay|ok|yeah|yep|yo|oh|ah|alright|hmm+|well|so|man|bro|dude"
                     + _pack_alt("content_openers") + r")\b[\s,.!-]*)+")


def content_key(sentence: str) -> str:
    """A sentence minus its interjection opener ('Ruh-roh,' / 'Rokay,' / 'Yeah,').
    Live 10-01: Pup answered a company search with his earlier weather line,
    re-voiced under a different opener. Only real sentences (>=5 words) count."""
    k = _OPENER.sub('', filler_key(sentence)).strip()
    return ('D:' + k) if len(k.split()) >= 5 else ''


def recent_content(history: list, last_n: int = 6) -> set:
    said = [_spoken(m) for m in history if m.get('role') == 'assistant'][-last_n:]
    out = set()
    for s in said:
        for sent in re.split(r'(?<=[.!?])\s+', s or ''):
            k = content_key(sent)
            if k:
                out.add(k)
    return out


def recent_fillers(history: list, last_n: int = 6) -> set:
    """Filler sentences the agent said in its last N replies (loop detector)."""
    said = [_spoken(m) for m in history if m.get('role') == 'assistant'][-last_n:]
    out = set()
    for i, s in enumerate(said):
        recent_reply = i >= len(said) - SHORT_REPEAT_WINDOW
        for sent in re.split(r'(?<=[.!?])\s+', s):
            if is_stock_filler(sent) or (recent_reply and filler_key(sent)):
                out.add(filler_key(sent))
    return out


# Live 09-29 17:44: "It's a fan." x3 to three different questions. Any short line the
# agent said in its last few replies counts as a loop if it comes out again verbatim.
SHORT_REPEAT_WINDOW = 3
SHORT_REPEAT_MAX_WORDS = 5


def is_short_line(sentence: str) -> bool:
    n = len(filler_key(sentence).split())
    return 0 < n <= SHORT_REPEAT_MAX_WORDS


def strip_filler_openers(history: list) -> list:
    """Model-facing history only: drop a leading "Yeah,"/"Alright," from agent
    lines that carry real content, so context isn't a wall of "Yeah, ..." for
    the model to copy. The UI transcript is untouched."""
    out = []
    for m in history:
        if m.get('role') == 'assistant':
            body = m.get('content', '')
            head = ''
            if body.startswith('['):
                end = body.find(']')
                if end != -1:
                    head, body = body[:end + 1] + ' ', body[end + 1:].lstrip()
            rest = _FILLER_OPENER_RE.sub('', body, count=1)
            if rest != body and len(rest.split()) >= 3:
                m = dict(m, content=head + rest[:1].upper() + rest[1:])
        out.append(m)
    return out


def opener_rut_note(history: list, last_n: int = 8, threshold: int = 3) -> str:
    """Name a first word the agent keeps opening with ("Yeah" 5 of 8)."""
    said = [_spoken(m) for m in history if m.get('role') == 'assistant'][-last_n:]
    firsts = [filler_key(s.split()[0]) for s in said if s.split()]
    if not firsts:
        return ""
    word = max(set(firsts), key=firsts.count)
    n = firsts.count(word)
    if n < threshold:
        return ""
    return (f'You opened {n} of your last {len(firsts)} replies with "{word.title()}". '
            "Start with the actual point instead of an acknowledgement.")


# ---------------------------------------------------------------------------
# Template loops (live 09-30 11:43: 'What's the "smelly"?' on ~every line).
# The words change every time, so exact-repeat checks miss it; the SHAPE is what
# repeats. A short sentence whose 2-word opener the agent already used in 2+ of
# its last few replies is a template loop.
TEMPLATE_MAX_WORDS = 14  # live 10-01: 'Then that's the verdict: it was tight.' x127 (7-12 words)
TEMPLATE_WINDOW = 6
TEMPLATE_MIN_REPEATS = 2


def template_key(sentence: str) -> str:
    words = filler_key(sentence).split()
    if not (2 <= len(words) <= TEMPLATE_MAX_WORDS):
        return ''
    return 'T:' + ' '.join(words[:2])


def _short_sentences(text: str):
    return [s for s in re.split(r'(?<=[.!?])\s+', text or '') if template_key(s)]


def recent_templates(history: list, last_n: int = TEMPLATE_WINDOW) -> set:
    said = [_spoken(m) for m in history if m.get('role') == 'assistant'][-last_n:]
    counts = {}
    for s in said:
        for k in {template_key(x) for x in _short_sentences(s)}:
            counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n >= TEMPLATE_MIN_REPEATS}


def collapse_openers(history: list) -> list:
    """Model-facing only. Of agent lines that open with the same two words, keep
    the latest. One mechanism for exact repeats, templates ('What's the "X"?',
    'Then that's the X: ...') and filler openers: the model never sees its own
    pattern stacked up, so it has nothing to copy. User turns untouched."""
    def key(m):
        w = filler_key(_spoken(m)).split()
        return ' '.join(w[:2]) if len(w) >= 2 else ''
    last = {}
    for i, m in enumerate(history):
        if m.get('role') == 'assistant' and key(m):
            last[key(m)] = i
    return [m for i, m in enumerate(history)
            if m.get('role') != 'assistant' or not key(m) or last[key(m)] == i]


SHORT_REPLY_WORDS = 5
SHORT_REPLY_KEEP = int(os.environ.get("ATLAS_SHORT_KEEP", "3"))


def thin_short_replies(history: list, keep: int = None) -> list:
    """Model-facing only. A wall of the agent's own one-liners ('Pip see it.',
    'Rokay, I'm here.') teaches it to answer in one-liners. Keep only the latest
    few short agent replies; longer ones and all user turns stay."""
    keep = SHORT_REPLY_KEEP if keep is None else keep
    if keep < 0:
        return history
    short = [i for i, m in enumerate(history) if m.get('role') == 'assistant'
             and 0 < len(_spoken(m).split()) <= SHORT_REPLY_WORDS]
    drop = set(short[:-keep] if keep else short)
    return [m for i, m in enumerate(history) if i not in drop]


OWN_KEEP = int(os.environ.get("ATLAS_OWN_KEEP", "1"))


def clip_own_replies(history: list, keep: int = None) -> list:
    """Model-facing only ("Clip Context", conversational-inertia research): models
    treat their own earlier replies as few-shot demos and imitate them. Keep only
    the latest `keep` agent replies verbatim; user turns all stay. keep<0 = off."""
    keep = OWN_KEEP if keep is None else keep
    if keep < 0:
        return history
    own = [i for i, m in enumerate(history) if m.get('role') == 'assistant']
    drop = set(own[:-keep] if keep else own)
    return [m for i, m in enumerate(history) if i not in drop]


def collapse_templates(history: list) -> list:
    """Model-facing only: of short agent lines sharing a template, keep the latest,
    so a loop can't teach itself. User turns and longer lines are untouched."""
    def key(m):
        body = _spoken(m).strip()
        sents = _short_sentences(body)
        return template_key(body) if len(sents) == 1 and sents[0].strip() == body else ''
    counts, last = {}, {}
    for i, m in enumerate(history):
        if m.get('role') == 'assistant':
            k = key(m)
            if k:
                counts[k] = counts.get(k, 0) + 1
                last[k] = i
    return [m for i, m in enumerate(history)
            if m.get('role') != 'assistant' or not key(m)
            or counts[key(m)] < TEMPLATE_MIN_REPEATS or last[key(m)] == i]


def _command_leads(text: str) -> bool:
    """A strong silence command opens the line (first sentence, <= 12 words in)."""
    from conversation_dynamics import strong_silence_request
    first = re.split(r"(?<=[.!?])\s+", strip_label(text).strip(), maxsplit=1)[0]
    return len(words(first)) <= 12 and strong_silence_request(first)


# The agent's own promise to go quiet ("Noted. I'll stay quiet until someone says
# my name.") must bind it -- live 10-02 Fae promised, then kept talking.
_SELF_QUIET = re.compile(
    r"\b(?:i(?:'|’)?ll|i\s+will|i(?:'|’)?m\s+gonna|gonna)\s+(?:stay|keep|be|go)\s+(?:quiet|silent)"
    r"|\b(?:i(?:'|’)?ll|i\s+will)\s+(?:shut\s+up|zip\s+it|pipe\s+down|hush|stop\s+talking)"
    r"|\b(?:i(?:'|’)?ll|i\s+will)\s+(?:only\s+)?(?:speak|talk)\s+(?:only\s+)?when\s+(?:i(?:'|’)?m\s+)?(?:spoken|talked)\s+to",
    re.IGNORECASE)


def promises_quiet(reply: str) -> bool:
    return bool(reply) and bool(_SELF_QUIET.search(reply))


# ---------------------------------------------------------------- status updates
# Demo take 4 (10-04): "Nice. OK. Loading in." got "Welcome, Riley." -- the
# PASSIVE prompt note says to let these pass, but a 14B model still answered.
# A short, unnamed, non-question status/afk line never needs a reply, so it
# skips the model call entirely (like laughter in filler_only).
_STATUS = re.compile(
    r"\b(loading( in| up)?|brb|be right back|one sec(ond)?|hold on|gimme a (sec|min)|"
    r"give me a (sec|second|minute)|afk|almost (done|there)|(\w+ )?more minutes?|"
    r"(downloading|updating|installing|restarting|rebooting|queu(e|ing)|joining|getting on|"
    r"hopping on|logging (in|on))|lagg?(ing|y)|my (internet|wifi|connection|ping)|"
    r"i'?m back|back now)\b", re.IGNORECASE)
_STATUS_FILL = re.compile(r"\b(nice|ok(ay)?|alright|all right|cool|yeah|yep|ya|k|so|"
                          r"like|just|um+|uh+|lol|haha|sorry|wait)\b", re.IGNORECASE)


_CONN_GRIPE = re.compile(r"\bmy (?:internet|wifi|wi-fi|connection|ping|router)\b", re.IGNORECASE)


def status_only(text: str, names: tuple[str, ...] = ()) -> bool:
    body = strip_label(text).strip()
    if not body or "?" in body or names_agent(text, names):
        return False
    # Live sim 10-06: "anyway I'm gonna go make tea" after "okay thanks" got
    # "Go make your tea." -- someone announcing they're stepping away needs no reply.
    lm = ConversationFloor._LEAVING.search(body)
    if lm and len(body.split()) <= 12:
        tail = re.sub(r"[^a-z' ]+", " ", body[lm.end():].lower()).split()
        # "gonna go make tea" / "brb" end the line; "gotta go to the dentist
        # tomorrow" is news someone may want to talk about.
        if len(tail) <= 2 and not any(w in ("to", "because", "cause", "but") for w in tail):
            return True
    # Live sim 10-06: "my ping is like three hundred right now" got "Three hundred?
    # That's brutal." A complaint about one's own connection, asked of nobody,
    # is a status line however it's phrased.
    if (_CONN_GRIPE.search(body) and len(body.split()) <= 12
            and not ConversationFloor._TO_YOU.search(body)):
        return True
    if len(body.split()) > 9 or not _STATUS.search(body):
        return False
    rest = _STATUS_FILL.sub(" ", _STATUS.sub(" ", body.lower()))
    rest = re.sub(r"[^a-z' ]+", " ", rest)
    # Only the status phrase plus filler: nothing else being said.
    return len(rest.split()) <= 3


# ---------------------------------------------------------------- acknowledgements
# Situation matrix 10-04: "lol okay" in a busy room and "okay back" got replies
# ("Great, Ben going first..."). A pure acknowledgement asks nothing; it only needs
# an answer when the agent just asked this person a question (ConversationFloor).
_ACK = {"ok", "okay", "k", "kk", "cool", "alright", "nice", "sure", "yeah", "yep", "yup",
        "ya", "yea", "got", "it", "gotcha", "sounds", "good", "bet", "word", "true", "fair",
        "right", "perfect", "great", "back", "same", "facts", "ight", "aight", "fine", "noted"}


_THANKS = {"thanks", "thank", "you", "ty", "thx", "tysm", "cheers", "appreciate", "it",
           "much", "so", "very", "awesome", "perfect", "great", "cool", "ok", "okay",
           "nice", "got", "gotcha", "that's", "thats", "all", "i", "needed", "bye", "later"}


def _thanks_only(text: str, names: tuple = ()) -> bool:
    """'ok thanks Fae' / 'cool, that's all I needed': closes a background exchange."""
    body = strip_label(text).strip()
    if not body or "?" in body:
        return False
    w = [t for t in re.sub(r"[^a-z' ]+", " ", body.lower()).split() if t not in names]
    if not w or len(w) > 6:
        return False
    return all(t in _THANKS for t in w) and any(t in ("thanks", "thank", "ty", "thx", "tysm",
                                                       "cheers", "appreciate", "needed", "bye")
                                                 for t in w)


_PARTING_WORDS = _ACK | _THANKS | {
    "bye", "byee", "byeee", "goodbye", "goodnight", "night", "nite", "gn", "later", "laters",
    "see", "ya", "you", "u", "cya", "peace", "out", "take", "care", "have", "a", "good", "one",
    "day", "evening", "sleep", "well", "sweet", "dreams", "talk", "soon", "catch", "love",
    "shh", "shhh", "hush", "quiet", "be", "stay", "shut", "up", "stop", "enough", "now", "for",
    "tonight", "everyone", "guys", "y'all", "yall", "bro", "dude", "man", "girl", "buddy",
    "lol", "haha", "xoxo", "mwah", "oh", "and", "too", "again", "the", "rest", "of", "night's",
    "just", "with", "me", "we're", "were", "done", "go", "to", "lil", "little", "friend",
}
_PARTING_MARK = {"bye", "byee", "byeee", "goodbye", "goodnight", "night", "nite", "gn", "later",
                 "laters", "cya", "peace", "care", "dreams", "love", "shh", "shhh", "hush",
                 "quiet", "shut", "stop", "thanks", "thank", "ty", "thx", "tysm", "cheers",
                 "see", "enough", "done"}


def parting_only(text: str, names: tuple = ()) -> bool:
    """'bye Fae' / 'goodnight Max, love you' / 'thanks Fae' / 'shh Fae':
    only a farewell, thanks or hush -- nothing asked, nothing requested."""
    from conversation_dynamics import _is_agent_name
    body = re.sub(r'"[^"]*"|“[^”]*”', '', strip_label(text)).strip()
    if not body or "?" in body:
        return False
    toks = re.sub(r"[^a-z' ]+", " ", body.lower().replace("’", "'")).split()
    w = [t for t in toks if not (len(t) >= 3 and _is_agent_name(t, names))
         and t not in ("hey", "yo", "aw", "aww", "awww", "ok", "okay")]
    if not w or len(w) > 10:
        return False
    return all(t in _PARTING_WORDS for t in w) and any(t in _PARTING_MARK for t in w)


def ack_only(text: str) -> bool:
    body = strip_label(text).strip()
    if not body or "?" in body:
        return False
    w = re.sub(r"[^a-z' ]+", " ", body.lower()).split()
    if not w or len(w) > 4:
        return False
    return all(t in _ACK or t in FILLER for t in w) and any(t in _ACK for t in w)
