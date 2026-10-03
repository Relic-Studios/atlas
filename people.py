"""Who's who in the call: diarizer voice tags (S1, S2...) -> human names.

S-tags are routing IDs for the control header. They must never be spoken, and a
voice the agent can't name is an unnamed person ("you"), not "S8".

Names are learned deterministically (no model call):
  * self-intros:        "my name is Jake", "call me Jake", "this is Jake", "hi, I'm Jake"
  * answers:            the agent asked "who's this?" and the same voice says "Jake." / "it's Jake"
  * corrections:        "it's not Jake, it's Jack" / "my name's Jack" (overwrites)
  * refusals:           "not telling you" after an ask -> stop asking
Names heard in the room ("Maya, pass the charger") are kept as unmatched
candidates so the agent can clarify ("wait, are you Maya?").

Optional cross-call memory: learned names are saved with the diarizer's voice
centroid (memory_db/voiceprints.json). A new voice that matches a saved print is
only a *probable* name that the agent is told to confirm; it is never asserted.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TAG_RE = re.compile(r"\b(?:speaker\s+)?(S\d{1,4})\b", re.IGNORECASE)
# Words that make "S23" a product/season name, not a voice label.
_PRODUCT_BEFORE = re.compile(
    r"(?:galaxy|samsung|note|model|audi|canon|season|series|iphone|tesla|mercedes|class|lumix|sony)\s*$",
    re.IGNORECASE)

# Words that follow "I'm"/"it's"/sit capitalized at a sentence start but aren't names.
_NOT_NAMES = set("""
a an the i im i'm me my mine you your yours he she they we it its it's this that
good fine great ok okay alright cool here there just not so gonna going gone sorry
tired back done ready sure like trying literally down busy playing bored hungry high
drunk dead joking kidding serious actually still also always never really pretty very
from in on at with about over out up off no nah nope yes yeah yep yup yo hey hi hello
sup why what who where when how bro dude man guy girl bruh bitch nobody none someone
somebody everyone anybody guess whatever idk nothing wait hold hmm uh um oh ah lol lmao
thanks thank please well hell damn fuck shit god jesus black white gay straight
american canadian british mexican asian human real ai bot robot the_ok single married
new old big little fucking honestly basically anyway anyways right left sick broke
home late early free lost confused scared happy sad mad angry fr deadass bet lowkey
highkey dad mom mommy daddy friend friends mate homie sir maam ma'am
atlas monday tuesday wednesday thursday friday saturday sunday
later tomorrow tonight today sometime soon anytime whenever maybe asap again
italian french german spanish english irish scottish welsh russian chinese japanese korean
indian african european australian brazilian polish dutch swedish norwegian danish greek
turkish arab arabic jewish muslim christian catholic filipino vietnamese thai persian
latino latina hispanic puerto rican cuban colombian nigerian kenyan egyptian israeli
ukrainian portuguese swiss belgian austrian hungarian czech finnish texan southern
""".split())

def _agent_words() -> set:
    """Names/nicknames of every installed agent: never learned as a human's name."""
    import agent_registry as _r
    return _r.all_agent_words()


_NAME_TOKEN = r"([A-Za-z][A-Za-z'\-]{1,14})"
_STRONG = [
    re.compile(r"\bmy\s+name(?:'s|\s+is)\s+(?:actually\s+)?" + _NAME_TOKEN, re.I),
    re.compile(r"\b(?:you\s+can\s+)?call\s+me\s+" + _NAME_TOKEN, re.I),
    re.compile(r"^\W*(?:(?:hey|hi|hello|yo|sup)\W+)?(?:this\s+is|it's|it\s+is)\s+" + _NAME_TOKEN + r"(?:\W+(?:by\s+the\s+way|btw|here|nice\s+to\s+meet\s+(?:you|y\'all|everyone)|pleasure|what\'s\s+up))*\W*$", re.I),
    re.compile(r"\bname's\s+" + _NAME_TOKEN, re.I),
    re.compile(r"\bit's\s+not\s+[A-Za-z'\-]+,?\s+it's\s+" + _NAME_TOKEN, re.I),
]
# "I'm Jake" only when it is clearly an introduction.
_IM_GREETED = re.compile(r"^\W*(?:hey|hi|hello|yo|sup)\W+(?:(?:guys|everyone|y'all|bro)\W+)?i'?m\s+" + _NAME_TOKEN, re.I)
_IM_BARE = re.compile(r"^\W*(?:(?:no|nah|yeah|yo|actually|oh)\W+)?i'?m\s+" + _NAME_TOKEN + r"(?:\W+(?:by\s+the\s+way|btw|here|nice\s+to\s+meet\s+(?:you|y\'all|everyone)|pleasure|what\'s\s+up))*\W*$", re.I)
_ANSWER_LEAD = re.compile(r"^\W*(?:(?:uh|um|oh|yeah|yo|bro|well|so|ok|okay)\W+)*(?:it's|its|it\s+is|i'?m|i\s+am|my\s+name(?:'s|\s+is)|name's|call\s+me|this\s+is)?\s*", re.I)
_INTRO_TAIL = re.compile(r"(?:\W+(?:by\s+the\s+way|btw|here|nice\s+to\s+meet\s+(?:you|y'all|everyone)|pleasure|what's\s+up))+\W*$", re.I)
_REFUSE = re.compile(r"\b(?:not\s+(?:telling|gonna\s+tell|saying)|none\s+of\s+your|why\s+do\s+you\s+(?:wanna|want\s+to)\s+know|guess|no\s+one|nobody|doesn'?t\s+matter)\b", re.I)
_FORGET = re.compile(r"\b(?:don'?t|do\s+not|stop)\s+call(?:ing)?\s+me\s+([A-Za-z'\-]+)", re.I)
_VOCATIVE = re.compile(r"^\W*(?:(?:yo|hey|oh)\W+)?([A-Z][a-z'\-]{1,14}),\s")
_ASK = re.compile(
    r"\b(?:what(?:'s|\s+is)\s+your\s+name|who(?:'s|\s+is)\s+(?:this|that|talking|speaking)|"
    r"who\s+am\s+i\s+(?:talking|speaking)\s+(?:to|with)|what\s+(?:do|should)\s+i\s+call\s+you|"
    r"what\s+do\s+(?:people|they|we)\s+call\s+you|and\s+you\s+are\??|who\s+are\s+you\b|"
    r"didn'?t\s+catch\s+your\s+name|your\s+name\b.*\?|are\s+you\s+[A-Z][a-z]+\?)", re.I)


def _clean_name(tok: str | None, need_cap: bool = False) -> str | None:
    if not tok:
        return None
    tok = tok.strip("'-").strip()
    if need_cap and not tok[:1].isupper():
        return None
    if not (2 <= len(tok) <= 15) or not re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tok):
        return None
    if (tok.lower() in _NOT_NAMES or tok.lower().rstrip("'s") in _NOT_NAMES
            or tok.lower() in _agent_words()):
        return None
    return tok[0].upper() + tok[1:]


@dataclass
class Person:
    label: str
    name: str | None = None
    source: str = ""            # intro | answer | correction | voice (unconfirmed)
    probable: str | None = None  # voice-print guess, never asserted
    asked_at: float | None = None
    asks: int = 0
    declined: bool = False
    first_seen: float = field(default_factory=time.monotonic)


class NameBook:
    def __init__(self, agent_names=(), store: str | Path | None = "memory_db/voiceprints.json",
                 clock=time.monotonic, answer_window_s: float = 45.0,
                 ask_cooldown_s: float = 240.0, max_asks: int = 2):
        self.people: dict[str, Person] = {}
        self.heard: dict[str, float] = {}      # names said in the room, not yet matched
        self.agent_names = {n.lower() for n in agent_names}
        self.clock = clock
        self.answer_window_s = answer_window_s
        self.ask_cooldown_s = ask_cooldown_s
        self.max_asks = max_asks
        self.voice_lookup = None                # label -> centroid (np.ndarray) | None
        self.store = Path(store) if store else None
        self._prints: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.store and self.store.exists():
            try:
                self._prints = json.loads(self.store.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("voiceprints unreadable: %s", e)

    def _save_print(self, label: str, name: str) -> None:
        if not (self.store and self.voice_lookup):
            return
        try:
            c = self.voice_lookup(label)
            if c is None:
                return
            self._prints[name] = [round(float(x), 5) for x in c]
            self.store.parent.mkdir(parents=True, exist_ok=True)
            self.store.write_text(json.dumps(self._prints), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("voiceprint save failed: %s", e)

    def _match_print(self, label: str, threshold: float = 0.70) -> str | None:
        if not (self._prints and self.voice_lookup):
            return None
        try:
            import numpy as np
            c = self.voice_lookup(label)
            if c is None:
                return None
            c = np.asarray(c, dtype=np.float32)
            taken = {p.name for p in self.people.values() if p.name}
            best, best_sim = None, threshold
            for name, v in self._prints.items():
                if name in taken:
                    continue
                v = np.asarray(v, dtype=np.float32)
                sim = float(np.dot(c, v) / ((np.linalg.norm(c) * np.linalg.norm(v)) or 1.0))
                if sim >= best_sim:
                    best, best_sim = name, sim
            return best
        except Exception as e:  # noqa: BLE001
            logger.warning("voiceprint match failed: %s", e)
            return None

    # ------------------------------------------------------------ mutation
    def reset(self) -> None:
        with self._lock:
            self.people.clear()
            self.heard.clear()

    def _person(self, label: str) -> Person:
        p = self.people.get(label)
        if p is None:
            p = self.people[label] = Person(label)
            guess = self._match_print(label)
            if guess:
                p.probable = guess
                logger.info("👤 %s sounds like %s (voiceprint, unconfirmed)", label, guess)
        return p

    def _assign(self, p: Person, name: str, source: str) -> str:
        for other in self.people.values():   # one name, one voice
            if other is not p and other.name == name and source != "voice":
                other.name, other.source = None, ""
        changed = p.name != name
        p.name, p.source, p.probable, p.declined = name, source, None, False
        self.heard.pop(name.lower(), None)
        if changed:
            logger.info("👤 %s is %s (%s)", p.label, name, source)
            self._save_print(p.label, name)
        return name

    def learn(self, label: str | None, text: str) -> str | None:
        """Update from one final user turn. Returns the name learned, if any."""
        text = (text or "").strip()
        if not text:
            return None
        body = re.sub(r"^\s*\[S\d+\]\s*", "", text)
        with self._lock:
            self._note_vocative(body)
            if not label or label in ("user", "self"):
                return None
            p = self._person(label)
            now = self.clock()
            f = _FORGET.search(body)
            if f:
                if p.name and f[1].lower() == p.name.lower():
                    logger.info("👤 %s asked not to be called %s", label, p.name)
                    p.name, p.source = None, ""
                return None
            for i, rx in enumerate(_STRONG):
                m = rx.search(body)
                # "this is crazy" / "it's lit": weak frames need a capitalized (proper) noun
                name = _clean_name(m[1], need_cap=(i == 2)) if m else None
                if name and name.lower() not in self.agent_names:
                    return self._assign(p, name, "intro")
            words = body.split()
            asked = p.asked_at is not None and now - p.asked_at <= self.answer_window_s
            core = _INTRO_TAIL.sub("", body).split()  # "I'm Dana, nice to meet you" is still 2 words of intro
            m = _IM_GREETED.search(body) or ((_IM_BARE.search(body) if asked or len(core) <= 3 else None))
            name = _clean_name(m[1], need_cap=True) if m else None
            if name and name.lower() not in self.agent_names:
                return self._assign(p, name, "intro")
            if asked:
                if _REFUSE.search(body):
                    p.declined, p.asked_at = True, None
                    logger.info("👤 %s declined to give a name", label)
                    return None
                if len(words) <= 4:
                    rest = _ANSWER_LEAD.sub("", body, count=1).strip()
                    tok = re.match(r"[A-Za-z][A-Za-z'\-]*", rest)
                    name = _clean_name(tok[0], need_cap=len(words) > 1) if tok else None
                    if name and name.lower() not in self.agent_names:
                        p.asked_at = None
                        return self._assign(p, name, "answer")
            if p.probable and asked and re.match(r"^\W*(?:yeah|yes|yep|yup|that's\s+me|correct|mhm)\b", body, re.I):
                return self._assign(p, p.probable, "confirmed")
            return None

    def _note_vocative(self, body: str) -> None:
        m = _VOCATIVE.match(body)
        name = _clean_name(m[1]) if m else None
        if name and name.lower() not in self.agent_names \
                and name not in {p.name for p in self.people.values()}:
            self.heard[name.lower()] = self.clock()

    def note_agent_reply(self, target: str | None, text: str) -> None:
        """Record that the agent asked this voice who they are."""
        if not target or target in ("user", "self") or not _ASK.search(text or ""):
            return
        with self._lock:
            p = self._person(target)
            p.asked_at = self.clock()
            p.asks += 1

    # ------------------------------------------------------------ reading
    def name_of(self, label: str | None) -> str | None:
        p = self.people.get(label or "")
        return p.name if p else None

    def display(self, label: str | None) -> str:
        """Human-facing label for UI/logs."""
        if not label or label in ("user", "you", "YOU"):
            return "you"
        p = self.people.get(label)
        if p and p.name:
            return p.name
        if p and p.probable:
            return f"{p.probable}?"
        return "unknown voice"

    def who_note(self, current: str | None, labels=(), roster=None) -> str:
        """Prompt block: who each voice tag is, and whether to ask for a name."""
        with self._lock:
            labels = [l for l in dict.fromkeys([current, *labels]) if l and l not in ("user", "self")]
            if not labels:
                return ""
            lines = ["PEOPLE (S-tags are internal voice IDs for the header ONLY. Never say an "
                     "S-tag out loud and never call anyone 'speaker'. Use real names below; "
                     "for someone unnamed just say 'you', or describe them by what they said):"]
            for l in labels[:8]:
                p = self._person(l)
                if p.name:
                    d = f"{p.name}"
                elif p.probable:
                    d = f"name not confirmed; the voice sounds like {p.probable} from before - check before using it"
                elif p.declined:
                    d = "won't say their name - don't ask again"
                else:
                    d = "unknown - you haven't learned their name"
                lines.append(f"- {l}{' (talking now)' if l == current else ''}: {d}")
            if self.heard:
                recent = [n.title() for n, t in sorted(self.heard.items(), key=lambda x: -x[1])
                          if self.clock() - t <= 600][:4]
                if recent:
                    lines.append("Names people used in the call but not matched to a voice yet: "
                                 + ", ".join(recent) + ".")
            cue = self._ask_cue(current, roster)
            if cue:
                lines.append(cue)
            return "\n".join(lines)

    def _ask_cue(self, current, roster) -> str:
        if not current or current in ("user", "self"):
            return ""
        p = self._person(current)
        if p.name or p.declined or p.asks >= self.max_asks:
            return ""
        now = self.clock()
        if p.asked_at is not None and now - p.asked_at < self.ask_cooldown_s:
            return ""
        engaged = 0
        if roster is not None and current in roster:
            r = roster[current]
            engaged = getattr(r, "addressed_agent", 0) + (getattr(r, "turns", 0) >= 3)
        if engaged < 1 and not p.probable:
            return ""
        if p.probable:
            return (f"If you answer {current}, you can casually check whether they're "
                    f"{p.probable} - in your own voice, only if it fits.")
        return (f"You don't know who {current} is yet. If you answer them, work in a quick "
                "ask for their name in your own style (e.g. 'wait, who's this?'), once, "
                "after actually responding.")

    def sanitize(self, text: str, target: str | None = None) -> str:
        """Replace any spoken S-tag with a name, 'you', or 'someone'."""
        def sub(m):
            lab = m[1].upper()
            if lab not in self.people and lab != (target or "").upper():
                # Live 10-02: Fae said "S130" aloud -- a voice label from
                # before a persona switch / beyond the book. Only product
                # names ("Galaxy S23") keep the literal.
                if _PRODUCT_BEFORE.search(m.string[:m.start()]):
                    return m[0]
                return "someone"
            name = self.name_of(lab)
            if name:
                return name
            return "you" if lab == (target or "").upper() else "someone"
        return TAG_RE.sub(sub, text or "")

    def snapshot(self) -> dict:
        with self._lock:
            return {l: {"name": p.name, "probable": p.probable, "source": p.source,
                        "declined": p.declined, "asks": p.asks}
                    for l, p in self.people.items()}
