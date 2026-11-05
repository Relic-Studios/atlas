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

The owner is different (owner_voice.py, docs/designs/owner_voice.md): `NameBook.owner`
is the session's owner_voice.Adopter. A label it has adopted reads the owner's name
at read time (nothing is copied into a Person), and no other label may take the
owner's name from anything said in the call. With `owner = None` nothing changes.
"""
from __future__ import annotations

import json
import logging
import os
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
hindu sikh buddhist atheist agnostic mormon christianity islam judaism moroccan algerian tunisian
pakistani bangladeshi nepali iranian iraqi syrian lebanese afghan ethiopian somali ghanaian jamaican
haitian peruvian chilean argentinian argentine venezuelan ecuadorian dominican romanian bulgarian
serbian croatian bosnian albanian lithuanian latvian estonian slovak icelandic scandinavian
muted unmuted afk brb back here ya buh huh larping
""".split())
# Slurs a recognizer can hand back as a "name" (stems, as in loopbait._CRUDE).
_SLUR_NAME = re.compile(r"(?:nigg|fag)\w*|tranny", re.I)

# One-word answers to "who's this?" were learned as names ("Close.", "All.", "Muted.",
# "Tell.", voiceprints 10-07). People on Discord go by handles, so rare words are always
# welcome ("Tangy", "Itsblazemf"); a word that is common in English only counts from a
# one-word answer when it is also a given name. Strong frames ("call me Batman",
# "my name's Queen") still take any handle.
_COMMON_GIVEN = frozenset("""
aaron adam alan albert alex alexander alice amanda amber amy andrew andy angel angela anna anne cameron earl
anthony april ashley austin barbara ben benjamin beth bill billy bob bobby brad brandon brian
bruce carl carlos carol caroline carter charles charlie chase chris christian christina christopher
claire cole connor craig dan daniel danny dave david dean dennis diana dominic don donald drew
dylan ed eddie edward eli elizabeth ella ellen emily emma eric eva evan eve frank fred gabriel
gary george grace grant greg hannah harry hayden heather helen henry holly hope hunter ian isaac
jack jackson jacob jake james jamie jane jason jay jeff jenny jeremy jerry jesse jessica jim
jimmy joe joel john johnny jon jordan jose joseph josh joshua joy juan julia julian justin karen
kate katie keith kelly ken kevin kim kyle larry laura lee leo lewis liam lily linda lisa logan
louis lucas lucy luke marcus maria mark martin mary mason matt matthew max may megan michael mike miles
miller mitchell molly morgan nancy nate nathan neil nick nicole noah oliver olivia owen parker
patrick paul peter phil rachel ray rebecca richard rick riley rob robert robin roger ron rose
ross ruby russell ryan sam samuel sara sarah scott sean seth shane simon sophia stephen steve
steven susan taylor ted terry max tim timothy todd tom tommy tony travis troy tyler victor
vincent walter wayne will william zach
""".split())


def _plausible_answer_name(tok: str) -> bool:
    w = (tok or "").lower()
    if w in _COMMON_GIVEN:
        return True
    if "'" in w or w.endswith("ly"):      # I'll, they're, let's, unironically
        return False
    try:
        from wordfreq import zipf_frequency
    except Exception:  # noqa: BLE001
        return True
    return zipf_frequency(w, "en") < 4.0

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
_IM_BARE = re.compile(r"^\W*(?:(?:no|nah|yeah|yo|actually|oh|and|also|so|me)\W+){0,2}i'?m\s+" + _NAME_TOKEN + r"(?:\W+(?:by\s+the\s+way|btw|here|nice\s+to\s+meet\s+(?:you|y\'all|everyone)|pleasure|what\'s\s+up))*\W*$", re.I)
_ANSWER_LEAD = re.compile(r"^\W*(?:(?:uh|um|oh|yeah|yo|bro|well|so|ok|okay)\W+)*(?:(?:it's|its|it\s+is|i'?m|i\s+am|my\s+name(?:'s|\s+is)|name's|call\s+me|this\s+is)\s+)?", re.I)
# Closing-clause intro: 'Hey Wren, you there? It's Sam.' (demo 10-04). Capitalized name only.
_CLAUSE_INTRO = re.compile(r"(?:^|[.,!?;]\s*)(?:this\s+is|it's|it\s+is)\s+" + _NAME_TOKEN + r"(?:\s+here)?\W*$", re.I)
_NAME_LEAD = re.compile(r"^\W*(?:(?:uh|um|oh|yeah|yo|bro|well|so|ok|okay)\W+)*(?:it's|its|it\s+is|i'?m|i\s+am|"
                        r"my\s+name(?:'s|\s+is)|name's|call\s+me|this\s+is)\s", re.I)
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
            or _SLUR_NAME.fullmatch(tok) or tok.lower() in _agent_words()):
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
        self.aliases: dict[str, str] = {}       # diarizer folded label -> surviving label
        self.owner = None                       # owner_voice.Adopter (the owner-name hook), or None
        self._lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------ the owner (owner_voice.Adopter)
    def _owner_of(self, label) -> str | None:
        """The owner's name if this label (after merges) is adopted as the owner, else None."""
        o = self.owner
        if o is None or not label:
            return None
        try:
            return o.owner_name(self._resolve(label))
        except Exception:  # noqa: BLE001 - the hook must never break naming
            return None

    def _owner_first(self) -> str:
        """First word of the stored owner name (casefolded) while a voice is stored, else ''."""
        o = self.owner
        try:
            name = o.stored_name() if o is not None else None
        except Exception:  # noqa: BLE001
            name = None
        w = str(name or "").split()
        return w[0].casefold() if w else ""

    def _is_owner_name(self, name) -> bool:
        first = self._owner_first()
        w = str(name or "").split()
        return bool(first) and bool(w) and w[0].casefold() == first

    def _ask_held(self, label) -> bool:
        """While a voice is stored, a label isn't asked its name until the owner check has had its turns (it may be
        the owner, whose spoken name would be refused before adoption)."""
        o = self.owner
        if o is None:
            return False
        try:
            if not o.has_voice():
                return False
            import owner_voice as _ov
            return o.seen(self._resolve(label)) < _ov.ADOPT_TURNS
        except Exception:  # noqa: BLE001
            return False

    def call_names(self) -> list[str]:
        """Every name this call has learned (bound to a voice) or heard (said in the room, not yet bound), for the
        name scrub (docs/designs/experience_memory.md, D2). The owner's saved name is handled separately."""
        with self._lock:
            names = [p.name for p in self.people.values() if p.name]
            names += list(self.heard)
        return names

    def known_names(self, stored: bool = False) -> list[str]:
        """Names Atlas knows belong to voices in this call: learned names, plus the owner's while a label is adopted
        (or, with `stored`, whenever the owner's voice is stored: name protection must not wait for adoption)."""
        names = [p.name for p in list(self.people.values()) if p.name]
        o = self.owner
        try:
            if o is not None and o.stored_name() and (stored or o.labels()):
                names.append(o.stored_name())
        except Exception:  # noqa: BLE001
            pass
        return names

    def owner_adopted(self, label: str) -> None:
        """A label was just adopted as the owner: their name is no longer an unmatched name heard in the room, and it
        is a room name (as `_assign` does for learned names)."""
        o = self.owner
        name = o.stored_name() if o is not None else None
        if not name:
            return
        with self._lock:
            for k in [k for k in self.heard if k.split()[:1] == [name.split()[0].casefold()]]:
                self.heard.pop(k, None)
            self.heard.pop(name.lower(), None)
        try:
            import conversation_dynamics as _cd
            _cd.note_room_name(name)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if self.store and self.store.exists():
            try:
                self._prints = json.loads(self.store.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                logger.warning("voiceprints unreadable: %s", e)

    def _write_prints(self) -> None:
        """voiceprints.json written atomically (temporary file, sync, replace)."""
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_name(self.store.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._prints, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.store)
        except Exception:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise

    def _save_print(self, label: str, name: str) -> None:
        if not (self.store and self.voice_lookup):
            return
        try:
            c = self.voice_lookup(label)
            if c is None:
                return
            self._prints[name] = [round(float(x), 5) for x in c]
            self._write_prints()
        except Exception as e:  # noqa: BLE001
            logger.warning("voiceprint save failed: %s", e)

    def drop_prints(self, name, voice, sim: float = 0.70) -> int:
        """"Forget my voice": remove saved voiceprints named like the owner (first word, any case) or sounding like
        them (similarity >= sim to `voice`). Returns how many were removed."""
        import numpy as np
        first = str(name or "").split()[:1]
        first = first[0].casefold() if first else ""
        v = None if voice is None else np.asarray(voice, dtype=np.float32).reshape(-1)
        with self._lock:
            drop = []
            for k, c in self._prints.items():
                if first and str(k).split()[:1] and str(k).split()[0].casefold() == first:
                    drop.append(k)
                    continue
                if v is not None:
                    c = np.asarray(c, dtype=np.float32).reshape(-1)
                    den = float(np.linalg.norm(c) * np.linalg.norm(v)) or 1.0
                    if c.shape == v.shape and float(c @ v) / den >= sim:
                        drop.append(k)
            for k in drop:
                self._prints.pop(k, None)
            if drop and self.store:
                try:
                    self._write_prints()
                except Exception as e:  # noqa: BLE001
                    logger.warning("voiceprint cleanup failed: %s", e)
        if drop:
            logger.info("👤 removed %d saved voiceprint(s) of the owner", len(drop))
        return len(drop)

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
                if name in taken or self._is_owner_name(name):    # the owner is named by voice only (owner_voice)
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
            self.aliases.clear()

    def merge(self, old: str, new: str) -> None:
        """The diarizer decided `old` was a fragment of `new` (one person split in
        two). A name learned on either side survives on `new`."""
        if not old or not new or old == new:
            return
        with self._lock:
            self.aliases[old] = new
            o = self.people.pop(old, None)
            if o is None or not o.name:
                return
            n = self.people.get(new)
            if n is None:
                n = self.people[new] = Person(new)
            if not n.name:
                n.name, n.source = o.name, o.source
                logger.info("👤 %s folded into %s: still %s", old, new, o.name)

    def _resolve(self, label):
        seen = set()
        while label in self.aliases and label not in seen:
            seen.add(label)
            label = self.aliases[label]
        return label

    def _person(self, label: str) -> Person:
        p = self.people.get(label)
        if p is None:
            p = self.people[label] = Person(label)
            guess = self._match_print(label)
            if guess:
                p.probable = guess
                logger.info("👤 %s sounds like %s (voiceprint, unconfirmed)", label, guess)
        return p

    def _assign(self, p: Person, name: str, source: str) -> str | None:
        if self._owner_of(p.label):
            return None                      # the owner's name changes only from the dashboard
        if self._is_owner_name(name):
            # Nobody else carries the owner's name, from any source (intro, answer, correction, a guess's "yes"):
            # stop asking this voice (it isn't refusing, so not "declined")
            logger.info("👤 %s said the owner's name; not naming that voice", p.label)
            p.asks = self.max_asks
            p.asked_at = None
            try:   # the pending file keeps the call's names for the post-call pass (docs/designs/experience_memory.md, D2)
                import call_boundary as _cbn
                _cbn.owner_event("namesake", name=name)
            except Exception:  # noqa: BLE001
                pass
            return None
        for other in self.people.values():   # one name, one voice
            if other is not p and other.name == name and source != "voice":
                other.name, other.source = None, ""
        try:   # the pending file keeps the call's names for the post-call pass (docs/designs/experience_memory.md, D2)
            import call_boundary as _cbn
            _cbn.name_event("learned", name)
        except Exception:  # noqa: BLE001
            pass
        changed = p.name != name
        try:
            import conversation_dynamics as _cd
            _cd.note_room_name(name)
        except Exception:  # noqa: BLE001
            pass
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
            if self._owner_of(label):
                return None                  # nothing said in the call renames the owner
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
            # only in a greeting turn, so 'what are we playing? it's Minecraft.' never names anyone
            m = _CLAUSE_INTRO.search(body) if re.match(r"^\W*(?:hey|hi|hello|yo|sup)\b", body, re.I) else None
            name = _clean_name(m[1], need_cap=True) if m else None
            if name and name.lower() not in self.agent_names:
                return self._assign(p, name, "intro")
            words = body.split()
            asked = p.asked_at is not None and now - p.asked_at <= self.answer_window_s
            # Intros usually lead the turn: "I'm Sam. how's it going", "hey Max, I'm Sam".
            # Judge the first sentence, with a leading "hey <agent>," vocative dropped.
            parts = re.split(r"(?<=[.!?])\s+", body)
            first = parts[0]
            # "Hi, Wren. I'm Riley. ..." (demo take 5): a sentence that is only a greeting
            # to the agent carries no intro, so judge the sentence after it.
            g0 = re.match(r"^\W*(?:hey|hi|hello|yo|sup)\W+([A-Za-z]+)\W*$", first, re.I)
            # A lone greeting sentence never carries an intro, whoever it greets.
            if g0 and len(parts) > 1:
                parts = parts[1:]
                body_rest = " ".join(parts)
                first = parts[0]
                if len(parts) <= 2:
                    body = body_rest
            v = re.match(r"^\W*((?:hey|hi|hello|yo|sup)\W+)?([A-Za-z]+),\s*", first, re.I)
            if v and v[2].lower() in self.agent_names:
                first = (v[1] or "") + first[v.end():]
            # "I'm Riley, hi Wren" / "and I'm Riley. hi Wren" (first_run sim 10-04): a trailing
            # greeting to the agent is not part of the intro.
            if len(re.split(r"(?<=[.!?])\s+", body)) <= 2:
                g = re.search(r"[,.!]?\s*(?:hi|hey|hello|yo)\s+([A-Za-z]+)\W*$", body, re.I)
                if g and g[1].lower() in self.agent_names:
                    first = re.split(r"(?<=[.!?])\s+", body[:g.start()].strip(), maxsplit=1)[0]
            core = _INTRO_TAIL.sub("", first).split()  # "I'm Dana, nice to meet you" is still 2 words of intro
            m = (_IM_GREETED.search(body) or _IM_GREETED.search(first)
                 or ((_IM_BARE.search(first) if asked or len(core) <= 3 else None)))
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
                    # "it's Marcus" / "I'm Tangy" fill the name slot: any handle. A bare word
                    # ("Close.", "Better.") has to look like a name or a handle.
                    if name and ("'" in name.rstrip("'s") or name.lower().endswith("ly")
                                 or (not _NAME_LEAD.match(body) and not _plausible_answer_name(name))):
                        name = None
                    if name and name.lower() not in self.agent_names:
                        p.asked_at = None
                        return self._assign(p, name, "answer")
            if p.probable and asked and re.match(r"^\W*(?:yeah|yes|yep|yup|that's\s+me|correct|mhm)\b", body, re.I):
                return self._assign(p, p.probable, "confirmed")
            return None

    def peek_intro(self, text: str) -> str | None:
        """Name a turn introduces, without changing this book (for UI labels)."""
        scratch = NameBook(store=None)
        scratch.agent_names = set(self.agent_names)
        try:
            name = scratch.learn("S0", text)
        except Exception:  # noqa: BLE001
            return None
        return None if name and self._is_owner_name(name) else name

    def _note_vocative(self, body: str) -> None:
        m = _VOCATIVE.match(body)
        name = _clean_name(m[1]) if m else None
        if name and name.lower() not in self.agent_names \
                and name not in {p.name for p in self.people.values()} \
                and not (self._is_owner_name(name) and self.owner is not None and self.owner.labels()):
            self.heard[name.lower()] = self.clock()
            try:
                import conversation_dynamics as _cd
                _cd.note_room_name(name)
            except Exception:  # noqa: BLE001
                pass
            try:   # the pending file keeps the call's names for the post-call pass (docs/designs/experience_memory.md, D2)
                import call_boundary as _cbn
                _cbn.name_event("heard", name)
            except Exception:  # noqa: BLE001
                pass
        # People call each other by name mid-line too ("You see, Jefe, even...", "See,
        # Prof?", "bye peach,"); sim 11-01: those names were never learned, so every
        # later line to Jefe/Prof/Peach reached the agent. Room names only (the gate's
        # address check), never a voice's name.
        try:
            import conversation_dynamics as _cd
            w = _cd._room_vocative(body, tuple(self.agent_names))
            if w and not _cd._is_room_name(w) and _clean_name(w):
                _cd.note_room_name(w)
        except Exception:  # noqa: BLE001
            pass

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
        own = self._owner_of(label)
        if own:
            return own
        p = self.people.get(self._resolve(label or ""))
        return p.name if p else None

    def display(self, label: str | None) -> str:
        """Human-facing label for UI/logs."""
        if not label or label in ("user", "you", "YOU"):
            return "you"
        own = self._owner_of(label)
        if own:
            return own
        p = self.people.get(self._resolve(label))
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
                own = self._owner_of(l)
                if own:
                    lines.append(f"- {l}{' (talking now)' if l == current else ''}: you recognize this voice: it's "
                                 f"{own}; if asked how, just say you know their voice.")
                    continue
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
        if self._owner_of(current) or self._ask_held(current):
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
        # sim_self_tools 10-05: 8B copied the quoted example ("wait, who's this?") as its
        # WHOLE reply and dropped the answer. No quoted example; answer comes first.
        return (f"You don't know who {current} is yet. Answer what they said first; then, "
                "only if it fits, add a short ask for their name in your own words (once).")

    def sanitize(self, text: str, target: str | None = None) -> str:
        """Replace any spoken S-tag with a name, 'you', or 'someone'."""
        def sub(m):
            lab = m[1].upper()
            own = self._owner_of(lab)
            if own:
                return "you" if lab == (target or "").upper() else own
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
