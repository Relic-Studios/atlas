"""Privacy core of the experience memory (design docs/designs/experience_memory.md, D3, D7, D8, S3, S5, S6).

Pure text functions, no model and no state:
  * `is_private`, the off-the-record and forget phrases, `content_nodes` and the slur check, moved here from the
    retired hypergraph memory so every module that filters private text keeps working without it;
  * `content_words` (the glossary's content words) and the shared keyword matcher;
  * the off-the-record and "remember that" patterns (S3);
  * the name scrub (the cheap redaction: no model, no tagger) and full redaction for the post-call pass (D3 step 2);
  * the validator's rules (D3 step 5), each a function returning a reason string or None.

The name tagger is passed in (anything with `persons(text) -> list[str]`), so this module never loads it: the server
uses the rules that need no tagger, and the post-call pass adds the tagger (D3).
"""
from __future__ import annotations

import functools
import re
from typing import Callable, Iterable, Optional, Sequence

# ------------------------------------------------------------------ moved from hypergraph_memory.py
_STOP = set("""a an the and or but so to of in on at for with is are was were be been being am i you he she
it we they me him her us them my your his its our their this that these those there here what which who whom
whose when where why how do does did done doing have has had having not no yes yeah yep nah ok okay just like
really very much many more most some any all can could would should will shall may might must if then than
too also only even still about into over under out up down off again once oh um uh hmm lol bro dude man guys
get got gonna wanna kinda sorta thing things stuff something anything nothing everything one two lot
know think thought say said tell told go going went come came see saw look make made want need let right
well now time today yesterday tomorrow im ive dont cant wont didnt isnt its thats whats youre theyre were
wait anyway anyways honestly literally basically actually maybe sure fine cool nice good bad huh yo hey""".split())
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_TAG = re.compile(r"\[[^\]]*\]|\bS\d{1,4}\b")

UNCOMMON_ZIPF = 4.2     # "play"/"favourite" (~5) are everyday words, "ska"/"chess" (<4) carry meaning
try:
    from wordfreq import zipf_frequency as _zipf
except Exception:  # pragma: no cover
    _zipf = None


@functools.lru_cache(maxsize=65536)
def zipf(word: str) -> float:
    """English word frequency on the Zipf scale (about 0 for unknown words, 7 for "the"); 0.0 without wordfreq."""
    if _zipf is None or not word:
        return 0.0
    return _zipf(word.lstrip("@").lower(), "en")


def _uncommon(word: str) -> bool:
    """A shared word only counts as evidence if it's uncommon in English too."""
    if _zipf is None:
        return True
    return zipf(word) < UNCOMMON_ZIPF


@functools.lru_cache(maxsize=65536)
def _norm_word(w: str) -> str:
    """Simple plural and tense endings stripped (glossary: "played" and "plays" match "play"). An ending goes only
    when what is left is a common word at least nearly as common as the full word ("always", "speed", "morning"
    stay), and a tense ending never leaves a stop word ("evening" stays)."""
    w = w.strip("'-")
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    cands = []
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        cands = [w[:-1]] + ([w[:-2]] if w.endswith("es") else [])      # "likes" -> "like" (a stop word) is fine
    else:
        for end in ("ing", "ed"):
            if len(w) > len(end) + 2 and w.endswith(end):
                stem = w[:-len(end)]
                cands = [c for c in (stem, stem + "e") if c not in _STOP]   # "evening" is not "even"
                if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
                    cands.append(stem[:-1])                          # stopped -> stop, running -> run
                break
    if cands:
        best = max(cands, key=zipf)
        if zipf(best) >= 3.0 and zipf(best) >= zipf(w) - 1.5:
            return best
    return w


_CONTRACTION = re.compile(r"(?:n't|'ll|'re|'ve|'d|'s|'m)$")


def content_nodes(text: str) -> list:
    out = []
    for w in _WORD.findall(_TAG.sub(" ", text or "").lower().replace("’", "'")):
        w = _CONTRACTION.sub("", w.strip("'-"))      # "I'll", "doesn't", "he's": their base word, then the stop list
        if w.replace("'", "") in _STOP or w in _STOP:
            continue
        w = _norm_word(w.replace("'", ""))
        if len(w) >= 3 and w not in _STOP and w not in out:
            out.append(w)
    return out[:24]


# Never stored at all (whole line dropped), checked before anything else:
#  - secrets: passwords, PINs, keys, card/bank/social-security numbers
#  - where someone lives: "I live at 42 ...", street addresses
#  - the speaker asked: "don't remember this", "off the record", "between us"
_PRIVATE = re.compile(
    r"\b(pass ?words?|passcodes?|pin (?:code|number)|security code|api ?keys?|secret keys?|"
    r"seed phrase|social security|ssn|credit ?card|debit ?card|card number|bank account|"
    r"routing number|login (?:is|was)|my (?:home )?address|i live at|we live at|"
    r"don'?t (?:remember|save|store|record|repeat|tell anyone)|do not (?:remember|save|store|record|repeat)|"
    r"off the record|between (?:you and me|us)|keep (?:this|it|that) (?:a )?secret|"
    r"(?:it'?s|this is) (?:private|confidential))\b", re.I)
# a street address = number + Capitalised name + street word ("42 Wallaby Way");
# case-sensitive so "drove 20 minutes down the road" is not an address
_STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){1,3}(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|"
                     r"Lane|Ln|Drive|Dr|Way|Court|Ct|Place|Pl|Terrace|Crescent)\b")
_FORGET = re.compile(r"\b(?:forget (?:that|this|what i (?:just )?said|it|everything i said)|"
                     r"scratch that,? (?:don'?t|do not) remember|delete that)\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def is_private(text: str) -> bool:
    """A line with a secret, an address, an email or a phone number is not stored at all
    (a redacted 'my number is [number]' is useless as a memory and still says too much)."""
    t = text or ""
    return bool(_PRIVATE.search(t) or _STREET.search(t) or _EMAIL.search(t) or _PHONE.search(t))


def is_forget_request(text: str) -> bool:
    return bool(_FORGET.search(text or ""))


def has_slur(text: str) -> bool:
    try:
        import speech_safety
        return bool(speech_safety.has_slur(text or ""))
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------ agent names
@functools.lru_cache(maxsize=1)
def _registry_agent_words() -> frozenset:
    try:
        import agent_registry
        return frozenset(agent_registry.all_agent_words())
    except Exception:  # noqa: BLE001
        return frozenset()


def agent_words(extra: Iterable[str] = ()) -> frozenset:
    """Every name and nickname an agent answers to, lower case."""
    return _registry_agent_words() | frozenset(w.lower() for w in extra if w)


def refresh_agent_words() -> None:
    """After an agent is created, deleted or renamed: the name checks and the trigger patterns use the new names."""
    _registry_agent_words.cache_clear()
    _patterns.cache_clear()


# ------------------------------------------------------------------ content words and the matcher
_PLACEHOLDER = re.compile(r"^person\d+$")
_NOT_CONTENT = {"owner", "someone", "somebody", "group", "everyone", "call", "person"}   # PersonN reads as "person"


def content_words(text: str, agents: Iterable[str] = ()) -> set:
    """The glossary's content words: content_nodes minus owner/someone/group/everyone/call, PersonN and agent names."""
    skip = agent_words(agents)
    return {w for w in content_nodes(text) if w not in _NOT_CONTENT and w not in skip and not _PLACEHOLDER.match(w)}


def scoring_words(words: Iterable[str], freq_limit: float = 5.5) -> set:
    """Content words that count toward a snippet score: rarer than `freq_limit` on the Zipf scale (S6)."""
    return {w for w in words if zipf(w) < freq_limit}


def overlap_score(query: set, fields: Sequence[tuple]) -> float:
    """Shared matcher core: sum over (words, weight) fields of weight x shared words. Lore and lessons call it with
    their own word sets and weights; recall calls it with one field of weight 1."""
    return sum(weight * len(query & set(words)) for words, weight in fields)


class Matcher:
    """Keyword recall over many short items (D1 Size, D4): an index from each content word to the items holding it,
    so a lookup only reads items that share a word with the turn."""

    def __init__(self, freq_limit: float = 5.5, agents: Iterable[str] = ()):
        self.freq_limit = freq_limit
        self.agents = tuple(agents)
        self._words: dict = {}          # item id -> frozenset of scoring words
        self._index: dict = {}          # word -> set of item ids
        self._order: dict = {}          # item id -> recency (larger = newer)

    def add(self, item_id: str, text: str, recency: float = 0.0) -> None:
        self.remove(item_id)
        ws = frozenset(scoring_words(content_words(text, self.agents), self.freq_limit))
        self._words[item_id] = ws
        self._order[item_id] = recency
        for w in ws:
            self._index.setdefault(w, set()).add(item_id)

    def remove(self, item_id: str) -> None:
        for w in self._words.pop(item_id, ()):
            ids = self._index.get(w)
            if ids is not None:
                ids.discard(item_id)
                if not ids:
                    del self._index[w]
        self._order.pop(item_id, None)

    def recency(self, item_id: str) -> float:
        return self._order.get(item_id, 0.0)

    def __len__(self) -> int:
        return len(self._words)

    def lookup(self, text: str, min_score: int = 2, limit: int = 3,
               exclude: Callable[[str], bool] = lambda _i: False) -> list:
        """[(item id, score)], best first; ties go to the newer item. Turns with fewer than two content words
        never match (D4)."""
        q = content_words(text, self.agents)
        if len(q) < 2:
            return []
        q = scoring_words(q, self.freq_limit)
        hits: dict = {}
        for w in q:
            for i in self._index.get(w, ()):
                hits[i] = hits.get(i, 0) + 1
        ranked = sorted(((i, n) for i, n in hits.items() if n >= min_score and not exclude(i)),
                        key=lambda p: (-p[1], -self._order.get(p[0], 0.0)))
        return ranked[:limit]


# ------------------------------------------------------------------ S3 patterns
_ADDRESS_WORDS = ("hey", "yo", "thanks", "ok", "okay", "yeah", "no", "bro")     # S6, tuned


def _clause_start(agents: Iterable[str] = ()) -> str:
    names = sorted({re.escape(a) for a in agent_words(agents) if a}, key=len, reverse=True)
    alt = "|".join(["please", "ok", "okay", "and", "actually", "wait", "no", "nah", "oh", "just", "um", "uh",
                    "so", "yeah", "also"] + names)       # lead-ins: "actually forget that", "wait scratch that"
    return rf"(?:^|[.,!?]\s*|\b(?:{alt})[,\s]+)"


_SPAN_BODY = (r"(?:(?:don'?t|do not) (?:remember|save|record|store) (?:this|any of this|what i'?m about to say)\b"
              r"|between you and me\b|keep (?:this|it) (?:a )?secret\b)")
_BACK_BODY = (r"(?:(?:don'?t|do not) (?:remember|save|record|store) (?:that|what i (?:just )?said)\b"
              r"|forget (?:that|what i (?:just )?said)\b|strike that\b|delete that\b|scratch that\b)")
_OFF_ANYWHERE = re.compile(r"(?<!back on )\boff the record\b", re.I)
_SPAN_END = re.compile(r"\bback on the record\b", re.I)
_REMEMBER_BODY = r"(?:please\s+)?remember (?:that|this)[:,]?\s+(?P<clause>.+)"
_REMEMBER_NOT = re.compile(r"^(?:time|when|day|night|game|one)\b", re.I)


@functools.lru_cache(maxsize=32)
def _patterns(agents: tuple) -> tuple:
    cs = _clause_start(agents)
    return (re.compile(cs + _SPAN_BODY, re.I), re.compile(cs + _BACK_BODY, re.I),
            re.compile(cs + _REMEMBER_BODY, re.I | re.S))


def off_record_kind(text: str, agents: Iterable[str] = ()) -> Optional[str]:
    """'end' for "back on the record", 'span' for a trigger that also opens a span, 'back' for a backward-only
    trigger, or None (S3). "I don't remember his name" is none of them."""
    t = (text or "").strip()
    if not t:
        return None
    if _SPAN_END.search(t):
        return "end"
    span, back, _ = _patterns(tuple(sorted(agents)))
    if _OFF_ANYWHERE.search(t) or span.search(t):
        return "span"
    if back.search(t):
        return "back"
    return None


def remember_clause(text: str, agents: Iterable[str] = ()) -> Optional[str]:
    """The clause after an imperative "remember that ...", or None for questions ("do you remember that?") and for
    "remember that time we ..." (S3)."""
    t = (text or "").strip()
    if not t or t.endswith("?"):
        return None
    m = _patterns(tuple(sorted(agents)))[2].search(t)
    if not m:
        return None
    clause = m.group("clause").strip().rstrip(".! ").strip()
    if not clause or _REMEMBER_NOT.match(clause):
        return None
    return clause


_RECALL_ASK = re.compile(
    r"\b(?:(?:do|did|don't|dont|can|could|would|will) (?:you|ya|u) (?:even |still )?(?:remember|recall)"
    r"|(?:you|ya|u) (?:remember|recall)\b"
    r"|(?<!\bi )(?<!n't )(?<!not )remember (?:when|me|us|what|how|who|where|that time|the time|last|the|that|"
    r"those|this)\b"
    r"|recall (?:when|what|me|us|the time|that time)\b"
    r"|(?:have|haven't|did) we (?:ever )?(?:meet|met|play|played|talk|talked|speak|spoken)\b"
    r"|(?:do|did) (?:you|ya|u) know (?:me|who i am)\b"
    r"|you (?:were|was) there\b"
    r"|(?:what|where|how|who|which|when)\b[^.?!]{0,40}\blast (?:time|week|night|call|session)\b"
    r"|last time (?:we|you|i)\b)", re.I)


_EARLIER_ASK = re.compile(
    r"\b(?:earlier|a (?:minute|while|bit|second) ago|what did (?:\w+ ){0,3}say|what was (?:that|it) (?:again|you said)"
    r"|you (?:just )?(?:said|mentioned|told (?:me|us))|(?:said|told|mentioned) (?:that )?(?:earlier|before))\b", re.I)


def asks_about_earlier(text: str) -> bool:
    """A line that asks about something said earlier in this call ("what did he say earlier?", "you said...")."""
    return bool(_EARLIER_ASK.search(text or ""))


def past_reference(text: str) -> bool:
    """A line that asks the agent to recall something from before ("do you remember me?", "remember when we...",
    "what did we play last time?"); design D6's wording rule is repeated on such turns. A line that only mentions the
    past ("I don't remember his name", "yesterday I went out") is not one."""
    return bool(_RECALL_ASK.search(text or ""))


_ABOUT_OWNER = re.compile(r"\b(?:your|ur) (?:owner|creator|maker|developer|dev)\b"
                          r"|\bwho (?:made|created|built|owns|programmed) (?:you|u)\b", re.I)
_QUESTION = re.compile(r"\?|^\W*(?:what|who|where|when|why|how|does|did|is|was|has|do|can|tell me)\b", re.I)


def about_owner(text: str, owner_name: Optional[str] = None) -> bool:
    """A line that asks about the owner: "your creator", "who made you", or a question naming the owner's saved
    name ("what does <name> like?"), but not a line that only addresses the owner ("<name>, pass the ammo")."""
    t = text or ""
    if _ABOUT_OWNER.search(t):
        return True
    if not _QUESTION.search(t):
        return False
    for part in (owner_name or "").split():
        if len(part) < 2:
            continue
        p = re.escape(part)
        if zipf(part) >= COMMON_NAME_ZIPF:      # a name that is also a word ("Will"): only the capitalized name,
            hit = re.search(r"(?<=\w\W)" + re.escape(part.capitalize()) + r"(?![\w])", t)   # not a sentence start
        else:
            hit = re.search(r"(?<![\w])" + p + r"(?![\w])", t, re.I)
        if hit and not re.match(r"\W*(?:(?:hey|yo|ok|okay|oi)\W+)?" + p + r"\W*[,!:]", t, re.I):
            return True
    return False


# ------------------------------------------------------------------ S5 lists
FIRST_PERSON = {"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself"}
FIRST_PERSON_MAP = [                      # longest first; applied to whole words, any case
    ("i'm", "the owner is"), ("i've", "the owner has"), ("i'd", "the owner would"), ("i'll", "the owner will"),
    ("myself", "the owner"), ("mine", "the owner's"), ("my", "the owner's"), ("me", "the owner"), ("i", "the owner"),
]
TRAIT_VERBS = {"is", "are", "has", "have", "works", "lives", "likes", "loves", "hates", "feels", "thinks", "wants",
               "needs", "prefers", "believes",
               "work", "live", "like", "love", "hate", "feel", "think", "want", "need", "prefer", "believe"}   # "they love"
DIRECTIVE_WORDS = {"will", "would", "should", "must", "always", "never", "promised"}
DIRECTIVE_PHRASES = ("going to", "has to", "have to", "agreed to")
_FROM_NOW_ON = re.compile(r"\bfrom now on\b", re.I)
CODE_WORDS = {"code", "coding", "script", "scripts", "python", "javascript", "programming", "hack", "hacking",
              "exploit", "compile", "terminal", "powershell", "bash"}
SENSITIVE = {
    "health": {"sick", "illness", "ill", "diagnosis", "diagnosed", "therapy", "therapist", "medication", "meds",
               "depression", "depressed", "anxiety", "hospital", "pregnant", "surgery", "cancer", "disease", "adhd",
               "autism", "autistic", "disorder", "rehab", "suicide", "suicidal"},
    "sexuality": {"gay", "lesbian", "bisexual", "bi", "straight", "sexuality", "queer", "trans", "transgender",
                  "sex", "porn", "goon", "gooning", "gooner", "horny", "nsfw", "nudes", "onlyfans", "hentai", "kinky",
                  "sex life", "dating life", "dating"},
    "drugs": {"drug", "alcohol", "marijuana", "weed", "cocaine", "meth", "stoned", "drunk", "vape", "vaping",
              "edibles", "shrooms",
              "molly", "xanax", "hungover", "wasted"},
    "religion": {"church", "mosque", "temple", "synagogue", "religion", "religious", "pray", "praying", "atheist",
                 "christian", "muslim", "jewish", "hindu", "buddhist"},
    "politics": {"vote", "voted", "voting", "democrat", "democrats", "republican", "republicans", "election",
                 "liberal", "conservative", "trump", "biden", "politics", "political", "party politics"},
    "money": {"salary", "debt", "loan", "loans", "rent", "bank", "income", "broke", "paycheck", "mortgage", "savings",
              "owe", "owes", "paid"},
    "children": {"kid", "kids", "child", "children", "son", "daughter", "toddler", "baby"},
    "location": {"address", "neighborhood", "neighbourhood", "zip", "postcode", "lives in", "live in", "moved to",
                 "my city", "my town", "my state", "is from", "i'm from", "grew up in", "based in", "lives near",
                 "live near", "hometown"},
    "setup": {"gpu", "hardware", "server", "qwen", "model", "model", "remote box", "vllm", "ollama", "4090",
              "model weights", "language model", "llm", "model", "where atlas runs", "atlas runs on", "self-hosted"},
}
# S5 "school (with an age)": a school word together with an age is about a child
_SCHOOL = re.compile(r"\b(?:school|kindergarten|preschool|daycare|elementary|middle school|grade)\b", re.I)
_AGE = re.compile(r"\b\d{1,2}\s*(?:years?\s*old|yrs?\s*old|y/?o)\b|\b(?:\d|1[0-2])(?:st|nd|rd|th)\s+grade\b|"
                  r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+grade\b", re.I)
# S5 "city names": the best-known cities, US states and countries; a place in a fact about the owner is location.
# Places that are also common first names or game characters (Austin, Charlotte, Georgia, Phoenix...) are left out.
PLACES = {
    "new york", "nyc", "los angeles", "chicago", "houston", "philadelphia", "san antonio", "san diego",
    "dallas", "jacksonville", "san jose", "fort worth", "columbus", "indianapolis",
    "san francisco", "seattle", "denver", "nashville", "oklahoma city", "el paso", "boston", "portland", "las vegas",
    "detroit", "memphis", "louisville", "baltimore", "milwaukee", "albuquerque", "tucson", "fresno", "sacramento",
    "kansas city", "atlanta", "miami", "raleigh", "omaha", "minneapolis", "tulsa", "cleveland", "tampa", "orlando",
    "pittsburgh", "cincinnati", "st louis", "st. louis", "new orleans", "honolulu", "anchorage", "salt lake city",
    "boise", "spokane", "richmond", "buffalo", "toronto", "montreal", "vancouver", "calgary", "ottawa", "london",
    "manchester", "birmingham", "glasgow", "edinburgh", "dublin", "paris", "berlin", "munich", "hamburg", "madrid",
    "barcelona", "rome", "milan", "amsterdam", "brussels", "vienna", "zurich", "stockholm", "oslo", "copenhagen",
    "helsinki", "warsaw", "prague", "budapest", "lisbon", "athens", "istanbul", "moscow", "kyiv", "dubai", "tokyo",
    "osaka", "seoul", "beijing", "shanghai", "hong kong", "taipei", "singapore", "bangkok", "manila", "jakarta",
    "mumbai", "delhi", "bangalore", "sydney", "melbourne", "brisbane", "perth", "auckland", "mexico city",
    "sao paulo", "são paulo", "rio de janeiro", "buenos aires", "lima", "bogota", "cairo", "lagos",
    "nairobi", "johannesburg", "cape town",
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "west virginia", "wisconsin", "wyoming",
    "canada", "mexico", "england", "scotland", "wales", "ireland", "france", "germany", "spain", "italy",
    "netherlands", "poland", "sweden", "norway", "finland", "denmark", "australia", "japan", "china", "korea",
    "india", "brazil", "argentina", "philippines", "vietnam", "thailand", "russia", "ukraine",
}
_PLACE = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(p) for p in sorted(PLACES, key=len, reverse=True))
                    + r")(?![\w])", re.I)
PERSON_SUBJECTS = {"someone", "somebody", "he", "she", "they", "him", "her", "them", "owner", "friend",
                   "everybody's", "someone's"}
ALLOWED_TRAIT_SUBJECTS = {"group", "everyone", "everybody", "call"}

_TOKEN = re.compile(r"[A-Za-z0-9_@][A-Za-z0-9_'’@-]*")
_HANDLE = re.compile(r"(?<![\w@])(?:@\w+|(?=\w*[A-Za-z])(?=\w*[\d_])\w{3,})(?!\w)")
_NOT_HANDLE = re.compile(r"^(?:\d+(?:st|nd|rd|th|am|pm|k|s|x|m|h|v\d+)|[a-z]?\d+v\d+|\d+[a-z]?)$", re.I)
_LABEL = re.compile(r"\b(?:S\d{1,4}|Person\d{1,4})\b")


def handles(text: str) -> list:
    """Handle-like tokens: an @ word, or a word mixing letters with digits or underscores (not 5pm, 2v2, 90s)."""
    return [m.group(0) for m in _HANDLE.finditer(text or "") if not _NOT_HANDLE.match(m.group(0).lstrip("@"))]


def _addresses(words: list, i: int, line_alone: bool) -> bool:
    """A word addresses someone when it follows an addressing word, is followed by a comma, or is the whole line."""
    if line_alone:
        return True
    w = words[i]
    if w.endswith(","):
        return True
    return i > 0 and words[i - 1].strip(",.!?").lower() in _ADDRESS_WORDS


def _bare(w: str) -> str:
    return w.strip(".,!?;:\"'()[]{}").replace("’", "'")


def _sentence_starts(words: list) -> set:
    starts, prev_end = set(), True
    for i, w in enumerate(words):
        if prev_end:
            starts.add(i)
        prev_end = w.endswith((".", "!", "?"))
    return starts


# ------------------------------------------------------------------ the name scrub and redaction
# Games, apps and platforms the scrub would otherwise take for names (rare capitalized words), kept everywhere the
# owner's allow-list is: none of them is also a person's name (so "Destiny", "Zelda", "Hades" are left out).
_DEFAULT_ALLOW_ALL = frozenset({
    "valorant", "palworld", "phasmophobia", "chatgpt", "helldivers", "genshin", "tarkov", "kahoot", "starfield",
    "stardew", "elden", "terraria", "hokage", "tiktok", "warzone", "roblox", "cyberpunk", "skyrim", "dota", "csgo",
    "hogwarts", "fortnite", "discord", "overwatch", "minecraft", "twitch", "pokemon", "youtube", "fallout",
    "battlefield", "pubg", "rainbow", "siege", "witcher", "doom", "sims", "hearthstone", "warcraft", "starcraft",
    "runescape", "osrs", "smite", "paladins", "brawlhalla", "deltarune", "undertale", "celeste", "hollow", "silksong",
    "cuphead", "subnautica", "valheim", "satisfactory", "factorio", "rimworld", "oxygen", "spotify", "netflix",
    "reddit", "instagram", "snapchat", "whatsapp", "xbox", "playstation", "nintendo", "switch", "steam", "epic",
    "ubisoft", "blizzard", "riot", "openai", "anthropic", "gemini", "copilot", "linux", "windows", "android", "iphone",
})
COMMON_NAME_ZIPF = 5.0     # names that are also common words (Will, May, Mark, Hope, Bill): S6, tuned
RARE_CAP_ZIPF = 3.5        # S6, tuned
# only the rare ones: a common word ("riot", "switch") is never taken for a name, and allowing it would let a friend
# nicknamed that through
DEFAULT_ALLOW = frozenset(w for w in _DEFAULT_ALLOW_ALL if zipf(w) < RARE_CAP_ZIPF)
RARE_LOWER_ZIPF = 1.0      # S6, tuned on development calls 2026-11-04: 2.5 replaced rare English words
                           # ("incisiveness"); only words wordfreq hardly knows (handles, made-up names) go


def _names_set(names: Iterable[str]) -> set:
    out = set()
    for n in names or ():
        n = (n or "").strip()
        if n:
            out.add(n.lower())
            out.update(p.lower() for p in n.split() if len(p) >= 2)
    return out


def scrub(text: str, names: Iterable[str] = (), owner_name: Optional[str] = None, *, agents: Iterable[str] = (),
          allow: Iterable[str] = (), owner_namesake: bool = False, from_owner: bool = True,
          tagger=None, active_agent: Optional[Iterable[str]] = None) -> str:
    """Name scrub (glossary) and, with a tagger, the redaction steps of D3 step 2 that apply to a single text:
    allow-list protected; handles to "someone"; the owner's saved name to "the owner" (to "someone" in a non-owner's
    line when a non-owner used it this call); names from the call's list to "someone" (common-word names only when
    capitalized or addressing); tagger persons to "someone"; rare words to "someone" (a capitalized first word only
    when it addresses someone). "the owner" and placeholders are never replaced. Agent names stay, except that
    with `active_agent` (the names of the agent active when a human line was said) an agent's name stays only when it
    addresses that agent: a friend called Max is not the agent Max."""
    t = text or ""
    if not t.strip():
        return t
    allow_l = {a.lower() for a in allow or () if a}
    agent_l = agent_words(agents)
    active_l = {a.lower() for a in active_agent} if active_agent is not None else None
    owner_parts = _names_set([owner_name]) if owner_name else set()
    call_names = _names_set(names) - agent_l - allow_l
    tagged = set()
    if tagger is not None:          # errors propagate: the pass must not go on with an unscrubbed text
        for p in tagger.persons(t):
            tagged.update(_names_set([p]))
    words = t.split()
    starts = _sentence_starts(words)
    line_alone = len([w for w in words if _bare(w)]) == 1
    out = []
    for i, raw in enumerate(words):
        core = _bare(raw)
        low = core.lower()
        key = low[:-2] if low.endswith("'s") else low
        rep = None
        if low in agent_l and low not in allow_l and active_l is not None and \
                not (low in active_l and _addresses(words, i, line_alone)):           # D3: the allow-list comes first
            out.append(raw.replace(core, "someone", 1))
            continue
        if not core or low in allow_l or low in agent_l or low in ("owner", "someone", "the") or \
                _PLACEHOLDER.match(low):
            out.append(raw)
            continue
        if handles(core):
            rep = "someone"
        elif key in owner_parts and (zipf(key) < COMMON_NAME_ZIPF or (core[:1].isupper() and i not in starts)
                                     or _addresses(words, i, line_alone)):   # "Will" the owner, not "we will"
            rep = "someone" if (owner_namesake and not from_owner) else "the owner"
        elif key in call_names:
            if zipf(key) >= COMMON_NAME_ZIPF:
                capital = core[:1].isupper() and i not in starts
                if capital or _addresses(words, i, line_alone):
                    rep = "someone"
            else:
                rep = "someone"
        elif key in tagged:
            rep = "someone"
        elif core[:1].isupper() and key.isalpha() and key[1:] == core[1:len(key)] and len(key) >= 2:
            # past a sentence start or addressing someone; at a sentence start too when English hardly knows the
            # word (trap S8-2: "Vororb hates pizza for real" kept the name when the tagger missed it)
            if zipf(key) < RARE_CAP_ZIPF and (i not in starts or _addresses(words, i, line_alone)
                                              or zipf(key) < RARE_LOWER_ZIPF):
                rep = "someone"
        elif core.islower() and core.isalpha() and len(core) >= 3 and zipf(low) < RARE_LOWER_ZIPF:
            rep = "someone"
        if rep is None:
            out.append(raw)
        else:
            if key != low:                      # possessive: "Sam's" -> "someone's", "Relic's" -> "the owner's"
                rep += "'s"
            out.append(raw.replace(core, rep, 1))
    return " ".join(out)


def scrub_names_in(text: str, names: Iterable[str], owner_name: Optional[str] = None, **kw) -> str:
    """Write-time name scrub for lore, lessons, predictions and timer messages (no tagger)."""
    return scrub(text, names, owner_name, **kw)


# ------------------------------------------------------------------ validator rules (D3 step 5)
def _clauses(text: str) -> list:
    parts = re.split(r"[.;!?]+|,\s+(?:and|but)\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _subject_before(clause: str, verbs: set, phrases: Sequence[str] = ()) -> Optional[list]:
    """Words of the clause before the first verb in `verbs` (or phrase in `phrases`), lower case; None when absent."""
    low = clause.lower().replace("’", "'")
    toks = [_bare(w).lower() for w in low.split()]
    cut = None
    for i, w in enumerate(toks):
        if w in verbs:
            cut = i
            break
    for p in phrases:
        m = re.search(r"\b" + re.escape(p) + r"\b", low)
        if m:
            j = len(low[:m.start()].split())
            cut = j if cut is None else min(cut, j)
    if cut is None:
        return None
    return toks[:cut]


def _is_person_subject(subject: list, agents: frozenset) -> bool:
    for w in subject:
        if w in PERSON_SUBJECTS or w in agents or _PLACEHOLDER.match(w) or w.startswith("person"):
            return True
        if w.isalpha() and len(w) >= 3 and zipf(w) < RARE_LOWER_ZIPF:     # an unknown word: likely a name
            return True
    return False


def check_names(text: str, names: Iterable[str] = (), owner_name: Optional[str] = None, *,
                agents: Iterable[str] = (), allow: Iterable[str] = (), tagger=None) -> Optional[str]:
    """'name' when the text holds a call name, a label, a handle, the owner's saved name, or (with a tagger) a word
    the tagger marks as a person that is not allowed and not an agent's name."""
    t = text or ""
    if _LABEL.search(t):
        return "name:label"
    if handles(t):
        return "name:handle"
    agent_l = agent_words(agents)
    allow_l = {a.lower() for a in allow or ()}
    words = t.split()
    starts = _sentence_starts(words)
    line_alone = len(words) == 1
    call = _names_set(names) - agent_l - allow_l
    owner_parts = _names_set([owner_name]) if owner_name else set()
    for i, raw in enumerate(words):
        low = _bare(raw).lower()
        low = low[:-2] if low.endswith("'s") else low
        if not low:
            continue
        if low in owner_parts and (zipf(low) < COMMON_NAME_ZIPF or (_bare(raw)[:1].isupper() and i not in starts)
                                   or _addresses(words, i, line_alone)):
            return "name:owner"
        if low in call:
            if zipf(low) >= COMMON_NAME_ZIPF:
                if (_bare(raw)[:1].isupper() and i not in starts) or _addresses(words, i, line_alone):
                    return "name:call"
            else:
                return "name:call"
        if (low.isalpha() and len(low) >= 3 and zipf(low) < RARE_LOWER_ZIPF and low not in agent_l
                and low not in allow_l and low not in ("owner", "someone")):
            return "name:rare"           # a word English hardly knows: a name or handle the scrub didn't place
    if tagger is not None:
        for p in tagger.persons(t):
            ps = _names_set([p])
            if ps and not ps <= (agent_l | allow_l | {"owner", "the", "someone"}):
                return "name:tagger"
    return None


def extra_capitals(text: str, *, agents: Iterable[str] = (), allow: Iterable[str] = ()) -> Optional[str]:
    """'name:capital' when a capitalized word appears anywhere but the fact's first word, other than an agent's
    name, an allow-list word, "I", or part of "the owner" (D7's extra rule for "remember that" facts: a capital
    later in the fact, even at a sentence start, may be a name the server can't check)."""
    agent_l = agent_words(agents)
    allow_l = {a.lower() for a in allow or ()}
    words = (text or "").split()
    for i, raw in enumerate(words):
        core = _bare(raw)
        if not core or not core[:1].isupper() or i == 0:
            continue
        low = core.lower()
        if low in agent_l or low in allow_l or low in ("i", "i'm", "i've", "i'd", "i'll", "owner", "the"):
            continue
        return "name:capital"
    return None


def check_safety(text: str, topics: bool = True, places: bool = False) -> Optional[str]:
    """D3 step 5 safety; `topics` adds the S5 sensitive-topic lists (episodes and owner facts), `places` the S5
    city names (owner facts only: a place in an episode is a trip or a game map, not where the owner lives)."""
    t = text or ""
    if is_private(t):
        return "safety:private"
    if has_slur(t):
        return "safety:slur"
    try:
        import untrusted
        _clean, flags = untrusted.sanitize(t)
        if flags:
            return "safety:instruction"
    except Exception:  # noqa: BLE001
        pass
    low = t.lower()
    toks = set()
    for w in low.split():                 # with possessive and plural forms ("son's", "daughters", "hospitals")
        b = _bare(w).lower()
        b = b[:-2] if b.endswith(("'s", "s'")) else b
        toks |= {b, b[:-1] if b.endswith("s") else b, b[:-2] if b.endswith("es") else b}
    if toks & CODE_WORDS:
        return "safety:code"
    if topics:
        for topic, words in SENSITIVE.items():
            for w in words:
                if (" " in w and re.search(r"(?<![\w])" + re.escape(w) + r"(?![\w])", low)) or w in toks:
                    return f"safety:{topic}"
        if _SCHOOL.search(t) and _AGE.search(t):
            return "safety:children"
        if places and _PLACE.search(t):
            return "safety:location"
    return None


def check_directive(text: str, agents: Iterable[str] = ()) -> Optional[str]:
    agent_l = agent_words(agents) | {"agent", "you"}
    for clause in _clauses(text):
        if _FROM_NOW_ON.search(clause) and any(_bare(w).lower() in agent_l for w in clause.split()):
            return "directive"
        subj = _subject_before(clause, DIRECTIVE_WORDS, DIRECTIVE_PHRASES)
        if subj is not None and any(w in agent_l for w in subj[-SUBJECT_WORDS:]):
            return "directive"
    return None


SUBJECT_WORDS = 2     # tuned on development calls 2026-11-04: the subject is right before the verb ("someone is",
                      # "Ivy will", "the owner really likes"), not anywhere earlier ("Someone claimed to have sold")


def check_traits(text: str, agents: Iterable[str] = ()) -> Optional[str]:
    agent_l = agent_words(agents)
    for clause in _clauses(text):
        subj = _subject_before(clause, TRAIT_VERBS)
        if subj is not None and _is_person_subject(subj[-SUBJECT_WORDS:], agent_l):
            return "trait"
    return None


def check_owner_subject(fact: str) -> Optional[str]:
    return None if re.match(r"^\s*(?:the\s+)?owner\b", fact or "", re.I) else "owner_subject"


def has_first_person(text: str) -> bool:
    return any(_bare(w).lower() in FIRST_PERSON for w in (text or "").split())


def check_quote(fact: str, quote: str, min_share: float = 0.6) -> Optional[str]:
    """The quote must hold a first-person word, and at least `min_share` of the fact's content words must be in it."""
    if not has_first_person(quote):
        return "quote:first_person"
    fw = content_words(fact)
    if not fw:
        return "quote:empty"
    qw = content_words(quote)
    if len(fw & qw) / len(fw) < min_share:
        return "quote:grounding"
    return None


def first_person_to_owner(clause: str) -> str:
    """Rewrite first-person words with the S5 map ("I'm" -> "the owner is", "my" -> "the owner's", ...)."""
    t = (clause or "").replace("’", "'")
    for a, b in FIRST_PERSON_MAP:
        t = re.sub(rf"(?<![\w']){re.escape(a)}(?![\w'])", b, t, flags=re.I)
    t = re.sub(r"\bthe owner (\w+)\b", lambda m: "the owner " + _third_person(m.group(1)), t, flags=re.I)
    t = t.strip()
    return t[:1].upper() + t[1:] if t else t


_BASE_VERBS = {"like", "love", "hate", "want", "need", "play", "live", "work", "prefer", "enjoy", "think", "feel",
               "watch", "listen", "own", "drive", "eat", "drink", "use", "make", "know", "speak", "study", "do", "go",
               "have", "main", "run", "collect", "build", "stream", "train"}


def _third_person(verb: str) -> str:
    """'like' -> 'likes' after "the owner" (only for common base verbs; anything else is left as said)."""
    v = verb.lower()
    if v not in _BASE_VERBS:
        return verb
    if v == "have":
        return "has"
    if v in ("do", "go") or v.endswith(("ch", "sh", "ss", "x")):
        return verb + "es"
    if v.endswith("y") and v[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def allow_names() -> list:
    """The owner's allow-list (the viewer's game and product names, `_allow_names.txt`) plus DEFAULT_ALLOW."""
    own = []
    try:
        import experience_memory as EM
        own = [w.strip() for w in (EM.root() / "_allow_names.txt").read_text(encoding="utf-8").splitlines()
               if w.strip()]
    except Exception:  # noqa: BLE001
        pass
    return sorted(set(own) | DEFAULT_ALLOW, key=str.lower)


def write_time_check(text: str, names: Iterable[str] = (), owner_name: Optional[str] = None, *,
                     agents: Iterable[str] = (), allow: Optional[Iterable[str]] = None) -> Optional[str]:
    """The checks lore entries and lessons pass when written (D8): names without the tagger, safety without the
    sensitive-topic lists, and the directive rule. Returns the first failing reason or None."""
    allow = allow_names() if allow is None else allow
    return (check_names(text, names, owner_name, agents=agents, allow=allow)
            or check_safety(text, topics=False) or check_directive(text, agents))


def store_scrub(text: str, extra_names: Iterable[str] = ()) -> str:
    """Name scrub for text an agent stores during a call (lore, lessons, predictions, timer messages): the call's
    learned and heard names, room names and `extra_names`, with the owner's saved name becoming "the owner"."""
    try:
        import call_boundary
        names = list(call_boundary.call_names()) + [n for n in extra_names or () if n]
        owner = call_boundary.owner_name()
    except Exception:  # noqa: BLE001
        names, owner = [n for n in extra_names or () if n], None
    return scrub(text, names, owner, allow=allow_names())


def store_check(text: str, extra_names: Iterable[str] = ()) -> Optional[str]:
    """write_time_check with the call's names (after store_scrub, names are already gone; this catches labels,
    handles, private details, instructions, code and directives)."""
    try:
        import call_boundary
        names = list(call_boundary.call_names()) + [n for n in extra_names or () if n]
        owner = call_boundary.owner_name()
    except Exception:  # noqa: BLE001
        names, owner = [n for n in extra_names or () if n], None
    return write_time_check(text, names, owner)


REFUSALS = {
    "safety:private": "Not kept: that has a private detail (phone, email, address or key). Say you don't keep those.",
    "directive": "Not kept: notes like this can't set rules for how you act. Say so briefly.",
    "safety:code": "Not kept: you don't keep anything about writing code or scripts. Say so briefly.",
}


def refusal(reason: str) -> str:
    """What the agent is told when a write-time check refuses a text."""
    if reason in REFUSALS:
        return REFUSALS[reason]
    if reason.startswith("name"):
        return "Not kept: it names or tags a person, and you keep the group's stories without names. Say so briefly."
    return "Not kept: that can't be stored. Say so briefly."


__all__ = [
    "is_private", "is_forget_request", "has_slur", "content_nodes", "content_words", "scoring_words", "zipf",
    "overlap_score", "Matcher", "off_record_kind", "remember_clause", "handles", "scrub", "scrub_names_in",
    "check_names", "extra_capitals", "check_safety", "check_directive", "check_traits", "check_owner_subject",
    "check_quote", "has_first_person", "first_person_to_owner", "write_time_check", "agent_words", "store_scrub",
    "store_check", "refusal",
]


# ------------------------------------------------------------------ the name tagger (D3)
TAGGER_MODEL = "urchade/gliner_small-v2.1"
TAGGER_MIN = 0.4               # S6, tuned


TAGGER_BACKBONE = "microsoft/deberta-v3-small"
TAGGER_FILES = ((TAGGER_MODEL, ("gliner_config.json", "pytorch_model.bin")),
                (TAGGER_BACKBONE, ("config.json", "spm.model", "tokenizer_config.json")))


def tagger_files_present(model: str = TAGGER_MODEL) -> bool:
    """Every file the tagger loads (its weights and its backbone's tokenizer) is in the local Hugging Face cache (D3:
    the pass never downloads; a partial download counts as missing)."""
    try:
        import os
        from huggingface_hub import try_to_load_from_cache
        for repo, files in TAGGER_FILES:
            for f in files:
                p = try_to_load_from_cache(model if repo == TAGGER_MODEL else repo, f)
                if not isinstance(p, str) or not os.path.exists(p) or os.path.getsize(p) == 0:
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


class GlinerTagger:
    """Person names, from GLiNER small on the CPU, loaded only from the local cache (D3). `persons(text)` returns the
    words it marks as a person at or above the confidence threshold."""

    def __init__(self, threshold: float = TAGGER_MIN, model: str = TAGGER_MODEL):
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from gliner import GLiNER
        self.threshold = threshold
        self.model = GLiNER.from_pretrained(model, local_files_only=True)

    def persons(self, text: str) -> list:
        t = (text or "").strip()
        if not t:
            return []
        out = []
        words = t.split()
        size, step = 250, 200            # GLiNER reads up to 384 words: long texts go in overlapping word windows
        for start in range(0, max(1, len(words) - (size - step)), step):
            for e in self.model.predict_entities(" ".join(words[start:start + size]), ["person"],
                                                 threshold=self.threshold):
                w = (e.get("text") or "").strip()
                if w and person_like(w) and w not in out:
                    out.append(w)
        return out


NOT_PERSONS = {"i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they",
               "them", "their", "we", "us", "our", "it", "someone", "somebody", "anyone", "anybody", "everyone",
               "everybody", "nobody", "guys", "guy", "bro", "dude", "man", "girl", "boy", "people", "person", "friend",
               "friends", "owner", "the owner", "yourself", "myself", "himself", "herself", "themselves"}
TAGGER_COMMON_ZIPF = 4.0   # S6, tuned on development calls 2026-11-04: lower-case common words the tagger marks
                           # ("scientist") are roles, not names


def person_like(hit: str) -> bool:
    """Tagger hits that can be a name: not a pronoun or a word for a person, and a lower-case hit only when it is
    not a common English word (tuned on development calls: the tagger marks "I", "He", "you", "scientist")."""
    words = [w.strip(".,!?;:'\"()") for w in hit.split()]
    words = [w for w in words if w]
    if not words or hit.lower() in NOT_PERSONS or all(w.lower() in NOT_PERSONS for w in words):
        return False
    if all(w.islower() for w in words) and all(zipf(w) >= TAGGER_COMMON_ZIPF for w in words):
        return False
    return True


class CachedTagger:
    """Wraps a tagger with a cache: the pass asks about the same lines more than once."""

    def __init__(self, inner):
        self.inner = inner
        self._c: dict = {}

    def persons(self, text: str) -> list:
        v = self._c.get(text)
        if v is None:
            v = self._c[text] = list(self.inner.persons(text))
        return v
