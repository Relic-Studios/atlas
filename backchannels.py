"""Search backchannels: short pre-rendered "hold on, let me look" lines.

Web search takes several seconds (tool-call round + fetch + answer round). Instead of
dead air, the agent says one of these in its own voice the moment a search starts.
Clips are rendered ONCE per voice by tools/render_backchannels.py into
voices/backchannels/<voice>/NN.wav (24 kHz mono int16) and preloaded into RAM here,
so playback costs zero TTS/GPU time and starts instantly.

They are never written to conversation history (the model never sees them), so they
can't feed a repetition loop. Selection avoids the last few used lines.
"""
from __future__ import annotations

import logging
import random
import re
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CLIP_DIR = ROOT / "voices" / "backchannels"

# Lines come from agent_registry (each agent's own "backchannels" list), falling
# back to GENERIC_LINES. Clips live in voices/backchannels/<clip_key>/NN.wav.
import agent_registry as _registry

GENERIC_LINES = [
    "Hold on, let me check.", "One sec, looking.", "Gimme a second.", "Hang on, checking that.",
    "Let me see.", "Okay, one moment.", "Hmm, let me look.", "Hold up, checking.",
]


class _Lines:
    """Live view {agent_id: [lines]} over the registry."""
    def get(self, aid, default=None):
        return _registry.backchannel_lines(aid) or (default if default is not None else [])

    def __getitem__(self, aid):
        v = _registry.backchannel_lines(aid)
        if not v:
            raise KeyError(aid)
        return v

    def __contains__(self, aid):
        return bool(_registry.backchannel_lines(aid))

    def items(self):
        return [(a, _registry.backchannel_lines(a)) for a in _registry.agent_ids() if _registry.backchannel_lines(a)]


LINES = _Lines()

# Voice -> persona line set (voice overrides reuse the closest persona's wording).
# Lines that only make sense for a web search (never used for a screen look).
SEARCH_ONLY_RE = re.compile(r"google|search|look(?:ing)?\s+(?:it|that)\s+up|pull(?:ing)?\s+(?:it|that)\s+up|"
                            r"verify|make\s+that\s+up|dig|good\s+question|lookup|find\s+it|find\s+out|"
                            r"research|internet|web", re.I)

class _VoiceLines:
    """clip_key -> agent id whose lines were rendered into that clip dir."""
    def get(self, key, default=None):
        return _registry.agent_for_clip_key(key) or default

    def items(self):
        return [(_registry.clip_key(a), a) for a in _registry.agent_ids() if _registry.backchannel_lines(a)]


VOICE_LINES = _VoiceLines()

# Deterministic "this turn will need the web" cue: lets us speak the filler at
# generation start instead of waiting ~5s for the model's tool-call round.
_SEARCH_INTENT_RE = re.compile(
    r"\b((?:can|could|would|will)\s+you\s+(?:go\s+)?research|(?:please|go)\s+research|\w+,\s*research\b|research\s+(?:this|that|it)\b|search|google|look\s*(?:it|that|this)?\s*up|look\s+into|check\s+online|"
    r"what'?s\s+the\s+latest|latest\s+news|news\s+(?:on|about)|who\s+won|"
    r"what\s+time\s+is\s+it\s+in|weather\s+in|how\s+much\s+(?:is|does|are)|"
    r"release\s+date|when\s+(?:does|did|is)\s+.+\s+(?:come\s+out|release|drop))\b",
    re.IGNORECASE,
)


# Live call 10-05: "I can ask Fae to search the internet" forced a junk search. Lines that
# only TALK ABOUT searching (third person / ability statements) aren't requests.
_SEARCH_MENTION_RE = re.compile(
    r"\b(?:(?:i|we|you|u|they|people|anyone|someone)\s+(?:can|could|should|might)\s+(?:just\s+)?ask\s+\w+\s+to"
    r"|(?:she|he|it|they|fae|\w+)\s+(?:can|could|is\s+able\s+to|knows\s+how\s+to)\s+(?:also\s+)?(?:search|google|look)"
    r"|(?:able|ability)\s+to\s+(?:search|google|look)"
    r"|(?:she|he|it|they)\s+(?:searches|googles|looks\s+(?:stuff|things)\s+up)"
    r"|searching\s+(?:is|was)\b)",
    re.IGNORECASE,
)
_DIRECT_ASK_RE = re.compile(r"\b(?:can|could|would|will)\s+you\b|^\W*(?:\[\w+\]\s*)?(?:\w+,\s*)?(?:please\s+)?(?:search|google|look)", re.I)


def mention_only(text: str) -> bool:
    """True when a line merely mentions searching rather than asking for one."""
    t = text or ""
    return bool(_SEARCH_MENTION_RE.search(t)) and not _DIRECT_ASK_RE.search(t)


def wants_search(text: str) -> bool:
    return bool(text and _SEARCH_INTENT_RE.search(text) and not mention_only(text))


class Backchannels:
    """Preloaded per-voice clips; thread-safe pick with recent-avoidance."""

    def __init__(self):
        self._clips: dict[str, list[tuple[str, bytes]]] = {}
        self._recent: dict[str, deque] = {}
        self._lock = threading.Lock()

    def _load(self, voice: str) -> list[tuple[str, bytes]]:
        import wave
        d = CLIP_DIR / voice
        out = []
        persona = VOICE_LINES.get(voice)
        lines = LINES.get(persona, []) if persona else []
        for i, text in enumerate(lines):
            f = d / f"{i:02d}.wav"
            if not f.exists():
                continue
            try:
                with wave.open(str(f), "rb") as w:
                    if w.getframerate() != 24000 or w.getsampwidth() != 2 or w.getnchannels() != 1:
                        continue
                    out.append((text, w.readframes(w.getnframes())))
            except Exception as e:  # noqa: BLE001
                logger.warning("backchannel %s unreadable: %s", f, e)
        logger.info("🗣️⏳ backchannels preloaded for '%s': %d clips", voice, len(out))
        return out

    def preload(self, voice: str) -> int:
        with self._lock:
            if voice not in self._clips:
                self._clips[voice] = self._load(voice)
                self._recent[voice] = deque(maxlen=6)
            return len(self._clips[voice])

    def pick(self, voice: str, kind: str = "search"):
        """Return (text, pcm_24k_int16_bytes) or None if no clips for this voice.

        kind="look": only lines that don't claim a web search (a screen look
        must not say "lemme google that").
        """
        self.preload(voice)
        with self._lock:
            clips = self._clips.get(voice) or []
            if not clips:
                return None
            recent = self._recent[voice]
            pool = list(range(len(clips)))
            if kind == "look":
                pool = [i for i in pool if not SEARCH_ONLY_RE.search(clips[i][0])] or pool
            fresh = [i for i in pool if i not in recent] or pool
            i = random.choice(fresh)
            recent.append(i)
            return clips[i]


# ---------------------------------------------------------------------------
# Search nudges (deterministic; the model decides nothing here, it gets told)
# ---------------------------------------------------------------------------
# The model often skips the tool on live-fact questions ("top AI stocks right
# now") and then improvises. A current-time marker + a factual topic is a
# strong, cheap signal that the answer needs the web.
_NOW_RE = re.compile(r"\b(right\s+now|today|tonight|currently|current|this\s+(?:week|month|year|season)|"
                     r"latest|newest|recent(?:ly)?|as\s+of|at\s+the\s+moment|these\s+days)\b", re.I)
_FACT_RE = re.compile(r"\b(stocks?|shares?|price[sd]?|cost|worth|market|crypto|bitcoin|score[sd]?|standings|"
                      r"rank(?:ing|ed)?s?|top\s+(?:\d+|three|five|ten)|most\s+(?:expensive|popular|valuable)|"
                      r"weather|news|election|president|ceo|champion|record|chapter|episode|patch|update|"
                      r"version|trailer|release|tour|album|game|movie|box\s+office)\b", re.I)
_FOLLOWUP_RE = re.compile(r"\b(did\s+you\s+(?:find|look|search|check)|find\s+(?:it|anything|out)|"
                          r"any\s+luck|what\s+did\s+you\s+(?:find|get)|you\s+(?:find|found)\s+it)\b", re.I)


def needs_lookup(text: str) -> bool:
    t = text or ""
    return wants_search(t) or bool(_NOW_RE.search(t) and _FACT_RE.search(t))


def search_note(text: str, have_recent_lookup: bool) -> str:
    """Per-turn nudge for the tool loop, or ''."""
    t = text or ""
    if _FOLLOWUP_RE.search(t) and not have_recent_lookup:
        return ("Nobody asked you to look anything up yet and you have NOT searched anything. "
                "If someone asks 'did you find it?', you don't know what they mean: answer them, say "
                "you haven't looked anything up, and ask what they want you to find. Never claim "
                "you found something.")
    if needs_lookup(t) and not _FOLLOWUP_RE.search(t):
        return ("This asks for current real-world facts. Call web_search FIRST, then answer "
                "with the actual specifics (names, numbers) from the results, in your own voice. "
                "Never invent numbers.")
    return ""
