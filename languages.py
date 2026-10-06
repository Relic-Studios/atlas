"""Multilingual conversation (owner 10-05): transcribe what people actually say in
their own language (Whisper auto-detect, no forced English), let agents answer in
that language, and speak it with the matching TTS language.

Pure module, no models. Three pieces:
  * Room      - which language the room is speaking right now, from Whisper's
                per-turn detection with confidence gating (a low-confidence "yeah"
                must not flip the room to Welsh).
  * note()    - one prompt line telling the agent which language to reply in.
  * tts_language(text, hint) - which Qwen3-TTS language to voice a reply with.

Live translation is NOT here; it is a separate opt-in plugin.
"""
from __future__ import annotations

import os
import re
import threading
import time

# Whisper code -> English name (subset that matters; anything else falls back to code)
NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "tr": "Turkish", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "sv": "Swedish",
    "no": "Norwegian", "da": "Danish", "fi": "Finnish", "cs": "Czech", "el": "Greek",
    "he": "Hebrew", "ro": "Romanian", "hu": "Hungarian", "tl": "Tagalog", "ms": "Malay",
}

# Languages the bundled Qwen3-TTS talker can voice (read from the GGUF language table).
TTS_NAMES = {"en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean",
             "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese",
             "es": "spanish", "it": "italian"}

CONFIDENT = 0.60      # Whisper language_probability needed to change the room language
MIN_UNITS = 2         # words (or CJK characters) needed to trust a detection


def _plugin() -> tuple[bool, dict]:
    try:
        import plugins
        return plugins.is_enabled("languages"), plugins.settings_of("languages")
    except Exception:  # noqa: BLE001
        return True, {}


def setting() -> str:
    """'auto' (default) or a fixed Whisper code. ATLAS_STT_LANGUAGE overrides, then the
    Languages plugin (off = fixed to the main language), then user settings."""
    v = (os.environ.get("ATLAS_STT_LANGUAGE") or "").strip().lower()
    if not v:
        on, s = _plugin()
        if not on:
            return primary()
        v = str(s.get("listen") or "").strip().lower()
    if not v:
        try:
            import user_settings
            v = str(user_settings.load().get("stt_language") or "").strip().lower()
        except Exception:  # noqa: BLE001
            v = ""
    return v or "auto"


def primary() -> str:
    """Fallback language for ambiguous turns before anything confident was heard."""
    v = (os.environ.get("ATLAS_PRIMARY_LANGUAGE") or "").strip().lower()
    if not v:
        v = str(_plugin()[1].get("primary") or "").strip().lower()
    if not v:
        try:
            import user_settings
            v = str(user_settings.load().get("primary_language") or "").strip().lower()
        except Exception:  # noqa: BLE001
            v = ""
    return v or "en"


def follow_room() -> bool:
    """Owner 10-06: replies stay in the main language unless someone asks for another
    one. 'follow' (opt-in) restores auto-switching to whatever the room speaks."""
    v = (os.environ.get("ATLAS_REPLY_LANGUAGE") or "").strip().lower()
    if not v:
        on, s = _plugin()
        v = str(s.get("reply") or "").strip().lower() if on else "ask"
    return v == "follow"


# "speak Spanish", "reply in Japanese", "can you talk in French?", "habla español",
# "back to English". Matched against NAMES plus a few native spellings.
_NATIVE = {"español": "es", "espanol": "es", "castellano": "es", "français": "fr",
           "francais": "fr", "deutsch": "de", "português": "pt", "portugues": "pt",
           "italiano": "it", "nihongo": "ja", "日本語": "ja", "中文": "zh", "한국어": "ko",
           "русский": "ru", "mandarin": "zh"}


def _lang_words() -> dict:
    m = {v.lower(): k for k, v in NAMES.items()}
    m.update(_NATIVE)
    return m


_REQ_VERB = (r"(?:speak|talk|reply|respond|answer|say (?:it|that|this)|switch|change|"
             r"habla|hablame|háblame|hablar|parle|parlez|sprich|sprechen|fala|parla)")


_NOT_ASKING = {"i", "we", "they", "he", "she", "it", "who", "people", "i'd", "i'll", "i'm",
               "we're", "they're", "gonna", "usually", "always", "often", "never", "can't"}


def requested_language(text: str) -> str | None:
    """Language code someone explicitly asked the agent to use, else None."""
    t = (text or "").lower()
    if not t.strip():
        return None
    words = _lang_words()
    alt = "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))
    pats = (
        rf"\b(?:back|go back|switch back|return)\s+to\s+({alt})\b",
        rf"\b{_REQ_VERB}\b[^.?!]{{0,25}}?\b(?:in|to|into|en|em|auf)\s+({alt})\b",
        rf"\b{_REQ_VERB}\s+({alt})\b",
        rf"\b(?:in|en)\s+({alt})\s*,?\s*(?:please|por favor|s'il (?:te|vous) pla[iî]t|bitte)\b",
    )
    for rx in pats:
        for m in re.finditer(rx, t):
            # "I speak Spanish at home", "do you speak French?" describe, not ask.
            before = re.findall(r"[a-z']+", t[:m.start()])[-2:]
            if before and before[-1] in _NOT_ASKING:
                continue
            if len(before) == 2 and before[0] in ("do", "does", "did") and before[1] in ("you", "he", "she", "they"):
                continue
            return words.get(m.group(1))
    if re.search(r"\b(?:stop|quit) (?:speaking|talking) (?:in )?(?:" + alt + r")\b", t):
        return "__back__"
    return None


def name(code: str | None) -> str:
    c = (code or "").lower()
    return NAMES.get(c, c or "English")


_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def units(text: str) -> int:
    t = text or ""
    return len(re.findall(r"[^\W\d_]+", t)) + len(_CJK.findall(t))


class Room:
    """Thread-safe current-language state for one call."""

    def __init__(self, fallback: str | None = None):
        self._lock = threading.Lock()
        self.fallback = (fallback or primary()).lower()
        self.code = self.fallback          # language the agent replies (and is voiced) in
        self.heard = self.fallback         # last confidently detected language (STT hint)
        self.seen_other = False            # reply language left the fallback this call
        self.requested = False             # someone explicitly asked for self.code
        self.last_detect: tuple[str | None, float] = (None, 0.0)
        self.at = 0.0

    def needs_retry(self, detected: str | None, prob: float, text: str) -> str | None:
        """Low-confidence detection that disagrees with the room: return the code to
        re-transcribe with (forced), else None."""
        d = (detected or "").lower()
        with self._lock:
            room = self.heard
        if not d or d == room:
            return None
        if prob >= CONFIDENT and units(text) >= MIN_UNITS:
            return None
        return room

    def observe(self, detected: str | None, prob: float, text: str) -> str:
        """Record one final turn; return the language to treat it as."""
        d = (detected or "").lower()
        with self._lock:
            self.last_detect = (d or None, float(prob or 0.0))
            confident = bool(d and prob >= CONFIDENT and units(text) >= MIN_UNITS)
            if confident:
                self.heard = d
            req = requested_language(text)
            if req == "__back__":
                self.code, self.requested = self.fallback, False
                self.at = time.time()
            elif req:
                self.code, self.requested = req, (req != self.fallback)
                self.at = time.time()
                if req != self.fallback:
                    self.seen_other = True
            elif confident and not self.requested and follow_room():
                self.code = d
                self.at = time.time()
                if d != self.fallback:
                    self.seen_other = True
            return self.code

    def current(self) -> str:
        with self._lock:
            return self.code

    def note(self, speaker: str | None = None) -> str:
        """Prompt line, or '' when replies are in the main language."""
        with self._lock:
            code, req, other = self.code, self.requested, self.seen_other
        if code == self.fallback:
            if not other:
                return ""
            lang = name(code)
            # Back to the main language after a switch: say so, or the model keeps
            # answering in the previous language from history.
            return f"LANGUAGE: reply in {lang} again (the conversation switched back to {lang})."
        if req:
            lang = name(code)
            return (f"LANGUAGE: you were asked to speak {lang}. Reply in {lang} until someone"
                    f" asks you to switch back.")
        who = speaker or "The person you're answering"
        lang = name(code)
        extra = ""
        if code not in TTS_NAMES:
            extra = (f" Your voice can't pronounce {lang} perfectly, but still answer in {lang}"
                     " unless they ask for another language.")
        return (f"LANGUAGE: {who} is speaking {lang}. Reply in {lang}, naturally, as a fluent"
                f" speaker would. Don't translate their words back to them, don't switch to"
                f" English unless asked, and keep names as they said them.{extra}")


ROOM = Room()


# ---------------------------------------------------------------- text -> TTS language
_SCRIPTS = [
    ("ja", re.compile(r"[぀-ヿ]")),          # kana beats han
    ("ko", re.compile(r"[가-힯]")),
    ("zh", re.compile(r"[一-鿿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
]
_STOP = {
    "en": ("the and is you that it to of what i not this with for are a an in on my your we" " he she they was be have do don't it's that's i'm can will just so but if at as" " all one got get like yeah okay sure good"),
    "es": "el la que de y es en los las un una por para no con qué sí pero muy",
    "fr": "le la les de et est un une que pas je vous il elle des du pour avec c'est",
    "de": "der die das und ist nicht ich du ein eine zu mit es wir sie auch was",
    "it": "il la che di e è non un una per sono ma con mi ti cosa anche",
    "pt": "o a que de e é não um uma para com eu você os as mas muito",
}
_STOPSETS = {k: set(v.split()) for k, v in _STOP.items()}


# Live 10-06: "That's a hell of a move." was voiced as Portuguese because the bare
# word "a" only appeared in the Portuguese list. Switching the voice away from the
# room's language now needs real evidence: a different script, or a clear lead of
# several stopwords over the room language.
SWITCH_MARGIN = 2


def guess_text(text: str, hint: str | None = None) -> str:
    """Best-guess Whisper-style code for a reply text (script first, then stopwords)."""
    t = text or ""
    for code, rx in _SCRIPTS:
        if len(rx.findall(t)) >= 2:
            return code
    h = (hint or "en").lower()
    words = re.findall(r"[a-z\u00e0-\u00ff']+", t.lower())
    if not words:
        return h
    scores = {k: sum(w in st for w in words) for k, st in _STOPSETS.items()}
    best = max(scores, key=scores.get)
    base = scores.get(h, 0)
    if best != h and scores[best] >= SWITCH_MARGIN and scores[best] - base >= SWITCH_MARGIN:
        return best
    return h if h in scores or h in TTS_NAMES else "en"


def tts_language(text: str, hint: str | None = None) -> str:
    """Qwen3-TTS language name for this text ('english' if unsupported).
    While the whole call has stayed in the main language, only a different script
    (kana, hangul, cyrillic, CJK) may change the voice: short English lines share too
    many little words with Spanish/Portuguese/Italian to guess from."""
    h = (hint or "en").lower()
    try:
        locked = (h == ROOM.fallback)
    except Exception:  # noqa: BLE001
        locked = True
    if locked:
        # Owner 10-06: stay in the main language until someone asks for another.
        # Only a different script may change the voice (the main-language voice
        # can't say kana/hangul/cyrillic anyway); Latin-script guesses are off.
        for code, rx in _SCRIPTS:
            if len(rx.findall(text or "")) >= 2:
                return TTS_NAMES.get(code, "english")
        return TTS_NAMES.get(h, "english")
    return TTS_NAMES.get(guess_text(text, h), "english")
