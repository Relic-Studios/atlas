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
        self.code = self.fallback          # language the room is speaking now
        self.seen_other = False            # any confident non-fallback language this call
        self.last_detect: tuple[str | None, float] = (None, 0.0)
        self.at = 0.0

    def needs_retry(self, detected: str | None, prob: float, text: str) -> str | None:
        """Low-confidence detection that disagrees with the room: return the code to
        re-transcribe with (forced), else None."""
        d = (detected or "").lower()
        with self._lock:
            room = self.code
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
            if d and prob >= CONFIDENT and units(text) >= MIN_UNITS:
                self.code = d
                self.at = time.time()
                if d != self.fallback:
                    self.seen_other = True
            return self.code

    def current(self) -> str:
        with self._lock:
            return self.code

    def note(self, speaker: str | None = None) -> str:
        """Prompt line, or '' when the whole call has been in the fallback language."""
        with self._lock:
            code, other = self.code, self.seen_other
        if code == self.fallback and not other:
            return ""
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
    "en": "the and is you that it to of what i not this with for are",
    "es": "el la que de y es en los las un una por para no con qué sí pero muy",
    "fr": "le la les de et est un une que pas je vous il elle des du pour avec c'est",
    "de": "der die das und ist nicht ich du ein eine zu mit es wir sie auch was",
    "it": "il la che di e è non un una per sono ma con mi ti cosa anche",
    "pt": "o a que de e é não um uma para com eu você os as mas muito",
}
_STOPSETS = {k: set(v.split()) for k, v in _STOP.items()}


def guess_text(text: str, hint: str | None = None) -> str:
    """Best-guess Whisper-style code for a reply text (script first, then stopwords)."""
    t = text or ""
    for code, rx in _SCRIPTS:
        if len(rx.findall(t)) >= 2:
            return code
    words = re.findall(r"[a-zà-ÿ']+", t.lower())
    if not words:
        return (hint or "en").lower()
    scores = {k: sum(w in s for w in words) for k, s in _STOPSETS.items()}
    h = (hint or "").lower()
    if h in scores:
        scores[h] += 0.5            # ties go to the language the room is speaking
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else (h or "en")


def tts_language(text: str, hint: str | None = None) -> str:
    """Qwen3-TTS language name for this text ('english' if unsupported)."""
    return TTS_NAMES.get(guess_text(text, hint), "english")
