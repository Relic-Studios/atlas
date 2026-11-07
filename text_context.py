import logging
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)
from colors import Colors # Assuming this is needed externally

# The quick answer is the first piece of a reply the voice speaks; it starts as soon as the first clause or sentence
# is there (owner 11-06: "it needs the fastest response time possible"). Waiting for the whole first sentence cost
# ~0.1-0.3 s, so the first piece still ends at a comma; tts_tail keeps almost no quiet after such a piece and no
# silence is added (audio_module), so the rest follows at once. Every LATER sentence is voiced whole (PLAY_FLOW).
SENTENCE_ENDS: Set[str] = {".", "!", "?", "\n", "。", "！", "？", "…"}
SOFT_BREAKS: Set[str] = {",", ";", ":", "、", "，", "；"}
_CLOSERS = "\"')]}»”’"
# a period after these is not a sentence end ("Mr. Smith", "e.g. a lot"); single letters (initials) neither
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "e.g", "i.e", "approx", "no", "fig", "prof", "mt"}


class TextContext:
    """
    Finds the first piece of a streamed reply the voice can start on: up to the first clause break or sentence end.
    """
    def __init__(self, split_tokens: Optional[Set[str]] = None, soft_tokens: Optional[Set[str]] = None) -> None:
        """
        Args:
            split_tokens: characters that end a sentence (default SENTENCE_ENDS).
            soft_tokens: clause breaks (default SOFT_BREAKS). Hyphens and dashes never cut ("well-known").
        """
        self.split_tokens: Set[str] = set(SENTENCE_ENDS if split_tokens is None else split_tokens)
        self.soft_tokens: Set[str] = set(SOFT_BREAKS if soft_tokens is None else soft_tokens)

    @staticmethod
    def _after(txt: str, i: int) -> Optional[int]:
        """End of the piece if the token ending at index i (exclusive) is a boundary, else None.

        Cuts as soon as the token arrives, except where it may not be one yet: after a digit ("10,000", "10:30",
        "3.5") and a period after an abbreviation or initial ("Mr.", "e.g.", "J."). Those wait for the next
        character: whitespace makes them a boundary (a period also needs the next word to start like a sentence)."""
        if txt[i - 1] == "\n":
            return i
        j = i
        while j < len(txt) and txt[j] in _CLOSERS:
            j += 1
        prev = txt[:i - 1]
        word = prev.rsplit(None, 1)[-1].lower().lstrip("\"'(“‘") if prev.strip() else ""
        period = txt[i - 1] == "."
        if period and (word in _ABBREV or (len(word) == 1 and word.isalpha())):
            return None
        if j >= len(txt):                       # nothing after it yet
            return None if prev[-1:].isdigit() else j
        if not txt[j].isspace():
            return None
        if period:
            k = j
            while k < len(txt) and txt[k].isspace():
                k += 1
            if k < len(txt) and not (txt[k].isupper() or txt[k].isdigit() or txt[k] in "\"'“‘(["):
                return None
        return j

    def get_context(self, txt: str, min_len: int = 6, max_len: int = 120, min_alnum_count: int = 10) -> Tuple[Optional[str], Optional[str]]:
        """
        The first piece of `txt` within max_len characters, as (piece, rest): up to the first clause break or
        sentence end once the piece has min_len characters and min_alnum_count letters/digits. (None, None) if
        there is none yet.
        """
        alnum_count = 0
        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1
            if i < min_len or alnum_count < min_alnum_count:
                continue
            if char in self.split_tokens or char in self.soft_tokens:
                end = self._after(txt, i)
                if end is not None:
                    logger.info(f"🧠 {Colors.MAGENTA}Context found after char no: {end}, context: {txt[:end]}")
                    return txt[:end], txt[end:]
        return None, None
