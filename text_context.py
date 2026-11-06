import logging
from typing import Optional, Set, Tuple

logger = logging.getLogger(__name__)
from colors import Colors # Assuming this is needed externally

# The quick answer (the first piece the voice speaks, docs: speech_pipeline_manager) ends at a sentence end. The voice
# speaks each piece as a finished utterance, so a piece cut at a comma came out with a sentence-final fall and a
# pause ("you won't change by next week, |0.84 s| Sam", demo 11-06). Commas only cut a first sentence that runs past
# max_len with no end yet, so a very long sentence doesn't hold the first audio back.
SENTENCE_ENDS: Set[str] = {".", "!", "?", "\n", "。", "！", "？", "…"}
SOFT_BREAKS: Set[str] = {",", ";", ":", "、", "，", "；"}
_CLOSERS = "\"')]}»”’"
# a period after these is not a sentence end ("Mr. Smith", "e.g. a lot"); single letters (initials) neither
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "vs", "e.g", "i.e", "approx", "no", "fig", "prof", "mt"}


class TextContext:
    """
    Finds the first piece of a streamed reply that the voice can speak on its own: the first full sentence.
    """
    def __init__(self, split_tokens: Optional[Set[str]] = None, soft_tokens: Optional[Set[str]] = None) -> None:
        """
        Args:
            split_tokens: characters that end a sentence (default SENTENCE_ENDS).
            soft_tokens: clause breaks used only for a first sentence longer than max_len (default SOFT_BREAKS).
        """
        self.split_tokens: Set[str] = set(SENTENCE_ENDS if split_tokens is None else split_tokens)
        self.soft_tokens: Set[str] = set(SOFT_BREAKS if soft_tokens is None else soft_tokens)

    @staticmethod
    def _after(txt: str, i: int) -> Optional[int]:
        """End of the piece if the token ending at index i (exclusive) is a real boundary, else None.

        A boundary needs whitespace after it (closing quotes/brackets first), so "3.5", "e.g.", "10,000" and "10:30"
        are never cut, and a token at the very end of what has arrived waits for the next characters (the caller
        speaks the whole reply once the stream is done)."""
        if txt[i - 1] == "\n":
            return i
        j = i
        while j < len(txt) and txt[j] in _CLOSERS:
            j += 1
        if not (j < len(txt) and txt[j].isspace()):
            return None
        if txt[i - 1] == ".":
            # A period ends a sentence when the next word starts like a sentence ("e.g. a lot" doesn't) and the
            # word before it isn't an abbreviation or an initial. Wait for that next word's first character.
            k = j
            while k < len(txt) and txt[k].isspace():
                k += 1
            if k >= len(txt):
                return None
            if not (txt[k].isupper() or txt[k].isdigit() or txt[k] in "\"'“‘(["):
                return None
            word = txt[:i - 1].rsplit(None, 1)[-1].lower().lstrip("\"'(“‘") if txt[:i - 1].strip() else ""
            if word in _ABBREV or (len(word) == 1 and word.isalpha()):
                return None
        return j

    def get_context(self, txt: str, min_len: int = 6, max_len: int = 140, min_alnum_count: int = 10) -> Tuple[Optional[str], Optional[str]]:
        """
        The first full sentence of `txt` within max_len characters, as (piece, rest).

        If the text already runs past max_len with no sentence end, the piece ends at the last clause break
        (comma, semicolon, colon) before max_len instead. (None, None) if neither is there yet.

        Args:
            txt: The streamed text so far.
            min_len: Minimum length of the piece.
            max_len: Longest first sentence to wait for before a clause break is used.
            min_alnum_count: Minimum number of letters and digits in the piece.
        """
        alnum_count = 0
        soft_end = None
        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1
            if i < min_len or alnum_count < min_alnum_count:
                continue
            if char in self.split_tokens:
                end = self._after(txt, i)
                if end is not None:
                    logger.info(f"🧠 {Colors.MAGENTA}Context found after char no: {end}, context: {txt[:end]}")
                    return txt[:end], txt[end:]
            elif char in self.soft_tokens and self._after(txt, i) is not None:
                soft_end = i
        if soft_end is not None and len(txt) >= max_len:
            logger.info(f"🧠 {Colors.MAGENTA}Long first sentence: context at clause break {soft_end}: {txt[:soft_end]}")
            return txt[:soft_end], txt[soft_end:]
        return None, None
