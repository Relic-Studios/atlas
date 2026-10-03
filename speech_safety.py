"""Hard backstop on what gets VOICED: a sentence containing a slur is never spoken.

Not a personality rule -- the model still decides how to handle bait (moment.py).
This exists because on 10-01, under the old notes, Ivy said the N-word aloud when
asked. Sentence-level, so the rest of the reply still plays. Words are stored
reversed so the source itself stays clean for public export / grep.
"""
import logging
import re

logger = logging.getLogger(__name__)

_STEMS = [s[::-1] for s in ("ggin", "aggin", "toggaf", "ggaf", "drater", "knihc", "ekik",
                            "ynnart", "kcapg", "koog")]
_SPIC = "cips"[::-1]
SLUR_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s) + r"\w*" for s in _STEMS) + r"|" + _SPIC + r"s?)\b",
    re.IGNORECASE,
)


def has_slur(text: str) -> bool:
    return bool(SLUR_RE.search(text or ""))


_STAR = re.compile(r"\*+([^*\n]{1,120}?)\*+")


def strip_stage(text: str) -> str:
    """Short starred spans (*evil laugh*, *baaa*, *beatbox*) are actions/sound effects TTS
    would read aloud or mangle: drop them. Agents don't perform sounds (owner 10-02).
    Long starred spans are emphasis on real speech: keep the words."""
    def sub(m):
        inner = m.group(1).strip()
        return "" if len(inner.split()) <= 4 else inner
    out = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", " times ", text or "")
    out = _STAR.sub(sub, out)
    out = out.replace("*", "")
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([,.!?])", r"\1", out)
    return out


# A character narrating itself in third person ("<Name> waves. <Name> checks
# you're not taking anything shiny.") reads as a second voice in one mouth.
# Spoken self-narration is dropped sentence by sentence (owner 10-01); the
# names to watch come from the agent pack's "narrator_names" trait.
def _narrator_alt() -> str:
    try:
        import agent_registry as _reg
        return "|".join(re.escape(n) for n in _reg.traits().get("narrator_names") or [] if n)
    except Exception:  # noqa: BLE001
        return ""


_NARR = _narrator_alt() or r"(?!x)x"   # never matches when the pack names nobody
_NARRATE = re.compile(
    r"^\W*(?:[A-Za-z']+[,!]\s+){0,2}(?:" + _NARR + r")\s+(?:says|said|thinks|waves|checks|agrees|asks|wants|"
    r"whispers|hisses|nods|laughs|cries|grins|smiles|sniffs|looks|watches|hides|"
    r"is\s+(?:listening|watching|thinking|here)|says\s+hello|knows|hates|likes|loves|"
    r"doesn'?t|does|will|won'?t)\b", re.I)


def narrates(sentence: str) -> bool:
    return bool(_NARRATE.search(sentence or ""))


def screen(source, dropped: list | None = None):
    """Wrap a stream of text chunks; drop any settled sentence containing a slur."""
    buf = ""
    for chunk in source:
        buf += chunk
        while True:
            m = re.search(r"[.!?]+\s|\n", buf)
            if not m:
                break
            sent, buf = buf[:m.end()], buf[m.end():]
            if has_slur(sent):
                if dropped is not None:
                    dropped.append(sent.strip())
                logger.info("Slur sentence not voiced")
                continue
            sent = strip_stage(sent)
            if narrates(sent):
                if dropped is not None:
                    dropped.append(sent.strip())
                logger.info("Self-narration not voiced: %r", sent.strip()[:80])
                continue
            if sent.strip():
                yield sent
    buf = strip_stage(buf)
    if narrates(buf):
        return
    if buf.strip():
        if has_slur(buf):
            if dropped is not None:
                dropped.append(buf.strip())
            logger.info("Slur sentence not voiced")
            return
        yield buf
