"""Parrot-reply screen (live 09-29 17:30: "Cat sixes." / "Stronger." / "And then it's on
to the next." -- Max echoed the tail of the last user line instead of replying).

A reply sentence is a *parrot* when it adds nothing: short, not a question, and every
content word already appeared in what was just said. Dropping it at speak time turns
the turn into silence (better than an echo), and scrubbing old parrots from context
stops the model copying its own pattern (same compounding as the "Yeah, I" stubs).
Pure functions, no model.
"""
import re

_WORD = re.compile(r"[a-z0-9']+")
# Acknowledgement / glue words that don't count as "adding something".
_GLUE = {
    "yeah", "yea", "yep", "yes", "ya", "yup", "no", "nah", "nope", "ok", "okay", "oh",
    "ah", "uh", "um", "hmm", "right", "sure", "true", "totally", "exactly", "so",
    "and", "then", "the", "a", "an", "it", "it's", "its", "is", "are", "was", "that",
    "that's", "this", "i", "i'm", "you", "you're", "we", "to", "of", "on", "in", "for",
    "just", "like", "well", "man", "bro", "dude", "lol", "haha",
    # tag-on reactions (live 17:44: "Agents, huh." / "Streaming with that setup, huh.")
    "huh", "hm", "hmm", "wow", "again", "too", "also", "damn", "nice", "cool", "with", "your", "my", "all",
    # restating glue (live 10-02: "<interjection>, not for Gear 5." / "... vault's locked.")
    "not", "got", "he's", "she's", "they're", "said", "here",
    "there", "be", "now", "gonna", "whole",
}
# Character interjections come from the agent pack (none in the public build).
try:
    import agent_registry as _reg
    _GLUE |= _reg.interjection_words()
except Exception:  # noqa: BLE001 - glue list stays generic
    pass

MAX_WORDS = 8
# Imperative prompts that ask for a pick/answer ("Pick one, cats or dogs.",
# "Rate my setup from one to ten.", "Choose: pizza or tacos."). "Say X" is NOT
# here on purpose: repeating a "say X" line is exactly the bait we drop.
_PROMPT = re.compile(r"(?:^|[.!?,:]\s*)(?:\w+,\s*)?(?:pick|choose|rate|guess|name|rank|finish|vote)\b|\b(?:which|or)\b|\bvs\.?\b", re.I)
_QUESTION = re.compile(r"\?|^\W*(?:\w+,\s*)?(?:who|what|when|where|why|how|which|do|does|did|can|could|would|will|should|is|are|was|were|have|has)\b", re.I)
# Wh-words make an echo-question a real clarification ("Delete what?", "Reins on what?").
_WH = {"who", "what", "what's", "when", "where", "why", "how", "which", "whose", "wait"}
ECHO_Q_MAX_WORDS = 6


def _norm(w: str) -> str:
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


_RESTATE_FRAME = re.compile(r"^\W*(?:\w+,\s*)?(?:so\s+)?(?:you(?:'re| are)\s+(?:saying|telling me)|so\s+you\s+mean|so\s+what\s+you(?:'re| are)\s+saying)\b", re.I)


def is_parrot(sentence: str, heard: str) -> bool:
    """True if `sentence` only repeats content words from `heard`."""
    s = (sentence or "").strip()
    if not s:
        return False
    words = _WORD.findall(s.lower())
    if _RESTATE_FRAME.match(s):
        # "So you're saying X?" / "You're telling me X?" restates by construction
        # (owner 10-02). The rest of the reply, if any, is still spoken.
        return True
    if s.endswith("?"):
        # Live 09-30: "Ridiculous?" / "Retarded?" / "The Green Goblin?" / "It was fine
        # to begin with?" -- repeating their line back with a question mark is still an
        # echo. A wh-word or any new content word makes it a real question; keep those.
        heard_words = set(_WORD.findall((heard or "").lower()))
        # A wh-word only makes it a real question if WE added it; "What the stigma?"
        # echoed back to "What the stigma?" is still an echo (live 10-02 ablation).
        if not words or len(words) > ECHO_Q_MAX_WORDS or any(w in _WH and w not in heard_words for w in words):
            return False
        content = [_norm(w) for w in words if w not in _GLUE and w not in _WH]
        pool = {_norm(w) for w in heard_words}
        if not content:
            return bool(words) and all(w in heard_words for w in words)
        return all(w in pool for w in content)
    if not words or len(words) > MAX_WORDS:
        return False
    content = [_norm(w) for w in words if w not in _GLUE]
    asked = _QUESTION.search((heard or "").strip()) or _PROMPT.search((heard or "").strip())
    if not content:
        # 3+ words of pure glue ("Rokay, he's got that.") is an empty acknowledgement;
        # a bare "Yeah." is left to the loop-filler screen; answers to questions stay.
        # A bare summons ("Pup.") still gets its "I'm here".
        return len(words) >= 3 and not asked and len(_WORD.findall((heard or "").lower())) > 2
    if asked:
        return False  # answering a question / choice prompt with its own words is an answer
    pool = {_norm(w) for w in _WORD.findall((heard or "").lower())}
    return all(w in pool for w in content)


def strip_label(text: str) -> str:
    return re.sub(r"^\s*\[[^\]]{1,24}\]\s*", "", text or "")


def drop_parrots(history: list) -> list:
    """Remove assistant replies that only echoed the user line right before them."""
    out, last_user = [], ""
    for m in history:
        role, content = m.get("role"), m.get("content") or ""
        if role == "user":
            last_user = strip_label(content)
        elif role == "assistant":
            spoken = re.sub(r"^\s*\[[^\]]*\]\s*", "", content)
            if last_user and is_parrot(spoken, last_user):
                continue
        out.append(m)
    return out
