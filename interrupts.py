"""Floor holding and resume-after-interruption. Pure logic, no model, no audio.

Two live complaints (09-29):
  * Any 3-word line from anyone stopped the agent mid-sentence, so side chatter
    and "wait what, no way" cut her off constantly and people got annoyed.
  * When she WAS cut off, the whole generated reply went into history as if she
    had said it, so she never knew what she didn't get to say and never came back
    to it.

should_yield()  -- does speech heard over the agent take the floor from her?
split_spoken()  -- how much of a reply was actually heard before the cut.
CutOff          -- what was said / unsaid, used for the resume note and gap cue.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

RESUME_TTL_S = 30.0         # after this the thought is stale; drop it
FINISH_S = 2.0              # this close to the end of her line: just finish it
                            # (live 10-05: 1.3-1.7s left still got cut; a 2s overlap
                            #  is normal in a group, a half-sentence is not)
DEFAULT_CPS = 14.0          # spoken chars/second when synthesis isn't done yet
MIN_UNSAID_WORDS = 4        # less left than this = she basically finished

RESUME_CUE_RE = re.compile(r"\(RESUME CUE\)")


PARTNER_KEEP_GOING = 6      # content words before the partner counts as taking over
_PUSHBACK = {"wait", "no", "nah", "nope", "but", "actually", "hold", "stop", "hang"}
_GROUP_VOCATIVE = {"guys", "everyone", "everybody", "yall", "y'all", "chat", "people"}


def _body(text: str) -> str:
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", text or "")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def takeover_words(talkativeness: float) -> int:
    """Words a third party must get out before she yields. Chattier characters
    hold the floor longer (0 -> 6 words, 0.35 -> 9, 1 -> 14)."""
    t = min(1.0, max(0.0, float(talkativeness)))
    return 6 + round(8 * t)


def should_yield(text: str, names: tuple[str, ...] = (), *, speaker: str | None = None,
                 partner: str | None = None, agent_left_s: float | None = None,
                 talkativeness: float = 0.35) -> tuple[bool, str]:
    """Should speech heard WHILE the agent talks stop her? Returns (yield, reason).

    Yield to: her name, a stop/shut-up aimed at her, her conversation partner
    taking the floor back, or anyone who clearly takes over (sustained speech).
    Talk over: backchannels, short remarks, side chatter addressed to someone
    else, and anything that lands when she's about to finish anyway.
    """
    from floor import BACKCHANNEL
    body = _body(text)
    w = _words(body)
    if not w:
        return False, "empty"
    lowered = {n.lower() for n in names}
    if any(t in lowered for t in w):
        return True, "named"
    from conversation_dynamics import detect_silence_request, detect_other_addressee
    to_other = bool(detect_other_addressee(body, names))
    if detect_silence_request(body) is not None and not to_other:
        return True, "stop"
    content = [t for t in w if t not in BACKCHANNEL]
    if agent_left_s is not None and agent_left_s <= FINISH_S:
        return False, "finishing her line"
    if to_other:
        return False, "side chatter"
    if speaker and partner and speaker == partner:
        # Live call 10-05: 23 of 37 cut-offs were "partner took the floor" on any
        # 3 words -- "I don't know.", "I can't do this.", "Guys, I'm hungry.",
        # "So would I-". Those are remarks, not a bid for the floor. Yield only to
        # a question, a pushback opener, or the partner clearly carrying on.
        # (Called on every partial, so a remark that keeps growing still yields.)
        if w[0] in _GROUP_VOCATIVE and len(content) < PARTNER_KEEP_GOING + 2:
            return False, "partner to the room"
        if "?" in body and len(content) >= 2:
            return True, "partner asked"
        if w[0] in _PUSHBACK and len(content) >= 2:
            return True, "partner pushback"
        if len(content) >= PARTNER_KEEP_GOING:
            return True, "partner kept going"
        return False, "partner remark"
    if len(w) >= takeover_words(talkativeness) and len(content) >= 5:
        return True, "takeover"
    return False, "talk over"


def split_spoken(text: str, played_s: float, total_s: float, synth_done: bool) -> tuple[str, str]:
    """Split a reply into (heard, not heard) from how much of its audio played.

    If synthesis finished, the reply's own chars/second is known exactly;
    otherwise use a typical speaking rate. Snaps to a word boundary."""
    text = " ".join((text or "").split())
    if not text:
        return "", ""
    if synth_done and total_s > 0.5:
        cps = len(text) / total_s
    else:
        cps = DEFAULT_CPS
    n = int(max(0.0, played_s) * cps)
    if n >= len(text):
        return text, ""
    cut = text.rfind(" ", 0, n + 1)
    cut = 0 if cut < 0 else cut
    return text[:cut].rstrip(" ,;-"), text[cut:].strip()


@dataclass
class CutOff:
    said: str
    unsaid: str
    target: str = ""
    by: str | None = None
    at: float = field(default_factory=time.time)
    gap_tried: bool = False

    def live(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return bool(self.unsaid) and now - self.at <= RESUME_TTL_S

    def note(self, now: float | None = None) -> str:
        """Context note for the next turn. The MODEL decides if the topic moved on."""
        now = time.time() if now is None else now
        said = self.said or "(nothing yet)"
        return ("CUT OFF: you got talked over mid-sentence %ds ago. You'd said: \"%s\" "
                "and were about to say: \"%s\". If the conversation is still on that, "
                "pick it back up from where you stopped, in your own words (a quick "
                "'anyway' or 'like I was saying' is fine; don't restart from the top). "
                "If they changed the subject or asked you something new, answer that "
                "instead and drop it." % (int(now - self.at), said[-160:], self.unsaid[:200]))

    def cue(self) -> str:
        lead = f"[{self.target}] " if re.fullmatch(r"S\d+", self.target or "") else ""
        return (f"{lead}(RESUME CUE) Nobody is talking right now. This is NOT a new line from "
                "anyone: it's your own cue. You got cut off earlier (see CUT OFF). If it still "
                "fits, finish your thought now in one or two lines, naturally. If the room has "
                "moved on, output [HOLD].")


def make_cutoff(text: str, played_s: float, total_s: float, synth_done: bool,
                target: str = "", by: str | None = None) -> CutOff | None:
    """CutOff for a reply stopped mid-way, or None if she'd basically finished."""
    said, unsaid = split_spoken(text, played_s, total_s, synth_done)
    if len(_words(unsaid)) < MIN_UNSAID_WORDS:
        return None
    return CutOff(said=said, unsaid=unsaid, target=target or "", by=by)
