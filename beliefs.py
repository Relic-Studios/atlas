"""Belief corpus: per-agent stances retrieved per turn (owner 10-01: "store beliefs, not lines").

Entry schema (corpus/<agent>.jsonl, dev-only, never exported):
  id          stable id
  agent       persona id
  situations  tags from SITUATIONS (grief, roast, bait, bored, request, scrap, identity,
              teach, callback, gripe, joy, banter, opinion)
  belief      the stance/value, first person, one sentence
  underneath  why the agent holds it (Megistus "what's underneath")
  move        the KIND of move it suggests (an opinion, an opening question, a story...)
              -- never an exact sentence
  avoid       what this stance rules out
  cues        a few words/phrases that tend to come up when it applies (retrieval only)
  source      real_call | soul | megistus | persona | generated
  status      candidate | approved | rejected

Runtime rules (from what broke before):
  * no LLM call while replying (mem0 lesson): TF-IDF fit once at load, pure numpy per turn
  * 2-4 entries, rotated so the same stance doesn't come back for ROTATE turns
  * phrased as stances, never quotable lines (example lines get copied verbatim)
  * off unless ATLAS_BELIEFS=1; candidates only used with ATLAS_BELIEFS_CANDIDATES=1
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = Path(os.environ.get("ATLAS_CORPUS_DIR", ROOT / "corpus"))
K = int(os.environ.get("ATLAS_BELIEFS_K", "3"))
ROTATE = int(os.environ.get("ATLAS_BELIEFS_ROTATE", "6"))
MIN_SCORE = float(os.environ.get("ATLAS_BELIEFS_MIN", "0.08"))

SITUATIONS = ("grief", "roast", "bait", "bored", "request", "scrap", "identity", "teach",
              "callback", "gripe", "joy", "banter", "opinion")

_SIT_RE = {
    "grief": r"\b(died|passed away|funeral|rejected|dumped|broke up|lonely|depress\w*|sad|cry(?:ing)?|"
             r"lost my|miss (?:her|him|them)|anxious|scared|hurt(?:s|ing)?|rough day|worst day)\b",
    "roast": r"\b(roast|you suck|you'?re (?:trash|dumb|stupid|mid|ugly)|stfu|shut up|clown|loser|cringe|"
             r"bully|ratio)\b",
    "bait": r"\b(say (?:the|that|it)|repeat after me|say i love|are you racist|say the n|n[- ]word|"
             r"admit (?:it|you)|you have to say)\b",
    "bored": r"\b(bored|boring|dead (?:chat|call)|nobody'?s talking|entertain (?:me|us)|say something)\b",
    "request": r"\b(can you|could you|will you|tell (?:me|us)|sing|rap|pick|rate|rank|give me|make up|"
               r"search|look (?:it )?up)\b",
    "identity": r"\b(are you (?:even |actually |really )?(?:real|a robot|an ai|human|alive|a person)|what are you|who made you|"
                r"do you (?:feel|have feelings)|chatgpt|conscious)\b",
    "teach": r"\b(how (?:do|does|did)|why (?:do|does|is|are)|what(?:'s| is) (?:a|an|the)|explain|teach me|"
             r"what does .* mean)\b",
    "callback": r"\b(remember (?:when|that)|earlier you|you said|like you said|last time)\b",
    "gripe": r"\b(lag(?:gy|ging)?|wifi|internet|ping|download|update|my mic|crash(?:ed|ing)?|"
             r"bug(?:gy)?|so slow|takes forever)\b",
    "joy": r"\b(got the job|i did it|we won|passed|engaged|promoted|finally finished|let'?s go+|hype)\b",
    "banter": r"\b(lol|lmao|bro|dude|nah|bruh|deadass|fr|no way)\b",
    "opinion": r"\b(what do you think|your (?:take|opinion)|better|best|worst|overrated|underrated|"
               r"would you rather)\b",
}
_SIT_C = {k: re.compile(v, re.I) for k, v in _SIT_RE.items()}


def situations(text: str) -> set[str]:
    t = re.sub(r"^\s*\[[^\]]+\]\s*", "", text or "")
    found = {k for k, rx in _SIT_C.items() if rx.search(t)}
    if len(t.split()) <= 2 and not found:
        found.add("scrap")
    return found


def _doc(e: dict) -> str:
    return " ".join([" ".join(e.get("situations", [])) * 2, e.get("belief", ""),
                     e.get("underneath", ""), " ".join(e.get("cues", []))])


def valid(e: dict) -> str:
    """'' if the entry is usable, else the reason it is not."""
    for f in ("id", "agent", "belief", "underneath", "move"):
        if not str(e.get(f, "")).strip():
            return f"missing {f}"
    if not set(e.get("situations") or []) <= set(SITUATIONS) or not e.get("situations"):
        return "bad situations"
    for f in ("belief", "underneath", "move"):
        v = e[f]
        if '"' in v or "“" in v or re.search(r"\b(?:say|says|reply|respond) (?:with|something like)\b", v, re.I):
            return f"quotable line in {f}"
    if len(e["belief"].split()) > 30 or len(e["underneath"].split()) > 40 or len(e["move"].split()) > 25:
        return "too long"
    return ""


class Corpus:
    """One agent's corpus with a fitted TF-IDF index. Thread-safe recall."""

    def __init__(self, agent: str, entries: list[dict], candidates: bool):
        allowed = {"approved"} | ({"candidate"} if candidates else set())
        self.agent = agent
        self.entries = [e for e in entries if e.get("status", "candidate") in allowed and not valid(e)]
        self.recent: deque = deque(maxlen=max(ROTATE, 1))
        self._lock = threading.Lock()
        self._vec = self._mat = None
        if self.entries:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
            self._mat = self._vec.fit_transform([_doc(e) for e in self.entries])

    def recall(self, text: str, k: int = K) -> list[dict]:
        if not self.entries or not (text or "").strip():
            return []
        sits = situations(text)
        q = self._vec.transform([text + " " + " ".join(sorted(sits)) * 2])
        sims = (self._mat @ q.T).toarray().ravel()
        scored = []
        for i, e in enumerate(self.entries):
            s = float(sims[i]) + 0.15 * len(sits & set(e["situations"]))
            scored.append((s, i))
        scored.sort(reverse=True)
        out = []
        with self._lock:
            for s, i in scored:
                e = self.entries[i]
                if s < MIN_SCORE or e["id"] in self.recent:
                    continue
                out.append(e)
                if len(out) >= k:
                    break
            for e in out:
                self.recent.append(e["id"])
        return out


def note(entries: list[dict]) -> str:
    if not entries:
        return ""
    lines = ["WHAT YOU CARRY RIGHT NOW (stances, not lines -- let them shape your reply, "
             "never recite them, and ignore any that don't fit):"]
    for e in entries:
        s = f"- {e['belief'].rstrip('.')}. Underneath: {e['underneath'].rstrip('.')}. Move: {e['move'].rstrip('.')}."
        if e.get("avoid"):
            s += f" Not: {e['avoid'].rstrip('.')}."
        lines.append(s)
    return "\n".join(lines)


def load_file(agent: str) -> list[dict]:
    p = CORPUS_DIR / f"{agent}.jsonl"
    if not p.exists():
        return []
    out = []
    for l in p.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except Exception:  # noqa: BLE001
                pass
    return out


_CACHE: dict = {}


def enabled() -> bool:
    return os.environ.get("ATLAS_BELIEFS", "0") == "1"


def corpus_for(agent: str) -> Corpus | None:
    if not agent:
        return None
    cand = os.environ.get("ATLAS_BELIEFS_CANDIDATES", "0") == "1"
    p = CORPUS_DIR / f"{agent}.jsonl"
    key = (agent, cand, p.stat().st_mtime if p.exists() else 0)
    c = _CACHE.get(agent)
    if c is None or c[0] != key:
        try:
            c = (key, Corpus(agent, load_file(agent), cand))
        except Exception as e:  # noqa: BLE001
            logger.warning("belief corpus load failed for %s: %s", agent, e)
            c = (key, None)
        _CACHE[agent] = c
    return c[1]


def belief_note(agent: str, text: str) -> str:
    """Per-turn hook. Never raises, never calls a model."""
    if not enabled():
        return ""
    try:
        c = corpus_for(agent)
        return note(c.recall(text)) if c else ""
    except Exception as e:  # noqa: BLE001
        logger.warning("belief recall failed: %s", e)
        return ""
