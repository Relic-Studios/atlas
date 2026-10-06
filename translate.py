"""Live translate plugin (owner 10-05: a separate add-on on top of Languages).

Two modes:
  * on request -- "what did she say?", "translate that", "what did Riley say in English?"
    The agent gets the real line (speaker, detected language, words) and translates it
    in its reply. Nothing is invented: if no foreign line exists, it says so.
  * captions -- every confident line not in the owner's language is translated in the
    background by the owner's model and shown in the plugin's activity list. Nothing is
    spoken; the reply path is never touched.

Lines live in memory for this session only (never written to disk by this module).
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, List, Optional

MAX_LINES = 400
MAX_AGE_S = 30 * 60
MIN_UNITS = 2
CAPTION_GAP_S = 1.0          # min seconds between background translations
CAPTION_DAILY = 500

_lock = threading.Lock()
_lines: deque = deque(maxlen=MAX_LINES)      # (t, who, text, lang)
_captions: deque = deque(maxlen=60)          # dicts for the feed
_q: deque = deque(maxlen=20)
_worker: Optional[threading.Thread] = None
_day = {"d": "", "n": 0}
_last_cap = [0.0]

_ASK_RE = re.compile(
    r"\b(?:translate|translation)\b"
    r"|\bwhat\s+(?:did|does|was)\s+(?!you\b|i\b|we\b)(?:he|she|they|that|it|\w+)\s+(?:just\s+)?(?:say|said|mean|saying)\b"
    r"|\bwhat\s+(?:is|was|were)\s+(?!you\b|i\b|we\b)(?:he|she|they|\w+)\s+saying\b"
    r"|\bwhat\s+does\s+(?:that|it|this)\s+mean\s+in\s+\w+", re.I)
_NOT_RE = re.compile(r"\b(?:translate\w*\s+(?:it\s+|that\s+|this\s+)?(?:to|into)\s+(?:action|reality|results|sales|money)|lost in translation|(?:say|said|saying)\s+about|google translate is)\b", re.I)
_TO_RE = re.compile(r"\b(?:in|into|to)\s+(english|spanish|french|german|italian|portuguese|russian|"
                    r"chinese|mandarin|japanese|korean)\b", re.I)
_WHO_RE = re.compile(r"\bwhat\s+(?:did|was|is)\s+([A-Z][a-z]{1,20})\s+(?:just\s+)?(?:say|said|saying)\b")
_TRANSL_WHO_RE = re.compile(r"\btranslate\s+(?:what\s+)?([A-Z][a-z]{1,20})(?:\s+(?:said|says|just said))?\b")
_PRON = {"He", "She", "They", "That", "It", "This", "What", "For", "Me", "Us"}

_LANG_CODE = {"english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
              "portuguese": "pt", "russian": "ru", "chinese": "zh", "mandarin": "zh",
              "japanese": "ja", "korean": "ko"}


def _units(text: str) -> int:
    try:
        import languages
        return languages.units(text)
    except Exception:  # noqa: BLE001
        return len((text or "").split())


def _lname(code: str) -> str:
    try:
        import languages
        return languages.name(code)
    except Exception:  # noqa: BLE001
        return code or "?"


def note(who: str, text: str, lang: str, clock: Callable[[], float] = time.time) -> None:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return
    with _lock:
        _lines.append((clock(), who or "", text, (lang or "").lower()))


def reset() -> None:
    with _lock:
        _lines.clear()
        _captions.clear()
        _q.clear()
        _day.update(d="", n=0)
        _last_cap[0] = 0.0


def intent(text: str):
    t = re.sub(r"^\s*\[S\d+\]\s*", "", text or "")
    if not _ASK_RE.search(t) or _NOT_RE.search(t):
        return None
    # Only a translate request if someone actually spoke another language lately:
    # "what did they say about the raid" in an all-English room is a recap, not this.
    with _lock:
        langs = {x[3] for x in _lines if x[3] and time.time() - x[0] <= MAX_AGE_S}
    if len(langs) < 2 and not re.search(r"\btranslat", t, re.I):
        return None
    args = {}
    m = _TO_RE.search(t)
    if m:
        args["to"] = m[1].lower()
    w = _WHO_RE.search(t) or _TRANSL_WHO_RE.search(t)
    if w and w[1] not in _PRON:
        args["who"] = w[1]
    return ("translate_line", args)


def _target(to: str, asker_line: str, primary: str) -> str:
    to = (to or "").strip().lower()
    if to in _LANG_CODE:
        return _LANG_CODE[to]
    if len(to) == 2:
        return to
    try:
        import languages
        return languages.guess_text(asker_line or "", hint=primary) or primary
    except Exception:  # noqa: BLE001
        return primary or "en"


def find(target: str, who: str = "", name_of: Optional[Callable] = None,
         clock: Callable[[], float] = time.time):
    now = clock()
    who = (who or "").strip().lower()
    with _lock:
        rows = list(_lines)
    for t, spk, text, lang in reversed(rows):
        if now - t > MAX_AGE_S:
            break
        if not lang or lang == target or _units(text) < MIN_UNITS:
            continue
        name = (name_of(spk) if name_of else "") or spk
        if who and who not in (name or "").lower():
            continue
        return name, text, lang, now - t
    return None


def translate_line(to: str = "", who: str = "", asker_line: str = "", primary: str = "en",
                   name_of: Optional[Callable] = None, clock: Callable[[], float] = time.time) -> str:
    tgt = _target(to, asker_line, primary)
    hit = find(tgt, who, name_of, clock)
    if not hit:
        who_s = f" from {who}" if who else ""
        return (f"No line{who_s} in another language in the last {MAX_AGE_S // 60} minutes. "
                f"Say so plainly; don't guess what anyone said.")
    name, text, lang, ago = hit
    tl = _lname(tgt)
    return (f'LINE TO TRANSLATE: {name or "Someone"} said, in {_lname(lang)} ({int(ago)}s ago): "{text}". '
            f"Translate it faithfully into {tl} and say who said it. Answer in {tl} this time, even if "
            f"the room is speaking another language. Keep their meaning and tone; don't add or soften "
            f"anything. If part of it is unclear, say which part.")


# ---------------------------------------------------------------- captions (background)
def _today(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(now))


def _work(ask: Callable[[str, str], str]) -> None:
    while True:
        with _lock:
            item = _q.popleft() if _q else None
        if item is None:
            break
        try:
            out = ask("You translate one spoken line. Output only the translation, nothing else.",
                      f"Translate from {_lname(item['lang'])} into {_lname(item['to'])}:\n{item['text']}")
            out = re.sub(r"(?s)<think>.*?</think>", "", out or "").strip().strip('"')
            item["translation"] = out[:600] or "(no translation)"
            item["status"] = "done"
        except Exception as e:  # noqa: BLE001
            item["translation"] = ""
            item["status"] = f"failed: {type(e).__name__}"


def _default_ask(system: str, user: str) -> str:
    import soul_reflection
    return soul_reflection.complete(system, user, timeout=60.0)


def caption(who: str, text: str, lang: str, prob: float, target: str,
            ask: Optional[Callable[[str, str], str]] = None, daily: int = CAPTION_DAILY,
            clock: Callable[[], float] = time.time, sync: bool = False) -> Optional[dict]:
    """Queue one line for a background translation. Returns the feed item, or None if skipped."""
    global _worker
    lang = (lang or "").lower()
    if not lang or lang == target or _units(text) < MIN_UNITS:
        return None
    try:
        import languages
        if float(prob or 0) < languages.CONFIDENT:
            return None
    except Exception:  # noqa: BLE001
        pass
    now = clock()
    with _lock:
        if _day["d"] != _today(now):
            _day.update(d=_today(now), n=0)
        if _day["n"] >= daily or now - _last_cap[0] < CAPTION_GAP_S:
            return None
        _day["n"] += 1
        _last_cap[0] = now
        item = {"t": now, "who": who or "", "text": text, "lang": lang, "to": target,
                "translation": "", "status": "pending"}
        _captions.append(item)
        _q.append(item)
    fn = ask or _default_ask
    if sync:
        _work(fn)
    elif _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target=_work, args=(fn,), daemon=True, name="atlas-captions")
        _worker.start()
    return item


def feed() -> List[dict]:
    with _lock:
        items = list(_captions)
    out = []
    for it in reversed(items):
        tr = it["translation"] or ("translating…" if it["status"] == "pending" else it["status"])
        out.append({"title": f'{it["who"] or "Someone"}: "{it["text"]}"',
                    "meta": time.strftime("%H:%M:%S", time.localtime(it["t"]))
                            + f' · {_lname(it["lang"])} → {_lname(it["to"])}',
                    "items": [{"text": tr}]})
    return out
