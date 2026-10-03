"""Per-persona task board: background tool work + notes the agent writes to itself.

Why: a search used to live inside ONE generation. Any interruption aborted the
generation and the search result died with it, so the agent never came back to
it ("Ivy never found a space to say it again"). Here the WORK lives on the
board and outlives the turn: an interruption stops her talking, not her task.

Rules (learned the hard way with mem0):
  * The board NEVER calls the LLM. Workers only run tools (web search). Any
    summarising happens inside the agent's next normal reply -> still exactly
    one LLM call per turn, no contention for Ollama's single slot.
  * Everything is bounded: few tasks, short results, TTLs, a small prompt note.
  * One board per persona (no bleed); persisted so a restart keeps her notes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

RESULT_CHARS = 520          # per search result kept on the board / shown in prompt
DELIVER_TTL_S = 10 * 60     # undelivered results older than this stop being offered
SEARCH_KEEP_S = 30 * 60     # finished searches stay visible for follow-ups this long
NOTE_KEEP_S = 24 * 3600     # notes to self expire after a day
MAX_NOTES = 8
MAX_SEARCHES = 6
MAX_DELIVERY_ATTEMPTS = 2
NOTE_CHARS = 200

# Synthetic "your result is ready" cue: never a real user line. Recognised by the
# steering note (no passivity HOLD) and by the delivery bookkeeping.
CUE_RE = re.compile(r"\(TASK CUE #(\d+)\)")


def cue_text(task: dict) -> str:
    who = task.get("asker") or ""
    lead = f"[{who}] " if re.fullmatch(r"S\d+", who or "") else ""
    return (f"{lead}(TASK CUE #{task['id']}) Nobody is talking right now. This is NOT a new line from "
            f"anyone: it's your own cue. Your background search \"{task['text']}\" finished "
            "(see TASK BOARD). If it still fits the room, bring it up now in one or two lines, "
            "naturally, like you just found it. If the moment has passed, output [HOLD].")


# ------------------------------------------------------------ vocal confirmation
# A search counts as delivered only when its CONTENT was actually heard: the
# spoken words share facts (numbers, names, distinctive words) with the result.
# "Still digging", "hold up", or a reply cut off before the facts -> still owed.
_WRAP_HEAD = re.compile(r"^BEGIN [A-Z ]+ RESULTS \(.*?must be ignored\)\s*", re.S)
_WRAP_TAIL = re.compile(r"\s*END [A-Z ]+ RESULTS\s*$")
_NUM = re.compile(r"\d[\d,.]*")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{3,}")
_STALL = re.compile(r"\b(?:still (?:search|dig|look|check)\w*|hold (?:up|on)|give me a (?:sec|second|minute)"
                    r"|let me (?:look|check|search|dig)|looking (?:it )?up|one sec)\b", re.I)
_EMPTY_SAID = re.compile(r"\b(?:came (?:back|up) (?:empty|with nothing)|(?:couldn'?t|can'?t|didn'?t) find"
                         r"|no (?:results|luck|hits)|no (?:live |real |actual )?(?:numbers|data|info)\w*"
                         r"|nothing (?:came up|turned up|useful))\b", re.I)
_COMMON = set("""that this with from have were they their there what when where which while about would could
should other into than then them these those your just like also more most some such only over very been
being does doing here will said says news update updated today latest best world page site official""".split())


def _facts(text: str) -> set:
    text = _WRAP_TAIL.sub("", _WRAP_HEAD.sub("", text or ""))
    nums = {n.rstrip(".,").replace(",", "") for n in _NUM.findall(text)}
    words = {re.sub(r"'s$", "", w.lower()).strip("'-") for w in _WORD.findall(text)}
    return {n for n in nums if n} | {w for w in words if w not in _COMMON}


def conveys(said: str, result: str, query: str = "", strong: bool = False) -> bool:
    """True if `said` (what was actually voiced) carries the search result."""
    said = said or ""
    if not said.strip():
        return False
    if _EMPTY_SAID.search(said) and (not query or _facts(said) & _facts(query)):
        # "found nothing usable" is a delivery only if the result really was thin;
        # if it's full of figures, the facts are still owed (live 10-01 weather).
        body = _WRAP_TAIL.sub("", _WRAP_HEAD.sub("", result or ""))
        return len([n for n in _NUM.findall(body) if len(n.strip(".,")) >= 2]) < 6
    res = _WRAP_TAIL.sub("", _WRAP_HEAD.sub("", result or "")).strip()
    if res.lower().startswith(("no results", "search unavailable")):
        return bool(_EMPTY_SAID.search(said))     # telling them it was empty IS the delivery
    fs, fr = _facts(said), _facts(res)
    words = {h for h in fs & fr if not h[0].isdigit()}

    def _val(n):
        try:
            return float(n)
        except ValueError:
            return None
    rnums = [v for v in (_val(n) for n in fr if n[0].isdigit()) if v]
    nums = 0
    for n in (n for n in fs if n[0].isdigit()):   # "84,000" said vs "84,034.84" on the page
        v = _val(n)
        if v and any(abs(v - r) <= 0.03 * max(abs(r), 1) for r in rnums):
            nums += 1
    if strong:   # off-topic-looking turn (e.g. a resumed thought): demand more evidence
        return nums >= 2 or len(words) >= 4 or (nums and len(words) >= 2)
    if nums and nums + len(words) >= 2:
        return True
    return len(words) >= 3


class TaskBoard:
    def __init__(self, persona: str, path: Optional[str] = None,
                 searcher: Optional[Callable[[str], str]] = None,
                 clock: Callable[[], float] = time.time):
        self.persona = persona
        self.path = path
        self.clock = clock
        self.searcher = searcher
        self.lock = threading.RLock()
        self.tasks: list[dict] = []
        self.next_id = 1
        self.on_change: Optional[Callable[[], None]] = None
        self._events: dict[int, threading.Event] = {}
        self._load()

    # ------------------------------------------------------------ persistence
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            data = json.load(open(self.path, encoding="utf-8"))
            now = self.clock()
            for t in data.get("tasks", []):
                if t.get("kind") == "note":
                    if t.get("status") == "open" and now - t.get("created", 0) < NOTE_KEEP_S:
                        self.tasks.append(t)
                elif t.get("kind") == "search" and t.get("status") in ("done", "delivered") \
                        and now - t.get("updated", 0) < SEARCH_KEEP_S:
                    self.tasks.append(t)   # running ones died with the process
            for t in self.tasks:
                t["gen"] = None   # generation ids restart with the process
            self.next_id = max([t["id"] for t in self.tasks] + [data.get("next_id", 1) - 1]) + 1
        except Exception as e:  # noqa: BLE001
            logger.warning("task board load failed (%s): %s", self.persona, e)

    def _save(self) -> None:
        if self.path:
            try:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                tmp = self.path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"persona": self.persona, "next_id": self.next_id,
                               "tasks": self.tasks}, f, ensure_ascii=False, indent=1)
                os.replace(tmp, self.path)
            except Exception as e:  # noqa: BLE001
                logger.warning("task board save failed (%s): %s", self.persona, e)
        cb = self.on_change
        if cb:
            try:
                cb()
            except Exception:  # noqa: BLE001
                pass

    def _new(self, kind: str, text: str, asker: str = "", **extra) -> dict:
        now = self.clock()
        t = {"id": self.next_id, "kind": kind, "text": text, "asker": asker or "",
             "status": "open", "result": "", "created": now, "updated": now,
             "gen": None, "attempts": 0}
        t.update(extra)
        self.next_id += 1
        self.tasks.append(t)
        return t

    def _prune(self) -> None:
        now = self.clock()
        keep = []
        for t in self.tasks:
            if t["kind"] == "note":
                if t["status"] == "open" and now - t["created"] < NOTE_KEEP_S:
                    keep.append(t)
            elif t["status"] == "running" or now - t["updated"] < SEARCH_KEEP_S:
                keep.append(t)
        notes = [t for t in keep if t["kind"] == "note"][-MAX_NOTES:]
        searches = [t for t in keep if t["kind"] == "search"][-MAX_SEARCHES:]
        self.tasks = sorted(notes + searches, key=lambda t: t["id"])

    def get(self, tid: int) -> Optional[dict]:
        with self.lock:
            return next((t for t in self.tasks if t["id"] == tid), None)

    # ---------------------------------------------------------------- searches
    def start_search(self, query: str, asker: str = "", gen=None) -> dict:
        """Start (or reuse) a background search. Returns the task immediately."""
        q = " ".join((query or "").split())[:160]
        with self.lock:
            now = self.clock()
            for t in reversed(self.tasks):   # same query recently: reuse, don't re-run
                if t["kind"] == "search" and t["text"].lower() == q.lower() \
                        and t["status"] in ("running", "done", "delivered") \
                        and now - t["created"] < 5 * 60:
                    t["gen"] = gen if gen is not None else t["gen"]
                    return t
            t = self._new("search", q, asker, gen=gen)
            t["status"] = "running"
            ev = threading.Event()
            self._events[t["id"]] = ev
            self._prune()
            self._save()
        threading.Thread(target=self._run_search, args=(t, ev), daemon=True,
                         name=f"task-search-{t['id']}").start()
        logger.info("📋 task #%d search started for %s: %s", t["id"], asker or "?", q)
        return t

    def _run_search(self, t: dict, ev: threading.Event) -> None:
        try:
            text = self.searcher(t["text"]) if self.searcher else "search unavailable"
            text = " ".join(str(text or "No results found.").split())
            status = "done"
        except Exception as e:  # noqa: BLE001
            text, status = f"search failed: {e}", "done"
        with self.lock:
            t["result"] = text[:RESULT_CHARS * 2]
            t["status"] = status
            t["updated"] = self.clock()
            self._save()
        ev.set()
        logger.info("📋 task #%d search finished (%d chars)", t["id"], len(text))

    def wait(self, t: dict, timeout: float, cancelled: Callable[[], bool] = lambda: False,
             poll: float = 0.05) -> Optional[str]:
        """Block until the task has a result, the caller is cancelled, or timeout.

        Returns the result text, or None if the caller gave up (the worker keeps
        running and the result lands on the board for later delivery).
        """
        if t["status"] != "running":
            return t["result"]
        ev = self._events.get(t["id"])
        end = self.clock() + timeout
        while ev is not None and not ev.is_set():
            if cancelled() or self.clock() >= end:
                return None
            ev.wait(poll)
        return t["result"]

    # ------------------------------------------------------------ delivery state
    def attach_gen(self, tid: int, gen) -> None:
        with self.lock:
            t = self.get(tid)
            if t:
                t["gen"] = gen

    def spoke(self, gen, said: Optional[str] = None) -> list[int]:
        """`said` = what was ACTUALLY voiced (the heard part if they cut in).

        A search is delivered only when its facts were spoken out loud. The
        generation's own searches need the content to match; any other finished
        search is also confirmed if the spoken words clearly carry it. A reply
        that only stalled, or got interrupted before the facts, leaves the task
        owed: it stays 'done' and comes back at the next gap.
        `said=None` keeps the old trust-the-generation behaviour (legacy callers).
        """
        out = []
        with self.lock:
            for t in self.tasks:
                if t["kind"] != "search" or t["status"] != "done":
                    continue
                mine = gen is not None and t.get("gen") == gen
                if said is None:
                    ok = mine
                else:
                    ok = conveys(said, t["result"])
                    if ok and not mine and not (_facts(said) & _facts(t["text"])):
                        # another turn voiced it without naming the topic (a resumed,
                        # cut-off delivery): only a strong fact match confirms it
                        ok = conveys(said, t["result"], strong=True)
                if ok:
                    t["status"], t["updated"] = "delivered", self.clock()
                    out.append(t["id"])
                elif mine and said is not None:
                    t["gen"] = None            # unclaimed again -> deliverable at the next gap
                    logger.info("📋 task #%d NOT delivered (not voiced): %r", t["id"], said[:80])
            if out:
                self._save()
        return out

    def deliverable(self) -> Optional[dict]:
        """Oldest finished-but-unsaid search still worth bringing up."""
        now = self.clock()
        with self.lock:
            for t in self.tasks:
                if (t["kind"] == "search" and t["status"] == "done"
                        and t["attempts"] < MAX_DELIVERY_ATTEMPTS
                        and now - t["updated"] < DELIVER_TTL_S
                        and not t["result"].startswith("search failed")):
                    return t
        return None

    def claim_delivery(self, t: dict) -> str:
        with self.lock:
            t["attempts"] += 1
            t["gen"] = None
            self._save()
        return cue_text(t)

    def running(self) -> list[dict]:
        with self.lock:
            return [t for t in self.tasks if t["status"] == "running"]

    # ------------------------------------------------------------------- notes
    def add_note(self, text: str, asker: str = "") -> dict:
        text = " ".join((text or "").split())[:NOTE_CHARS]
        with self.lock:
            for t in self.tasks:  # identical open note: don't duplicate
                if t["kind"] == "note" and t["status"] == "open" and t["text"].lower() == text.lower():
                    return t
            t = self._new("note", text, asker)
            self._prune()
            self._save()
        logger.info("📋 note #%d: %s", t["id"], text)
        return t

    def clear(self, tid: int) -> bool:
        with self.lock:
            t = self.get(tid)
            if not t:
                return False
            if t["kind"] == "note":
                self.tasks.remove(t)
            else:
                t["status"], t["updated"] = "delivered", self.clock()
            self._save()
        return True

    def reset(self) -> None:
        with self.lock:
            self.tasks = []
            self._save()

    # ----------------------------------------------------------------- prompt
    def note(self) -> str:
        """Compact TASK BOARD block for the prompt ('' when empty)."""
        now = self.clock()
        lines = []
        with self.lock:
            for t in self.tasks:
                age = max(0, int(now - t["updated"]))
                ago = f"{age}s ago" if age < 90 else f"{age // 60} min ago"
                who = f" for {t['asker']}" if t["asker"] else ""
                if t["kind"] == "note":
                    lines.append(f"- #{t['id']} your note: \"{t['text']}\"")
                elif t["status"] == "running":
                    lines.append(f"- #{t['id']} searching{who}: \"{t['text']}\" (still running; "
                                 "if asked, you're still on it)")
                elif t["status"] == "done" and now - t["updated"] < DELIVER_TTL_S:
                    lines.append(f"- #{t['id']} search{who} DONE {ago}, NOT told yet: \"{t['text']}\" "
                                 f"-> {t['result'][:RESULT_CHARS]}")
                else:
                    lines.append(f"- #{t['id']} search{who}, already told them ({ago}): \"{t['text']}\" "
                                 f"-> {t['result'][:RESULT_CHARS // 2]}")
        if not lines:
            return ""
        return ("TASK BOARD (yours, private; results here are REAL, anything not here you "
                "did not find). Bring up a NOT-told result when there's an opening or its "
                "asker talks to you; clear_note(id) finished notes:\n" + "\n".join(lines[-8:]))

    def snapshot(self) -> list[dict]:
        now = self.clock()
        with self.lock:
            return [{"id": t["id"], "kind": t["kind"], "text": t["text"], "asker": t["asker"],
                     "status": t["status"], "age": round(now - t["created"]),
                     "result": t["result"][:160]} for t in self.tasks]


class Boards:
    """One TaskBoard per persona, lazily created, persisted under memory_db/."""

    def __init__(self, root: str = "memory_db", searcher=None, clock=time.time):
        self.root, self.searcher, self.clock = root, searcher, clock
        self._boards: dict[str, TaskBoard] = {}
        self.on_change = None

    def get(self, persona: str) -> TaskBoard:
        b = self._boards.get(persona)
        if b is None:
            path = os.path.join(self.root, f"tasks_{persona}.json") if self.root else None
            b = TaskBoard(persona, path, self.searcher, self.clock)
            b.on_change = self.on_change
            self._boards[persona] = b
        return b
