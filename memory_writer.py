"""The post-call pass (docs/designs/experience_memory.md, D3, S1, S2, S6): turns finished calls into episodes and
owner facts. Runs as its own process, below normal priority, two CPU threads, no GPU:

    python memory_writer.py --mode remote
    python memory_writer.py --mode local --local-url http://127.0.0.1:11434 --local-model model --ctx 4096

The server starts it between calls and kills it on any speech activity; the desktop app starts it after quit (not in
local mode). A killed run leaves the stores as they were; the next run does the same call again and replaces its
records. Exit codes: 0 done, 2 a run failed (counted in <call>.fails), 3 the name tagger is missing.

For each pending file of an ended call (step numbers from D3):
  1. owner lines: owner turns whose label (or the label it merged into) was not refused later in the call;
  2. redaction in the D3 order, with the call's name list from the file;
  3. per agent active for 6+ human turns: chunk requests (JSON schema), validated, then one merge request;
  4. owner facts from qualifying owner lines (OWNER_FACTS_FROM_PASS);
  5. the validator on every item;
  6. write under the store lock, then the tagger check of unchecked lore, lessons, predictions and said facts, then
     delete the pending file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("memory_writer")

OWNER_FACTS_FROM_PASS = True      # from tools/voice_id_eval.py --turn (owner_turn_eval.md: PASS on 2026-11-04)
EPISODE_MIN_TURNS = 6             # S6
CHUNK_TOKENS = 2500               # S6
CONTEXT_BEFORE, CONTEXT_AFTER = 2, 1
MAX_FAILS = 3
LOG_KEEP = 500

# Bounded so an output always fits its token budget: cited lines per item (an item citing dozens of lines once ran a
# chunk past 900 tokens) and text lengths (S2: 60 words for the whole summary).
TEXT_MAX, SHORT_MAX = 300, 60
TURNS = {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 8}
TEXT_ITEM = {"type": "object", "properties": {"text": {"type": "string", "maxLength": TEXT_MAX}, "turns": TURNS},
             "required": ["text", "turns"]}
SHORT = {"type": "string", "maxLength": SHORT_MAX}
EPISODE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": TEXT_ITEM, "maxItems": 3},
        "topics": {"type": "array", "items": SHORT, "maxItems": 5},
        "games": {"type": "array", "items": SHORT, "maxItems": 5},
        "decisions": {"type": "array", "items": TEXT_ITEM, "maxItems": 5},
        "jokes": {"type": "array", "items": TEXT_ITEM, "maxItems": 5},
        "moments": {"type": "array", "items": TEXT_ITEM, "maxItems": 6},
    },
    "required": ["summary", "topics", "games", "decisions", "jokes", "moments"],
}
FACT_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "maxItems": 10, "items": {
        "type": "object", "properties": {"fact": {"type": "string", "maxLength": 200},
                                         "quote": {"type": "string", "maxLength": 200},
                                         "turns": {"type": "array", "items": {"type": "integer"}, "minItems": 1,
                                                   "maxItems": 3}},
        "required": ["fact", "quote", "turns"]}}},
    "required": ["facts"],
}

EPISODE_RULES = """You write a short memory of one voice call for {agent}, an AI agent who was in the call.
The transcript below has numbered lines. People appear only as "Owner" (the person who runs {agent}), Person1,
Person2, ... (other people) and "Someone" (a voice not told apart from the others); their names were removed and
became "someone". Agent lines are marked "(agent)".

Write only what happened: what was talked about, played, decided and joked about, and memorable moments.
- Write about "the group", "someone", "the owner" or the agents by name. Never write Person1, Person2 or any name.
- Never describe what a person is like (their job, looks, likes, feelings, where they live): only events.
- Every summary sentence, decision, joke and moment lists the numbers of the human lines it comes from in "turns"
  (up to 8, the main ones; only numbers shown in brackets; agent lines have none). Never write line numbers in the
  text itself.
  Lines marked "(recalled)" repeat old memories: they are not new events.
- Nothing about health, sex, drugs or alcohol, religion, politics, money, children, code or scripts.
- Short: a summary of 1 to 3 sentences (60 words in all), at most 5 topics, games, decisions and jokes, 6 moments.
Answer with JSON only."""

MERGE_RULES = """These are memories of consecutive parts of one call, for {agent}. Merge them into one memory with the
same fields and the same rules: keep the most memorable things, drop repeats, keep each item's "turns" (only numbers
that appear in the parts below). Answer with JSON only."""

FACT_RULES = """Below are lines from a voice call. "Owner" is the person who runs the agent; context lines are marked
"(context)" and cannot be cited. List facts the owner stated about themself, each as a sentence starting "The owner"
(for example "The owner plays bass"), with "quote": the owner's own words copied exactly from one cited Owner line
(first person, like "I play bass"), and "turns": that line's number. Only what the owner said about themself; nothing
others said about the owner; nothing about health, sex, religion, politics, money, children, other people, or where
the owner lives. If there is nothing, return {"facts": []}. Answer with JSON only."""


# ------------------------------------------------------------------ the log (never line text)
def _log(call: str, step: str, **counts) -> None:
    import experience_memory as EM
    line = time.strftime("%Y-%m-%d %H:%M:%S") + f" {call} {step} " + " ".join(f"{k}={v}" for k, v in counts.items())
    p = EM.root() / "_pass_log.txt"
    try:
        lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        lines = (lines + [line])[-LOG_KEEP:]
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass
    logger.info(line)


# ------------------------------------------------------------------ the model client
class ModelError(Exception):
    pass


class Model:
    """Streamed chat requests to the configured model (D3 "Which model"): never through RemoteSession's local fallback;
    closing the connection (the process being killed) makes the model server drop the request."""

    def __init__(self, mode: str, local_url: str = "", local_model: str = "", ctx: int = 4096,
                 url: Optional[str] = None, model: Optional[str] = None):
        self.mode = mode
        self.ctx = ctx
        self.cfg = None
        self.last: dict = {}                  # the last request's finish reason and token counts
        self.tokens = {"in": 0, "out": 0, "requests": 0}
        if mode == "remote":
            import remote_llm
            cfg = remote_llm.load_config()
            if not cfg:
                raise ModelError("no remote model configured")
            if url and cfg.get("url", "").rstrip("/") != url.rstrip("/"):
                # D3: the run talks to the model the server was using; a configuration that changed since is refused
                raise ModelError("the model configuration changed since Atlas started")
            self.cfg = dict(cfg, model=model or cfg["model"])
            self.ctx = int(self.cfg.get("context") or 8192)
        else:
            self.local_url, self.local_model = local_url.rstrip("/"), local_model

    @property
    def cloud(self) -> bool:
        return bool(self.cfg and self.cfg.get("cloud"))

    def chat(self, system: str, user: str, schema: dict, max_tokens: int = 900) -> str:
        import requests
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.last = {"finish": None, "in": 0, "out": 0}
        try:
            if self.mode == "remote":
                import remote_llm
                body = remote_llm.translate_payload(
                    {"messages": msgs, "stream": True, "think": False, "format": schema,
                     "options": {"temperature": 0.2, "num_predict": max_tokens}},
                    self.cfg["model"], vllm=bool(self.cfg.get("vllm", True)) and not self.cloud)
                if not self.cloud:      # token counts; some cloud providers reject the field (remote_llm skips it)
                    body["stream_options"] = {"include_usage": True}
                headers = {"Authorization": "Bearer " + self.cfg["key"]} if self.cfg.get("key") else {}
                r = requests.post(self.cfg["url"].rstrip("/") + "/v1/chat/completions", json=body, headers=headers,
                                  stream=True, timeout=(10, 120))
                r.raise_for_status()
                out = []
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    s = raw.decode("utf-8", "replace")
                    if not s.startswith("data:"):
                        continue
                    s = s[5:].strip()
                    if s == "[DONE]":
                        break
                    try:
                        d = json.loads(s)
                    except ValueError:
                        continue
                    for ch in d.get("choices") or []:
                        out.append((ch.get("delta") or {}).get("content") or "")
                        if ch.get("finish_reason"):
                            self.last["finish"] = ch["finish_reason"]
                    if d.get("usage"):
                        self.last["in"] = int(d["usage"].get("prompt_tokens") or 0)
                        self.last["out"] = int(d["usage"].get("completion_tokens") or 0)
                return self._count("".join(out))
            r = requests.post(self.local_url + "/api/chat", stream=True, timeout=(10, 120), json={
                "model": self.local_model, "messages": msgs, "stream": True, "format": schema, "think": False,
                "options": {"num_ctx": self.ctx, "temperature": 0.2, "num_predict": max_tokens}})
            r.raise_for_status()
            out = []
            for raw in r.iter_lines():
                if raw:
                    try:
                        d = json.loads(raw)
                    except ValueError:
                        continue
                    out.append((d.get("message") or {}).get("content") or "")
                    if d.get("done"):
                        self.last.update(finish=d.get("done_reason"), **{"in": int(d.get("prompt_eval_count") or 0),
                                                                         "out": int(d.get("eval_count") or 0)})
            return self._count("".join(out))
        except ModelError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ModelError(f"{type(e).__name__}: {e}") from e

    def _count(self, text: str) -> str:
        self.tokens["in"] += self.last.get("in", 0)
        self.tokens["out"] += self.last.get("out", 0)
        self.tokens["requests"] += 1
        return text


def _parse(text: str) -> Optional[dict]:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < i:
        return None
    try:
        d = json.loads(t[i:j + 1])
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


PARSE_FAILS: list = []        # (finish reason, output tokens, characters) of outputs that didn't parse, for the log


MAX_TOKENS = 1200


def _shape_ok(value, schema: dict) -> bool:
    """The output has the schema's shape (types only): servers that don't enforce the schema (cloud json_object) can
    return "turns": 3 or a summary object, which the validator must never see."""
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        return all(_shape_ok(value[k], sub) for k, sub in props.items() if k in value)
    if t == "array":
        return isinstance(value, list) and all(_shape_ok(v, schema.get("items", {})) for v in value)
    if t == "string":
        return isinstance(value, str)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return True


def ask(model, system: str, user: str, schema: dict) -> Optional[dict]:
    """One request, retried once when the output does not parse (D3 Failures); an output cut off by the token
    budget is retried with twice the budget. None if it still doesn't parse."""
    budget = out_budget(int(getattr(model, "ctx", 8192) or 8192))
    for _ in range(2):
        text = model.chat(system, user, schema, max_tokens=budget)
        d = _parse(text)
        if d is not None and _shape_ok(d, schema):
            return d
        last = getattr(model, "last", None) or {}
        PARSE_FAILS.append((last.get("finish"), last.get("out"), len(text or "")))
        if last.get("finish") == "length":
            budget *= 2
    return None


# ------------------------------------------------------------------ reading a pending file (D3 step 1)
class Call:
    def __init__(self, path: Path):
        import call_boundary as CB
        self.path = path
        self.id = path.stem
        self.recs = CB.read_records(path)
        self.lines = [r for r in self.recs if r.get("k") in ("turn", "reply")]
        self.by_n = {r["n"]: r for r in self.lines if "n" in r}
        self.names = [r["name"] for r in self.recs if r.get("k") == "name" and r.get("name")]
        self.namesake = any(r.get("k") == "namesake" for r in self.recs)
        self.ended = any(r.get("k") == "end" for r in self.recs)
        times = [float(r.get("t", 0)) for r in self.lines]
        self.first = min(times) if times else 0.0
        self.last = max(times) if times else 0.0
        self._owner_lines()

    def _owner_lines(self) -> None:
        """An owner line: an owner turn whose label (or the label it merged into) has no later refusal; a "Forget
        my voice" record ends ownership for every later line."""
        merged: dict = {}
        order = []
        for i, r in enumerate(self.recs):
            if r.get("k") == "merge":
                merged[r.get("old")] = r.get("new")
            order.append(r)

        def final(label):
            seen = set()
            while label in merged and label not in seen:
                seen.add(label)
                label = merged[label]
            return label
        refused_after: dict = {}          # label -> index of its last refusal
        forget_at = None
        for i, r in enumerate(self.recs):
            if r.get("k") == "refuse":
                refused_after[final(r.get("label"))] = i
            elif r.get("k") == "forget_voice" and forget_at is None:
                forget_at = i
        self.owner_ns = set()
        for i, r in enumerate(self.recs):
            if r.get("k") != "turn" or not r.get("owner"):
                continue
            lab = final(r.get("label"))
            if lab in refused_after and refused_after[lab] > i:
                continue
            if forget_at is not None and i > forget_at:
                continue
            self.owner_ns.add(r["n"])

    def human_turns(self, agent: Optional[str] = None) -> list:
        return [r for r in self.lines if r.get("k") == "turn" and (agent is None or r.get("agent") == agent)]

    def too_small(self) -> bool:
        import call_boundary as CB
        turns = self.human_turns()
        if len(turns) < CB.TOO_SMALL_TURNS:
            return True
        ts = [float(r.get("t", 0)) for r in turns]
        return max(ts) - min(ts) < CB.TOO_SMALL_S

    def qualifying(self, n: int) -> bool:
        import owner_voice as ov
        r = self.by_n.get(n) or {}
        dur, aud, sc = r.get("dur"), r.get("audible"), r.get("score")
        return bool(n in self.owner_ns and dur is not None and dur >= ov.QUALIFY_MIN_S and aud is False
                    and sc is not None and sc >= ov.TURN_SIM)


# ------------------------------------------------------------------ redaction (D3 step 2)
def redact(call: Call, owner_name: Optional[str], allow, tagger) -> dict:
    """{n: "Label: text"} for every line, redacted before any model sees it."""
    import memory_redact as MR
    import agent_registry
    persons: dict = {}
    out = {}
    names = list(call.names)
    for r in call.lines:
        n = r["n"]
        if r["k"] == "reply":
            agent = r.get("agent") or "agent"
            text = MR.scrub(r.get("text", ""), names, owner_name, allow=allow, tagger=tagger)
            who = f"{agent_registry.display_name(agent) if agent else 'agent'} (agent)"
            if r.get("recalled"):
                who += " (recalled)"
        else:
            owner = n in call.owner_ns
            agent = r.get("agent") or ""
            try:
                active = {agent.lower(), agent_registry.display_name(agent).lower(),
                          *agent_registry.nicknames(agent)} if agent else set()
            except Exception:  # noqa: BLE001
                active = {agent.lower()} if agent else set()
            text = MR.scrub(r.get("text", ""), names, owner_name, allow=allow, owner_namesake=call.namesake,
                            from_owner=owner, tagger=tagger, active_agent=active)
            if owner:
                who = "Owner"
            elif not r.get("label"):
                who = "Someone"           # an unplaced voice: never folded into one of the call's people
            else:
                lab = r.get("label")
                if lab not in persons:
                    persons[lab] = f"Person{len(persons) + 1}"
                who = persons[lab]
        out[n] = (who, text)
    return out


# ------------------------------------------------------------------ chunks (D3 step 3)
PROMPT_RULES_TOKENS = 450      # the longest instructions (EPISODE_RULES, MERGE_RULES) with room to spare


def chunk_budget(ctx: int) -> int:
    """Input tokens per chunk or merge group: the request, the rules and twice the output budget (the retry of a
    cut-off output) fit the model's context window, up to CHUNK_TOKENS and never under 800."""
    return min(CHUNK_TOKENS, max(800, ctx - PROMPT_RULES_TOKENS - 2 * out_budget(ctx)))


def out_budget(ctx: int) -> int:
    return MAX_TOKENS if ctx >= 8192 else 700


def merge_groups(recs: list, budget: int) -> list:
    """Consecutive chunk records packed into groups whose JSON fits `budget` tokens; every group but a last
    single one has at least two records, so each round at least nearly halves the count."""
    from conversation_log import est_tokens
    groups, cur = [], []
    for r in recs:
        if len(cur) >= 2 and est_tokens(json.dumps(cur + [r], ensure_ascii=False)) > budget:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    return groups


def chunks(ns: list, red: dict, budget: int) -> list:
    from conversation_log import est_tokens
    out, cur, used = [], [], 0
    for n in ns:
        who, text = red[n]
        c = est_tokens(f"[{n}] {who}: {text}") + 1
        if cur and used + c > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append(n)
        used += c
    if cur:
        out.append(cur)
    return out


def transcript(ns, red) -> str:
    """Numbered human lines; agent lines carry no number, so they cannot be cited (tuned on development calls)."""
    return "\n".join(f"[{'-' if red[n][0].endswith(')') else n}] {red[n][0]}: {red[n][1]}" for n in ns)


_TURN_REF = re.compile(r"\s*\((?:turns?|lines?)\s*[\d,\s\-and]+\)", re.I)


_PERSON_LABEL = re.compile(r"\b(?:Person|Speaker)\s?\d{1,4}\b|\bS\d{1,4}\b")


def _clean_text(t: str) -> str:
    """Turn references the model wrote into the text ("(turns 3, 4)") go; the citations are in "turns". A transcript
    placeholder the model copied ("Person3", "S2") becomes "someone": it is our own label, not a name (tuned on
    development calls 2026-11-04: such items were the largest group dropped)."""
    t = " ".join(_TURN_REF.sub("", t or "").split())

    def someone(m):
        start = m.start() == 0 or t[:m.start()].rstrip().endswith((".", "!", "?"))
        return "Someone" if start else "someone"
    return _PERSON_LABEL.sub(someone, t)


# ------------------------------------------------------------------ the validator over records (D3 step 5)
ITEM_LIMITS = {"summary": 3, "topics": 5, "games": 5, "decisions": 5, "jokes": 5, "moments": 6}   # S2
SUMMARY_WORDS = 60                                                                                 # S2


class Checker:
    def __init__(self, call: Call, red: dict, owner_name, allow, tagger, agents=()):
        import memory_redact as MR
        self.MR = MR
        self.call, self.red, self.owner_name, self.allow, self.tagger = call, red, owner_name, allow, tagger
        self.agents = tuple(agents)
        self.drops: dict = {}
        self.dropped: Optional[list] = None

    def _drop(self, reason: str) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + 1

    def text_ok(self, text: str, traits: bool = True, topics: bool = True, places: bool = False) -> bool:
        MR = self.MR
        reason = (MR.check_names(text, self.call.names, self.owner_name, agents=self.agents, allow=self.allow,
                                 tagger=self.tagger)
                  or MR.check_safety(text, topics=topics, places=places) or MR.check_directive(text, self.agents)
                  or (MR.check_traits(text, self.agents) if traits else None))
        if reason:
            self._drop(reason)
            if self.dropped is not None:      # tools/memory_eval.py only (private folder): what each rule drops
                self.dropped.append((reason, text))
            return False
        return True

    def cites_human(self, turns, allowed: Optional[set]) -> bool:
        ok = [n for n in (turns or []) if isinstance(n, int) and n in self.call.by_n
              and self.call.by_n[n].get("k") == "turn" and (allowed is None or n in allowed)]
        if not ok or len(ok) != len(turns or []):
            self._drop("grounding:cite")
            return False
        return True

    def shares_word(self, text: str, ns) -> bool:
        words = set()
        for n in ns:
            if self.call.by_n.get(n, {}).get("k") == "turn":
                words |= self.MR.content_words(self.red[n][1])
        if not (self.MR.content_words(text) & words):
            self._drop("grounding:words")
            return False
        return True

    def episode(self, rec: dict, ns: list, allowed_turns: Optional[dict] = None) -> Optional[dict]:
        """Validate one chunk or merged record; returns the kept record (still with turns) or None when no summary
        sentence survives. allowed_turns: per field, the turns a merged item may cite."""
        out = {}
        human = set(n for n in ns if self.call.by_n.get(n, {}).get("k") == "turn")
        for field in ("summary", "decisions", "jokes", "moments"):
            keep, words = [], 0
            for it in (rec.get(field) or [])[:ITEM_LIMITS[field]]:
                if not isinstance(it, dict):
                    self._drop("format")
                    continue
                raw = str(it.get("text") or "")
                text, turns = _clean_text(raw), it.get("turns") or []
                allowed = (allowed_turns or {}).get(field, human) if allowed_turns is not None else human
                if len(raw) >= TEXT_MAX - 2:
                    self._drop("length")                       # cut off at the schema's limit, mid-sentence
                    continue
                if field == "summary" and words + len(text.split()) > SUMMARY_WORDS:
                    self._drop("length")                       # S2: 60 words in all
                    continue
                if any(k["text"].lower() == text.lower() for k in keep):
                    continue                                   # the same item twice (a placeholder became "someone")
                if text and self.cites_human(turns, allowed) and self.text_ok(text):
                    keep.append({"text": text, "turns": [int(n) for n in turns]})
                    words += len(text.split())
            out[field] = keep
        for field in ("topics", "games"):
            keep = []
            for t in (rec.get(field) or [])[:ITEM_LIMITS[field]]:
                t = str(t or "").strip()
                if len(t) >= SHORT_MAX - 2 or re.search(r"\bsomeone\b", t, re.I):
                    self._drop("format")                       # cut off, or a name scrubbed out of it
                    continue
                if t and self.shares_word(t, ns) and self.text_ok(t):          # D3: traits apply to every item
                    keep.append(t)
            out[field] = keep
        if not out["summary"]:
            self._drop("no_summary")
            return None
        return out

    def facts(self, rec: dict, qualifying: set) -> list:
        MR = self.MR
        out = []
        for f in (rec.get("facts") or [])[:10]:
            if not isinstance(f, dict):
                continue
            fact, quote, turns = str(f.get("fact") or "").strip(), str(f.get("quote") or "").strip(), \
                f.get("turns") or []
            if not fact or not self.cites_human(turns, qualifying):
                continue
            if not any(quote and quote in self.red[n][1] for n in turns):
                self._drop("grounding:quote")
                continue
            reason = (MR.check_owner_subject(fact) or MR.check_quote(fact, quote))
            if reason:
                self._drop(reason)
                continue
            if not (self.text_ok(fact, traits=False, places=True) and self.text_ok(quote, traits=False, places=True)):
                continue
            out.append({"fact": fact, "quote": quote, "turns": [int(n) for n in turns]})
        return out


# ------------------------------------------------------------------ one call
def _stored(rec: dict) -> dict:
    return {k: [x["text"] if isinstance(x, dict) else x for x in rec.get(k, [])]
            for k in ("summary", "topics", "games", "decisions", "jokes", "moments")}


def process(path: Path, model, tagger, owner_name: Optional[str] = None, allow=(),
            stop: Callable[[], bool] = lambda: False, audit: Optional[dict] = None) -> dict:
    """One pending file -> episodes and extracted owner facts. Raises ModelError on a failed run."""
    import experience_memory as EM
    import memory_redact as MR
    call = Call(path)
    stats = {"episodes": 0, "facts": 0}
    if not call.ended:
        return {"skipped": "not ended"}
    if call.too_small():
        _delete_pending(path)
        return {"skipped": "too small"}
    tag = MR.CachedTagger(tagger) if tagger is not None else None
    red = redact(call, owner_name, allow, tag)
    if audit is not None:          # tools/memory_eval.py: what the model saw, and what each kept item cited
        audit["red"] = {n: list(v) for n, v in red.items()}
        audit["owner_ns"] = sorted(call.owner_ns)
    agents = sorted({r.get("agent") for r in call.human_turns() if r.get("agent")})
    checker = Checker(call, red, owner_name, allow, tag, agents)
    if audit is not None:
        checker.dropped = audit.setdefault("dropped", [])
    budget = chunk_budget(int(getattr(model, "ctx", 8192)))
    episodes = []
    for agent in agents:
        part = [r["n"] for r in call.lines if r.get("agent") == agent]
        if len([n for n in part if call.by_n[n]["k"] == "turn"]) < EPISODE_MIN_TURNS:
            continue
        import agent_registry
        name = agent_registry.display_name(agent) if agent else "the agent"
        recs = []
        for ns in chunks(part, red, budget):
            if stop():
                return {"stopped": True}
            d = ask(model, EPISODE_RULES.format(agent=name), transcript(ns, red), EPISODE_SCHEMA)
            if d is None:
                raise ModelError("chunk output did not parse")
            v = checker.episode(d, ns)
            if v is not None:
                recs.append(v)
        if not recs:
            continue
        while len(recs) > 1:          # merged in rounds of groups that fit the context (a long call has many chunks)
            if stop():
                return {"stopped": True}
            merged_round = []
            for group in merge_groups(recs, budget):
                if len(group) == 1:
                    merged_round.append(group[0])
                    continue
                cited = set(n for r in group for f in ("summary", "decisions", "jokes", "moments")
                            for it in r.get(f, []) for n in it["turns"])
                allowed = {f: cited for f in ("summary", "decisions", "jokes", "moments")}
                d = ask(model, MERGE_RULES.format(agent=name), json.dumps(group, ensure_ascii=False), EPISODE_SCHEMA)
                if d is None:
                    raise ModelError("merge output did not parse")
                merged = checker.episode(d, part, allowed_turns=allowed)
                if merged is not None:
                    merged_round.append(merged)
            recs = merged_round
        if not recs:
            continue
        final = recs[0]
        ident = next((r.get("ident") for r in call.lines if r.get("agent") == agent and r.get("ident")), None)
        ep = {"id": f"{call.id}.{agent}", "call": call.id, "agent": agent, "ident": ident,
              "date": time.strftime("%Y-%m-%d", time.localtime(call.first)), "start": call.first, "end": call.last,
              **_stored(final)}
        episodes.append((agent, ep, ident))
    facts = []
    qual = {n for n in call.owner_ns if call.qualifying(n)}
    if OWNER_FACTS_FROM_PASS and qual:
        human = [r["n"] for r in call.lines if r["k"] == "turn"]
        ctx_ns = set()
        for n in qual:
            i = human.index(n)
            ctx_ns |= set(human[max(0, i - CONTEXT_BEFORE):i + CONTEXT_AFTER + 1])
        ns = sorted(ctx_ns)
        for part_ns in chunks(ns, red, budget):          # a talkative owner's lines go in pieces that fit
            if stop():
                return {"stopped": True}
            text = "\n".join(f"[{n}] {red[n][0]}{'' if n in qual else ' (context)'}: {red[n][1]}" for n in part_ns)
            d = ask(model, FACT_RULES, text, FACT_SCHEMA)
            if d is None:
                raise ModelError("owner fact output did not parse")
            facts += checker.facts(d, qual)
    if audit is not None:
        audit["facts"] = facts
        audit["episodes"] = [ep for _, ep, _ in episodes]
        audit["drops"] = dict(checker.drops)
    if stop():
        return {"stopped": True}
    # step 6: write, then the pending file goes, all under the store lock
    with EM.store_lock(timeout=30):
        deleted = EM.deleted_ids()
        for agent, ep, ident in episodes:
            if EM.put_episode(agent, ep, ident=ident):
                stats["episodes"] += 1
        if not EM.wiped("_owner", call.first, call.last):
            for i, f in enumerate(facts):
                fid = f"f-{call.id}-{i}"
                if fid in deleted:
                    continue
                if EM.add_owner_fact(f["fact"], f["quote"], "extracted", True, call=call.id,
                                     src_t=float(call.by_n[f["turns"][0]].get("t", call.first)), tagger_checked=True,
                                     fid=fid):
                    stats["facts"] += 1
        _delete_pending(path)
    stats["drops"] = sum(checker.drops.values())
    _log(call.id, "done", **stats, **{f"drop_{k}": v for k, v in sorted(checker.drops.items())})
    return stats


def _delete_pending(path: Path) -> None:
    for p in (path, path.with_suffix(".fails")):
        try:
            p.unlink()
        except OSError:
            pass


def _fails(path: Path) -> int:
    try:
        return int(path.with_suffix(".fails").read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def _count_fail(path: Path) -> None:
    path.with_suffix(".fails").write_text(str(_fails(path) + 1), encoding="utf-8")


# ------------------------------------------------------------------ the tagger check of things stored in calls (D3 6)
def tagger_check(tagger, allow=()) -> dict:
    """Lore entries, lessons, predictions and said facts stored with tagger_checked false get the tagger check:
    lore and lessons that name a person go to "review"; predictions have the name replaced; said facts go to
    "to confirm"."""
    import experience_memory as EM
    import memory_redact as MR
    import lore
    import self_tools
    import teach
    out = {"lore": 0, "lessons": 0, "predictions": 0, "facts": 0}
    agents = MR.agent_words()
    allow_l = {a.lower() for a in allow}

    def named(text):
        for p in tagger.persons(text or ""):
            ps = {w.lower() for w in p.split()}
            if ps and not ps <= (agents | allow_l | {"owner", "the", "someone"}):
                return True
        return False
    base = EM.root()
    for d in sorted(base.iterdir()) if base.exists() else []:
        if not d.is_dir() or d.name.startswith("_"):
            continue
        for mod, key, fields in ((lore, "lore", ("title", "story")), (teach, "lessons", ("topic", "content"))):
            with mod._lock:
                items = mod._load(d.name)
                dirty = False
                for it in items:
                    if it.get("tagger_checked", True):
                        continue
                    if named(" ".join(str(it.get(f, "")) for f in fields)) and it.get("status") != "rejected":
                        for f in fields:               # the name goes even while the entry waits for review (D9 gate)
                            it[f] = MR.scrub(str(it.get(f, "")), [], tagger=tagger, allow=allow)
                        it["status"] = "review"
                        out[key] += 1
                    it["tagger_checked"] = True
                    dirty = True
                if dirty:
                    mod._save(d.name, items)
        if (d / "predictions.json").exists():
            with self_tools._lock:
                items = self_tools._load(d.name)
                dirty = False
                for it in items:
                    if it.get("tagger_checked", True):
                        continue
                    for k in ("claim", "check_when", "note"):
                        if it.get(k):
                            new = MR.scrub(it[k], [], tagger=tagger, allow=allow)
                            if new != it[k]:
                                it[k] = new
                                out["predictions"] += 1
                    it["tagger_checked"] = True
                    dirty = True
                if dirty:
                    self_tools._save(d.name, items)
    for f in EM.owner_facts():
        if f.get("source") == "said" and not f.get("tagger_checked", True):
            if named(f.get("fact", "")):
                EM.update_facts([f["id"]], tagger_checked=True, confirmed=False)
                out["facts"] += 1
            else:
                EM.update_facts([f["id"]], tagger_checked=True)
    return out


# ------------------------------------------------------------------ the run
def run(model, tagger, owner_name: Optional[str] = None, stop: Callable[[], bool] = lambda: False,
        only: Optional[str] = None) -> int:
    """Every pending file of an ended call; returns an exit code (0, 2)."""
    import call_boundary as CB
    current = None
    cur = CB.read_current()
    if cur and CB.in_progress():
        current = cur.get("call")
    CB.close_leftovers(current=current)
    CB.expire(current=current)
    allow = _allow_list()
    code = 0
    for path in sorted(CB.pending_dir().glob("*.jsonl")) if CB.pending_dir().exists() else []:
        if path.stem == current or (only and path.stem != only):
            continue
        if _fails(path) >= MAX_FAILS:
            continue
        if stop():
            break
        del PARSE_FAILS[:]
        try:
            process(path, model, tagger, owner_name, allow, stop)
        except Exception as e:  # noqa: BLE001 - one call's failure must not stop the others (or the run)
            _count_fail(path)
            extra = {}
            if "parse" in str(e) and PARSE_FAILS:     # counts only: why the output didn't parse
                finish, out_tokens, chars = PARSE_FAILS[-1]
                extra = {"finish": finish, "out_tokens": out_tokens, "chars": chars}
            _log(path.stem, "failed", fails=_fails(path), error="parse" if "parse" in str(e) else type(e).__name__,
                 **extra)
            code = 2
    try:
        n = tagger_check(tagger, allow)
        if any(n.values()):
            _log("-", "tagger_check", **n)
    except Exception as e:  # noqa: BLE001
        logger.warning("tagger check failed: %s", e)
    return code


def _allow_list() -> list:
    """The owner's allow-list (the viewer's game and product names) plus memory_redact.DEFAULT_ALLOW."""
    import memory_redact as MR
    return MR.allow_names()


def _owner_name() -> Optional[str]:
    """The owner's saved name (D3 step 2 turns it into "the owner"); read from the file even when the saved voice
    itself doesn't load, since the name is still the owner's."""
    try:
        import owner_voice
        s = owner_voice.Store()
        if s.load():
            return s.name
        d = json.loads(owner_voice.PATH.read_text(encoding="utf-8"))
        return owner_voice.valid_name(d.get("name") if isinstance(d, dict) else None)
    except Exception:  # noqa: BLE001
        return None


def _below_normal() -> None:
    try:
        import psutil
        p = psutil.Process()
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:  # noqa: BLE001
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("remote", "local"), required=True)
    ap.add_argument("--local-url", default="http://127.0.0.1:11434")
    ap.add_argument("--local-model", default="")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--call", default=None)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    _below_normal()
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:  # noqa: BLE001
        pass
    import memory_redact as MR
    _write_worker_pid()
    if not MR.tagger_files_present():
        _log("-", "waiting", reason="tagger_missing")
        return 3
    try:
        tagger = MR.GlinerTagger()
    except Exception as e:  # noqa: BLE001
        _log("-", "waiting", reason="tagger_failed", error=type(e).__name__)
        return 3
    try:
        model = Model(a.mode, a.local_url, a.local_model, a.ctx, url=a.url, model=a.model)
    except ModelError as e:
        _log("-", "failed", error=str(e)[:60])
        return 2
    import experience_memory as EM
    if model.cloud and not EM.settings()["cloud_allowed"]:
        _log("-", "waiting", reason="cloud_not_allowed")
        return 3
    return run(model, tagger, _owner_name(), only=a.call)



# ------------------------------------------------------------------ worker.pid (a run started at quit, S1)
def _worker_pid_path() -> Path:
    import call_boundary as CB
    return CB.pending_dir() / "worker.pid"


def _write_worker_pid() -> None:
    try:
        import psutil
        p = psutil.Process()
        path = _worker_pid_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": p.pid, "created": p.create_time()}), encoding="utf-8")
        import atexit
        atexit.register(lambda: path.unlink() if path.exists() and json.loads(
            path.read_text(encoding="utf-8")).get("pid") == p.pid else None)
    except Exception:  # noqa: BLE001
        pass


def kill_leftover_run() -> bool:
    """Server startup (D3): a run left from a quit is killed, after checking that its process id and creation time
    still match (Windows reuses process ids)."""
    path = _worker_pid_path()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    try:
        import psutil
        p = psutil.Process(int(d["pid"]))
        if abs(p.create_time() - float(d["created"])) < 1.0 and "memory_writer" in " ".join(p.cmdline()):
            p.kill()
            killed = True
        else:
            killed = False
    except Exception:  # noqa: BLE001
        killed = False
    try:
        path.unlink()
    except OSError:
        pass
    return killed


# ------------------------------------------------------------------ when a run may start (D3 "When it runs")
QUIET_S = 300.0                 # S6: 5 minutes without speech activity before an in-session run


def runnable_calls() -> list:
    """Pending files of ended calls that are not marked failed."""
    import call_boundary as CB
    d = CB.pending_dir()
    if not d.exists():
        return []
    cur = CB.read_current() or {}
    out = []
    for p in sorted(d.glob("*.jsonl")):
        if p.stem == cur.get("call") and CB.in_progress():
            continue
        if _fails(p) >= MAX_FAILS:
            continue
        out.append(p.stem)
    return out


def may_run(cloud: bool) -> Optional[str]:
    """None when a run may start; otherwise why not (paused, cloud not allowed, tagger missing, nothing to do)."""
    import experience_memory as EM
    import memory_redact as MR
    st = EM.settings()
    if st["paused"]:
        return "paused"
    if cloud and not st["cloud_allowed"]:
        return "cloud_not_allowed"
    if not MR.tagger_files_present():
        return "tagger_missing"
    if not runnable_calls():
        return "nothing"
    return None


BACKOFF_S = (60.0, 240.0, 960.0, 3600.0)   # waits after 1, 2, 3 and 4+ failed runs in a row
TAGGER_WAIT_S = 3600.0                     # after a run that found the tagger missing or failing to load


class Runner:
    """The server's side of the pass (D3): starts `memory_writer.py` between calls, inside a job object so it ends with
    the server, and kills it at once on speech activity (the audio path only sets a flag; a thread kills). A failed
    run (exit 2 or any other non-zero exit that wasn't a kill) makes the next start wait (BACKOFF_S), and a run that
    found no usable tagger (exit 3) waits an hour, so a broken setup never respawns the pass every few seconds."""

    def __init__(self, args_fn: Callable[[], list], cloud_fn: Callable[[], bool] = lambda: False,
                 on_done: Callable[[int], None] = lambda code: None, quiet_s: float = QUIET_S,
                 popen=None, clock: Callable[[], float] = time.time):
        import threading
        self.args_fn, self.cloud_fn, self.on_done = args_fn, cloud_fn, on_done
        self.quiet_s = quiet_s
        self.clock = clock
        self._popen = popen
        self.proc = None
        self._job = None
        self._kill = threading.Event()
        self._lock = threading.Lock()
        self.kills = 0
        self._kills_at_start = 0
        self.fails = 0
        self.not_before = 0.0
        self.last_reason: Optional[str] = None
        t = threading.Thread(target=self._watch, name="MemoryPassKiller", daemon=True)
        t.start()

    def on_activity(self, kind: str) -> None:
        if self.proc is not None:
            self._kill.set()

    def _watch(self) -> None:
        while True:
            self._kill.wait()
            self._kill.clear()
            with self._lock:
                p = self.proc
                if p is not None and p.poll() is None:
                    try:
                        p.kill()
                        self.kills += 1
                        logger.info("memory pass stopped: speech activity")
                    except Exception as e:  # noqa: BLE001
                        logger.warning("memory pass not killed: %s", e)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def tick(self, in_progress: bool, last_activity: float) -> Optional[str]:
        """From the server's lifespan task: reap a finished run, maybe start a new one. Returns what happened."""
        with self._lock:
            if self.proc is not None:
                code = self.proc.poll()
                if code is None:
                    return "running"
                self.proc, self._job = None, None
                now = self.clock()
                if self.kills > self._kills_at_start:
                    pass                                  # stopped for speech: not a failure
                elif code == 0:
                    self.fails, self.not_before = 0, 0.0
                elif code == 3:
                    self.not_before = now + TAGGER_WAIT_S
                else:
                    self.fails += 1
                    self.not_before = now + BACKOFF_S[min(self.fails, len(BACKOFF_S)) - 1]
                try:
                    self.on_done(code)
                except Exception as e:  # noqa: BLE001
                    logger.warning("memory pass end hook failed: %s", e)
                return f"exited {code}"
            if in_progress or self.clock() - last_activity < self.quiet_s:
                return None
            if self.clock() < self.not_before:
                self.last_reason = "backoff"
                return None
            why = may_run(self.cloud_fn())
            self.last_reason = why
            if why is not None:
                return None
            return self._start()

    def start_now(self) -> Optional[str]:
        """"Process now" in the viewer (refused while a call is in progress by the caller); the owner asking skips any
        back-off."""
        with self._lock:
            if self.proc is not None:
                return "running"
            self.not_before = 0.0
            why = may_run(self.cloud_fn())
            if why is not None:
                return why
            return self._start()

    def _start(self) -> str:
        import subprocess
        args = [sys.executable, str(Path(__file__).resolve())] + list(self.args_fn())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
        flags = 0
        if sys.platform == "win32":
            flags = 0x00004000 | 0x08000000           # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
        popen = self._popen or subprocess.Popen
        self._kills_at_start = self.kills
        self.proc = popen(args, cwd=str(Path(__file__).resolve().parent), env=env, creationflags=flags,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            from live_screen.sidecar_client import _job_object_for
            self._job = _job_object_for(self.proc)
        except Exception:  # noqa: BLE001
            self._job = None
        logger.info("memory pass started")
        return "started"


# last, so every function above exists when the pass runs as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
