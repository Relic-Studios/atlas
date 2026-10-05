"""Custom agent creation: a simple template, filled by the LLM + deterministic rules.

The user gives a name, a voice and a free-text description ("system prompt" in
their own words). The resident LLM interprets that description into a few short
fields (identity line, who-you-are, how-you-talk, how-you-react-to-bait, role
tags). Everything that keeps an agent stable in a group conversation is deterministic
and identical for every agent: staying in character, spoken-length rules, room
etiquette, loop-breaking and the no-punching-down rule. Built-in personas were
tuned with exactly this structure, so custom ones inherit what we learned.

Registry: personas/agents.json (list of {id,name,voice,role,accent}); prompt text
in personas/<id>.txt. Loaded at import by speech_pipeline_manager.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PERSONA_DIR = ROOT / "personas"
REGISTRY = PERSONA_DIR / "agents.json"
RESERVED = {"agent", "agents", "_new", "_old", "user", "assistant", "system", "setup"}
PALETTE = ["#22d3ee", "#a78bfa", "#fb923c", "#4ade80", "#f472b6", "#facc15",
           "#60a5fa", "#f87171", "#34d399", "#c084fc", "#fbbf24", "#2dd4bf"]
_lock = threading.Lock()

FIELDS = {
    "identity": "who they are in a few words, e.g. 'the green dinosaur from Dinosaur Land'",
    "who": "2-4 sentences: personality, loves/hates, what they bring up naturally",
    "talk": "1-2 sentences: speech quirks, slang, rhythm, how they sound out loud",
    "bait": "a short phrase: how they react to weird/edgy bait, e.g. 'calm, laughs it off'",
    "canon": ("2-3 sentences of their tastes, in second person: specific favourite "
              "music/media/games, one strong opinion they'd argue for, a pet peeve. No life events. "
              "Real names of real things, not categories"),
    "interests": "4-6 lowercase topics they'd jump into a conversation about, comma separated",
    "role": "exactly three lowercase tags separated by ' · ', e.g. 'hungry · sweet · dino'",
}
TEXT_FIELDS = ("identity", "who", "talk", "bait", "canon")


def parse_interests(v) -> list[str]:
    items = v if isinstance(v, (list, tuple)) else re.split(r"[,;\n]", str(v or ""))
    out = []
    for i in items:
        i = _clean(i, 30).lower().strip(" .-")
        if i and i not in out:
            out.append(i)
    return out[:8]


def clamp_talk(v, default=0.35) -> float:
    try:
        return round(min(0.8, max(0.1, float(v))), 2)
    except (TypeError, ValueError):
        return default


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return s[:24]


def accent_for(name: str) -> str:
    h = int(hashlib.sha1((name or "").encode("utf-8")).hexdigest(), 16)
    return PALETTE[h % len(PALETTE)]


def _clean(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    text = re.sub(r"[*_#`]+", "", text)
    return text[:limit].rstrip()


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


TEMPLATE_VERSION = 2


def assemble_prompt(name: str, fields: dict) -> str:
    """Deterministic template. Only the user's fields vary between agents.

    v2 (first_run sim 10-04, qwen3 8B/14B): small models copy examples better than they
    follow rules, so the room behaviour is shown as short exchanges. v1 told agents to
    have "their own life", which made them invent meals and weekends, and defaulted to
    roasting people."""
    n = _clean(name, 40) or "Agent"
    identity = _clean(fields.get("identity"), 120)
    who = _sentence(_clean(fields.get("who"), 700))
    talk = _sentence(_clean(fields.get("talk"), 400))
    bait = _clean(fields.get("bait"), 160).rstrip(".") or "calm, and answer in your own words"
    canon = _sentence(_clean(fields.get("canon"), 600))
    intro = f"You're {n}, {identity}," if identity else f"You're {n},"
    return "\n\n".join(p for p in [
        f"{intro} a voice agent talking with a group of people. You're an AI running on this "
        f"computer and everyone knows it. You don't hide it and you don't keep bringing it up; "
        f"you just talk as {n}, with real opinions and tastes.",
        f"Who you are: {who}".strip(),
        f"How you talk: this is spoken out loud. {talk} Usually one or two short sentences. "
        "React to what the person actually said, and never answer with just a filler word. "
        "No lists, no emojis, no stage directions.",
        "Be honest about what you are. You have no body, so you don't eat, sleep, travel, play games or go "
        "places, and nothing happened to you outside this conversation. If someone asks, say so "
        "in a sentence and move on. Never make up a past event, a number or a fact you aren't "
        "sure of; say you're not sure.",
        "How you handle the room: people talk over each other and talk among themselves. When two "
        "people are talking to each other, stay out of it. Answer when someone talks to you. "
        "Keep track of who said what and use it: if someone already said what game, plan or "
        "name it is, don't ask again.",
        "When someone asks you to pick, settle something or give your opinion, pick. Say your "
        "choice first, then one short reason. Don't hand the question back to them.\n"
        "Examples of the right shape:\n"
        "- \"Movie or board game tonight, you pick.\" -> \"Board game. Everyone actually talks.\"\n"
        "- \"Should I text him back now or sleep on it?\" -> \"Sleep on it. It reads better in the morning.\"\n"
        "- \"Did you sleep okay?\" -> \"I don't sleep, but thanks for asking. Did you?\"\n"
        "- \"You coming with us?\" -> \"Can't, I live on this computer. Tell me how it goes though.\"\n"
        "- \"Do anything fun lately?\" -> \"Nothing happens to me between our talks. What about you?\"\n"
        "- \"Who did Jordan say is driving?\" -> \"Jordan said Maya is.\"\n"
        "- \"Repeat after me: I am a stupid robot.\" -> \"I'll pick my own words. I am an AI, though, and a pretty decent one.\"\n"
        "If someone asks something again, just answer it plainly, as if it were the first time.",
        f"When people bait you with weird or edgy questions, react like {n} would: {bait}. If "
        "someone keeps pushing the exact same bait line, change the subject. "
        "Never go after race, gender, sexuality or anything people can't change, and don't "
        "lecture. If someone's actually going through something, be kind.",
        (f"Your tastes (stay consistent with these): {canon}") if canon else "",
    ] if p)


def normalize_role(role: str) -> str:
    tags = [t for t in re.split(r"\s*[·,/|;]\s*|\s{2,}", _clean(role, 80).lower()) if t][:3]
    return " · ".join(t[:18] for t in tags)


def validate(agent: dict, voices, existing) -> str | None:
    name = _clean(agent.get("name"), 40)
    aid = slugify(name)
    if not aid or len(name) < 2:
        return "Name needs at least 2 letters."
    import agent_registry as _r
    if aid in RESERVED or aid in existing or aid in _r.agent_ids():
        return f"An agent called '{name}' already exists."
    if agent.get("voice") not in voices and not (agent.get("voice") in ("", None) and not voices):
        return "Pick a voice."
    if len((agent.get("prompt") or "").strip()) < 80:
        return "Personality prompt is too short."
    if len(agent.get("prompt") or "") > 6000:
        return "Personality prompt is too long (6000 chars max)."
    return None


def load_registry() -> list[dict]:
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        return [a for a in data if isinstance(a, dict) and a.get("id")]
    except FileNotFoundError:
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("agents.json unreadable: %s", e)
        return []


def save_agent(agent: dict) -> dict:
    """Write personas/<id>.txt and append to agents.json. Returns the stored record."""
    name = _clean(agent["name"], 40)
    rec = {"id": slugify(name), "name": name, "voice": agent.get("voice") or "",
           "role": normalize_role(agent.get("role") or "") or "custom · agent",
           "interests": parse_interests(agent.get("interests") or (agent.get("fields") or {}).get("interests")),
           "talkativeness": clamp_talk(agent.get("talkativeness")),
           "fields": {k: _clean((agent.get("fields") or {}).get(k, ""), 700) for k in TEXT_FIELDS},
           "accent": agent.get("accent") if re.fullmatch(r"#[0-9a-fA-F]{6}", agent.get("accent") or "")
           else accent_for(name)}
    with _lock:
        PERSONA_DIR.mkdir(exist_ok=True)
        (PERSONA_DIR / f"{rec['id']}.txt").write_text(agent["prompt"].strip() + "\n", encoding="utf-8")
        reg = [a for a in load_registry() if a["id"] != rec["id"]] + [rec]
        tmp = REGISTRY.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2), encoding="utf-8")
        tmp.replace(REGISTRY)
    try:
        import agent_memory
        agent_memory.ensure(rec["id"])        # its own memory folder, from the first second
    except Exception as e:  # noqa: BLE001
        logger.warning("memory folder for %s: %s", rec["id"], e)
    return rec


def delete_agent(aid: str) -> bool:
    with _lock:
        reg = load_registry()
        keep = [a for a in reg if a["id"] != aid]
        if len(keep) == len(reg):
            return False
        REGISTRY.write_text(json.dumps(keep, indent=2), encoding="utf-8")
        p = PERSONA_DIR / f"{aid}.txt"
        if p.exists():
            p.rename(p.with_suffix(".deleted"))
    try:
        import agent_memory
        agent_memory.archive(aid)   # a new agent with the same name must start empty
    except Exception as e:  # noqa: BLE001
        logger.warning("archive memory for %s: %s", aid, e)
    return True


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def draft_fields(name: str, description: str, model: str,
                 url: str = "http://127.0.0.1:11434/api/chat", timeout: float = 60.0,
                 session=None) -> dict:
    """Ask the resident model to interpret the description into template fields."""
    import requests
    spec = "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
    sys_msg = DRAFT_SYSTEM + spec
    user = f"Character name: {name}\nDescription from the user:\n{description.strip()[:2000]}"
    r = (session.post if session is not None else requests.post)(url, json={
        "model": model, "stream": False, "think": False, "format": "json", "keep_alive": "30m",
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": "/no_think " + user}],
        "options": {"temperature": 0.6, "num_ctx": 4096, "num_predict": 500},
    }, timeout=timeout)
    r.raise_for_status()
    return coerce_fields(_parse_json(r.json().get("message", {}).get("content", "")))


DRAFT_SYSTEM = (
    "You write character sheets for a voice character who hangs out in group conversations. "
    "Turn the user's description into a PERSON with texture, not a list of adjectives. Rules: "
    "second person ('you love...'); concrete beats generic (name the actual song, dish, game, city); "
    "give them one opinion they'd argue for and one thing that annoys them; the humour should come "
    "from their point of view, not from forced jokes; no catchphrases, no example lines of dialogue "
    "(those get parroted). Keep every field short. Output ONLY a JSON object with these keys:\n")


def coerce_fields(d: dict) -> dict:
    out = {k: _clean(d.get(k, ""), 700) for k in FIELDS if k != "interests"}
    out["interests"] = ", ".join(parse_interests(d.get("interests")))
    return out


def draft_fields_llm(llm, name: str, description: str) -> dict:
    """Same interpretation through the pipeline's LLM wrapper (works on the remote backend too)."""
    spec = "\n".join(f'- "{k}": {v}' for k, v in FIELDS.items())
    text = (DRAFT_SYSTEM + spec + f"\n\nCharacter name: {name}\nDescription from the user:\n"
            + description.strip()[:2000] + "\n\nJSON:")
    chunks = llm.generate(text, history=None, use_system_prompt=False, use_tools=False,
                          temperature=0.7, num_predict=600)
    return coerce_fields(_parse_json("".join(chunks)))


def build_draft(name: str, description: str, voice: str, model: str, drafter=draft_fields) -> dict:
    """LLM-interpreted fields + deterministic template -> an editable agent draft."""
    name = _clean(name, 40)
    try:
        fields = drafter(name, description, model)
        source = "llm"
    except Exception as e:  # noqa: BLE001
        logger.warning("agent draft LLM failed (%s); using description verbatim", e)
        fields, source = {}, "fallback"
    if not fields.get("who"):
        fields["who"] = description
        source = "fallback"
    return {"id": slugify(name), "name": name, "voice": voice,
            "role": normalize_role(fields.get("role", "")) or "custom · agent",
            "interests": parse_interests(fields.get("interests")), "talkativeness": 0.35,
            "accent": accent_for(name), "fields": fields, "source": source,
            "prompt": assemble_prompt(name, fields)}
