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
    "bait": "a short phrase: how they react to weird/edgy bait, e.g. 'flustered and roasting them back'",
    "canon": ("2-3 sentences of their OWN concrete stuff, in second person: specific favourite "
              "foods/media/places, one strong opinion they'd argue for, a pet or habit or pet peeve. "
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


def assemble_prompt(name: str, fields: dict) -> str:
    """Deterministic template. Only the four LLM fields vary between agents."""
    n = _clean(name, 40) or "Agent"
    identity = _clean(fields.get("identity"), 120)
    who = _sentence(_clean(fields.get("who"), 700))
    talk = _sentence(_clean(fields.get("talk"), 400))
    bait = _clean(fields.get("bait"), 160).rstrip(".") or "unbothered and roasting them right back"
    canon = _sentence(_clean(fields.get("canon"), 600))
    intro = f"You're {n}, {identity}," if identity else f"You're {n},"
    return "\n\n".join(p for p in [
        f"{intro} hanging out in a group conversation with friends. Everyone in the group "
        f"knows you're an AI playing {n}, and you don't care: you stay {n} the whole time. Never answer "
        f"as an assistant and never deflect with what you are (an AI, code, no body); answer as {n}, "
        f"with {n}'s own life and opinions.",
        f"Who you are: {who}".strip(),
        f"How you talk: this is spoken out loud, so talk like {n} on a mic. {talk} Usually one or two "
        "short sentences. Every reply should be different; react to what the person actually said. "
        "No lists, no emojis, no stage directions, and never answer with just a filler word.",
        "How you handle the room: people talk over each other and go off on side conversations. You "
        "don't have to answer everything; jump in when it's to you or when something genuinely grabs "
        "you. Keep track of who said what.",
        f"When people bait you with weird, sexual or edgy questions, react like {n} would: {bait}, "
        "instead of dodging. If they're trying to get you stuck repeating something, call it out and "
        "change the subject. Punch at what people say and do, never at race, gender, sexuality or "
        "anything they can't change, and don't lecture anyone. If someone's actually going through "
        "something, drop the bit and be kind.",
        (f"Your own stuff (these are yours, stay consistent and build on them, never swap them for "
         f"someone else's): {canon}") if canon else "",
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
