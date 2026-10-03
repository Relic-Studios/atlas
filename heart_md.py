"""heart.md: a plain-markdown character sheet users can write, upload and export.

    # Gordon
    ## Identity
    angry celebrity chef
    ## Who
    You roast everyone's food but you're secretly soft on beginners...
    ## Talk
    Short, loud, British swearing...
    ## Bait
    disgusted and roasting them back
    ## Own stuff
    Your favourite dish is beef wellington; you think pineapple pizza is a crime...
    ## Interests
    cooking, restaurants, football
    ## Role
    loud · picky · chef
    ## Talkativeness
    0.5
    ## Voice
    my-imported-voice
    ## Prompt
    (optional: a full system prompt, used verbatim instead of the template)

Headings are case-insensitive and have aliases. A file with no headings at all is
treated as a raw system prompt. Unknown headings are folded into "who".
"""
from __future__ import annotations

import re

ALIASES = {
    "identity": "identity", "what": "identity", "tagline": "identity",
    "who": "who", "who they are": "who", "who you are": "who", "personality": "who", "about": "who", "character": "who",
    "talk": "talk", "speech": "talk", "how they talk": "talk", "voice style": "talk", "style": "talk",
    "bait": "bait", "when baited": "bait", "when baited they react": "bait",
    "their own stuff": "canon", "your own stuff": "canon",
    "own stuff": "canon", "canon": "canon", "favourites": "canon", "favorites": "canon",
    "opinions": "canon", "tastes": "canon",
    "interests": "interests", "topics": "interests", "loves": "interests",
    "role": "role", "tags": "role",
    "talkativeness": "talkativeness", "chattiness": "talkativeness",
    "voice": "voice",
    "prompt": "prompt", "system prompt": "prompt",
}
MAX_BYTES = 20_000


def parse(text: str) -> dict:
    """-> {name, fields{identity,who,talk,bait,canon}, interests[], role, talkativeness,
    voice, prompt}. Missing keys are simply absent/empty."""
    text = (text or "")[:MAX_BYTES].replace("\r\n", "\n").lstrip("﻿")
    out: dict = {"fields": {}}
    fm = re.match(r"^\s*---\n(.*?)\n---\s*\n", text, re.S)  # optional YAML-ish front matter
    meta = fm.group(1) if fm else ""
    if fm:
        text = text[fm.end():]
    m = re.search(r"^\s*#\s+(.+?)\s*$", text, re.M)
    if m and not m.group(0).lstrip().startswith("##"):
        out["name"] = re.sub(r"[*_`]", "", m.group(1)).strip()[:40]
    # "key: value" lines in front matter, or before the first ## section, are metadata.
    has_sections = bool(re.search(r"^\s*##\s", text, re.M))
    pre = text.split("\n##", 1)[0] if has_sections else ""
    meta_sections = []
    for line in (meta + "\n" + pre).splitlines():
        mm = re.match(r"^\s*([A-Za-z][A-Za-z ]{1,24}):\s*(.+?)\s*$", line)
        if not mm:
            continue
        k = mm.group(1).strip().lower()
        if k == "name":
            out.setdefault("name", mm.group(2).strip()[:40])
        elif ALIASES.get(k) in ("interests", "role", "talkativeness", "voice", "identity"):
            meta_sections.append((mm.group(1).strip(), mm.group(2)))
    parts = re.split(r"^\s*##\s+(.+?)\s*$", text, flags=re.M)
    if meta_sections:
        parts = parts[:1] + [x for kv in meta_sections for x in kv] + parts[1:]
    if len(parts) < 3:  # no sections: the whole thing is a raw prompt
        body = re.sub(r"^\s*#\s+.+$", "", text, count=1, flags=re.M).strip()
        if body:
            out["prompt"] = body
        return out
    extra = []
    for head, body in zip(parts[1::2], parts[2::2]):
        key = ALIASES.get(re.sub(r"[^a-z ]", "", head.lower()).strip())
        body = body.strip()
        if not body:
            continue
        if key is None:
            extra.append(f"{head.strip()}: {body}")
        elif key == "interests":
            items = re.split(r"[,\n;]|^\s*[-*]\s*", body, flags=re.M)
            out["interests"] = [i.strip(" -*.").lower() for i in items if i.strip(" -*.")][:8]
        elif key == "talkativeness":
            try:
                v = float(re.findall(r"[\d.]+", body)[0])
                out["talkativeness"] = v / 100 if v > 1 else v
            except (IndexError, ValueError):
                pass
        elif key in ("role", "voice", "prompt"):
            out[key] = body if key == "prompt" else body.splitlines()[0].strip()
        else:
            out["fields"][key] = " ".join(body.split())
    if extra:
        out["fields"]["who"] = " ".join(filter(None, [out["fields"].get("who", "")] + extra))
    return out


def dump(name: str, fields: dict, interests=(), role="", talkativeness=None, voice="",
         prompt=None) -> str:
    """Inverse of parse (round-trips the fields the creator edits)."""
    lines = [f"# {name}", ""]
    for head, key in (("Identity", "identity"), ("Who", "who"), ("Talk", "talk"),
                      ("Bait", "bait"), ("Own stuff", "canon")):
        if (fields or {}).get(key):
            lines += [f"## {head}", fields[key].strip(), ""]
    if interests:
        lines += ["## Interests", ", ".join(interests), ""]
    if role:
        lines += ["## Role", role, ""]
    if talkativeness is not None:
        lines += ["## Talkativeness", f"{float(talkativeness):.2f}", ""]
    if voice:
        lines += ["## Voice", voice, ""]
    if prompt:
        lines += ["## Prompt", prompt.strip(), ""]
    return "\n".join(lines).rstrip() + "\n"
