"""Drop saved voiceprint "names" that are ordinary words (Close, All, Muted, I'll, ...).

Name learning took one-word answers to "who's this?" as names before 10-31. Handles that
merely look like words (Batman, Queen, Link) are kept. Run only while ATLAS is stopped:
the live NameBook writes its in-memory copy back. A backup is written next to the file.
  python tools/clean_voiceprints.py [--dry-run]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import people as P  # noqa: E402

EXTRA = {"bigger", "nasty"}   # adjectives seen as one-word "answers"


def junk(name: str) -> bool:
    w = name.lower()
    try:
        from wordfreq import zipf_frequency as z
        very_common = z(w, "en") >= 5.0 and w not in P._COMMON_GIVEN
    except Exception:  # noqa: BLE001
        very_common = False
    return (P._clean_name(name) is None or "'" in w or w.endswith("ly") or very_common or w in EXTRA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", default=str(ROOT / "memory_db" / "voiceprints.json"))
    a = ap.parse_args()
    path = Path(a.path)
    prints = json.loads(path.read_text(encoding="utf-8"))
    drop = sorted(n for n in prints if junk(n))
    print(f"{len(prints)} saved, dropping {len(drop)}: {drop}")
    if a.dry_run or not drop:
        return
    backup = path.with_name(f"{path.stem}.bak-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    shutil.copy2(path, backup)
    for n in drop:
        prints.pop(n)
    path.write_text(json.dumps(prints), encoding="utf-8")
    print(f"kept {len(prints)}; backup: {backup.name}")


if __name__ == "__main__":
    main()
