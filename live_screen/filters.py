"""What the screen log may keep (docs/designs/live_screen.md, section 6): personal details are scrubbed from every
line, nothing is logged while a mail client or password manager is in front, and all screen text is cleaned of
instruction-like sentences and wrapped as data before a prompt sees it."""
from __future__ import annotations

import re

SKIP_PROCESSES = frozenset({"outlook.exe", "olk.exe", "thunderbird.exe", "mailspring.exe", "1password.exe",
                            "bitwarden.exe", "keepass.exe", "keepassxc.exe", "dashlane.exe", "lastpass.exe"})
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<![\w+])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?!\w)")
_LONG_DIGITS = re.compile(r"(?<![\d,])(?:\d[ -]?){8,}\d(?![\d,])")       # card/account numbers, codes; not 1,234,567
REMOVED = "(removed)"


def skip(process: str) -> bool:
    """Never log what's on screen while this program is in front."""
    return (process or "").lower() in SKIP_PROCESSES


def scrub(text: str) -> str:
    text = _EMAIL.sub(REMOVED, text or "")
    text = _PHONE.sub(REMOVED, text)
    return _LONG_DIGITS.sub(REMOVED, text)


def clean(text: str, limit: int = 200) -> str:
    """Scrubbed and with instruction-like sentences replaced (untrusted.sanitize)."""
    text = scrub(text)
    try:
        import untrusted
        text, _ = untrusted.sanitize(text, limit=limit)
    except Exception:  # noqa: BLE001
        text = text[:limit]
    return text.strip()


def wrap(text: str) -> str:
    import untrusted
    return untrusted.wrap(text, source="screen log", origin="text read from a screen")
