"""detect_other_addressee across languages and interjections (live 10-31).

A German side chat was held as "addressed to Ja" x7 ("Nein", "Digga", "Mmh" too), and
"Aw, come on, man." as addressed to "Aw": interjections and words that are common in
the turn's own language are not names. Real vocatives in any language still count."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conversation_dynamics as CD  # noqa: E402

NAMES = ("pup",)
needs_wordfreq = pytest.mark.skipif(CD._zipf is None, reason="wordfreq not installed")


@pytest.mark.parametrize("line", [
    "Aw, come on, man.",
    "Mmh, Cosmo ist da.",
    "[S2] Digga, sie haben mich wirklich da komplett gefickt.",
    "Ja, das ist sehr.",
    "[S2] Ja, Bruder, manche sind halt, ähm, manche sind halt ein",
    "[S4] Ja, das ist doch eine Schweigepflicht.",
    "Ja, ich hatte auch schon mal eine Ausdurchsuchung.",
    "Sí, claro, vamos ahora mismo.",
    "Oui, je pense que c'est vrai.",
])
def test_interjections_and_foreign_common_words_are_not_names(line):
    assert CD.detect_other_addressee(line, NAMES) is None


@needs_wordfreq
@pytest.mark.parametrize("line,name", [
    ("Maya, did you bring your charger?", "Maya"),
    ("Ben, are you coming tonight?", "Ben"),
    ("Hey Perma, I think you were muted before. Are you back?", "Perma"),
    ("Kim, kannst du mir mal helfen?", "Kim"),
    ("Jake, hast du das gesehen?", "Jake"),
])
def test_real_vocatives_still_count(line, name):
    assert CD.detect_other_addressee(line, NAMES) == name


@needs_wordfreq
def test_turn_language():
    assert CD._turn_language("Ja, das ist doch eine Schweigepflicht.") == "de"
    assert CD._turn_language("Maya, did you bring your charger?") is None
    assert CD._turn_language("Ja, ok.") is None          # too short to judge


@needs_wordfreq
def test_room_handles_beat_the_language_check():
    # Discord people go by handles; one the room has used is trusted even when it is
    # a very common word in the turn's language ("Welt" as someone's tag).
    line = "Welt, kannst du mich hören?"
    assert CD.detect_other_addressee(line, NAMES) is None
    CD.note_room_name("Welt")
    try:
        assert CD.detect_other_addressee(line, NAMES) == "Welt"
    finally:
        CD._ROOM_NAMES.pop("welt", None)


@needs_wordfreq
@pytest.mark.parametrize("line,name", [
    ("Hey Perma, you back?", "Perma"),
    ("Tangy, did you queue up yet?", "Tangy"),
    ("Itsblazemf, are you muted?", "Itsblazemf"),
])
def test_unseen_handles_still_count(line, name):
    assert CD.detect_other_addressee(line, NAMES) == name
