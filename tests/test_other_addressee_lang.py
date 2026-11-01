"""detect_other_addressee across languages and interjections (live 10-31).

A German side chat was held as "addressed to Ja" x7 ("Nein", "Digga", "Mmh" too), and
"Aw, come on, man." as addressed to "Aw": interjections and words that are common in
the turn's own language are not names. Real vocatives in any language still count.
Plain unittest: the public CI runs `python -m unittest` without pytest."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import conversation_dynamics as CD  # noqa: E402

NAMES = ("pup",)
needs_wordfreq = unittest.skipIf(CD._zipf is None, "wordfreq not installed")


class NotNames(unittest.TestCase):
    def test_interjections_and_foreign_common_words_are_not_names(self):
        for line in ("Aw, come on, man.",
                     "Mmh, Cosmo ist da.",
                     "[S2] Digga, sie haben mich wirklich da komplett gefickt.",
                     "Ja, das ist sehr.",
                     "[S2] Ja, Bruder, manche sind halt, ähm, manche sind halt ein",
                     "[S4] Ja, das ist doch eine Schweigepflicht.",
                     "Ja, ich hatte auch schon mal eine Ausdurchsuchung.",
                     "Sí, claro, vamos ahora mismo.",
                     "Oui, je pense que c'est vrai."):
            with self.subTest(line=line):
                self.assertIsNone(CD.detect_other_addressee(line, NAMES))


@needs_wordfreq
class StillNames(unittest.TestCase):
    def test_real_vocatives_still_count(self):
        for line, name in (("Maya, did you bring your charger?", "Maya"),
                           ("Ben, are you coming tonight?", "Ben"),
                           ("Hey Perma, I think you were muted before. Are you back?", "Perma"),
                           ("Kim, kannst du mir mal helfen?", "Kim"),
                           ("Jake, hast du das gesehen?", "Jake")):
            with self.subTest(line=line):
                self.assertEqual(CD.detect_other_addressee(line, NAMES), name)

    def test_turn_language(self):
        self.assertEqual(CD._turn_language("Ja, das ist doch eine Schweigepflicht."), "de")
        self.assertIsNone(CD._turn_language("Maya, did you bring your charger?"))
        self.assertIsNone(CD._turn_language("Ja, ok."))          # too short to judge

    def test_room_handles_beat_the_language_check(self):
        # Discord people go by handles; one the room has used is trusted even when it is
        # a very common word in the turn's language ("Welt" as someone's tag).
        line = "Welt, kannst du mich hören?"
        self.assertIsNone(CD.detect_other_addressee(line, NAMES))
        CD.note_room_name("Welt")
        try:
            self.assertEqual(CD.detect_other_addressee(line, NAMES), "Welt")
        finally:
            CD._ROOM_NAMES.pop("welt", None)

    def test_unseen_handles_still_count(self):
        for line, name in (("Hey Perma, you back?", "Perma"),
                           ("Tangy, did you queue up yet?", "Tangy"),
                           ("Itsblazemf, are you muted?", "Itsblazemf")):
            with self.subTest(line=line):
                self.assertEqual(CD.detect_other_addressee(line, NAMES), name)


if __name__ == "__main__":
    unittest.main()
