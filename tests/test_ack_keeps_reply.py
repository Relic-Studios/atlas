"""Live 10-06: someone said 'Yeah.' while Max was mid-answer (Graham Hancock).
Barge-in correctly chose 'talk over', but the acknowledgement HOLD then aborted the
running generation -> he was cut off mid-sentence. An ack/veto may retire an
UNSPOKEN draft, never a reply already committed to audio."""
import threading
import unittest
from types import SimpleNamespace

from tests.test_fast_vad_guards import make_callbacks


def gen(playing: bool):
    ev = threading.Event()
    if playing:
        ev.set()
    return SimpleNamespace(tts_quick_allowed_event=ev, quick_answer_provided=playing,
                           abortion_started=False, text="")


class AckOverReply(unittest.TestCase):
    def _run(self, playing):
        cb, mgr = make_callbacks()
        aborts = []
        mgr.abort_generation = lambda **k: aborts.append(k.get("reason"))
        mgr.running_generation = gen(playing)
        cb._live_speaker = lambda: "S3"
        cb._text_echo = lambda t: False
        cb.on_potential_sentence("Yeah.")
        return aborts

    def test_ack_does_not_cut_playing_reply(self):
        self.assertEqual(self._run(playing=True), [])

    def test_ack_still_retires_unspoken_draft(self):
        self.assertTrue(any("pre_llm_veto" in (r or "") for r in self._run(playing=False)))

    def test_bridge_speaking_counts_as_in_flight(self):
        cb, mgr = make_callbacks()
        cb.app.state.CallBridge = SimpleNamespace(is_speaking=lambda: True)
        self.assertTrue(cb._reply_in_flight(gen(False)))


if __name__ == "__main__":
    unittest.main()
