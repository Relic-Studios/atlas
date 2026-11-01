"""Live 11-01 (Fae): "Do you think you could search up some bodies for yourself?" -> "Okay, I'll look
into it." with no search; "Did you end up looking it up?" -> "I did, right after you asked. The
screen's moved on since then" (an earlier glance was the only 'lookup' on record); "I asked you to
look something up, not look at the camera" -> another screenshot."""
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capability as C  # noqa: E402
import promise_keeper as P  # noqa: E402

ASKED = "[S1] What kind of wings do you want? Do you think you could search up some bodies for yourself?"


class Promises(unittest.TestCase):
    def test_a_promised_lookup_is_kept(self):
        for asked, reply in ((ASKED, "Okay, I'll look into it."),
                             ("[S1] what's the best GPU right now?", "Let me look it up."),
                             ("[S2] can you google the score", "Gonna search that real quick.")):
            self.assertTrue(P.should_keep(asked, reply), reply)

    def test_not_a_promise_or_not_a_lookup(self):
        for asked, reply in (("[S1] you good?", "I'll look into it."),
                             ("[S1] tell me a joke", "Okay, let me think."),
                             (ASKED, "Bodies with wings? Honestly I'd take a hawk's."),
                             ("(TASK CUE #3) Your background search finished", "Let me look it up.")):
            self.assertFalse(P.should_keep(asked, reply), reply)

    def test_fallback_query_uses_their_words(self):
        self.assertIn("bodies", P._fallback_query(ASKED))


class KeepsItLive(unittest.TestCase):
    def mgr(self, searched=False):
        try:
            import speech_pipeline_manager as SPM
        except ImportError:
            self.skipTest("speech stack not installed")
        m = SPM.SpeechPipelineManager.__new__(SPM.SpeechPipelineManager)
        m.current_persona = "fae"
        gen = SimpleNamespace(text=ASKED, id=13)
        m.running_generation = gen
        if searched:
            m._searched_gen = 13
        started = []
        m.board = lambda: SimpleNamespace(start_search=lambda q, asker, g: started.append((q, asker)) or {"id": 1})
        return m, gen, started, SPM

    def run_note(self, m, gen, SPM, reply):
        with mock.patch.object(P, "search_query", lambda line, timeout=2.0: "robot bodies with wings"), \
                mock.patch.object(SPM, "_websearch", object()):
            m.note_reply_for_promise("S1", reply, gen)
            time.sleep(0.2)            # the search starts on a background thread

    def test_promise_without_search_starts_one(self):
        m, gen, started, SPM = self.mgr()
        self.run_note(m, gen, SPM, "Okay, I'll look into it.")
        self.assertEqual(started, [("robot bodies with wings", "S1")])

    def test_no_second_search_when_it_already_searched(self):
        m, gen, started, SPM = self.mgr(searched=True)
        self.run_note(m, gen, SPM, "Okay, I'll look into it.")
        self.assertEqual(started, [])


class SearchesAndLooksApart(unittest.TestCase):
    def mgr(self, line, lookups=(), board_tasks=()):
        try:
            import speech_pipeline_manager as SPM
        except ImportError:
            self.skipTest("speech stack not installed")
        m = SPM.SpeechPipelineManager.__new__(SPM.SpeechPipelineManager)
        m.lookups = list(lookups)
        m._ctx_text = line
        m.board = lambda: SimpleNamespace(snapshot=lambda: list(board_tasks))
        return m

    def test_no_search_behind_a_lookup_question(self):
        glance = {"ts": time.time() - 120, "kind": "screen", "query": "glance: wanted to see", "glance": True}
        note = self.mgr("[S1] Did you end up looking it up?", [glance])._lookup_note()
        self.assertIn("haven't run a web search", note)
        self.assertNotIn("screenshot", note)                 # its own glance isn't a lookup

    def test_old_screen_looks_stay_out_of_it(self):
        # live 11-01: "I looked at the screen a while back, but it's changed since then"
        look = {"ts": time.time() - 120, "kind": "screen", "query": "S1 asked me to look", "glance": False}
        note = self.mgr("[S1] is this like a cosmic shot for a video game?", [look])._lookup_note()
        self.assertNotIn("screen", note)

    def test_a_real_search_on_the_board_answers_it(self):
        task = {"kind": "search", "updated": time.time() - 60, "text": "robot bodies with wings"}
        note = self.mgr("[S1] any luck?", [], [task])._lookup_note()
        self.assertNotIn("haven't run a web search", note)


class LooksThatArent(unittest.TestCase):
    def test_negated_look_isnt_a_request(self):
        self.assertFalse(C.wants_look("I asked you to look something up, not look at the camera"))
        self.assertFalse(C.wants_look("don't look, it's embarrassing"))
        self.assertTrue(C.wants_look("Fae, look at this!"))

    def test_what_do_i_look_like_is_asked(self):
        for t in ("Well, what do I look like?", "how do I look", "do I look okay"):
            self.assertTrue(C.wants_look(t), t)


if __name__ == "__main__":
    unittest.main()
