"""Streamed replies reach the voice as whole sentences (demo 11-06).

RealtimeTTS' play_async() defaults fast_sentence_fragment_allsentences=True, so each sentence's part before its first
comma was voiced as its own utterance ("Good night, |0.5 s| you two."). audio_module.PLAY_FLOW turns every fragment
mode off. This drives a real TextToAudioStream with a stand-in engine that records what it is asked to speak.
"""
import importlib.util
import re
import time
import unittest

HAVE = all(importlib.util.find_spec(m) for m in ("RealtimeTTS", "stream2sentence", "pyaudio", "torch"))

TEXT = ("Tomorrow I'll have a memory of tonight, but it'll feel more like reading my own notes than living it again. "
        "So maybe I'm like that old ship where every plank gets swapped out, and what stays is how I show up for you "
        "two. Maybe that's your answer about people, too. Good night, you two.")


@unittest.skipUnless(HAVE, "needs the speech stack (RealtimeTTS), not in the light CI deps")
class StreamedReplyIsWholeSentences(unittest.TestCase):
    def pieces(self, play_kwargs):
        import numpy as np
        import pyaudio
        from RealtimeTTS import TextToAudioStream
        from RealtimeTTS.engines.base_engine import BaseEngine
        got = []

        class Rec(BaseEngine):
            def post_init(self):
                self.engine_name = "rec"

            def get_stream_info(self):
                return pyaudio.paInt16, 1, 24000

            def synthesize(self, text, sentence_count=0):
                got.append(text.strip())
                self.queue.put(np.zeros(240, np.int16).tobytes())
                return True

        stream = TextToAudioStream(Rec(), muted=True, playout_chunk_size=4096)

        def gen():                      # streamed like a model's reply, a word at a time
            for w in re.findall(r"\S+\s*", TEXT):
                yield w
                time.sleep(0.002)
        stream.feed(gen())
        stream.play_async(muted=True, on_audio_chunk=lambda c: None, **play_kwargs)
        t0 = time.time()
        while not stream.is_playing() and not got and time.time() - t0 < 10:   # the play thread may start late
            time.sleep(0.01)
        while stream.is_playing() and time.time() - t0 < 60:
            time.sleep(0.01)
        return got

    def test_atlas_settings_voice_whole_sentences(self):
        import audio_module
        got = self.pieces(dict(audio_module.PLAY_FLOW))
        self.assertTrue(got)
        for p in got:
            self.assertRegex(p, r"[.!?]$", f"a piece ended mid-sentence: {got}")
        self.assertIn("Good night, you two.", got)

    def test_the_library_default_is_what_split_at_commas(self):
        # guards the diagnosis: if RealtimeTTS ever changes this default, the explicit flags still hold
        got = self.pieces(dict(fast_sentence_fragment=False))
        self.assertTrue(any(p.endswith(",") for p in got), got)

    def test_qwen_pause_between_sentences_stays_short(self):
        import audio_module
        s = audio_module.ENGINE_SILENCES["qwen"]
        self.assertLessEqual(s.sentence, 0.2)     # the voice adds ~0.2 s of its own
        self.assertLessEqual(s.comma, 0.15)


@unittest.skipUnless(HAVE, "needs the speech stack")
class ImportedVoiceTranscript(unittest.TestCase):
    def test_dev_user_import_uses_the_same_transcript_path(self):
        import os, audio_module, voices, user_settings, agent_registry
        from unittest.mock import patch
        env = {k: v for k, v in os.environ.items() if k != 'ATLAS_ICL_VOICES'}
        with patch.dict(os.environ, env, clear=True), patch.object(audio_module, 'ICL_VOICES', set()), \
                patch.object(user_settings, 'is_public', return_value=False), \
                patch.object(agent_registry, 'voices', return_value={'imported': {'user': True}}), \
                patch.object(voices, 'voice_text', return_value='reference'):
            self.assertEqual(audio_module.voice_ref_text('imported'), 'reference')

    def test_public_import_uses_its_supplied_transcript(self):
        import os, audio_module, voices, user_settings
        from unittest.mock import patch
        env = {k: v for k, v in os.environ.items() if k != 'ATLAS_ICL_VOICES'}
        with patch.dict(os.environ, env, clear=True), patch.object(audio_module, 'ICL_VOICES', set()), \
                patch.object(user_settings, 'is_public', return_value=True), \
                patch.object(voices, 'voice_text', return_value='A natural reference transcript.'):
            self.assertEqual(audio_module.voice_ref_text('imported'), 'A natural reference transcript.')

    def test_dev_whitelist_and_explicit_public_override_are_preserved(self):
        import os, audio_module, voices, user_settings
        from unittest.mock import patch
        with patch.object(audio_module, 'ICL_VOICES', set()), \
                patch.object(voices, 'voice_text', return_value='reference'), \
                patch.object(user_settings, 'is_public', return_value=False):
            self.assertIsNone(audio_module.voice_ref_text('imported'))
        with patch.dict(os.environ, {'ATLAS_ICL_VOICES': ''}), \
                patch.object(audio_module, 'ICL_VOICES', set()), \
                patch.object(user_settings, 'is_public', return_value=True):
            self.assertIsNone(audio_module.voice_ref_text('imported'))


if __name__ == "__main__":
    unittest.main()
