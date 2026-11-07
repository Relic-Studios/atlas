"""Call recorder: JSONL events, best-effort, replayable by sim_long_call."""
import json, os, sys, tempfile, unittest
import threading
import time
import asyncio
from unittest.mock import patch
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import call_recorder as CR


class RecorderTests(unittest.TestCase):
    def test_roundtrip_and_replay(self):
        with tempfile.TemporaryDirectory() as d:
            r = CR.CallRecorder(Path(d)); r.enabled = True
            r.event("persona", persona="ivy")
            r.event("user", spk="S2", text="hey ivy what's up")
            r.event("decision", action="SPEAK", target="S2", ms=300.0)
            r.event("agent", persona="ivy", text="Not much, painting.")
            r.event("user", spk="S3", text="  ")
            r.close()
            ev = CR.load(r.path)
            self.assertEqual(ev, [("user", "S2", "hey ivy what's up"), ("agent", "ivy", "Not much, painting.")])
            # The public suite verifies the production replay loader above.
            # Its dev-only simulator is also checked when that harness is present.
            if (Path(__file__).resolve().parent / 'sim_long_call.py').is_file():
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                import sim_long_call
                self.assertEqual(sim_long_call.parse(str(r.path)), ev)

    def test_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            r = CR.CallRecorder(Path(d)); r.enabled = False
            r.event("user", spk="S1", text="x")
            self.assertEqual(list(Path(d).rglob("*")), [])

    def test_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = CR.CallRecorder(Path(tmp)); r.enabled = True
            # A nonexistent mapped drive can hang Windows filesystem lookup.
            # Exercise the intended unwritable-file failure without network I/O.
            with patch.object(r, '_append_event', side_effect=PermissionError('unwritable')):
                r.event("user", spk="S1", text="x")
                r.audio(np.zeros(160, dtype=np.float32))
                r.close()
            self.assertEqual(r.write_errors, 1)

    def test_audio_optional(self):
        with tempfile.TemporaryDirectory() as d:
            r = CR.CallRecorder(Path(d)); r.enabled = True; r.audio_enabled = True
            r.audio(np.zeros(1600, dtype=np.float32)); r.close()
            wavs = list(Path(d).rglob("*.wav"))
            self.assertEqual(len(wavs), 1)


class RecorderLatencyTests(unittest.TestCase):
    def recorder(self, root):
        rec = CR.CallRecorder(root)
        rec.enabled = True
        return rec

    def test_close_attempts_both_handles_and_propagates_first_failure(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            first_error = OSError('text flush failed during close')
            text = rec._fh = Mock()
            wav = rec._wav = Mock()
            text.close.side_effect = first_error
            wav.close.side_effect = OSError('audio close also failed')
            with self.assertRaises(OSError) as raised:
                rec.close()
            self.assertIs(raised.exception, first_error)
            text.close.assert_called_once()
            wav.close.assert_called_once()
            self.assertIsNone(rec._fh)
            self.assertIsNone(rec._wav)
            rec.close()  # A failed barrier must leave the writer reusable.

    def test_stalled_write_does_not_block_submissions_or_async_work_and_close_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            started, release = threading.Event(), threading.Event()
            original = rec._append_event
            def slow(line):
                started.set()
                release.wait(5)
                original(line)
            try:
                with patch.object(rec, '_append_event', slow):
                    rec.event('user', text='first')
                    self.assertTrue(started.wait(2))
                    async def scenario():
                        before = time.monotonic()
                        rec.event('user', text='second')
                        await asyncio.sleep(.01)
                        self.assertLess(time.monotonic() - before, .5)
                    asyncio.run(scenario())
                    with self.assertRaises(TimeoutError):
                        rec.close(timeout=.02)
            finally:
                release.set()
                rec.close()
            rows = [json.loads(x) for x in rec.path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['text'] for r in rows], ['first', 'second'])

    def test_submission_captures_timestamp_fields_and_span_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            rec.audio_enabled = True
            started, release = threading.Event(), threading.Event()
            original = rec._append_event
            def slow(line):
                started.set(); release.wait(5); original(line)
            fields = ['original']
            samples = np.ones(160, dtype=np.float32) * .2
            try:
                with patch.object(rec, '_append_event', slow):
                    with patch.object(CR.time, 'time', return_value=123.456), \
                            patch.object(CR, '_off_record_open', return_value=True):
                        rec.event('user', text='private', extra=fields)
                        rec.audio(samples)
                    self.assertTrue(started.wait(2))
                    fields.append('changed')
                    samples[:] = .9
                    with patch.object(CR, '_off_record_open', return_value=False):
                        rec.audio(samples)
                    samples[:] = 0
            finally:
                release.set(); rec.close()
            row = json.loads(rec.path.read_text(encoding='utf-8').splitlines()[0])
            self.assertEqual((row['t'], row['text'], row['extra']), (123.456, CR.MASKED, ['original']))
            import soundfile as sf
            audio = sf.read(next(Path(tmp).rglob('*.wav')))[0]
            self.assertEqual(abs(audio[:160]).max(), 0)
            self.assertGreater(audio[160:].min(), .89)

    def test_overload_drops_data_but_accepts_mask_and_close_barriers(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            started, release, masked = threading.Event(), threading.Event(), []
            original = rec._append_event
            def slow(line):
                started.set(); release.wait(5); original(line)
            mask_thread = None
            try:
                with patch.object(CR, 'MAX_PENDING_DATA', 1), patch.object(rec, '_append_event', slow):
                    rec.event('user', text='sensitive')
                    self.assertTrue(started.wait(2))
                    rec.event('user', text='over capacity')
                    self.assertEqual(rec.dropped_events, 1)
                    mask_thread = threading.Thread(target=lambda: masked.append(rec.mask_since(0, time.time() + 1)))
                    mask_thread.start()
                    time.sleep(.03)
                    self.assertTrue(mask_thread.is_alive())
                    release.set(); mask_thread.join(2)
                    self.assertFalse(mask_thread.is_alive())
            finally:
                release.set()
                if mask_thread: mask_thread.join(2)
                rec.close()
            self.assertEqual(masked, [1])
            rows = [json.loads(x) for x in rec.path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['text'] for r in rows], [CR.MASKED])

    def test_late_submission_is_masked_using_original_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            capturing, release = threading.Event(), threading.Event()
            original = CR.copy.deepcopy
            def slow(fields):
                capturing.set(); release.wait(5); return original(fields)
            producer = None
            try:
                with patch.object(CR.copy, 'deepcopy', slow), patch.object(CR.time, 'time', return_value=100):
                    producer = threading.Thread(target=lambda: rec.event('user', text='late private line'))
                    producer.start()
                    self.assertTrue(capturing.wait(2))
                    self.assertEqual(rec.mask_since(99, 101), 0)
                    release.set(); producer.join(2)
            finally:
                release.set()
                if producer: producer.join(2)
                rec.close()
            self.assertEqual(json.loads(rec.path.read_text(encoding='utf-8').strip())['text'], CR.MASKED)

    def test_mask_failure_propagates_and_can_be_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            rec.event('user', text='private')
            self.assertTrue(rec.flush())
            original = CR._mask_tail
            with patch.object(CR, '_mask_tail', side_effect=PermissionError('temporarily locked')):
                with self.assertRaises(PermissionError):
                    rec.mask_since(0, time.time() + 1)
            self.assertEqual(rec.mask_since(0, time.time() + 1), 1)
            rec.close()
            self.assertEqual(json.loads(rec.path.read_text(encoding='utf-8').strip())['text'], CR.MASKED)

    def test_payload_bytes_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = self.recorder(Path(tmp))
            rec.audio_enabled = True
            with patch.object(CR, 'MAX_PENDING_BYTES', 1):
                rec.event('user', text='too large')
                rec.audio(np.ones(10, dtype=np.float32))
            rec.close()
            self.assertEqual((rec.dropped_events, rec.dropped_audio), (1, 1))
            self.assertEqual(list(Path(tmp).rglob('*.jsonl')), [])


if __name__ == "__main__":
    unittest.main()
