"""Stutter-loop guard (owner 10-05: 'r-r-r-r-r a-a-a-a f-f-f-f' cascades)."""
import glob, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tts_loop_guard as G

ROOT = Path(__file__).resolve().parents[1]


def _speechlike(seconds, seed=0):
    """Non-repeating voiced noise with a varying envelope (stand-in when no wavs)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * G.SR)
    t = np.arange(n) / G.SR
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    x = 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / G.SR) + 0.05 * rng.standard_normal(n)
    env = np.abs(np.convolve(rng.standard_normal(n), np.ones(2400) / 2400, "same")) * 8
    return (x * np.clip(env, 0, 1.5)).astype(np.float32)


def _real_clips(limit=40):
    out = []
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
    except Exception:
        return out
    for f in sorted(glob.glob(str(ROOT / "voices" / "backchannels" / "*" / "*.wav")))[:limit]:
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != G.SR:
            x = resample_poly(x, G.SR, sr).astype(np.float32)
        out.append(x)
    return out


def _with_loop(x, k, reps, start=None):
    L = k * G.FRAME
    s = start if start is not None else len(x) // 2
    seg = x[s:s + L]
    return np.concatenate([x[:s]] + [seg] * reps)


def _run(x, chunk=1920):
    d = G.LoopDetector()
    for i in range(0, len(x), chunk):
        if d.feed(x[i:i + chunk]):
            return d, i + chunk
    return d, None


class LoopGuard(unittest.TestCase):
    def test_exact_frame_loop_is_cut_quickly(self):
        x = _speechlike(2.0)
        for k in (1, 2, 3):
            y = _with_loop(x, k, 12)
            d, at = _run(y)
            self.assertIsNotNone(at, f"k={k} loop not detected (max {d.max_score:.2f})")
            loop_start = len(x) // 2
            # caught within REPS+2 periods of the loop starting (+ chunk/check slack)
            self.assertLess(at - loop_start, (G.REPS + 2) * k * G.FRAME + 2 * 1920, f"k={k}")

    def test_loop_with_sampled_variation_is_cut(self):
        rng = np.random.default_rng(3)
        x = _speechlike(2.0, seed=3)
        seg = x[20000:20000 + G.FRAME]
        reps = [seg * (1 + 0.05 * rng.standard_normal()) + 0.003 * rng.standard_normal(seg.size) for _ in range(12)]  # ~-40 dB jitter; heavier noise (-27 dB) is a known miss, re-check on logs/tts_suspect evidence
        d, at = _run(np.concatenate([x[:20000]] + reps).astype(np.float32))
        self.assertIsNotNone(at, f"max {d.max_score:.2f}")

    def test_steady_vowel_is_not_a_loop(self):
        t = np.arange(int(1.5 * G.SR)) / G.SR
        x = (0.3 * np.sin(2 * np.pi * 125 * t) + 0.1 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        d, at = _run(x)
        self.assertIsNone(at, f"held vowel cut (score {d.max_score:.2f})")

    def test_silence_is_not_a_loop(self):
        d, at = _run(np.zeros(G.SR * 2, np.float32))
        self.assertIsNone(at)

    def test_real_tts_speech_never_cut(self):
        clips = _real_clips()
        if not clips:
            self.skipTest("no rendered clips in this tree")
        for x in clips:
            d, at = _run(x)
            self.assertIsNone(at, f"real speech cut (score {d.max_score:.2f})")

    def test_stream_wrapper_stops_and_closes_inner(self):
        closed = []
        y = _with_loop(_speechlike(1.5), 1, 30)

        def inner():
            try:
                for i in range(0, len(y), 1920):
                    yield y[i:i + 1920], G.SR
            finally:
                closed.append(True)
        G.SUSPECT_DIR = Path(__import__("tempfile").mkdtemp())
        out = list(G.guard_stream(inner(), "No."))
        got = sum(c.size for c, _ in out)
        self.assertLess(got, len(y) - 10 * G.FRAME)   # most of the loop never played
        self.assertTrue(closed)                         # native stream closed -> GPU cancel
        self.assertTrue(list(G.SUSPECT_DIR.glob("*.wav")))  # evidence saved

    def test_clean_stream_passes_through_unchanged(self):
        x = _speechlike(1.5, seed=7)
        G.SUSPECT_DIR = Path(__import__("tempfile").mkdtemp())
        chunks = [x[i:i + 1920] for i in range(0, len(x), 1920)]
        out = list(G.guard_stream(iter([(c, G.SR) for c in chunks]), "a normal length sentence here"))
        self.assertEqual(len(out), len(chunks))

    def test_warmup_not_wrapped(self):
        class E:
            def _stream_native(self, kw, *, stream_started_ns=None):
                return iter(["raw"])
        e = E()
        G.install(e)
        self.assertEqual(list(e._stream_native({"text": "hi", "max_new_tokens": 16})), ["raw"])


if __name__ == "__main__":
    unittest.main()
