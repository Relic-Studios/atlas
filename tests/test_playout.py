"""Playout: jitter buffer absorbs synthesis hiccups; soft edges; stats; barge-in."""
import sys, random, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playout import Playout

SR = 48000
BLOCK = 960  # 20 ms


class Clock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def tone(sec, amp=0.1):
    n = int(SR * sec)
    return (amp * np.sin(np.arange(n) * 2 * np.pi * 220 / SR)).astype(np.float32)


def run(chunks, arrivals, playout=None, naive=False, end=True, tail=1.5):
    """chunks: list of arrays; arrivals: wall time of each (s from reply start).
    Returns (gaps_ms_inside_reply, first_audio_s, output)."""
    clk = Clock(); t0 = clk.t
    p = playout or Playout(SR, clock=clk)
    if playout is not None:
        p.clock = clk
    q = []  # naive queue
    out = []
    i = 0
    total = arrivals[-1] + sum(c.size for c in chunks) / SR + tail
    steps = int(total / (BLOCK / SR))
    for _ in range(steps):
        while i < len(chunks) and arrivals[i] <= clk.t - t0 + 1e-9:
            if naive: q.append(chunks[i].copy())
            else: p.push(chunks[i])
            i += 1
            if i == len(chunks) and end and not naive: p.end_reply()
        if naive:
            b = np.zeros(BLOCK, np.float32); f = 0
            while f < BLOCK and q:
                h = q[0]; k = min(h.size, BLOCK - f); b[f:f+k] = h[:k]; f += k
                if k == h.size: q.pop(0)
                else: q[0] = h[k:]
        else:
            b = p.drain(BLOCK)
        out.append(b)
        clk.t += BLOCK / SR
    o = np.concatenate(out)
    nz = np.flatnonzero(np.abs(o) > 1e-6)
    first = nz[0] / SR
    # silent runs >= 10 ms between first and last audible sample = audible gaps
    seg = np.abs(o[nz[0]:nz[-1]]) > 1e-6
    gaps, run_ = [], 0
    for v in seg:
        if not v: run_ += 1
        else:
            if run_ >= SR * 0.010: gaps.append(run_ / SR * 1000)
            run_ = 0
    return gaps, first, o, p


def jittery(n=30, chunk_s=0.2, rtf=0.6, seed=0, spikes=(0.18, 0.47, 0.96), p_spike=0.1):
    """TTS faster than realtime on average (rtf<1) with occasional stalls from the log."""
    rnd = random.Random(seed)
    chunks, arr, t = [], [], 0.25
    for k in range(n):
        chunks.append(tone(chunk_s))
        t += chunk_s * rtf
        if rnd.random() < p_spike:
            t += rnd.choice(spikes)
        arr.append(t)
    return chunks, arr


class PlayoutTests(unittest.TestCase):
    def test_naive_stutters_playout_does_not(self):
        """Stall sizes from the live log (median 0.18 s, p95 0.47 s, max 0.96 s)."""
        def jit(seed, rtf=0.6, p=0.1, n=30):
            r = random.Random(seed); ch, ar, t = [], [], 0.25
            for _ in range(n):
                ch.append(tone(0.2)); t += 0.2 * rtf
                if r.random() < p:
                    t += r.choices([0.18, 0.47, 0.96], [70, 25, 5])[0]
                ar.append(t)
            return ch, ar
        nc = pc = 0
        for seed in range(40):
            ch, ar = jit(seed)
            nc += len(run(ch, ar, naive=True)[0]); pc += len(run(ch, ar)[0])
        self.assertGreater(nc, 5)
        self.assertLessEqual(pc, nc * 0.65, (nc, pc))

    def test_adaptive_prebuffer_learns_bad_engine(self):
        clk = Clock(); p = Playout(SR, clock=clk)
        start = p.prebuffer_s
        for seed in range(4):
            ch, ar = jittery(seed=seed, rtf=0.9, spikes=(0.5, 0.8), p_spike=0.2)
            run(ch, ar, playout=p)
        self.assertGreater(p.prebuffer_s, start)
        before = p.prebuffer_s
        for seed in range(10):
            ch, ar = jittery(seed=seed, rtf=0.5, spikes=(0.0,))
            run(ch, ar, playout=p)
        self.assertLess(p.prebuffer_s, before)

    def test_latency_bounded(self):
        ch, ar = jittery(seed=1)
        _, first, _, p = run(ch, ar)
        self.assertLess(first - ar[0], p.max_pre + 0.05)

    def test_short_reply_not_held(self):
        # a single 0.15 s "Yep." must start once end_reply arrives, not wait the cushion
        _, first, _, _ = run([tone(0.15)], [0.1])
        self.assertLess(first, 0.1 + 0.05)

    def test_reply_end_is_not_an_underrun(self):
        ch, ar = jittery(seed=2, rtf=0.5, spikes=(0.0,))
        *_, p = run(ch, ar, end=False, tail=2.0)
        p.tick()
        self.assertEqual(p.underruns, 0)

    def test_no_clicks_at_edges(self):
        _, _, o, _ = run([tone(0.3, 0.3)], [0.0])
        nz = np.flatnonzero(np.abs(o) > 1e-6)
        self.assertLess(abs(o[nz[0]]), 0.02)       # fade-in
        d = np.abs(np.diff(o))
        self.assertLess(d.max(), 0.05)              # no step anywhere

    def test_joint_jump_smoothed(self):
        a = np.full(4800, 0.4, np.float32); b = np.full(4800, -0.4, np.float32)
        _, _, o, _ = run([a, b], [0.0, 0.0])
        self.assertLess(np.abs(np.diff(o)).max(), 0.05)

    def test_soft_stop(self):
        clk = Clock(); p = Playout(SR, clock=clk, prebuffer_s=0.25)
        p.push(tone(2.0, 0.3)); p.end_reply()
        p.drain(BLOCK)
        dropped = p.stop(80)
        self.assertGreater(dropped, 1.5)
        tail = np.concatenate([p.drain(BLOCK) for _ in range(8)])
        nz = np.flatnonzero(np.abs(tail) > 1e-6)
        self.assertGreater(nz.size, 0)                   # fades, not instant cut
        self.assertLess(nz[-1] / SR, 0.1)                # gone within ~80 ms
        self.assertLess(abs(tail[nz[-1]]), 0.02)
        self.assertEqual(p.pending_seconds(), 0.0)

    def test_loudness_levelled_and_limited(self):
        clk = Clock(); p = Playout(SR, clock=clk, prebuffer_s=0.25)
        for amp in [0.05] * 6 + [0.15] * 6 + [3.0]:
            p.push(tone(0.2, amp))
        p.end_reply()
        o = np.concatenate([p.drain(BLOCK) for _ in range(160)])
        self.assertLessEqual(np.abs(o).max(), 1.0)          # limiter
        c = int(SR * 0.2)
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        quiet, loud = rms(o[3 * c:6 * c]), rms(o[9 * c:12 * c])
        self.assertLess(loud / quiet, 2.0)                  # raw ratio is 3.0


if __name__ == "__main__":
    unittest.main()


class ReplyStartAttackTests(unittest.TestCase):
    def test_first_consonant_not_swallowed(self):
        import numpy as np
        from playout import Playout
        p = Playout(24000, prebuffer_s=0.0, min_prebuffer_s=0.0, clock=lambda: 0.0)
        x = np.full(2400, 0.3, dtype=np.float32)   # loud attack at sample 0
        p.push(x); p.end_reply()
        out = p.drain(2400)
        # 10 ms in, the attack must already be at full level (old 20 ms fade: ~half)
        self.assertGreater(abs(out[240]), 0.25)
