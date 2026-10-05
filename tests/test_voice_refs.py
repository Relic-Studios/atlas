"""Reference clips must end (and start) in silence: ICL cloning continues from
the last reference sample, so a clip cut mid-word leaks that sound onto the
start of every reply (live 10-01: Pup 'Hey, okay...' on every line)."""
import glob, os, unittest
import sys
import numpy as np, soundfile as sf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

class RefEdgeTests(unittest.TestCase):
    def test_refs_end_quiet(self):
        for f in glob.glob(os.path.join(ROOT, "voices", "*.wav")):
            from audio_module import ICL_VOICES
            if os.path.basename(f)[:-4] not in ICL_VOICES:
                continue  # only ICL refs continue from the tail
            x, sr = sf.read(f, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            tail = np.abs(x[-int(sr * 0.03):]).max()
            self.assertLess(tail, 0.15, f"{os.path.basename(f)} ends mid-sound ({tail:.2f})")

if __name__ == "__main__":
    unittest.main()
