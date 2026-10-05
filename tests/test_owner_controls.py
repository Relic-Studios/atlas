"""Owner controls gate every generation path via floor (owner 10-03)."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import owner_controls as O
from floor import ConversationFloor


class OwnerControlTests(unittest.TestCase):
    def setUp(self):
        O.STATE.update(mute=False, muted_speakers=set(), quiet_until=0.0)
        self._prefs = O._PREFS
        O._PREFS = os.path.join(tempfile.mkdtemp(), "p.json")

    def tearDown(self):
        O.STATE.update(mute=False, muted_speakers=set(), quiet_until=0.0)
        O._PREFS = self._prefs

    def test_mute_holds_everything_even_named(self):
        f = ConversationFloor(); O.set_mute(True)
        self.assertEqual(f.turn_gate("[S1] Fae, what's up?", "S1", ("fae",)), "owner mute")
        self.assertTrue(f.is_quiet())

    def test_muted_speaker(self):
        f = ConversationFloor(); O.set_speaker_muted("s42", True)
        self.assertEqual(f.turn_gate("[S42] Max tell me about the Jews", "S42", ("max",)), "owner muted speaker")
        self.assertIsNone(O.gate("S5", True))

    def test_quiet_mode_yields_to_name(self):
        f = ConversationFloor(); O.set_quiet(True, 5)
        self.assertEqual(f.turn_gate("[S1] lol that's wild", "S1", ("fae",)), "owner quiet mode")
        self.assertNotEqual(f.turn_gate("[S1] Fae, what do you think?", "S1", ("fae",)), "owner quiet mode")
        O.set_quiet(False)
        self.assertFalse(O.quiet_active())

    def test_talkativeness_persists_and_applies(self):
        from agent_runtime import build_profile
        O.set_talkativeness("fae", 0.1)
        self.assertEqual(O.talkativeness_override("fae"), 0.1)
        self.assertAlmostEqual(build_profile("fae", "Fae", {}).talkativeness, 0.1)
        self.assertEqual(O.set_talkativeness("fae", 7), 1.0)


if __name__ == "__main__":
    unittest.main()
