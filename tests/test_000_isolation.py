"""Loaded first by `unittest discover` (alphabetical): point shared on-disk state at a
throwaway location so no test reads or writes the owner's live choices.

Found 10-06: the owner switched plugins off on the Plugins page and a server test that
assumed "timers on" failed, because it read the real user/plugins.json.
"""
import tempfile
import unittest
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="atlas_test_state_"))

try:
    import plugins as _plugins
    _plugins.STATE_PATH = _TMP / "plugins.json"
except Exception:  # noqa: BLE001  light envs without plugins deps
    _plugins = None


class Isolation(unittest.TestCase):
    def test_plugin_state_is_not_the_owners(self):
        if _plugins is None:
            self.skipTest("plugins not importable")
        self.assertNotIn("user", Path(_plugins.STATE_PATH).parent.name)
        self.assertTrue(str(_plugins.STATE_PATH).startswith(str(_TMP)))


if __name__ == "__main__":
    unittest.main()
