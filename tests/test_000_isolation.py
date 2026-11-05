"""Loaded first by `unittest discover` (alphabetical): point shared on-disk state at a
throwaway location so no test reads or writes the owner's live choices.

Found 10-06: the owner switched plugins off on the Plugins page and a server test that
assumed "timers on" failed, because it read the real user/plugins.json.
"""
import os
import tempfile
import unittest
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="atlas_test_state_"))

# Memory stores, the call boundary's current.json, task boards and room state (docs/designs/experience_memory.md):
# tests that commit turns or end calls must never touch the owner's agent_state/ or memory_db/.
os.environ["ATLAS_MEMORY_DIR"] = str(_TMP / "memory")
os.environ["ATLAS_TASKS_DIR"] = str(_TMP / "tasks")
import sys as _sys  # noqa: E402
if "agent_memory" in _sys.modules:
    _sys.modules["agent_memory"].ROOT = _TMP / "memory"
try:   # a call some test starts on the shared boundary must not look live to later tests (no current.json)
    import call_boundary as _call_boundary
    _call_boundary.BOUNDARY.write_files = False
except Exception:  # noqa: BLE001
    pass
try:
    import room_tools as _room_tools
    _room_tools.STATE_PATH = _TMP / "room.json"
except Exception:  # noqa: BLE001
    _room_tools = None

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

    def test_memory_tasks_and_room_state_are_not_the_owners(self):
        import agent_memory
        import call_boundary
        self.assertTrue(str(agent_memory.ROOT).startswith(str(_TMP)))
        self.assertTrue(str(call_boundary.current_path()).startswith(str(_TMP)))
        self.assertTrue(os.environ["ATLAS_TASKS_DIR"].startswith(str(_TMP)))
        if _room_tools is not None:
            self.assertTrue(str(_room_tools.STATE_PATH).startswith(str(_TMP)))


if __name__ == "__main__":
    unittest.main()
