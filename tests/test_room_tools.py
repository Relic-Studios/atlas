"""Room-tool plugins (owner-approved 10-05): dice/coins/polls, timers & reminders, weather."""
import collections
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import room_tools as RT  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._path = RT.STATE_PATH
        RT.STATE_PATH = Path(self.tmp.name) / "room.json"

    def tearDown(self):
        RT.STATE_PATH = self._path
        self.tmp.cleanup()


class Dice(Base):
    def test_d20_in_range_and_roughly_uniform(self):
        c = collections.Counter()
        for _ in range(4000):
            out = RT.roll_dice("d20")
            n = int(out.split(": ")[1].split(".")[0].split(" ")[0])
            self.assertTrue(1 <= n <= 20, out)
            c[n] += 1
        self.assertEqual(len(c), 20)
        self.assertLess(max(c.values()) / min(c.values()), 2.0)

    def test_notation(self):
        out = RT.roll_dice("3d6+2")
        total = int(out.split(": ")[1].split(" ")[0])
        self.assertTrue(5 <= total <= 20, out)
        self.assertIn("rolls:", out)
        self.assertIn("d20", RT.roll_dice(""))
        self.assertIn("d6", RT.roll_dice("6"))
        self.assertIn("Couldn't read", RT.roll_dice("banana"))
        self.assertIn("Keep it", RT.roll_dice("999d6"))

    def test_coin_and_pick(self):
        seen = {RT.flip_coin().split("on ")[1].split(".")[0] for _ in range(200)}
        self.assertEqual(seen, {"heads", "tails"})
        self.assertIn("Flipped 3 coins", RT.flip_coin(3))
        picks = {RT.pick_one("Valorant, Minecraft or Among Us").split(": ")[1].split(" (")[0] for _ in range(300)}
        self.assertEqual(picks, {"Valorant", "Minecraft", "Among Us"})
        self.assertIn("at least two", RT.pick_one("just one"))


class Polls(Base):
    def test_full_vote(self):
        self.assertIn("Options: pizza, tacos", RT.start_poll("Dinner?", "pizza, tacos"))
        RT.cast_vote("pizza", "Sam")
        RT.cast_vote("Tacos", "Riley")
        out = RT.cast_vote("pizza please", "Riley")         # changes her vote, fuzzy match
        self.assertIn("Changed", out)
        self.assertIn("pizza 2, tacos 0", out)
        self.assertIn("isn't one of the options", RT.cast_vote("sushi", "Max"))
        self.assertIn("pizza wins with 2", RT.poll_results(close=True))
        self.assertIn("No poll", RT.poll_results())
        self.assertIn("no poll open", RT.cast_vote("pizza", "Sam"))

    def test_tie_and_number_vote(self):
        RT.start_poll("", ["a", "b", "c"])
        RT.cast_vote("1", "x")
        RT.cast_vote("2", "y")
        self.assertIn("Tie between a and b", RT.poll_results())


class Timers(Base):
    def test_set_fire_claim(self):
        now = [1000.0]
        clock = lambda: now[0]  # noqa: E731
        out = RT.set_timer(10, "start the raid", "S2", clock=clock)
        self.assertIn("10 minutes", out)
        self.assertIsNone(RT.due(clock=clock))
        self.assertIn("start the raid", RT.list_timers(clock=clock))
        now[0] += 601
        t = RT.due(clock=clock)
        self.assertEqual(t["message"], "start the raid")
        cue = RT.claim(t["id"], clock=clock)
        self.assertTrue(cue.startswith("[S2] (ROOM CUE #"))
        self.assertTrue(RT.ROOM_CUE_RE.search(cue))
        self.assertIn("start the raid", cue)
        self.assertIsNone(RT.due(clock=clock))            # announced once only
        self.assertIn("No timers", RT.list_timers(clock=clock))

    def test_survives_restart_and_units(self):
        RT.set_timer("90 seconds", "eggs")
        self.assertIn("eggs", RT.list_timers())           # read back from disk
        self.assertIn("How long", RT.set_timer("soon", "x"))
        self.assertIn("at most 24 hours", RT.set_timer(5000, "x"))

    def test_cancel(self):
        RT.set_timer(5, "pizza")
        RT.set_timer(6, "laundry")
        self.assertIn("Which timer", RT.cancel_timer())
        self.assertIn("laundry", RT.cancel_timer(message="laund"))
        self.assertIn("pizza", RT.cancel_timer())          # only one left -> that one

    def test_stale_dropped(self):
        now = [0.0]
        RT.set_timer(1, "old", clock=lambda: now[0])
        now[0] = 60 + RT.TIMER_STALE_S + 5
        self.assertIsNone(RT.due(clock=lambda: now[0]))

    def test_cue_note(self):
        self.assertIn("Do not [HOLD]", RT.cue_note("(ROOM CUE #3) blah"))
        self.assertEqual(RT.cue_note("[S1] hey"), "")


class Weather(Base):
    def fake(self, url):
        if "geocoding" in url:
            return {"results": [{"name": "Seattle", "admin1": "Washington", "country": "United States",
                                 "country_code": "US", "latitude": 47.6, "longitude": -122.3}]}
        self.assertIn("fahrenheit", url)
        return {"current": {"temperature_2m": 58.4, "apparent_temperature": 55.0, "weather_code": 61,
                            "wind_speed_10m": 9, "relative_humidity_2m": 80},
                "daily": {"weather_code": [61, 3, 0], "temperature_2m_min": [50, 49, 47],
                          "temperature_2m_max": [60, 62, 66], "precipitation_probability_max": [80, 30, 5]}}

    def test_parse(self):
        out = RT.check_weather("Seattle", fetch=self.fake)
        self.assertIn("Now in Seattle, Washington", out)
        self.assertIn("light rain, 58°F", out)
        self.assertIn("Tomorrow: overcast, 49-62°F, 30% chance of rain", out)

    def test_default_place_and_errors(self):
        self.assertIn("Which place", RT.check_weather(""))
        self.assertIn("Now in Seattle", RT.check_weather("", default_place="Seattle", fetch=self.fake))
        self.assertIn("Couldn't find", RT.check_weather("Zzz", fetch=lambda u: {"results": []}))

        def boom(u):
            raise TimeoutError()
        self.assertIn("didn't answer", RT.check_weather("Seattle", fetch=boom))


class Wiring(Base):
    def test_every_room_tool_has_a_plugin(self):
        import plugins
        owned = set()
        for p in plugins.BUILTIN:
            owned |= set(plugins._builtin_tools(p))
        self.assertTrue(RT.NAMES <= owned, RT.NAMES - owned)

    def test_execute_routes_voter_name(self):
        RT.execute("start_poll", {"options": "a, b"})
        out = RT.execute("cast_vote", {"choice": "a"}, asker="S1", voter_name="Sam")
        self.assertIn("Counted", out)
        import json
        self.assertEqual(json.loads(RT.STATE_PATH.read_text())["poll"]["votes"], {"Sam": "a"})

    def test_steering_note_treats_room_cue_as_cue(self):
        from floor import ConversationFloor, steering_note
        note = steering_note("(ROOM CUE #1) timer", None, ConversationFloor(), ("fae",))
        self.assertIn("REMINDER", note)

    def test_server_start_timer(self):
        import server
        RT.set_timer(0.5, "stretch", "S1", clock=lambda: __import__("time").time() - 60)
        now = __import__("time").time() + 5
        started = []
        mgr = SimpleNamespace(prepare_generation=started.append)
        cb = SimpleNamespace(reset_state=lambda: None)
        orig = server.delivery_ready
        try:
            server.delivery_ready = lambda *a, **k: False
            self.assertIsNone(server.start_timer(mgr, cb, None, now))   # busy room: waits
            server.delivery_ready = lambda *a, **k: True
            self.assertIsNotNone(server.start_timer(mgr, cb, None, now))
        finally:
            server.delivery_ready = orig
        self.assertEqual(len(started), 1)
        self.assertIn("stretch", started[0])
        self.assertIsNone(server.start_timer(mgr, cb, None, now))       # only once


if __name__ == "__main__":
    unittest.main()
