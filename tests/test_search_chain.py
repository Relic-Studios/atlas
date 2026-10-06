"""Search provider chain: Exa -> Brave, keyed APIs only (owner 10-06: no scraping from
the user's PC). Keys never bundled."""
import os
import unittest
from unittest import mock

import websearch as W


class Chain(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        for k in ("ATLAS_EXA_KEY", "ATLAS_BRAVE_KEY"):
            os.environ.pop(k, None)
        self.nofile = mock.patch.object(W, "_HERE", W._Path(os.devnull).parent / "no_such_atlas_dir")
        self.nofile.start()

    def tearDown(self):
        for p in (self.nofile, self.env):
            p.stop()

    def test_no_keys_means_no_search(self):
        self.assertEqual(W.providers(), [])
        self.assertFalse(W.available())
        with mock.patch.object(W._ureq, "urlopen") as net:
            r = W.search("x")
        net.assert_not_called()          # nothing leaves the PC without a key
        self.assertFalse(r["ok"])
        self.assertIn("API key", r["text"])

    def test_no_scraping_backends_left(self):
        for name in ("_search_ddgs", "_ddgs_backend", "_search_searxng", "_searxng_available"):
            self.assertFalse(hasattr(W, name), name)

    def test_unknown_preference_falls_back_to_keyed(self):
        os.environ["ATLAS_BRAVE_KEY"] = "k"
        with mock.patch.object(W, "PREFERRED", "free"):
            self.assertEqual(W.providers(), ["brave"])

    def test_exa_first_when_keyed(self):
        os.environ["ATLAS_EXA_KEY"] = "k"
        os.environ["ATLAS_BRAVE_KEY"] = "k"
        with mock.patch.object(W, "_search_exa", return_value=[{"title": "e", "url": "http://e", "snippet": "e"}]) as e, \
             mock.patch.object(W, "_search_brave") as b:
            r = W.search("x")
        self.assertEqual(r["backend"], "exa"); e.assert_called_once(); b.assert_not_called()

    def test_exa_error_falls_to_brave(self):
        os.environ["ATLAS_EXA_KEY"] = "k"
        os.environ["ATLAS_BRAVE_KEY"] = "k"
        with mock.patch.object(W, "_search_exa", side_effect=OSError("down")), \
             mock.patch.object(W, "_search_brave", return_value=[{"title": "b", "url": "http://b", "snippet": "b"}]):
            self.assertEqual(W.search("x")["backend"], "brave")

    def test_empty_results_fall_through(self):
        os.environ["ATLAS_EXA_KEY"] = "k"
        with mock.patch.object(W, "_search_exa", return_value=[]):
            r = W.search("x")
        self.assertEqual(r["results"], [])
        self.assertIn("No results", r["text"])

    def test_brave_strips_markup(self):
        payload = b'{"web":{"results":[{"title":"<strong>Hi</strong>","url":"http://h","description":"a <strong>b</strong>","age":"2 days ago"}]}}'
        resp = mock.MagicMock(); resp.read.return_value = payload
        with mock.patch.object(W._ureq, "urlopen", return_value=resp):
            r = W._search_brave("k", "q", 3)
        self.assertEqual(r[0]["title"], "Hi")
        self.assertEqual(r[0]["snippet"], "[2 days ago] a b")


if __name__ == "__main__":
    unittest.main()
