import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from core.health import HealthServer, HealthState


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.state = HealthState()
        self.server = HealthServer(self.state, "127.0.0.1", 0)
        self.server.start()
        self.base_url = f"http://127.0.0.1:{self.server.port}"

    def tearDown(self):
        self.server.shutdown()

    def _read(self, path: str) -> tuple[int, dict]:
        try:
            response = urlopen(f"{self.base_url}{path}", timeout=2)
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())
        with response:
            return response.status, json.loads(response.read())

    def test_liveness_is_available_during_startup(self):
        status, payload = self._read("/health/live")

        self.assertEqual(status, 200)
        self.assertTrue(payload["live"])
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["status"], "starting")

    def test_readiness_tracks_runtime_lifecycle(self):
        status, _ = self._read("/health/ready")
        self.assertEqual(status, 503)

        self.state.mark_ready(active_tickers=3)
        status, payload = self._read("/health/ready")

        self.assertEqual(status, 200)
        self.assertEqual(payload["active_tickers"], 3)
        self.assertEqual(payload["status"], "ready")

        self.state.mark_stopping()
        status, payload = self._read("/health/ready")
        self.assertEqual(status, 503)
        self.assertEqual(payload["status"], "stopping")

    def test_unknown_path_returns_not_found(self):
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{self.base_url}/unknown", timeout=2)

        self.assertEqual(raised.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
