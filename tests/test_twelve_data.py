from datetime import datetime, timezone
import unittest
from unittest.mock import Mock, patch

from pydantic import SecretStr

from data import twelve_data
from models.schemas import OHLCVPoint, Ticker


def bars() -> list[dict[str, str]]:
    return [
        {
            "datetime": "2026-07-16 14:00:00",
            "open": "99.0",
            "high": "101.0",
            "low": "98.0",
            "close": "100.0",
            "volume": "1000",
        },
        {
            "datetime": "2026-07-16 14:15:00",
            "open": "104.0",
            "high": "106.0",
            "low": "103.0",
            "close": "105.0",
            "volume": "1200",
        },
    ]


class TwelveDataTests(unittest.TestCase):
    def test_get_twelve_data_api_key_reads_secret_value(self):
        with patch.object(
            twelve_data.settings, "twelve_data_api_key", SecretStr("secret")
        ):
            self.assertEqual(twelve_data.get_twelve_data_api_key(), "secret")

    def test_get_twelve_data_api_key_requires_config(self):
        with patch.object(twelve_data.settings, "twelve_data_api_key", None):
            with self.assertRaises(twelve_data.MissingTwelveDataConfigError):
                twelve_data.get_twelve_data_api_key()

    def test_parse_ohlcv_points_maps_sorts_and_limits_bars(self):
        points = twelve_data.parse_ohlcv_points("nvda", list(reversed(bars())), limit=1)

        self.assertEqual(len(points), 1)
        self.assertIsInstance(points[0], OHLCVPoint)
        self.assertEqual(points[0].ticker, "NVDA")
        self.assertEqual(points[0].close, 105.0)
        self.assertEqual(points[0].timestamp.tzinfo, timezone.utc)

    def test_parse_ohlcv_points_rejects_missing_fields(self):
        payload = bars()
        del payload[0]["close"]

        with self.assertRaises(twelve_data.TwelveDataError):
            twelve_data.parse_ohlcv_points("NVDA", payload, limit=2)

    def test_fetch_intraday_ohlcv_builds_request_and_returns_points(self):
        request = Mock()
        request.as_json.return_value = tuple(reversed(bars()))
        client = Mock()
        client.time_series.return_value = request

        points = twelve_data.fetch_intraday_ohlcv(
            "nvda",
            limit=2,
            to_date=datetime(2026, 7, 16, 20, 0, tzinfo=timezone.utc),
            client=client,
        )

        self.assertEqual([point.close for point in points], [100.0, 105.0])
        client.time_series.assert_called_once_with(
            symbol="NVDA",
            interval="15min",
            outputsize=2,
            timezone="UTC",
            order="DESC",
            start_date="2026-07-06 20:00:00",
            end_date="2026-07-16 20:00:00",
        )
        request.as_json.assert_called_once_with()

    def test_fetch_intraday_ohlcv_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            twelve_data.fetch_intraday_ohlcv("NVDA", limit=0)

        with self.assertRaises(ValueError):
            twelve_data.fetch_intraday_ohlcv("NVDA", lookback_days=0)

    def test_fetch_intraday_ohlcv_wraps_client_errors(self):
        client = Mock()
        client.time_series.side_effect = RuntimeError("down")

        with self.assertRaises(twelve_data.TwelveDataError):
            twelve_data.fetch_intraday_ohlcv("NVDA", client=client)

    def test_refresh_ticker_data_writes_through_database_layer(self):
        points = twelve_data.parse_ohlcv_points("NVDA", bars(), limit=1)

        with patch("data.twelve_data.fetch_intraday_ohlcv", return_value=points):
            with patch("data.twelve_data.database.upsert_ticker") as upsert_ticker:
                with patch(
                    "data.twelve_data.database.upsert_ticker_data"
                ) as upsert_ticker_data:
                    with patch(
                        "data.twelve_data.database.delete_old_ticker_data"
                    ) as delete_old:
                        returned = twelve_data.refresh_ticker_data("nvda", limit=2)

        self.assertEqual(returned, points)
        upsert_ticker_data.assert_called_once_with(points)
        upserted_ticker = upsert_ticker.call_args.args[0]
        self.assertIsInstance(upserted_ticker, Ticker)
        self.assertEqual(upserted_ticker.ticker, "NVDA")
        self.assertEqual(upserted_ticker.current_price, 105.0)
        delete_old.assert_called_once_with("NVDA", keep=2)

    def test_refresh_ticker_data_does_not_write_when_no_points_returned(self):
        with patch("data.twelve_data.fetch_intraday_ohlcv", return_value=[]):
            with patch("data.twelve_data.database.upsert_ticker") as upsert_ticker:
                returned = twelve_data.refresh_ticker_data("NVDA")

        self.assertEqual(returned, [])
        upsert_ticker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
