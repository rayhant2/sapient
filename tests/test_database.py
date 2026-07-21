from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from pydantic import ValidationError

from data import database
from models.schemas import (
    AgentOutput,
    Alert,
    AlertType,
    Confidence,
    EventType,
    HypothesisOutput,
    Motive,
    OHLCVPoint,
    Subscription,
    Ticker,
    UpdateInterval,
    User,
)


class FakeTable:
    def __init__(self, table_name: str, response_data):
        self.table_name = table_name
        self.response_data = response_data
        self.calls = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self

    def insert(self, payload):
        return self._record("insert", payload)

    def upsert(self, payload, **kwargs):
        return self._record("upsert", payload, **kwargs)

    def select(self, columns):
        return self._record("select", columns)

    def update(self, payload):
        return self._record("update", payload)

    def delete(self):
        return self._record("delete")

    def eq(self, column, value):
        return self._record("eq", column, value)

    def lte(self, column, value):
        return self._record("lte", column, value)

    def order(self, column, **kwargs):
        return self._record("order", column, **kwargs)

    def limit(self, count):
        return self._record("limit", count)

    def range(self, start, end):
        return self._record("range", start, end)

    def execute(self):
        self.calls.append(("execute", (), {}))
        return SimpleNamespace(data=self.response_data)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.tables = []

    def table(self, table_name: str):
        response_data = self.responses.get(table_name, [])
        if isinstance(response_data, list) and response_data and isinstance(response_data[0], list):
            response_data = response_data.pop(0)
        table = FakeTable(table_name, response_data)
        self.tables.append(table)
        return table


def user_row():
    return {
        "user_id": "user-1",
        "whatsapp_number": "whatsapp:+15551234567",
        "email": "person@example.com",
        "created_at": "2026-01-01T00:00:00+00:00",
        "preferences": {"tone": "brief"},
    }


def ticker_row():
    return {
        "ticker": "NVDA",
        "last_fetched": "2026-01-01T00:00:00+00:00",
        "next_fetch_time": "2026-01-01T00:15:00+00:00",
        "current_price": 500.25,
    }


def subscription_row():
    return {
        "user_id": "user-1",
        "ticker": "NVDA",
        "avg_price": 400.0,
        "shares": 2.0,
        "motive": "holding",
        "update_interval": "daily",
        "sharp_move_threshold": 0.025,
    }


def point_row(timestamp: str = "2026-01-01T00:00:00+00:00"):
    return {
        "ticker": "NVDA",
        "timestamp": timestamp,
        "open": 100.0,
        "high": 105.0,
        "low": 99.0,
        "close": 104.0,
        "volume": 1000.0,
    }


def update_row():
    return {
        "ticker": "NVDA",
        "user_id": "user-1",
        "event_type": "sharp_move",
        "summary": "NVDA moved sharply.",
        "recommendation": "Review the position.",
        "confidence": "medium",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "price_at_update": 500.25,
        "searched_web": True,
        "metadata": {},
    }


def alert_row():
    return {
        "user_id": "user-1",
        "ticker": "NVDA",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "alert_type": "sharp_move",
        "message": "NVDA moved sharply.",
        "trigger_details": {"move_pct": 0.03},
    }


class DatabaseTests(unittest.TestCase):
    def test_create_user_inserts_and_returns_model(self):
        client = FakeClient({"users": [user_row()]})
        user = User(
            user_id="user-1",
            whatsapp_number="whatsapp:+15551234567",
            email="person@example.com",
            preferences={"tone": "brief"},
        )

        created = database.create_user(user, client=client)

        self.assertEqual(created, User.model_validate(user_row()))
        self.assertEqual(client.tables[0].table_name, "users")
        self.assertEqual(client.tables[0].calls[0][0], "insert")
        self.assertEqual(client.tables[0].calls[0][1][0]["user_id"], "user-1")

    def test_get_user_returns_none_when_no_row_exists(self):
        client = FakeClient({"users": []})

        self.assertIsNone(database.get_user("missing-user", client=client))
        self.assertIn(("eq", ("user_id", "missing-user"), {}), client.tables[0].calls)
        self.assertIn(("limit", (1,), {}), client.tables[0].calls)

    def test_upsert_ticker_uppercases_symbol(self):
        client = FakeClient({"tickers": [ticker_row()]})

        ticker = database.upsert_ticker(
            Ticker(ticker="nvda", current_price=500.25),
            client=client,
        )

        self.assertEqual(ticker.ticker, "NVDA")
        self.assertEqual(client.tables[0].calls[0][0], "upsert")
        self.assertEqual(client.tables[0].calls[0][1][0]["ticker"], "NVDA")
        self.assertEqual(client.tables[0].calls[0][2], {"on_conflict": "ticker"})

    def test_subscription_queries_use_user_and_ticker_filters(self):
        client = FakeClient({"subscriptions": [subscription_row()]})

        subscription = database.get_subscription("user-1", "nvda", client=client)

        self.assertEqual(subscription.ticker, "NVDA")
        self.assertIn(("eq", ("user_id", "user-1"), {}), client.tables[0].calls)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), client.tables[0].calls)

    def test_list_subscriptions_for_ticker_parses_models(self):
        client = FakeClient({"subscriptions": [[subscription_row()]]})

        subscriptions = database.list_subscriptions_for_ticker("nvda", client=client)

        self.assertEqual(len(subscriptions), 1)
        self.assertIsInstance(subscriptions[0], Subscription)
        self.assertEqual(subscriptions[0].motive, Motive.HOLDING)

    def test_insert_ticker_data_upserts_point(self):
        client = FakeClient({"ticker_data": [point_row()]})
        point = OHLCVPoint(
            ticker="nvda",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000.0,
        )

        inserted = database.insert_ticker_data(point, client=client)

        self.assertEqual(inserted.ticker, "NVDA")
        self.assertEqual(client.tables[0].calls[0][0], "upsert")
        self.assertEqual(client.tables[0].calls[0][1][0]["ticker"], "NVDA")

    def test_upsert_ticker_data_writes_all_points_in_one_request(self):
        older = point_row("2026-01-01T00:00:00+00:00")
        newer = point_row("2026-01-01T00:15:00+00:00")
        client = FakeClient({"ticker_data": [older, newer]})
        points = [OHLCVPoint.model_validate(older), OHLCVPoint.model_validate(newer)]

        returned = database.upsert_ticker_data(points, client=client)

        self.assertEqual(returned, points)
        self.assertEqual(len(client.tables), 1)
        method, args, kwargs = client.tables[0].calls[0]
        self.assertEqual(method, "upsert")
        self.assertEqual(len(args[0]), 2)
        self.assertTrue(all(row["ticker"] == "NVDA" for row in args[0]))
        self.assertEqual(kwargs, {"on_conflict": "ticker,timestamp"})

    def test_upsert_ticker_data_skips_database_for_empty_list(self):
        client = FakeClient({})

        returned = database.upsert_ticker_data([], client=client)

        self.assertEqual(returned, [])
        self.assertEqual(client.tables, [])

    def test_get_latest_ticker_data_returns_chronological_points(self):
        newer = point_row("2026-01-01T00:15:00+00:00")
        older = point_row("2026-01-01T00:00:00+00:00")
        client = FakeClient({"ticker_data": [[newer, older]]})

        points = database.get_latest_ticker_data("nvda", limit=2, client=client)

        self.assertEqual([p.timestamp.isoformat() for p in points], [
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:15:00+00:00",
        ])
        self.assertIn(("order", ("timestamp",), {"desc": True}), client.tables[0].calls)
        self.assertIn(("limit", (2,), {}), client.tables[0].calls)

    def test_delete_old_ticker_data_deletes_at_and_before_cutoff(self):
        client = FakeClient(
            {
                "ticker_data": [
                    [{"timestamp": "2026-01-01T00:00:00+00:00"}],
                    [],
                ]
            }
        )

        database.delete_old_ticker_data("nvda", keep=150, client=client)

        select_table = client.tables[0]
        delete_table = client.tables[1]
        self.assertIn(("range", (150, 150), {}), select_table.calls)
        self.assertIn(("delete", (), {}), delete_table.calls)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), delete_table.calls)
        self.assertIn(
            ("lte", ("timestamp", "2026-01-01T00:00:00+00:00"), {}),
            delete_table.calls,
        )

    def test_insert_agent_output_preserves_subclass_fields_in_metadata(self):
        client = FakeClient({"updates": [update_row()]})
        output = HypothesisOutput(
            ticker="nvda",
            user_id="user-1",
            summary="Pattern developing.",
            recommendation="Watch closely.",
            confidence=Confidence.MEDIUM,
            recommended_next_scan_days=1,
            flagged=True,
            searched_web=True,
        )

        returned = database.insert_agent_output(output, client=client)

        payload = client.tables[0].calls[0][1][0]
        self.assertIsInstance(returned, AgentOutput)
        self.assertEqual(payload["ticker"], "NVDA")
        self.assertEqual(payload["event_type"], EventType.HYPOTHESIS_SCAN.value)
        self.assertEqual(
            payload["metadata"],
            {"flagged": True, "recommended_next_scan_days": 1},
        )

    def test_list_updates_for_user_ticker_parses_agent_outputs(self):
        client = FakeClient({"updates": [[update_row()]]})

        outputs = database.list_updates_for_user_ticker(
            "user-1",
            "nvda",
            client=client,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].event_type, EventType.SHARP_MOVE)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), client.tables[0].calls)

    def test_insert_alert_returns_alert_model(self):
        client = FakeClient({"alerts": [alert_row()]})
        alert = Alert(
            user_id="user-1",
            ticker="nvda",
            alert_type=AlertType.SHARP_MOVE,
            message="NVDA moved sharply.",
            trigger_details={"move_pct": 0.03},
        )

        inserted = database.insert_alert(alert, client=client)

        self.assertEqual(inserted.alert_type, AlertType.SHARP_MOVE)
        self.assertEqual(client.tables[0].calls[0][1][0]["ticker"], "NVDA")

    def test_single_row_rejects_multiple_rows(self):
        with self.assertRaises(database.DatabaseError):
            database._single_row([user_row(), user_row()])

    def test_parse_model_surfaces_schema_mismatch(self):
        bad_row = subscription_row()
        bad_row["sharp_move_threshold"] = 0.9

        with self.assertRaises(ValidationError):
            database._parse_model(Subscription, bad_row)


if __name__ == "__main__":
    unittest.main()
