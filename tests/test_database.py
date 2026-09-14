import base64
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import SecretStr
from pydantic import ValidationError

from data import database
from models.schemas import (
    AgentOutput,
    AgentType,
    Alert,
    AlertType,
    Confidence,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    Motive,
    OHLCVPoint,
    Subscription,
    Ticker,
    UpdateInterval,
    User,
)
from security.credentials import CredentialCipher


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

    def gte(self, column, value):
        return self._record("gte", column, value)

    def in_(self, column, values):
        return self._record("in_", column, values)

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
        "agent_type": "sharp_move",
        "event_type": "sharp_move",
        "summary": "NVDA moved sharply.",
        "recommendation": "Review the position.",
        "confidence": "medium",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "price_at_update": 500.25,
        "searched_web": True,
        "metadata": {},
    }


def hypothesis_row(*, flagged: bool = True):
    return {
        "ticker": "NVDA",
        "user_id": "user-1",
        "agent_type": "hypothesis",
        "event_type": "hypothesis_scan",
        "summary": "Pattern developing." if flagged else None,
        "recommendation": "Watch closely." if flagged else "",
        "confidence": "medium",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "price_at_update": 500.25,
        "searched_web": flagged,
        "metadata": {
            "flagged": flagged,
            "recommended_next_scan_days": 1 if flagged else 3,
        },
    }


def cross_portfolio_row():
    return {
        "ticker": None,
        "user_id": "user-1",
        "agent_type": "cross_portfolio",
        "event_type": None,
        "summary": "Semiconductor exposure is concentrated.",
        "recommendation": None,
        "confidence": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "price_at_update": None,
        "searched_web": False,
        "metadata": {
            "correlations_flagged": ["NVDA and AMD moved together"],
            "tickers_analyzed": ["NVDA", "AMD"],
        },
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


def credential_cipher():
    key = base64.urlsafe_b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")
    return CredentialCipher(active_key=key, active_key_id="test-v1")


def credential_row(encrypted_api_key: str):
    return {
        "user_id": "user-1",
        "provider": "anthropic",
        "encrypted_api_key": encrypted_api_key,
        "encryption_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "last_validated_at": None,
    }


class DatabaseTests(unittest.TestCase):
    def test_store_and_resolve_user_api_key_never_writes_plaintext(self):
        cipher = credential_cipher()
        stored_row = credential_row(
            cipher.encrypt("sk-user-secret", user_id="user-1", provider="anthropic")
        )
        client = FakeClient({"user_api_keys": [[stored_row], [stored_row]]})

        metadata = database.store_user_api_key(
            "user-1",
            "ANTHROPIC",
            SecretStr("sk-user-secret"),
            client=client,
            cipher=cipher,
        )
        resolved = database.resolve_user_api_key(
            "user-1",
            "anthropic",
            client=client,
            cipher=cipher,
        )

        write_payload = client.tables[0].calls[0][1][0]
        resolver_selection = client.tables[1].calls[0][1][0]
        self.assertEqual(metadata.provider, "anthropic")
        self.assertNotIn("sk-user-secret", write_payload["encrypted_api_key"])
        self.assertEqual(resolved.get_secret_value(), "sk-user-secret")
        self.assertIn("encrypted_api_key", resolver_selection)
        self.assertNotEqual(resolver_selection, "*")

    def test_credential_metadata_query_never_selects_ciphertext(self):
        row = credential_row("unused")
        client = FakeClient({"user_api_keys": [[row]]})

        metadata = database.list_user_api_key_metadata("user-1", client=client)

        selected_columns = client.tables[0].calls[0][1][0]
        self.assertEqual(len(metadata), 1)
        self.assertNotIn("encrypted_api_key", selected_columns)

    def test_delete_user_api_key_scopes_user_and_provider(self):
        client = FakeClient({"user_api_keys": []})

        database.delete_user_api_key("user-1", "ANTHROPIC", client=client)

        self.assertIn(("eq", ("user_id", "user-1"), {}), client.tables[0].calls)
        self.assertIn(("eq", ("provider", "anthropic"), {}), client.tables[0].calls)

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

    def test_delete_user_scopes_delete_to_user_id(self):
        client = FakeClient({"users": []})

        database.delete_user("user-1", client=client)

        self.assertIn(("delete", (), {}), client.tables[0].calls)
        self.assertIn(("eq", ("user_id", "user-1"), {}), client.tables[0].calls)

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
        client = FakeClient({"updates": [hypothesis_row()]})
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
        self.assertIsInstance(returned, HypothesisOutput)
        self.assertTrue(returned.flagged)
        self.assertEqual(returned.recommended_next_scan_days, 1)
        self.assertEqual(payload["ticker"], "NVDA")
        self.assertEqual(payload["agent_type"], AgentType.HYPOTHESIS.value)
        self.assertEqual(payload["event_type"], EventType.HYPOTHESIS_SCAN.value)
        self.assertEqual(
            payload["metadata"],
            {"flagged": True, "recommended_next_scan_days": 1},
        )

    def test_unflagged_hypothesis_round_trip_allows_null_summary(self):
        client = FakeClient({"updates": [hypothesis_row(flagged=False)]})
        output = HypothesisOutput(
            ticker="nvda",
            user_id="user-1",
            summary=None,
            confidence=Confidence.MEDIUM,
            recommended_next_scan_days=3,
        )

        returned = database.insert_agent_output(output, client=client)

        payload = client.tables[0].calls[0][1][0]
        self.assertIsNone(payload.get("summary"))
        self.assertIsInstance(returned, HypothesisOutput)
        self.assertFalse(returned.flagged)

    def test_cross_portfolio_output_round_trip(self):
        client = FakeClient({"updates": [cross_portfolio_row()]})
        output = CrossPortfolioOutput(
            user_id="user-1",
            summary="Semiconductor exposure is concentrated.",
            correlations_flagged=["NVDA and AMD moved together"],
            tickers_analyzed=["nvda", "amd"],
        )

        returned = database.insert_agent_output(output, client=client)

        payload = client.tables[0].calls[0][1][0]
        self.assertIsInstance(returned, CrossPortfolioOutput)
        self.assertIsNone(payload["ticker"])
        self.assertIsNone(payload["event_type"])
        self.assertEqual(payload["agent_type"], AgentType.CROSS_PORTFOLIO.value)
        self.assertEqual(payload["metadata"]["tickers_analyzed"], ["NVDA", "AMD"])

    def test_unknown_agent_type_is_rejected(self):
        row = update_row()
        row["agent_type"] = "unknown"

        with self.assertRaises(database.DatabaseError):
            database._parse_agent_output_row(row)

    def test_hypothesis_missing_required_metadata_is_rejected(self):
        row = hypothesis_row()
        row["metadata"] = {"flagged": True}

        with self.assertRaises(ValidationError):
            database._parse_agent_output_row(row)

    def test_agent_type_must_match_event_type(self):
        row = update_row()
        row["agent_type"] = "motive"

        with self.assertRaises(database.DatabaseError):
            database._parse_agent_output_row(row)

    def test_plain_hypothesis_agent_output_is_rejected(self):
        output = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.HYPOTHESIS_SCAN,
            summary="Incomplete hypothesis output.",
            recommendation="Watch.",
            confidence=Confidence.LOW,
        )

        with self.assertRaises(database.DatabaseError):
            database._agent_output_payload(output)

    def test_list_updates_for_user_ticker_parses_agent_outputs(self):
        client = FakeClient({"updates": [[update_row(), hypothesis_row()]]})

        outputs = database.list_updates_for_user_ticker(
            "user-1",
            "nvda",
            client=client,
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].event_type, EventType.SHARP_MOVE)
        self.assertIsInstance(outputs[1], HypothesisOutput)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), client.tables[0].calls)

    def test_get_latest_update_filters_by_user_agent_and_ticker(self):
        client = FakeClient({"updates": [[update_row()]]})

        output = database.get_latest_update(
            "user-1",
            AgentType.SHARP_MOVE,
            "nvda",
            client=client,
        )

        self.assertIsInstance(output, AgentOutput)
        calls = client.tables[0].calls
        self.assertIn(("eq", ("user_id", "user-1"), {}), calls)
        self.assertIn(("eq", ("agent_type", "sharp_move"), {}), calls)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), calls)
        self.assertIn(("limit", (1,), {}), calls)

    def test_get_latest_update_returns_none_when_history_is_empty(self):
        client = FakeClient({"updates": []})

        output = database.get_latest_update(
            "user-1",
            AgentType.SCHEDULED_REVIEW,
            client=client,
        )

        self.assertIsNone(output)

    def test_list_recent_updates_applies_optional_filters(self):
        client = FakeClient({"updates": [[hypothesis_row()]]})
        since = datetime(2026, 1, 1, tzinfo=timezone.utc)

        outputs = database.list_recent_updates(
            "user-1",
            ticker="nvda",
            agent_type=AgentType.HYPOTHESIS,
            limit=5,
            since=since,
            client=client,
        )

        self.assertIsInstance(outputs[0], HypothesisOutput)
        calls = client.tables[0].calls
        self.assertIn(("eq", ("user_id", "user-1"), {}), calls)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), calls)
        self.assertIn(("eq", ("agent_type", "hypothesis"), {}), calls)
        self.assertIn(("gte", ("timestamp", since.isoformat()), {}), calls)
        self.assertIn(("limit", (5,), {}), calls)

    def test_history_queries_reject_invalid_limits_and_naive_since(self):
        client = FakeClient({})

        with self.assertRaises(ValueError):
            database.list_recent_updates("user-1", limit=0, client=client)
        with self.assertRaises(ValueError):
            database.list_recent_alerts("user-1", limit=101, client=client)
        with self.assertRaises(ValueError):
            database.list_recent_updates(
                "user-1",
                since=datetime(2026, 1, 1),
                client=client,
            )

        self.assertEqual(client.tables, [])

    def test_list_latest_portfolio_updates_returns_one_per_subscription(self):
        amd_subscription = subscription_row() | {"ticker": "AMD"}
        latest_nvda = update_row() | {
            "timestamp": "2026-01-03T00:00:00+00:00",
        }
        older_nvda = update_row() | {
            "timestamp": "2026-01-01T00:00:00+00:00",
        }
        latest_amd = update_row() | {
            "ticker": "AMD",
            "timestamp": "2026-01-02T00:00:00+00:00",
        }
        client = FakeClient(
            {
                "subscriptions": [[subscription_row(), amd_subscription]],
                "updates": [[latest_nvda, latest_amd, older_nvda]],
            }
        )

        outputs = database.list_latest_portfolio_updates("user-1", client=client)

        self.assertEqual([output.ticker for output in outputs], ["NVDA", "AMD"])
        update_calls = client.tables[1].calls
        self.assertIn(("eq", ("user_id", "user-1"), {}), update_calls)
        self.assertIn(("in_", ("ticker", ["AMD", "NVDA"]), {}), update_calls)

    def test_list_latest_portfolio_updates_skips_query_without_subscriptions(self):
        client = FakeClient({"subscriptions": []})

        outputs = database.list_latest_portfolio_updates("user-1", client=client)

        self.assertEqual(outputs, [])
        self.assertEqual(len(client.tables), 1)

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

    def test_list_recent_alerts_applies_optional_filters(self):
        client = FakeClient({"alerts": [[alert_row()]]})
        since = datetime(2025, 12, 31, 19, tzinfo=timezone.utc)

        alerts = database.list_recent_alerts(
            "user-1",
            ticker="nvda",
            alert_type=AlertType.SHARP_MOVE,
            limit=7,
            since=since,
            client=client,
        )

        self.assertEqual(alerts, [Alert.model_validate(alert_row())])
        calls = client.tables[0].calls
        self.assertIn(("eq", ("user_id", "user-1"), {}), calls)
        self.assertIn(("eq", ("ticker", "NVDA"), {}), calls)
        self.assertIn(("eq", ("alert_type", "sharp_move"), {}), calls)
        self.assertIn(("gte", ("timestamp", since.isoformat()), {}), calls)
        self.assertIn(("limit", (7,), {}), calls)

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
