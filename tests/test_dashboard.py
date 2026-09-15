from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, patch

from dashboard.services import (
    DashboardPosition,
    load_dashboard_data,
    remove_subscription,
    resolve_mvp_user_id,
    save_subscription,
)
from models.schemas import (
    AgentOutput,
    Confidence,
    EventType,
    Motive,
    OHLCVPoint,
    Subscription,
    Ticker,
    UpdateInterval,
    User,
)


def subscription() -> Subscription:
    return Subscription(
        user_id="user-1",
        ticker="NVDA",
        avg_price=100.0,
        shares=2.0,
        motive=Motive.HOLDING,
        update_interval=UpdateInterval.DAILY,
        sharp_move_threshold=0.01,
    )


def points() -> list[OHLCVPoint]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        OHLCVPoint(
            ticker="NVDA",
            timestamp=start + timedelta(minutes=15 * index),
            open=close - 1,
            high=close + 1,
            low=close - 2,
            close=close,
            volume=1000 + index,
        )
        for index, close in enumerate((105.0, 110.0))
    ]


def output() -> AgentOutput:
    return AgentOutput(
        ticker="NVDA",
        user_id="user-1",
        event_type=EventType.SCHEDULED_UPDATE,
        summary="Position remains stable.",
        recommendation="Continue monitoring.",
        confidence=Confidence.MEDIUM,
        price_at_update=110.0,
    )


class DashboardServiceTests(unittest.TestCase):
    def test_position_calculates_value_and_return(self):
        position = DashboardPosition(
            subscription=subscription(),
            current_price=110.0,
            datapoints=points(),
            latest_output=output(),
        )

        self.assertEqual(position.cost_value, 200.0)
        self.assertEqual(position.current_value, 220.0)
        self.assertEqual(position.unrealized_pnl, 20.0)
        self.assertAlmostEqual(position.unrealized_pnl_pct, 0.10)

    @patch("dashboard.services.list_users")
    def test_resolve_mvp_user_uses_only_database_user(self, list_users):
        list_users.return_value = [
            User(user_id="user-1", whatsapp_number="whatsapp:+15551234567")
        ]

        self.assertEqual(resolve_mvp_user_id(), "user-1")
        list_users.assert_called_once_with(limit=2, client=None)

    @patch("dashboard.services.list_users")
    def test_resolve_mvp_user_prefers_configured_override(self, list_users):
        self.assertEqual(resolve_mvp_user_id(" user-2 "), "user-2")
        list_users.assert_not_called()

    @patch("dashboard.services.list_users")
    def test_resolve_mvp_user_rejects_ambiguous_database(self, list_users):
        list_users.return_value = [
            User(user_id="user-1", whatsapp_number="whatsapp:+15551234567"),
            User(user_id="user-2", whatsapp_number="whatsapp:+15557654321"),
        ]

        with self.assertRaisesRegex(ValueError, "MVP_USER_ID"):
            resolve_mvp_user_id()

    def test_load_dashboard_data_builds_position_view(self):
        user = User(
            user_id="user-1",
            whatsapp_number="whatsapp:+15551234567",
        )
        client = MagicMock()
        latest = output()
        alerts = []
        with (
            patch("dashboard.services.get_user", return_value=user),
            patch(
                "dashboard.services.list_subscriptions_for_user",
                return_value=[subscription()],
            ),
            patch(
                "dashboard.services.list_latest_portfolio_updates",
                return_value=[latest],
            ),
            patch(
                "dashboard.services.get_latest_ticker_data",
                return_value=points(),
            ) as get_points,
            patch(
                "dashboard.services.list_updates_for_user",
                return_value=[latest],
            ),
            patch(
                "dashboard.services.list_alerts_for_user",
                return_value=alerts,
            ),
        ):
            data = load_dashboard_data("user-1", client=client)

        self.assertEqual(data.user, user)
        self.assertEqual(len(data.positions), 1)
        self.assertEqual(data.positions[0].current_price, 110.0)
        self.assertEqual(data.positions[0].latest_output, latest)
        self.assertEqual(data.updates, [latest])
        get_points.assert_called_once_with("NVDA", limit=150, client=client)

    def test_save_subscription_creates_new_ticker_and_normalizes_symbol(self):
        client = MagicMock()
        saved = subscription()
        with (
            patch("dashboard.services.get_ticker", return_value=None),
            patch("dashboard.services.upsert_ticker") as upsert_ticker,
            patch(
                "dashboard.services.upsert_subscription",
                return_value=saved,
            ) as upsert_subscription,
        ):
            result = save_subscription(
                "user-1",
                " nvda ",
                avg_price=100.0,
                shares=2.0,
                motive=Motive.HOLDING,
                update_interval=UpdateInterval.DAILY,
                sharp_move_threshold=0.01,
                client=client,
            )

        self.assertEqual(result, saved)
        self.assertEqual(upsert_ticker.call_args.args[0], Ticker(ticker="NVDA"))
        self.assertEqual(upsert_ticker.call_args.kwargs, {"client": client})
        written = upsert_subscription.call_args.args[0]
        self.assertEqual(written.ticker, "NVDA")
        self.assertEqual(upsert_subscription.call_args.kwargs, {"client": client})

    @patch("dashboard.services.delete_subscription")
    def test_remove_subscription_delegates_to_database(self, delete_subscription):
        client = MagicMock()

        remove_subscription("user-1", "NVDA", client=client)

        delete_subscription.assert_called_once_with(
            "user-1", "NVDA", client=client
        )


if __name__ == "__main__":
    unittest.main()
