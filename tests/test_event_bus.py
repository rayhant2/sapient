from datetime import datetime, timezone
import unittest
from unittest.mock import Mock

from core.event_bus import (
    EventBus,
    HandlerNotRegisteredError,
    MissingTickerDataError,
    UnknownTickerError,
)
from models.schemas import (
    AgentContext,
    EventType,
    Motive,
    OHLCVPoint,
    Subscription,
    Ticker,
    UpdateInterval,
)


def subscription(
    user_id: str,
    *,
    ticker: str = "NVDA",
    avg_price: float = 100.0,
    shares: float = 2.0,
) -> Subscription:
    return Subscription(
        user_id=user_id,
        ticker=ticker,
        avg_price=avg_price,
        shares=shares,
        motive=Motive.HOLDING,
        update_interval=UpdateInterval.DAILY,
    )


def datapoints(ticker: str = "NVDA") -> list[OHLCVPoint]:
    return [
        OHLCVPoint(
            ticker=ticker,
            timestamp=datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc),
            open=105.0,
            high=108.0,
            low=104.0,
            close=106.0,
            volume=1000.0,
        ),
        OHLCVPoint(
            ticker=ticker,
            timestamp=datetime(2026, 7, 20, 14, 15, tzinfo=timezone.utc),
            open=106.0,
            high=111.0,
            low=105.0,
            close=110.0,
            volume=1200.0,
        ),
    ]


class EventBusTests(unittest.TestCase):
    def test_load_registry_groups_subscriptions_and_ticker_state(self):
        fetched_at = datetime(2026, 7, 20, 14, 15, tzinfo=timezone.utc)
        ticker_loader = Mock(
            return_value=[
                Ticker(
                    ticker="NVDA",
                    current_price=110.0,
                    last_fetched=fetched_at,
                ),
                Ticker(ticker="AAPL", current_price=210.0),
            ]
        )
        subscription_loader = Mock(
            return_value=[
                subscription("user-1", ticker="nvda"),
                subscription("user-2", ticker="NVDA"),
            ]
        )
        bus = EventBus(
            ticker_loader=ticker_loader,
            subscription_loader=subscription_loader,
        )

        registry = bus.load_registry()

        self.assertEqual(set(registry), {"NVDA"})
        self.assertEqual(registry["NVDA"].subscriber_count, 2)
        self.assertEqual(registry["NVDA"].current_price, 110.0)
        self.assertEqual(registry["NVDA"].last_fetched, fetched_at)
        ticker_loader.assert_called_once_with()
        subscription_loader.assert_called_once_with()

    def test_refresh_subscription_adds_and_replaces_user(self):
        bus = EventBus()

        bus.refresh_subscription(subscription("user-1", avg_price=100.0))
        updated = bus.refresh_subscription(subscription("user-1", avg_price=120.0))

        self.assertEqual(updated.subscriber_count, 1)
        self.assertEqual(updated.get_subscriber("user-1").avg_price, 120.0)

    def test_remove_subscription_removes_inactive_registry(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1"))

        removed = bus.remove_subscription("user-1", "nvda")

        self.assertTrue(removed)
        self.assertNotIn("NVDA", bus.registry)
        self.assertFalse(bus.remove_subscription("user-1", "NVDA"))

    def test_update_ticker_state_normalizes_symbol(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1"))
        fetched_at = datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)

        registry = bus.update_ticker_state(
            "nvda",
            current_price=115.0,
            last_fetched=fetched_at,
        )

        self.assertEqual(registry.current_price, 115.0)
        self.assertEqual(registry.last_fetched, fetched_at)

    def test_emit_loads_history_once_and_builds_personal_contexts(self):
        datapoint_loader = Mock(return_value=list(reversed(datapoints())))
        bus = EventBus(datapoint_loader=datapoint_loader)
        bus.refresh_subscription(
            subscription("user-1", avg_price=100.0, shares=2.0)
        )
        bus.refresh_subscription(
            subscription("user-2", avg_price=120.0, shares=3.0)
        )
        received: list[AgentContext] = []
        bus.register_handler(EventType.SCHEDULED_UPDATE, received.append)

        result = bus.emit(EventType.SCHEDULED_UPDATE, "nvda")

        self.assertEqual(result.attempted, 2)
        self.assertEqual(result.succeeded, 2)
        self.assertEqual(result.failed, 0)
        datapoint_loader.assert_called_once_with("NVDA", 150)
        self.assertEqual(len(received), 2)
        self.assertTrue(all(context.current_price == 110.0 for context in received))
        self.assertTrue(
            all(
                context.datapoints[0].timestamp < context.datapoints[-1].timestamp
                for context in received
            )
        )
        contexts = {context.subscription.user_id: context for context in received}
        self.assertEqual(contexts["user-1"].unrealized_pnl, 20.0)
        self.assertAlmostEqual(contexts["user-1"].unrealized_pnl_pct, 0.10)
        self.assertEqual(contexts["user-2"].unrealized_pnl, -30.0)
        self.assertAlmostEqual(contexts["user-2"].unrealized_pnl_pct, -1 / 12)
        self.assertEqual(bus.registry["NVDA"].current_price, 110.0)

    def test_emit_can_target_specific_users(self):
        datapoint_loader = Mock(return_value=datapoints())
        bus = EventBus(datapoint_loader=datapoint_loader)
        bus.refresh_subscription(subscription("user-1"))
        bus.refresh_subscription(subscription("user-2"))
        received: list[AgentContext] = []
        bus.register_handler(EventType.SHARP_MOVE, received.append)

        result = bus.emit(
            EventType.SHARP_MOVE,
            "NVDA",
            target_user_ids={"user-2"},
        )

        self.assertEqual(result.attempted, 1)
        self.assertEqual(received[0].subscription.user_id, "user-2")

    def test_emit_skips_data_load_when_no_targets_match(self):
        datapoint_loader = Mock(return_value=datapoints())
        bus = EventBus(datapoint_loader=datapoint_loader)
        bus.refresh_subscription(subscription("user-1"))
        bus.register_handler(EventType.SHARP_MOVE, Mock())

        result = bus.emit(
            EventType.SHARP_MOVE,
            "NVDA",
            target_user_ids={"missing-user"},
        )

        self.assertEqual(result.attempted, 0)
        datapoint_loader.assert_not_called()

    def test_emit_isolates_handler_failures(self):
        bus = EventBus(datapoint_loader=Mock(return_value=datapoints()))
        bus.refresh_subscription(subscription("user-1"))
        bus.refresh_subscription(subscription("user-2"))
        received: list[str] = []

        def handler(context: AgentContext) -> None:
            if context.subscription.user_id == "user-1":
                raise RuntimeError("job failed")
            received.append(context.subscription.user_id)

        bus.register_handler(EventType.MOTIVE_CHECK, handler)

        result = bus.emit(EventType.MOTIVE_CHECK, "NVDA")

        self.assertEqual(result.attempted, 2)
        self.assertEqual(result.succeeded, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.failures[0].user_id, "user-1")
        self.assertEqual(result.failures[0].error_type, "RuntimeError")
        self.assertEqual(received, ["user-2"])

    def test_emit_requires_registered_handler(self):
        bus = EventBus(datapoint_loader=Mock(return_value=datapoints()))
        bus.refresh_subscription(subscription("user-1"))

        with self.assertRaises(HandlerNotRegisteredError):
            bus.emit(EventType.HYPOTHESIS_SCAN, "NVDA")

    def test_emit_requires_price_history(self):
        bus = EventBus(datapoint_loader=Mock(return_value=[]))
        bus.refresh_subscription(subscription("user-1"))
        bus.register_handler(EventType.SCHEDULED_UPDATE, Mock())

        with self.assertRaises(MissingTickerDataError):
            bus.emit(EventType.SCHEDULED_UPDATE, "NVDA")

    def test_unknown_ticker_raises_clear_error(self):
        bus = EventBus()

        with self.assertRaises(UnknownTickerError):
            bus.get_registry("nvda")

    def test_unregister_handler_reports_whether_handler_existed(self):
        bus = EventBus()
        bus.register_handler(EventType.SHARP_MOVE, Mock())

        self.assertTrue(bus.unregister_handler(EventType.SHARP_MOVE))
        self.assertFalse(bus.unregister_handler(EventType.SHARP_MOVE))


if __name__ == "__main__":
    unittest.main()
