from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from models.schemas import (
    AgentContext,
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
    PortfolioContext,
    ResearchSource,
    Subscription,
    TickerRegistry,
    UpdateInterval,
    User,
)


def make_subscription(user_id: str = "user-1", ticker: str = "NVDA") -> Subscription:
    return Subscription(
        user_id=user_id,
        ticker=ticker,
        avg_price=100.0,
        shares=2.0,
        motive=Motive.HOLDING,
        update_interval=UpdateInterval.DAILY,
    )


def make_datapoint(ticker: str = "NVDA") -> OHLCVPoint:
    return OHLCVPoint(
        ticker=ticker,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        open=100.0,
        high=105.0,
        low=99.0,
        close=104.0,
        volume=1_000_000,
    )


def make_agent_context(user_id: str = "user-1", ticker: str = "NVDA") -> AgentContext:
    subscription = make_subscription(user_id=user_id, ticker=ticker)
    return AgentContext(
        ticker=ticker,
        datapoints=[make_datapoint(ticker=ticker)],
        subscription=subscription,
        event_type=EventType.SCHEDULED_UPDATE,
        current_price=110.0,
        unrealized_pnl=20.0,
        unrealized_pnl_pct=0.10,
    )


class SchemaTests(unittest.TestCase):
    def test_agent_types_identify_all_planned_agents(self):
        self.assertEqual(
            {agent_type.value for agent_type in AgentType},
            {
                "scheduled_review",
                "sharp_move",
                "motive",
                "hypothesis",
                "cross_portfolio",
            },
        )

    def test_ohlcv_rejects_negative_prices_and_volume(self):
        with self.assertRaises(ValidationError):
            OHLCVPoint(
                ticker="NVDA",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=-1.0,
                close=104.0,
                volume=1_000_000,
            )

    def test_subscription_defaults_and_computed_position_value(self):
        subscription = make_subscription()

        self.assertEqual(subscription.sharp_move_threshold, 0.01)
        self.assertEqual(subscription.position_value, 200.0)

    def test_subscription_validates_threshold_bounds(self):
        with self.assertRaises(ValidationError):
            Subscription(
                user_id="user-1",
                ticker="NVDA",
                avg_price=100.0,
                shares=2.0,
                motive=Motive.HOLDING,
                update_interval=UpdateInterval.DAILY,
                sharp_move_threshold=0.0005,
            )

        with self.assertRaises(ValidationError):
            Subscription(
                user_id="user-1",
                ticker="NVDA",
                avg_price=100.0,
                shares=2.0,
                motive=Motive.HOLDING,
                update_interval=UpdateInterval.DAILY,
                sharp_move_threshold=0.51,
            )

    def test_ticker_registry_subscriber_management(self):
        registry = TickerRegistry(ticker="NVDA")
        first = make_subscription(user_id="user-1")
        replacement = make_subscription(user_id="user-1")
        second = make_subscription(user_id="user-2")

        registry.add_subscriber(first)
        registry.add_subscriber(second)
        registry.add_subscriber(replacement)

        self.assertTrue(registry.is_active)
        self.assertEqual(registry.subscriber_count, 2)
        self.assertEqual(registry.get_subscriber("user-1"), replacement)

        registry.remove_subscriber("user-1")

        self.assertIsNone(registry.get_subscriber("user-1"))
        self.assertEqual(registry.subscriber_count, 1)

    def test_mutable_defaults_are_not_shared_between_instances(self):
        first_user = User(user_id="user-1", whatsapp_number="whatsapp:+10000000000")
        second_user = User(user_id="user-2", whatsapp_number="whatsapp:+10000000001")
        first_user.preferences["theme"] = "dark"

        first_alert = Alert(
            user_id="user-1",
            ticker="NVDA",
            alert_type=AlertType.SHARP_MOVE,
            message="Price moved sharply.",
        )
        second_alert = Alert(
            user_id="user-2",
            ticker="AAPL",
            alert_type=AlertType.SCHEDULED,
            message="Scheduled update.",
        )
        first_alert.trigger_details["move_pct"] = 0.03

        self.assertEqual(second_user.preferences, {})
        self.assertEqual(second_alert.trigger_details, {})

        first_output = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.SHARP_MOVE,
            summary="Summary",
            recommendation="Monitor",
            confidence=Confidence.MEDIUM,
        )
        second_output = first_output.model_copy(deep=True)
        first_output.sources.append(
            ResearchSource(title="Source", url="https://example.com")
        )

        self.assertEqual(second_output.sources, [])

    def test_agent_output_validates_web_search_request_count(self):
        with self.assertRaises(ValidationError):
            AgentOutput(
                ticker="NVDA",
                user_id="user-1",
                event_type=EventType.SHARP_MOVE,
                summary="Summary",
                recommendation="Monitor",
                confidence=Confidence.MEDIUM,
                web_search_requests=-1,
            )

    def test_portfolio_context_properties(self):
        nvda = make_agent_context(user_id="user-1", ticker="NVDA")
        aapl = make_agent_context(user_id="user-1", ticker="AAPL")

        portfolio = PortfolioContext(user_id="user-1", positions=[nvda, aapl])

        self.assertEqual(portfolio.tickers, ["NVDA", "AAPL"])
        self.assertEqual(portfolio.total_portfolio_value, 400.0)
        self.assertEqual(portfolio.total_unrealized_pnl, 40.0)

    def test_dynamic_timestamps_are_timezone_aware_and_per_instance(self):
        first = CrossPortfolioOutput(user_id="user-1", summary="First")
        second = CrossPortfolioOutput(user_id="user-1", summary="Second")

        self.assertIsNot(first.timestamp, second.timestamp)
        self.assertIsNotNone(first.timestamp.tzinfo)
        self.assertIsNotNone(second.timestamp.tzinfo)

    def test_hypothesis_output_extends_agent_output_shape(self):
        output = HypothesisOutput(
            ticker="NVDA",
            user_id="user-1",
            summary=None,
            confidence=Confidence.MEDIUM,
            recommended_next_scan_days=3,
        )

        self.assertEqual(output.event_type, EventType.HYPOTHESIS_SCAN)
        self.assertFalse(output.flagged)
        self.assertEqual(output.recommendation, "")


if __name__ == "__main__":
    unittest.main()
