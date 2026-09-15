import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from core.event_bus import EventBus
from core.runtime import AgentRuntime, RuntimeConfigurationError
from models.schemas import (
    AgentContext,
    AgentOutput,
    Confidence,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    Motive,
    OHLCVPoint,
    Subscription,
    UpdateInterval,
)


def subscription(
    ticker: str,
    *,
    interval: UpdateInterval = UpdateInterval.DAILY,
) -> Subscription:
    return Subscription(
        user_id="user-1",
        ticker=ticker,
        avg_price=100.0,
        shares=2.0,
        motive=Motive.HOLDING,
        update_interval=interval,
    )


def points(ticker: str) -> list[OHLCVPoint]:
    return [
        OHLCVPoint(
            ticker=ticker,
            timestamp=datetime(2026, 9, 14, 19, 30, tzinfo=timezone.utc),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000.0,
        ),
        OHLCVPoint(
            ticker=ticker,
            timestamp=datetime(2026, 9, 14, 19, 45, tzinfo=timezone.utc),
            open=100.0,
            high=102.0,
            low=99.5,
            close=101.0,
            volume=1200.0,
        ),
    ]


def context(ticker: str, event_type: EventType) -> AgentContext:
    ticker_points = points(ticker)
    return AgentContext(
        ticker=ticker,
        datapoints=ticker_points,
        subscription=subscription(ticker),
        event_type=event_type,
        current_price=101.0,
        unrealized_pnl=2.0,
        unrealized_pnl_pct=0.01,
    )


def ticker_output(ticker: str, event_type: EventType) -> AgentOutput:
    return AgentOutput(
        ticker=ticker,
        user_id="user-1",
        event_type=event_type,
        summary=f"{ticker} summary.",
        recommendation="Continue monitoring.",
        confidence=Confidence.MEDIUM,
        price_at_update=101.0,
    )


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus(
            datapoint_loader=lambda ticker, _limit: points(ticker)
        )
        self.bus.refresh_subscription(subscription("NVDA"))
        self.scheduler = MagicMock(name="scheduler")
        self.scheduler.event_bus = self.bus
        self.sink = MagicMock(name="output-sink")
        self.runtime = AgentRuntime(
            self.bus,
            self.scheduler,
            output_sink=self.sink,
            now_provider=lambda: datetime(
                2026,
                9,
                14,
                20,
                15,
                tzinfo=timezone.utc,
            ),
        )

    def test_runtime_requires_scheduler_to_share_event_bus(self):
        scheduler = MagicMock()
        scheduler.event_bus = EventBus()

        with self.assertRaises(RuntimeConfigurationError):
            AgentRuntime(self.bus, scheduler)

    def test_start_registers_handlers_before_starting_scheduler(self):
        self.runtime.start()

        self.assertEqual(set(self.bus._handlers), set(EventType))
        self.scheduler.start.assert_called_once_with(load_registry=True)

    def test_registered_handlers_run_agents_and_forward_outputs(self):
        sharp = ticker_output("NVDA", EventType.SHARP_MOVE)
        motive = ticker_output("NVDA", EventType.MOTIVE_CHECK)
        hypothesis = HypothesisOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.HYPOTHESIS_SCAN,
            summary=None,
            recommendation="",
            confidence=Confidence.LOW,
            price_at_update=101.0,
            flagged=False,
            recommended_next_scan_days=3,
        )
        self.runtime.register_handlers()

        with (
            patch("core.runtime.run_sharp_move", return_value=sharp) as run_sharp,
            patch(
                "core.runtime.run_motive_reassessment",
                return_value=motive,
            ) as run_motive,
            patch(
                "core.runtime.run_hypothesis_scan",
                return_value=hypothesis,
            ) as run_hypothesis,
        ):
            self.bus.emit(EventType.SHARP_MOVE, "NVDA")
            self.bus.emit(EventType.MOTIVE_CHECK, "NVDA")
            self.bus.emit(EventType.HYPOTHESIS_SCAN, "NVDA")

        run_sharp.assert_called_once()
        run_motive.assert_called_once()
        run_hypothesis.assert_called_once()
        self.assertIs(
            run_hypothesis.call_args.kwargs["schedule_next"],
            self.scheduler.schedule_hypothesis_scan,
        )
        self.assertEqual(
            [call.args[0] for call in self.sink.call_args_list],
            [sharp, motive, hypothesis],
        )

    def test_scheduled_cycle_runs_cross_portfolio_once_when_complete(self):
        self.bus.refresh_subscription(subscription("AMD"))
        nvda_context = context("NVDA", EventType.SCHEDULED_UPDATE)
        amd_context = context("AMD", EventType.SCHEDULED_UPDATE)
        nvda_output = ticker_output("NVDA", EventType.SCHEDULED_UPDATE)
        amd_output = ticker_output("AMD", EventType.SCHEDULED_UPDATE)
        portfolio_output = CrossPortfolioOutput(
            user_id="user-1",
            summary="The portfolio has shared semiconductor exposure.",
            correlations_flagged=["NVDA and AMD moved together."],
            tickers_analyzed=["AMD", "NVDA"],
        )

        with (
            patch(
                "core.runtime.run_scheduled_review",
                side_effect=[nvda_output, amd_output, amd_output],
            ),
            patch(
                "core.runtime.run_cross_portfolio",
                return_value=portfolio_output,
            ) as run_portfolio,
        ):
            self.runtime._handle_scheduled_update(nvda_context)
            run_portfolio.assert_not_called()
            self.runtime._handle_scheduled_update(amd_context)
            self.runtime._handle_scheduled_update(amd_context)

        run_portfolio.assert_called_once()
        portfolio_context = run_portfolio.call_args.args[0]
        self.assertEqual(portfolio_context.tickers, ["AMD", "NVDA"])
        self.assertEqual(
            [output.ticker for output in portfolio_context.latest_outputs],
            ["AMD", "NVDA"],
        )
        self.assertEqual(self.sink.call_count, 4)
        self.assertEqual(self.sink.call_args_list[2].args[0], portfolio_output)

    def test_cross_portfolio_failure_allows_cycle_retry(self):
        nvda_context = context("NVDA", EventType.SCHEDULED_UPDATE)
        nvda_output = ticker_output("NVDA", EventType.SCHEDULED_UPDATE)
        portfolio_output = CrossPortfolioOutput(
            user_id="user-1",
            summary="Portfolio review completed.",
            tickers_analyzed=["NVDA"],
        )

        with (
            patch(
                "core.runtime.run_scheduled_review",
                return_value=nvda_output,
            ),
            patch(
                "core.runtime.run_cross_portfolio",
                side_effect=[RuntimeError("temporary"), portfolio_output],
            ) as run_portfolio,
        ):
            with self.assertRaises(RuntimeError):
                self.runtime._handle_scheduled_update(nvda_context)
            self.runtime._handle_scheduled_update(nvda_context)

        self.assertEqual(run_portfolio.call_count, 2)

    def test_friday_cycle_waits_for_daily_and_weekly_subscriptions(self):
        self.bus.refresh_subscription(
            subscription("AMD", interval=UpdateInterval.WEEKLY)
        )
        friday_runtime = AgentRuntime(
            self.bus,
            self.scheduler,
            output_sink=self.sink,
            now_provider=lambda: datetime(
                2026,
                9,
                18,
                20,
                15,
                tzinfo=timezone.utc,
            ),
        )
        nvda_context = context("NVDA", EventType.SCHEDULED_UPDATE)
        amd_context = context("AMD", EventType.SCHEDULED_UPDATE).model_copy(
            update={"subscription": subscription("AMD", interval=UpdateInterval.WEEKLY)}
        )
        nvda_output = ticker_output("NVDA", EventType.SCHEDULED_UPDATE)
        amd_output = ticker_output("AMD", EventType.SCHEDULED_UPDATE)

        with (
            patch(
                "core.runtime.run_scheduled_review",
                side_effect=[nvda_output, amd_output],
            ),
            patch("core.runtime.run_cross_portfolio") as run_portfolio,
        ):
            friday_runtime._handle_scheduled_update(nvda_context)
            run_portfolio.assert_not_called()
            friday_runtime._handle_scheduled_update(amd_context)

        run_portfolio.assert_called_once()


if __name__ == "__main__":
    unittest.main()
