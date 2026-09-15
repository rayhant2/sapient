import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from agents.core import (
    AgentExecutionError,
    CrossPortfolioAnalysisDraft,
    build_portfolio_agent_state,
)
from agents.cross_portfolio import (
    _portfolio_payload,
    build_cross_portfolio_graph,
    run_cross_portfolio,
)
from models.schemas import (
    AgentContext,
    AgentOutput,
    Confidence,
    CrossPortfolioOutput,
    EventType,
    Motive,
    OHLCVPoint,
    PortfolioContext,
    Subscription,
    UpdateInterval,
)


def position(ticker: str, changes: list[float], shares: float) -> AgentContext:
    start = datetime(2026, 9, 14, 13, 30, tzinfo=timezone.utc)
    closes = [100.0]
    for change in changes:
        closes.append(closes[-1] * (1 + change))
    points = [
        OHLCVPoint(
            ticker=ticker,
            timestamp=start + timedelta(minutes=15 * index),
            open=close,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=1000.0 + index,
        )
        for index, close in enumerate(closes)
    ]
    current = closes[-1]
    return AgentContext(
        ticker=ticker,
        datapoints=points,
        subscription=Subscription(
            user_id="user-1",
            ticker=ticker,
            avg_price=100.0,
            shares=shares,
            motive=Motive.HOLDING,
            update_interval=UpdateInterval.DAILY,
        ),
        event_type=EventType.SCHEDULED_UPDATE,
        current_price=current,
        unrealized_pnl=(current - 100.0) * shares,
        unrealized_pnl_pct=(current - 100.0) / 100.0,
    )


def portfolio_context() -> PortfolioContext:
    changes = [0.01, -0.005, 0.008, -0.003, 0.012, -0.004]
    nvda = position("NVDA", changes, 8.0)
    amd = position("AMD", changes, 2.0)
    latest = AgentOutput(
        ticker="NVDA",
        user_id="user-1",
        event_type=EventType.SCHEDULED_UPDATE,
        summary="NVDA remains constructive.",
        recommendation="Monitor concentration.",
        confidence=Confidence.MEDIUM,
    )
    return PortfolioContext(
        user_id="user-1",
        positions=[nvda, amd],
        latest_outputs=[latest],
    )


class CrossPortfolioPayloadTests(unittest.TestCase):
    def test_payload_calculates_weights_and_overlapping_correlation(self):
        payload = _portfolio_payload(portfolio_context())

        self.assertEqual(payload["portfolio"]["ticker_count"], 2)
        self.assertGreater(payload["portfolio"]["largest_position_weight"], 0.7)
        self.assertEqual(len(payload["overlapping_return_correlations"]), 1)
        correlation = payload["overlapping_return_correlations"][0]
        self.assertAlmostEqual(correlation["correlation"], 1.0)
        self.assertEqual(correlation["overlapping_returns"], 6)
        self.assertEqual(len(payload["current_cycle_outputs"]), 1)
        self.assertNotIn("user_id", str(payload))

    def test_payload_omits_correlation_for_constant_returns(self):
        flat_changes = [0.0] * 6
        context = PortfolioContext(
            user_id="user-1",
            positions=[
                position("NVDA", flat_changes, 2.0),
                position("AMD", flat_changes, 2.0),
            ],
        )

        payload = _portfolio_payload(context)

        self.assertEqual(payload["overlapping_return_correlations"], [])

    def test_payload_filters_outputs_outside_trusted_portfolio(self):
        context = portfolio_context()
        context.latest_outputs.append(
            context.latest_outputs[0].model_copy(
                update={"ticker": "AAPL", "user_id": "another-user"}
            )
        )

        payload = _portfolio_payload(context)

        self.assertEqual(
            [item["ticker"] for item in payload["current_cycle_outputs"]],
            ["NVDA"],
        )

    def test_payload_rejects_duplicate_tickers(self):
        context = portfolio_context()
        context.positions.append(context.positions[0])

        with self.assertRaisesRegex(AgentExecutionError, "one position per ticker"):
            _portfolio_payload(context)


class CrossPortfolioGraphTests(unittest.TestCase):
    def test_graph_produces_trusted_cross_portfolio_output(self):
        context = portfolio_context()
        model = MagicMock(name="model")
        output_runnable = MagicMock(name="output-runnable")
        model.with_structured_output.return_value = output_runnable
        output_runnable.invoke.return_value = CrossPortfolioAnalysisDraft(
            summary="NVDA dominates tracked value and moves closely with AMD.",
            correlations_flagged=["NVDA and AMD returns are highly correlated."],
        )

        graph = build_cross_portfolio_graph(model)
        result = graph.invoke(build_portfolio_agent_state(context))

        output = result["output"]
        self.assertIsInstance(output, CrossPortfolioOutput)
        self.assertEqual(output.user_id, "user-1")
        self.assertEqual(output.tickers_analyzed, ["NVDA", "AMD"])
        self.assertEqual(len(output.correlations_flagged), 1)


class CrossPortfolioRunnerTests(unittest.TestCase):
    def test_runner_uses_shared_execution_boundary(self):
        context = portfolio_context()
        database_client = MagicMock(name="database-client")
        credential_client = MagicMock(name="credential-client")
        cipher = MagicMock(name="cipher")
        expected = CrossPortfolioOutput(
            user_id="user-1",
            summary="The tracked portfolio is concentrated.",
            correlations_flagged=[],
            tickers_analyzed=["NVDA", "AMD"],
        )

        with patch(
            "agents.cross_portfolio.execute_cross_portfolio_agent",
            return_value=expected,
        ) as execute:
            result = run_cross_portfolio(
                context,
                database_client=database_client,
                credential_client=credential_client,
                cipher=cipher,
            )

        self.assertEqual(result, expected)
        execute.assert_called_once_with(
            context,
            build_cross_portfolio_graph,
            database_client=database_client,
            credential_client=credential_client,
            cipher=cipher,
        )


if __name__ == "__main__":
    unittest.main()
