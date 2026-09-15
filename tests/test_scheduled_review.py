import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agents.core import (
    TickerAnalysisDraft,
    WebResearchError,
    WebResearchFocus,
    WebResearchResult,
    build_ticker_agent_state,
)
from agents.scheduled_review import (
    ScheduledResearchDecision,
    _review_payload,
    build_scheduled_review_graph,
    run_scheduled_review,
)
from models.schemas import (
    AgentContext,
    AgentOutput,
    AgentType,
    Confidence,
    EventType,
    Motive,
    OHLCVPoint,
    ResearchSource,
    Subscription,
    UpdateInterval,
)


def scheduled_context(candle_count: int = 16) -> AgentContext:
    start = datetime(2026, 9, 14, 14, 30, tzinfo=timezone.utc)
    points = []
    for index in range(candle_count):
        close = 100.0 + index
        points.append(
            OHLCVPoint(
                ticker="NVDA",
                timestamp=start + timedelta(minutes=15 * index),
                open=close - 0.5,
                high=close + 1.0,
                low=close - 1.0,
                close=close,
                volume=1000.0 + (index * 100.0),
            )
        )
    return AgentContext(
        ticker="NVDA",
        datapoints=points,
        subscription=Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=95.0,
            shares=3.0,
            motive=Motive.HOLDING,
            update_interval=UpdateInterval.DAILY,
        ),
        event_type=EventType.SCHEDULED_UPDATE,
        current_price=points[-1].close,
        unrealized_pnl=(points[-1].close - 95.0) * 3.0,
        unrealized_pnl_pct=(points[-1].close - 95.0) / 95.0,
    )


class ScheduledResearchDecisionTests(unittest.TestCase):
    def test_search_requires_focus(self):
        with self.assertRaises(ValidationError):
            ScheduledResearchDecision(
                search_web=True,
                reason="A catalyst may explain the move.",
            )

    def test_no_search_rejects_focus(self):
        with self.assertRaises(ValidationError):
            ScheduledResearchDecision(
                search_web=False,
                focus=WebResearchFocus.COMPANY_NEWS,
                reason="No external context is needed.",
            )


class ScheduledReviewPayloadTests(unittest.TestCase):
    def test_payload_summarizes_full_window_and_bounds_recent_candles(self):
        payload = _review_payload(build_ticker_agent_state(scheduled_context()))

        self.assertEqual(payload["market_window"]["candle_count"], 16)
        self.assertAlmostEqual(payload["market_window"]["window_change_pct"], 0.15)
        self.assertEqual(len(payload["recent_candles"]), 12)
        self.assertEqual(payload["recent_candles"][0]["close"], 104.0)
        self.assertEqual(payload["position"]["motive"], "holding")
        self.assertNotIn("user_id", str(payload))


class ScheduledReviewGraphTests(unittest.TestCase):
    def setUp(self):
        self.context = scheduled_context()
        self.model = MagicMock(name="model")
        self.decision_runnable = MagicMock(name="decision-runnable")
        self.output_runnable = MagicMock(name="output-runnable")
        self.model.with_structured_output.side_effect = [
            self.decision_runnable,
            self.output_runnable,
        ]
        self.output_runnable.invoke.return_value = TickerAnalysisDraft(
            summary="The position remains constructive.",
            recommendation="Monitor the next session for confirmation.",
            confidence=Confidence.MEDIUM,
        )

    def invoke_graph(self):
        graph = build_scheduled_review_graph(self.model, [])
        return graph.invoke(build_ticker_agent_state(self.context))

    def test_review_skips_unnecessary_research(self):
        self.decision_runnable.invoke.return_value = ScheduledResearchDecision(
            search_web=False,
            reason="The market data does not require external context.",
        )

        with patch("agents.scheduled_review.perform_web_research") as research:
            result = self.invoke_graph()

        research.assert_not_called()
        output = result["output"]
        self.assertIsInstance(output, AgentOutput)
        self.assertEqual(output.event_type, EventType.SCHEDULED_UPDATE)
        self.assertFalse(output.searched_web)
        self.assertEqual(output.web_search_requests, 0)

    def test_review_attaches_successful_research_metadata(self):
        self.decision_runnable.invoke.return_value = ScheduledResearchDecision(
            search_web=True,
            focus=WebResearchFocus.COMPANY_NEWS,
            reason="Unusual volume warrants current company context.",
        )
        source = ResearchSource(
            title="Company filing",
            url="https://example.com/filing",
        )
        research_result = WebResearchResult(
            summary="The company published a material filing.",
            sources=[source],
            search_requests=1,
            messages=[AIMessage(content="Research complete.")],
        )

        with patch(
            "agents.scheduled_review.perform_web_research",
            return_value=research_result,
        ) as research:
            result = self.invoke_graph()

        research.assert_called_once_with(
            self.model,
            self.context,
            WebResearchFocus.COMPANY_NEWS,
        )
        output = result["output"]
        self.assertTrue(output.searched_web)
        self.assertEqual(output.sources, [source])
        self.assertEqual(output.web_search_requests, 1)

    def test_review_continues_when_optional_research_is_unavailable(self):
        self.decision_runnable.invoke.return_value = ScheduledResearchDecision(
            search_web=True,
            focus=WebResearchFocus.PRICE_CATALYST,
            reason="The move may have an external catalyst.",
        )

        with patch(
            "agents.scheduled_review.perform_web_research",
            side_effect=WebResearchError("private provider failure"),
        ):
            result = self.invoke_graph()

        self.assertFalse(result["output"].searched_web)
        analysis_messages = self.output_runnable.invoke.call_args.args[0]
        self.assertIn(
            "Current web research was unavailable.",
            analysis_messages[-1].content,
        )
        self.assertNotIn("private provider failure", str(result))


class ScheduledReviewRunnerTests(unittest.TestCase):
    def test_runner_uses_shared_execution_boundary(self):
        context = scheduled_context()
        database_client = MagicMock(name="database-client")
        credential_client = MagicMock(name="credential-client")
        cipher = MagicMock(name="cipher")
        expected = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.SCHEDULED_UPDATE,
            summary="Position remains stable.",
            recommendation="Continue monitoring.",
            confidence=Confidence.MEDIUM,
            price_at_update=context.current_price,
        )

        with patch(
            "agents.scheduled_review.execute_ticker_agent",
            return_value=expected,
        ) as execute:
            result = run_scheduled_review(
                context,
                database_client=database_client,
                credential_client=credential_client,
                cipher=cipher,
            )

        self.assertEqual(result, expected)
        execute.assert_called_once_with(
            context,
            AgentType.SCHEDULED_REVIEW,
            build_scheduled_review_graph,
            database_client=database_client,
            credential_client=credential_client,
            cipher=cipher,
        )


if __name__ == "__main__":
    unittest.main()
