import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agents.core import (
    AgentExecutionError,
    TickerAnalysisDraft,
    WebResearchError,
    WebResearchFocus,
    WebResearchResult,
    build_ticker_agent_state,
)
from agents.motive import (
    MotiveAlignment,
    MotiveAssessment,
    _motive_payload,
    build_motive_graph,
    run_motive_reassessment,
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


def motive_context(motive: Motive = Motive.SHORT_TERM) -> AgentContext:
    start = datetime(2026, 9, 8, 13, 30, tzinfo=timezone.utc)
    points = []
    for index in range(32):
        close = 112.0 - (index * 0.4)
        points.append(
            OHLCVPoint(
                ticker="NVDA",
                timestamp=start + timedelta(minutes=15 * index),
                open=close + 0.1,
                high=close + 0.5,
                low=close - 0.5,
                close=close,
                volume=1000.0 + (index * 20.0),
            )
        )
    current_price = points[-1].close
    return AgentContext(
        ticker="NVDA",
        datapoints=points,
        subscription=Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=110.0,
            shares=2.0,
            motive=motive,
            update_interval=UpdateInterval.WEEKLY,
        ),
        event_type=EventType.MOTIVE_CHECK,
        current_price=current_price,
        unrealized_pnl=(current_price - 110.0) * 2.0,
        unrealized_pnl_pct=(current_price - 110.0) / 110.0,
    )


class MotiveAssessmentTests(unittest.TestCase):
    def test_research_decision_requires_matching_focus(self):
        with self.assertRaises(ValidationError):
            MotiveAssessment(
                alignment=MotiveAlignment.AT_RISK,
                search_web=True,
                reason="Material deterioration needs current context.",
            )

        with self.assertRaises(ValidationError):
            MotiveAssessment(
                alignment=MotiveAlignment.ALIGNED,
                search_web=False,
                focus=WebResearchFocus.COMPANY_NEWS,
                reason="No search is necessary.",
            )


class MotivePayloadTests(unittest.TestCase):
    def test_payload_quantifies_window_and_declares_missing_thesis_data(self):
        payload = _motive_payload(build_ticker_agent_state(motive_context()))

        self.assertEqual(payload["motive"]["stated_motive"], "short-term")
        self.assertEqual(payload["available_market_window"]["candle_count"], 32)
        self.assertLess(payload["available_market_window"]["window_change_pct"], 0)
        self.assertLess(
            payload["available_market_window"]["drawdown_from_window_high_pct"],
            0,
        )
        self.assertIn("No position entry timestamp is stored.", payload["limitations"])
        self.assertNotIn("user_id", str(payload))

    def test_payload_requires_two_candles(self):
        context = motive_context().model_copy(
            update={"datapoints": motive_context().datapoints[:1]}
        )
        with self.assertRaisesRegex(AgentExecutionError, "at least two"):
            _motive_payload(build_ticker_agent_state(context))


class MotiveGraphTests(unittest.TestCase):
    def setUp(self):
        self.context = motive_context()
        self.model = MagicMock(name="model")
        self.assessment_runnable = MagicMock(name="assessment-runnable")
        self.output_runnable = MagicMock(name="output-runnable")
        self.model.with_structured_output.side_effect = [
            self.assessment_runnable,
            self.output_runnable,
        ]
        self.output_runnable.invoke.return_value = TickerAnalysisDraft(
            summary="The available trend conflicts with the short-term motive.",
            recommendation="Clarify the intended time horizon and monitor momentum.",
            confidence=Confidence.MEDIUM,
        )

    def invoke_graph(self):
        graph = build_motive_graph(self.model, [])
        return graph.invoke(build_ticker_agent_state(self.context))

    def test_aligned_path_skips_unnecessary_research(self):
        self.assessment_runnable.invoke.return_value = MotiveAssessment(
            alignment=MotiveAlignment.ALIGNED,
            search_web=False,
            reason="Observed conditions remain compatible with the motive.",
        )

        with patch("agents.motive.perform_web_research") as research:
            result = self.invoke_graph()

        research.assert_not_called()
        output = result["output"]
        self.assertIsInstance(output, AgentOutput)
        self.assertEqual(output.event_type, EventType.MOTIVE_CHECK)
        self.assertFalse(output.searched_web)

    def test_at_risk_path_can_add_current_company_research(self):
        self.assessment_runnable.invoke.return_value = MotiveAssessment(
            alignment=MotiveAlignment.AT_RISK,
            search_web=True,
            focus=WebResearchFocus.COMPANY_NEWS,
            reason="The deterioration may reflect a material company change.",
        )
        source = ResearchSource(
            title="Company update",
            url="https://example.com/company-update",
        )
        research_result = WebResearchResult(
            summary="The company changed its near-term guidance.",
            sources=[source],
            search_requests=1,
            messages=[AIMessage(content="Research complete.")],
        )

        with patch(
            "agents.motive.perform_web_research",
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

    def test_optional_research_failure_does_not_abort_reassessment(self):
        self.assessment_runnable.invoke.return_value = MotiveAssessment(
            alignment=MotiveAlignment.INSUFFICIENT_EVIDENCE,
            search_web=True,
            focus=WebResearchFocus.EARNINGS,
            reason="Recent earnings context is needed.",
        )

        with patch(
            "agents.motive.perform_web_research",
            side_effect=WebResearchError("private provider detail"),
        ):
            result = self.invoke_graph()

        self.assertFalse(result["output"].searched_web)
        messages = self.output_runnable.invoke.call_args.args[0]
        self.assertIn("Current motive research was unavailable.", messages[-1].content)
        self.assertNotIn("private provider detail", str(result))

    def test_watching_prompt_does_not_treat_reference_pnl_as_owned(self):
        self.context = motive_context(Motive.WATCHING)
        self.assessment_runnable.invoke.return_value = MotiveAssessment(
            alignment=MotiveAlignment.INSUFFICIENT_EVIDENCE,
            search_web=False,
            reason="The available window does not establish an opportunity.",
        )

        self.invoke_graph()

        messages = self.output_runnable.invoke.call_args.args[0]
        self.assertIn(
            "do not describe unrealized P&L as an owned-position",
            messages[-1].content,
        )


class MotiveRunnerTests(unittest.TestCase):
    def test_runner_uses_shared_execution_boundary(self):
        context = motive_context()
        database_client = MagicMock(name="database-client")
        credential_client = MagicMock(name="credential-client")
        cipher = MagicMock(name="cipher")
        expected = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.MOTIVE_CHECK,
            summary="The short-term motive is under pressure.",
            recommendation="Clarify the intended horizon.",
            confidence=Confidence.MEDIUM,
            price_at_update=context.current_price,
        )

        with patch(
            "agents.motive.execute_ticker_agent",
            return_value=expected,
        ) as execute:
            result = run_motive_reassessment(
                context,
                database_client=database_client,
                credential_client=credential_client,
                cipher=cipher,
            )

        self.assertEqual(result, expected)
        execute.assert_called_once_with(
            context,
            AgentType.MOTIVE,
            build_motive_graph,
            database_client=database_client,
            credential_client=credential_client,
            cipher=cipher,
        )


if __name__ == "__main__":
    unittest.main()
