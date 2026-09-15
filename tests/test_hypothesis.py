import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from agents.core import (
    AgentExecutionError,
    HypothesisAnalysisDraft,
    WebResearchError,
    WebResearchFocus,
    WebResearchResult,
    build_ticker_agent_state,
)
from agents.hypothesis import (
    HypothesisSchedulingError,
    HypothesisScreen,
    _hypothesis_payload,
    build_hypothesis_graph,
    run_hypothesis_scan,
)
from models.schemas import (
    AgentContext,
    AgentType,
    Confidence,
    EventType,
    HypothesisOutput,
    Motive,
    OHLCVPoint,
    ResearchSource,
    Subscription,
    UpdateInterval,
)


def hypothesis_context() -> AgentContext:
    start = datetime(2026, 9, 10, 13, 30, tzinfo=timezone.utc)
    points = []
    for index in range(35):
        close = 100.0 + (index * 0.08)
        if index >= 31:
            close += (index - 30) * 0.7
        points.append(
            OHLCVPoint(
                ticker="NVDA",
                timestamp=start + timedelta(minutes=15 * index),
                open=close - 0.1,
                high=close + 0.3,
                low=close - 0.3,
                close=close,
                volume=1000.0 if index < 31 else 2500.0,
            )
        )
    current = points[-1].close
    return AgentContext(
        ticker="NVDA",
        datapoints=points,
        subscription=Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=100.0,
            shares=2.0,
            motive=Motive.WATCHING,
            update_interval=UpdateInterval.DAILY,
        ),
        event_type=EventType.HYPOTHESIS_SCAN,
        current_price=current,
        unrealized_pnl=(current - 100.0) * 2.0,
        unrealized_pnl_pct=(current - 100.0) / 100.0,
    )


class HypothesisScreenTests(unittest.TestCase):
    def test_candidate_contract_requires_pattern_and_focus(self):
        with self.assertRaises(ValidationError):
            HypothesisScreen(
                candidate_detected=True,
                confidence=Confidence.MEDIUM,
                reason="Momentum appears to be accelerating.",
            )

        with self.assertRaises(ValidationError):
            HypothesisScreen(
                candidate_detected=False,
                pattern="Possible breakout.",
                focus=WebResearchFocus.COMPANY_NEWS,
                confidence=Confidence.LOW,
                reason="Evidence is too weak.",
            )


class HypothesisPayloadTests(unittest.TestCase):
    def test_payload_builds_bounded_multi_window_features(self):
        payload = _hypothesis_payload(
            build_ticker_agent_state(hypothesis_context())
        )
        window = payload["market_window"]

        self.assertEqual(window["candle_count"], 35)
        self.assertGreater(window["recent_volume_ratio"], 1.0)
        self.assertGreater(window["recent_four_candle_change_pct"], 0)
        self.assertLessEqual(len(payload["recent_candles"]), 20)
        self.assertNotIn("user_id", str(payload))

    def test_payload_requires_enough_candles_for_pattern_screening(self):
        context = hypothesis_context().model_copy(
            update={"datapoints": hypothesis_context().datapoints[:11]}
        )
        with self.assertRaisesRegex(AgentExecutionError, "at least twelve"):
            _hypothesis_payload(build_ticker_agent_state(context))


class HypothesisGraphTests(unittest.TestCase):
    def setUp(self):
        self.context = hypothesis_context()
        self.model = MagicMock(name="model")
        self.screen_runnable = MagicMock(name="screen-runnable")
        self.output_runnable = MagicMock(name="output-runnable")
        self.model.with_structured_output.side_effect = [
            self.screen_runnable,
            self.output_runnable,
        ]

    def invoke_graph(self):
        graph = build_hypothesis_graph(self.model, [])
        return graph.invoke(build_ticker_agent_state(self.context))

    def test_empty_screen_skips_research_and_second_model_call(self):
        self.screen_runnable.invoke.return_value = HypothesisScreen(
            candidate_detected=False,
            confidence=Confidence.LOW,
            reason="The movement is ordinary noise.",
        )

        with patch("agents.hypothesis.perform_web_research") as research:
            result = self.invoke_graph()

        research.assert_not_called()
        self.output_runnable.invoke.assert_not_called()
        output = result["output"]
        self.assertIsInstance(output, HypothesisOutput)
        self.assertFalse(output.flagged)
        self.assertIsNone(output.summary)
        self.assertEqual(output.recommended_next_scan_days, 3)

    def test_candidate_is_researched_and_can_remain_flagged(self):
        self.screen_runnable.invoke.return_value = HypothesisScreen(
            candidate_detected=True,
            pattern="Price and volume may be building above the prior range.",
            focus=WebResearchFocus.COMPANY_NEWS,
            confidence=Confidence.MEDIUM,
            reason="Several independent features support a testable pattern.",
        )
        source = ResearchSource(
            title="Company announcement",
            url="https://example.com/announcement",
        )
        research_result = WebResearchResult(
            summary="A recent announcement may support the developing pattern.",
            sources=[source],
            search_requests=1,
            messages=[AIMessage(content="Research complete.")],
        )
        self.output_runnable.invoke.return_value = HypothesisAnalysisDraft(
            summary="Price and volume show an early-stage buildup.",
            recommendation="Rescan tomorrow for confirmation or failure.",
            confidence=Confidence.MEDIUM,
            flagged=True,
            recommended_next_scan_days=1,
        )

        with patch(
            "agents.hypothesis.perform_web_research",
            return_value=research_result,
        ) as research:
            result = self.invoke_graph()

        research.assert_called_once_with(
            self.model,
            self.context,
            WebResearchFocus.COMPANY_NEWS,
        )
        output = result["output"]
        self.assertTrue(output.flagged)
        self.assertTrue(output.searched_web)
        self.assertEqual(output.sources, [source])
        self.assertEqual(output.recommended_next_scan_days, 1)

    def test_candidate_can_be_rejected_after_research(self):
        self.screen_runnable.invoke.return_value = HypothesisScreen(
            candidate_detected=True,
            pattern="The stock may be breaking out.",
            focus=WebResearchFocus.SECTOR,
            confidence=Confidence.LOW,
            reason="The chart contains a tentative breakout signal.",
        )
        self.output_runnable.invoke.return_value = HypothesisAnalysisDraft(
            summary=None,
            recommendation="",
            confidence=Confidence.LOW,
            flagged=False,
            recommended_next_scan_days=3,
        )

        with patch(
            "agents.hypothesis.perform_web_research",
            return_value=WebResearchResult(
                summary="Sector evidence contradicts the proposed breakout.",
                search_requests=1,
            ),
        ):
            result = self.invoke_graph()

        self.assertFalse(result["output"].flagged)
        self.assertEqual(result["output"].recommended_next_scan_days, 3)

    def test_research_outage_still_allows_candidate_rejection(self):
        self.screen_runnable.invoke.return_value = HypothesisScreen(
            candidate_detected=True,
            pattern="Volume may be accumulating.",
            focus=WebResearchFocus.PRICE_CATALYST,
            confidence=Confidence.LOW,
            reason="The pattern is testable but weak.",
        )
        self.output_runnable.invoke.return_value = HypothesisAnalysisDraft(
            summary=None,
            recommendation="",
            confidence=Confidence.LOW,
            flagged=False,
            recommended_next_scan_days=3,
        )

        with patch(
            "agents.hypothesis.perform_web_research",
            side_effect=WebResearchError("private provider detail"),
        ):
            result = self.invoke_graph()

        messages = self.output_runnable.invoke.call_args.args[0]
        self.assertIn(
            "Current hypothesis research was unavailable.",
            messages[-1].content,
        )
        self.assertNotIn("private provider detail", str(result))


class HypothesisRunnerTests(unittest.TestCase):
    def setUp(self):
        self.context = hypothesis_context()
        self.output = HypothesisOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.HYPOTHESIS_SCAN,
            summary="A developing pattern remains worth watching.",
            recommendation="Rescan tomorrow.",
            confidence=Confidence.MEDIUM,
            price_at_update=self.context.current_price,
            flagged=True,
            recommended_next_scan_days=1,
        )

    def test_runner_persists_then_schedules_recommended_cadence(self):
        schedule_next = MagicMock(name="schedule-next")

        with patch(
            "agents.hypothesis.execute_ticker_agent",
            return_value=self.output,
        ) as execute:
            result = run_hypothesis_scan(
                self.context,
                schedule_next=schedule_next,
            )

        self.assertEqual(result, self.output)
        execute.assert_called_once_with(
            self.context,
            AgentType.HYPOTHESIS,
            build_hypothesis_graph,
            database_client=None,
            credential_client=None,
            cipher=None,
        )
        schedule_next.assert_called_once_with("user-1", "NVDA", 1)

    def test_scheduler_failure_is_sanitized(self):
        schedule_next = MagicMock(side_effect=RuntimeError("private scheduler detail"))
        with patch(
            "agents.hypothesis.execute_ticker_agent",
            return_value=self.output,
        ):
            with self.assertRaisesRegex(
                HypothesisSchedulingError,
                "could not be scheduled",
            ) as raised:
                run_hypothesis_scan(
                    self.context,
                    schedule_next=schedule_next,
                )

        self.assertNotIn("private scheduler detail", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
