import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

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
from agents.sharp_move import (
    SectorResearchDecision,
    _move_payload,
    build_sharp_move_graph,
    run_sharp_move,
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


def sharp_move_context() -> AgentContext:
    start = datetime(2026, 9, 15, 13, 30, tzinfo=timezone.utc)
    closes = [100.05 if index % 2 else 100.0 for index in range(25)]
    closes.append(103.0)
    points = [
        OHLCVPoint(
            ticker="NVDA",
            timestamp=start + timedelta(minutes=15 * index),
            open=close,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=1000.0 if index < 25 else 4000.0,
        )
        for index, close in enumerate(closes)
    ]
    return AgentContext(
        ticker="NVDA",
        datapoints=points,
        subscription=Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=101.0,
            shares=2.0,
            motive=Motive.SHORT_TERM,
            update_interval=UpdateInterval.DAILY,
            sharp_move_threshold=0.02,
        ),
        event_type=EventType.SHARP_MOVE,
        current_price=103.0,
        unrealized_pnl=4.0,
        unrealized_pnl_pct=2.0 / 101.0,
    )


def research_result(label: str, url: str) -> WebResearchResult:
    return WebResearchResult(
        summary=f"{label} research found relevant evidence.",
        sources=[ResearchSource(title=f"{label} source", url=url)],
        search_requests=1,
        messages=[AIMessage(content=f"{label} research complete.")],
    )


class SectorResearchDecisionTests(unittest.TestCase):
    def test_reason_is_required_and_trimmed(self):
        decision = SectorResearchDecision(
            search_sector=False,
            reason="  Catalyst is clearly company-specific.  ",
        )
        self.assertEqual(decision.reason, "Catalyst is clearly company-specific.")

        with self.assertRaises(ValidationError):
            SectorResearchDecision(search_sector=True, reason="   ")


class SharpMovePayloadTests(unittest.TestCase):
    def test_payload_reconstructs_trigger_and_position_significance(self):
        payload = _move_payload(build_ticker_agent_state(sharp_move_context()))
        trigger = payload["trigger"]

        self.assertAlmostEqual(trigger["move_pct"], 0.03)
        self.assertEqual(trigger["direction"], "up")
        self.assertTrue(trigger["absolute_threshold_triggered"])
        self.assertTrue(trigger["volatility_triggered"])
        self.assertTrue(trigger["crossed_cost_basis"])
        self.assertAlmostEqual(trigger["latest_volume_ratio"], 4.0)
        self.assertEqual(len(payload["recent_candles"]), 12)
        self.assertNotIn("user_id", str(payload))

    def test_payload_requires_two_positive_close_candles(self):
        context = sharp_move_context().model_copy(
            update={"datapoints": sharp_move_context().datapoints[:1]}
        )
        with self.assertRaisesRegex(AgentExecutionError, "at least two"):
            _move_payload(build_ticker_agent_state(context))


class SharpMoveGraphTests(unittest.TestCase):
    def setUp(self):
        self.context = sharp_move_context()
        self.model = MagicMock(name="model")
        self.decision_runnable = MagicMock(name="sector-decision-runnable")
        self.output_runnable = MagicMock(name="output-runnable")
        self.model.with_structured_output.side_effect = [
            self.decision_runnable,
            self.output_runnable,
        ]
        self.output_runnable.invoke.return_value = TickerAnalysisDraft(
            summary="NVDA moved sharply after a company-specific catalyst.",
            recommendation="Monitor whether the move holds above VWAP.",
            confidence=Confidence.HIGH,
        )
        self.portfolio_tool = MagicMock(name="portfolio-tool")
        self.portfolio_tool.name = "get_portfolio_snapshot"
        self.portfolio_tool.invoke.return_value = json.dumps(
            [
                {
                    "ticker": "AMD",
                    "summary": "AMD was also weak.",
                    "recommendation": "Monitor semiconductors.",
                    "user_id": "must-not-reach-the-prompt",
                }
            ]
        )

    def invoke_graph(self):
        graph = build_sharp_move_graph(self.model, [self.portfolio_tool])
        return graph.invoke(build_ticker_agent_state(self.context))

    def test_company_specific_path_uses_one_catalyst_search(self):
        self.decision_runnable.invoke.return_value = SectorResearchDecision(
            search_sector=False,
            reason="The confirmed catalyst is company-specific.",
        )
        catalyst = research_result("Catalyst", "https://example.com/catalyst")

        with patch(
            "agents.sharp_move.perform_web_research",
            return_value=catalyst,
        ) as research:
            result = self.invoke_graph()

        research.assert_called_once_with(
            self.model,
            self.context,
            WebResearchFocus.PRICE_CATALYST,
        )
        output = result["output"]
        self.assertEqual(output.event_type, EventType.SHARP_MOVE)
        self.assertTrue(output.searched_web)
        self.assertEqual(output.sources, catalyst.sources)
        self.assertEqual(output.web_search_requests, 1)
        self.assertEqual(result["portfolio_snapshot"][0]["ticker"], "AMD")
        self.assertNotIn("user_id", str(result["portfolio_snapshot"]))

    def test_broad_move_path_adds_sector_research(self):
        self.decision_runnable.invoke.return_value = SectorResearchDecision(
            search_sector=True,
            reason="Related holdings suggest a broader semiconductor move.",
        )
        catalyst = research_result("Catalyst", "https://example.com/catalyst")
        sector = research_result("Sector", "https://example.com/sector")

        with patch(
            "agents.sharp_move.perform_web_research",
            side_effect=[catalyst, sector],
        ) as research:
            result = self.invoke_graph()

        self.assertEqual(
            research.call_args_list,
            [
                call(self.model, self.context, WebResearchFocus.PRICE_CATALYST),
                call(self.model, self.context, WebResearchFocus.SECTOR),
            ],
        )
        output = result["output"]
        self.assertEqual(output.sources, catalyst.sources + sector.sources)
        self.assertEqual(output.web_search_requests, 2)

    def test_research_outage_still_produces_an_unsearched_output(self):
        self.decision_runnable.invoke.return_value = SectorResearchDecision(
            search_sector=False,
            reason="No second search is useful while research is unavailable.",
        )

        with patch(
            "agents.sharp_move.perform_web_research",
            side_effect=WebResearchError("private provider detail"),
        ):
            result = self.invoke_graph()

        self.assertFalse(result["output"].searched_web)
        analysis_messages = self.output_runnable.invoke.call_args.args[0]
        self.assertIn(
            "Current catalyst research was unavailable.",
            analysis_messages[-1].content,
        )
        self.assertNotIn("private provider detail", str(result))


class SharpMoveRunnerTests(unittest.TestCase):
    def test_runner_uses_shared_execution_boundary(self):
        context = sharp_move_context()
        database_client = MagicMock(name="database-client")
        credential_client = MagicMock(name="credential-client")
        cipher = MagicMock(name="cipher")
        expected = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.SHARP_MOVE,
            summary="A catalyst explains the move.",
            recommendation="Monitor follow-through.",
            confidence=Confidence.HIGH,
            price_at_update=context.current_price,
        )

        with patch(
            "agents.sharp_move.execute_ticker_agent",
            return_value=expected,
        ) as execute:
            result = run_sharp_move(
                context,
                database_client=database_client,
                credential_client=credential_client,
                cipher=cipher,
            )

        self.assertEqual(result, expected)
        execute.assert_called_once_with(
            context,
            AgentType.SHARP_MOVE,
            build_sharp_move_graph,
            database_client=database_client,
            credential_client=credential_client,
            cipher=cipher,
        )


if __name__ == "__main__":
    unittest.main()
