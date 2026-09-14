import unittest
from datetime import datetime, timezone
from typing import get_type_hints

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents.core import TickerAgentState, build_ticker_agent_state
from models.schemas import (
    AgentContext,
    EventType,
    Motive,
    OHLCVPoint,
    Subscription,
    UpdateInterval,
)


def agent_context() -> AgentContext:
    return AgentContext(
        ticker="NVDA",
        datapoints=[
            OHLCVPoint(
                ticker="NVDA",
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=104.0,
                volume=1000.0,
            )
        ],
        subscription=Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=100.0,
            shares=2.0,
            motive=Motive.HOLDING,
            update_interval=UpdateInterval.DAILY,
        ),
        event_type=EventType.SCHEDULED_UPDATE,
        current_price=104.0,
        unrealized_pnl=8.0,
        unrealized_pnl_pct=0.04,
    )


class TickerAgentStateTests(unittest.TestCase):
    def test_state_contains_no_credential_fields(self):
        field_names = get_type_hints(
            TickerAgentState,
            include_extras=True,
        )

        self.assertEqual(
            set(field_names),
            {"context", "messages", "previous_updates", "recent_alerts", "output"},
        )
        self.assertFalse(
            any(
                "key" in name or "credential" in name or "secret" in name
                for name in field_names
            )
        )

    def test_builder_returns_independent_mutable_collections(self):
        first = build_ticker_agent_state(agent_context())
        second = build_ticker_agent_state(agent_context())

        first["messages"].append(HumanMessage(content="Review this position."))

        self.assertEqual(len(first["messages"]), 1)
        self.assertEqual(second["messages"], [])
        self.assertEqual(first["previous_updates"], [])
        self.assertEqual(first["recent_alerts"], [])
        self.assertIsNone(first["output"])

    def test_state_compiles_and_flows_through_langgraph(self):
        graph = StateGraph(TickerAgentState)
        graph.add_node(
            "prepare",
            lambda state: {"messages": [HumanMessage(content=state["context"].ticker)]},
        )
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", END)

        result = graph.compile().invoke(build_ticker_agent_state(agent_context()))

        self.assertEqual(result["context"].ticker, "NVDA")
        self.assertEqual(result["messages"][0].content, "NVDA")


if __name__ == "__main__":
    unittest.main()
