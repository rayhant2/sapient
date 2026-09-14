import unittest
from datetime import datetime, timezone
from typing import get_type_hints
from unittest.mock import MagicMock, call, patch

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr

from agents import core
from agents.core import (
    AgentModelInitializationError,
    MissingUserApiKeyError,
    TickerAgentState,
    build_ticker_agent_state,
)
from data.database import DatabaseError
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


class UserModelFactoryTests(unittest.TestCase):
    @patch("agents.core.ChatAnthropic")
    @patch("agents.core.resolve_user_api_key")
    def test_factory_uses_user_secret_and_configured_limits(self, resolve_key, model):
        secret = SecretStr("sk-ant-user-secret")
        database_client = MagicMock()
        cipher = MagicMock()
        resolve_key.return_value = secret
        model.return_value = MagicMock(name="user-model")

        created = core.create_user_model(
            "user-1",
            client=database_client,
            cipher=cipher,
        )

        resolve_key.assert_called_once_with(
            "user-1",
            "anthropic",
            client=database_client,
            cipher=cipher,
        )
        model.assert_called_once_with(
            model=core.settings.anthropic_model,
            api_key=secret,
            max_tokens=core.settings.agent_model_max_tokens,
            timeout=core.settings.agent_model_timeout_seconds,
            max_retries=core.settings.agent_model_max_retries,
        )
        self.assertIs(created, model.return_value)

    @patch("agents.core.ChatAnthropic")
    @patch("agents.core.resolve_user_api_key")
    def test_factory_creates_a_fresh_model_for_each_run(self, resolve_key, model):
        resolve_key.return_value = SecretStr("sk-ant-user-secret")
        first_model = MagicMock(name="first-model")
        second_model = MagicMock(name="second-model")
        model.side_effect = [first_model, second_model]

        first = core.create_user_model("user-1")
        second = core.create_user_model("user-1")

        self.assertIs(first, first_model)
        self.assertIs(second, second_model)
        expected_call = call("user-1", "anthropic", client=None, cipher=None)
        self.assertEqual(resolve_key.call_args_list, [expected_call, expected_call])

    @patch("agents.core.ChatAnthropic")
    @patch("agents.core.resolve_user_api_key", return_value=None)
    def test_factory_rejects_missing_user_key(self, resolve_key, model):
        with self.assertRaisesRegex(MissingUserApiKeyError, "must add"):
            core.create_user_model("user-1")

        resolve_key.assert_called_once()
        model.assert_not_called()

    @patch("agents.core.ChatAnthropic")
    @patch("agents.core.resolve_user_api_key")
    def test_factory_wraps_credential_resolution_failure(self, resolve_key, model):
        resolve_key.side_effect = DatabaseError("sensitive database detail")

        with self.assertRaisesRegex(
            AgentModelInitializationError,
            "securely resolved",
        ) as raised:
            core.create_user_model("user-1")

        self.assertNotIn("sensitive database detail", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)
        model.assert_not_called()

    @patch("agents.core.ChatAnthropic")
    @patch("agents.core.resolve_user_api_key")
    def test_factory_wraps_model_initialization_failure(self, resolve_key, model):
        resolve_key.return_value = SecretStr("sk-ant-user-secret")
        model.side_effect = ValueError("secret-bearing SDK failure")

        with self.assertRaisesRegex(
            AgentModelInitializationError,
            "could not be initialized",
        ) as raised:
            core.create_user_model("user-1")

        self.assertNotIn("secret-bearing SDK failure", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    @patch("agents.core.resolve_user_api_key")
    def test_factory_rejects_blank_user_before_database_access(self, resolve_key):
        with self.assertRaisesRegex(ValueError, "user_id"):
            core.create_user_model("   ")

        resolve_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
