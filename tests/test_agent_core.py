import json
import unittest
from datetime import datetime, timezone
from typing import get_type_hints
from unittest.mock import MagicMock, call, patch

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr, ValidationError

from agents import core
from agents.core import (
    AgentModelInitializationError,
    AgentToolConfigurationError,
    MissingUserApiKeyError,
    TickerAgentState,
    build_ticker_agent_tools,
    build_ticker_agent_state,
)
from data.database import DatabaseError
from models.schemas import (
    AgentContext,
    AgentOutput,
    AgentType,
    Alert,
    AlertType,
    Confidence,
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


class TickerAgentToolTests(unittest.TestCase):
    def setUp(self):
        self.context = agent_context()
        self.client = MagicMock()

    def tools_by_name(self):
        return {
            tool.name: tool
            for tool in build_ticker_agent_tools(self.context, client=self.client)
        }

    def test_tools_do_not_expose_user_or_ticker_arguments(self):
        tools = self.tools_by_name()

        self.assertEqual(
            set(tools),
            {
                "get_market_history",
                "get_latest_agent_update",
                "get_recent_agent_updates",
                "get_recent_alerts",
            },
        )
        for tool in tools.values():
            self.assertNotIn("user_id", tool.args)
            self.assertNotIn("ticker", tool.args)

        with self.assertRaises(ValidationError):
            tools["get_market_history"].invoke(
                {"limit": 5, "ticker": "AMD", "user_id": "user-2"}
            )

    def test_tools_enforce_agent_specific_result_limits(self):
        tools = self.tools_by_name()

        with self.assertRaises(ValidationError):
            tools["get_market_history"].invoke({"limit": 151})
        with self.assertRaises(ValidationError):
            tools["get_recent_agent_updates"].invoke({"limit": 21})

    @patch("agents.core.get_latest_ticker_data")
    def test_market_tool_uses_bound_ticker_and_serializes_points(self, get_history):
        get_history.return_value = self.context.datapoints

        result = self.tools_by_name()["get_market_history"].invoke({"limit": 25})

        get_history.assert_called_once_with("NVDA", limit=25, client=self.client)
        payload = json.loads(result)
        self.assertEqual(payload[0]["ticker"], "NVDA")
        self.assertEqual(payload[0]["close"], 104.0)

    @patch("agents.core.get_latest_update")
    def test_latest_update_tool_enforces_bound_user_and_ticker(self, get_update):
        output = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.SCHEDULED_UPDATE,
            summary="Position remains stable.",
            recommendation="Continue monitoring.",
            confidence=Confidence.MEDIUM,
        )
        get_update.return_value = output

        result = self.tools_by_name()["get_latest_agent_update"].invoke(
            {"agent_type": AgentType.SCHEDULED_REVIEW.value}
        )

        get_update.assert_called_once_with(
            "user-1",
            AgentType.SCHEDULED_REVIEW,
            "NVDA",
            client=self.client,
        )
        self.assertEqual(json.loads(result)["summary"], "Position remains stable.")

    @patch("agents.core.list_recent_updates")
    def test_recent_updates_tool_passes_only_safe_filters(self, list_updates):
        list_updates.return_value = []
        since = "2026-01-01T00:00:00Z"

        result = self.tools_by_name()["get_recent_agent_updates"].invoke(
            {"agent_type": "sharp_move", "limit": 4, "since": since}
        )

        self.assertEqual(result, "[]")
        list_updates.assert_called_once_with(
            "user-1",
            ticker="NVDA",
            agent_type=AgentType.SHARP_MOVE,
            limit=4,
            since=datetime(2026, 1, 1, tzinfo=timezone.utc),
            client=self.client,
        )

    @patch("agents.core.list_recent_alerts")
    def test_recent_alert_tool_returns_typed_alert_history(self, list_alerts):
        alert = Alert(
            user_id="user-1",
            ticker="NVDA",
            alert_type=AlertType.SHARP_MOVE,
            message="NVDA moved sharply.",
        )
        list_alerts.return_value = [alert]

        result = self.tools_by_name()["get_recent_alerts"].invoke(
            {"alert_type": "sharp_move", "limit": 3}
        )

        list_alerts.assert_called_once_with(
            "user-1",
            ticker="NVDA",
            alert_type=AlertType.SHARP_MOVE,
            limit=3,
            since=None,
            client=self.client,
        )
        self.assertEqual(json.loads(result)[0]["message"], "NVDA moved sharply.")

    @patch("agents.core.get_latest_ticker_data")
    def test_tool_errors_are_sanitized(self, get_history):
        get_history.side_effect = DatabaseError("private database detail")

        result = self.tools_by_name()["get_market_history"].invoke({"limit": 5})

        self.assertEqual(result, "Market history is temporarily unavailable.")
        self.assertNotIn("private database detail", result)

    def test_tool_factory_rejects_mismatched_subscription_ticker(self):
        mismatched = self.context.model_copy(
            update={
                "subscription": self.context.subscription.model_copy(
                    update={"ticker": "AMD"}
                )
            }
        )

        with self.assertRaisesRegex(AgentToolConfigurationError, "must match"):
            build_ticker_agent_tools(mismatched, client=self.client)


if __name__ == "__main__":
    unittest.main()
