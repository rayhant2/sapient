from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, NotRequired, TypeAlias, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator
from supabase import Client

from config.settings import settings
from data.database import (
    DatabaseError,
    get_latest_ticker_data,
    get_latest_update,
    list_recent_alerts,
    list_recent_updates,
    resolve_user_api_key,
)
from models.schemas import (
    AgentContext,
    AgentOutput,
    AgentType,
    Alert,
    AlertType,
    HypothesisOutput,
)
from security.credentials import CredentialCipher, CredentialSecurityError


TickerAgentOutput: TypeAlias = AgentOutput | HypothesisOutput
MAX_AGENT_MARKET_RESULTS = 150
MAX_AGENT_MEMORY_RESULTS = 20


class AgentModelError(RuntimeError):
    """Base exception for per-user model setup failures."""


class MissingUserApiKeyError(AgentModelError):
    """Raised when a user has no stored API key for the requested provider."""


class AgentModelInitializationError(AgentModelError):
    """Raised when a user credential or model client cannot be initialized."""


class AgentToolConfigurationError(RuntimeError):
    """Raised when tools cannot be safely bound to an agent context."""


class AgentContextAssemblyError(RuntimeError):
    """Raised when an agent's bounded memory cannot be safely assembled."""


class AgentToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketHistoryToolInput(AgentToolInput):
    limit: int = Field(default=50, ge=1, le=MAX_AGENT_MARKET_RESULTS)


class LatestUpdateToolInput(AgentToolInput):
    agent_type: AgentType

    @field_validator("agent_type")
    @classmethod
    def agent_type_must_be_ticker_scoped(cls, value: AgentType) -> AgentType:
        if value == AgentType.CROSS_PORTFOLIO:
            raise ValueError("cross_portfolio updates are not ticker-scoped")
        return value


class RecentUpdatesToolInput(AgentToolInput):
    agent_type: AgentType | None = None
    limit: int = Field(default=5, ge=1, le=MAX_AGENT_MEMORY_RESULTS)
    since: datetime | None = None

    @field_validator("agent_type")
    @classmethod
    def agent_type_must_be_ticker_scoped(
        cls, value: AgentType | None
    ) -> AgentType | None:
        if value == AgentType.CROSS_PORTFOLIO:
            raise ValueError("cross_portfolio updates are not ticker-scoped")
        return value

    @field_validator("since")
    @classmethod
    def since_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("since must be timezone-aware")
        return value


class RecentAlertsToolInput(AgentToolInput):
    alert_type: AlertType | None = None
    limit: int = Field(default=5, ge=1, le=MAX_AGENT_MEMORY_RESULTS)
    since: datetime | None = None

    @field_validator("since")
    @classmethod
    def since_must_be_timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("since must be timezone-aware")
        return value


class TickerAgentState(TypedDict):
    """Shared, credential-free state for all single-ticker agent graphs."""

    context: AgentContext
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    previous_updates: NotRequired[list[TickerAgentOutput]]
    recent_alerts: NotRequired[list[Alert]]
    output: NotRequired[TickerAgentOutput | None]


def build_ticker_agent_state(context: AgentContext) -> TickerAgentState:
    """Create isolated state for one user and ticker agent run."""
    return {
        "context": context,
        "messages": [],
        "previous_updates": [],
        "recent_alerts": [],
        "output": None,
    }


def _agent_memory_limit(value: int, field_name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= MAX_AGENT_MEMORY_RESULTS
    ):
        raise ValueError(
            f"{field_name} must be an integer between 1 and "
            f"{MAX_AGENT_MEMORY_RESULTS}."
        )
    return value


def hydrate_ticker_agent_state(
    context: AgentContext,
    agent_type: AgentType | str,
    *,
    update_limit: int = 5,
    alert_limit: int = 5,
    client: Client | None = None,
) -> TickerAgentState:
    """Build state with bounded, user-scoped memory for one ticker agent run."""
    user_id, ticker = _bound_ticker_identity(context)
    normalized_agent_type = AgentType(agent_type)
    if normalized_agent_type == AgentType.CROSS_PORTFOLIO:
        raise ValueError("cross_portfolio does not use ticker agent state.")
    validated_update_limit = _agent_memory_limit(update_limit, "update_limit")
    validated_alert_limit = _agent_memory_limit(alert_limit, "alert_limit")

    try:
        latest_same_agent = get_latest_update(
            user_id,
            normalized_agent_type,
            ticker,
            client=client,
        )
        recent_updates = list_recent_updates(
            user_id,
            ticker=ticker,
            limit=validated_update_limit,
            client=client,
        )
        recent_alert_history = list_recent_alerts(
            user_id,
            ticker=ticker,
            limit=validated_alert_limit,
            client=client,
        )
    except Exception:
        raise AgentContextAssemblyError(
            "Agent memory could not be loaded for this position."
        ) from None

    previous_updates: list[TickerAgentOutput] = []
    if latest_same_agent is not None:
        previous_updates.append(latest_same_agent)
    for output in recent_updates:
        if len(previous_updates) >= validated_update_limit:
            break
        if output not in previous_updates:
            previous_updates.append(output)

    state = build_ticker_agent_state(context)
    state["previous_updates"] = previous_updates
    state["recent_alerts"] = recent_alert_history
    return state


def create_user_model(
    user_id: str,
    *,
    client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> ChatAnthropic:
    """Create a fresh Anthropic client billed to one user-owned credential."""
    if not user_id.strip():
        raise ValueError("user_id must not be blank.")

    try:
        api_key = resolve_user_api_key(
            user_id,
            "anthropic",
            client=client,
            cipher=cipher,
        )
    except (DatabaseError, CredentialSecurityError):
        raise AgentModelInitializationError(
            "The user's Anthropic credential could not be securely resolved."
        ) from None

    if api_key is None:
        raise MissingUserApiKeyError(
            "The user must add an Anthropic API key before an agent can run."
        )

    try:
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=api_key,
            max_tokens=settings.agent_model_max_tokens,
            timeout=settings.agent_model_timeout_seconds,
            max_retries=settings.agent_model_max_retries,
        )
    except Exception:
        raise AgentModelInitializationError(
            "The Anthropic model client could not be initialized."
        ) from None


def _serialize_tool_models(value: BaseModel | list[BaseModel] | None) -> str:
    if isinstance(value, list):
        payload = [item.model_dump(mode="json") for item in value]
    elif value is None:
        payload = None
    else:
        payload = value.model_dump(mode="json")
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _bound_ticker_identity(context: AgentContext) -> tuple[str, str]:
    user_id = context.subscription.user_id.strip()
    ticker = context.ticker.strip().upper()
    subscription_ticker = context.subscription.ticker.strip().upper()
    if not user_id:
        raise AgentToolConfigurationError("Agent context user_id must not be blank.")
    if not ticker or ticker != subscription_ticker:
        raise AgentToolConfigurationError(
            "Agent context ticker must match the subscription ticker."
        )
    return user_id, ticker


def build_ticker_agent_tools(
    context: AgentContext,
    *,
    client: Client | None = None,
) -> list[BaseTool]:
    """Build database tools locked to one user and ticker."""
    user_id, ticker = _bound_ticker_identity(context)

    def market_history(limit: int = 50) -> str:
        try:
            points = get_latest_ticker_data(ticker, limit=limit, client=client)
        except Exception:
            raise ToolException("Market history is temporarily unavailable.") from None
        return _serialize_tool_models(points)

    def latest_agent_update(agent_type: AgentType) -> str:
        try:
            output = get_latest_update(
                user_id,
                agent_type,
                ticker,
                client=client,
            )
        except Exception:
            raise ToolException("Previous agent output is temporarily unavailable.") from None
        return _serialize_tool_models(output)

    def recent_agent_updates(
        agent_type: AgentType | None = None,
        limit: int = 5,
        since: datetime | None = None,
    ) -> str:
        try:
            outputs = list_recent_updates(
                user_id,
                ticker=ticker,
                agent_type=agent_type,
                limit=limit,
                since=since,
                client=client,
            )
        except Exception:
            raise ToolException("Recent agent outputs are temporarily unavailable.") from None
        return _serialize_tool_models(outputs)

    def recent_alert_history(
        alert_type: AlertType | None = None,
        limit: int = 5,
        since: datetime | None = None,
    ) -> str:
        try:
            alerts = list_recent_alerts(
                user_id,
                ticker=ticker,
                alert_type=alert_type,
                limit=limit,
                since=since,
                client=client,
            )
        except Exception:
            raise ToolException("Recent alerts are temporarily unavailable.") from None
        return _serialize_tool_models(alerts)

    return [
        StructuredTool.from_function(
            func=market_history,
            name="get_market_history",
            description="Load recent 15-minute OHLCV candles for the current ticker.",
            args_schema=MarketHistoryToolInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=latest_agent_update,
            name="get_latest_agent_update",
            description="Load the newest output from one agent for the current position.",
            args_schema=LatestUpdateToolInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=recent_agent_updates,
            name="get_recent_agent_updates",
            description="Load recent agent outputs for the current position.",
            args_schema=RecentUpdatesToolInput,
            handle_tool_error=True,
        ),
        StructuredTool.from_function(
            func=recent_alert_history,
            name="get_recent_alerts",
            description="Load recent alert history for the current position.",
            args_schema=RecentAlertsToolInput,
            handle_tool_error=True,
        ),
    ]
