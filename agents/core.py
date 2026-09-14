from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from operator import add
from typing import Annotated, Any, Iterator, NotRequired, TypeAlias, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
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
    ResearchSource,
)
from security.credentials import CredentialCipher, CredentialSecurityError


TickerAgentOutput: TypeAlias = AgentOutput | HypothesisOutput
MAX_AGENT_MARKET_RESULTS = 150
MAX_AGENT_MEMORY_RESULTS = 20
WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
_PUBLIC_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.:-]{0,19}$")


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


class WebResearchError(RuntimeError):
    """Raised when server-side web research cannot produce a usable result."""


class WebResearchFocus(str, Enum):
    PRICE_CATALYST = "price_catalyst"
    COMPANY_NEWS = "company_news"
    EARNINGS = "earnings"
    SECTOR = "sector"
    REGULATORY = "regulatory"


class WebResearchResult(BaseModel):
    summary: str
    sources: list[ResearchSource] = Field(default_factory=list)
    search_requests: int = Field(default=0, ge=0)
    messages: list[AIMessage] = Field(default_factory=list, exclude=True)


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
    research_sources: NotRequired[Annotated[list[ResearchSource], add]]
    web_search_requests: NotRequired[Annotated[int, add]]
    output: NotRequired[TickerAgentOutput | None]


def build_ticker_agent_state(context: AgentContext) -> TickerAgentState:
    """Create isolated state for one user and ticker agent run."""
    return {
        "context": context,
        "messages": [],
        "previous_updates": [],
        "recent_alerts": [],
        "research_sources": [],
        "web_search_requests": 0,
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


def bind_web_search(
    model: BaseChatModel,
    *,
    max_uses: int | None = None,
) -> Runnable[list[BaseMessage], AIMessage]:
    """Bind only Anthropic's server-side web search to a model."""
    search_limit = (
        settings.agent_web_search_max_uses if max_uses is None else max_uses
    )
    if (
        isinstance(search_limit, bool)
        or not isinstance(search_limit, int)
        or not 1 <= search_limit <= 5
    ):
        raise ValueError("max_uses must be an integer between 1 and 5.")
    return model.bind_tools(
        [
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "max_uses": search_limit,
            }
        ]
    )


def _public_research_prompt(ticker: str, focus: WebResearchFocus) -> str:
    instructions = {
        WebResearchFocus.PRICE_CATALYST: (
            "Find current public news, filings, or market events that may explain "
            "the ticker's recent price movement."
        ),
        WebResearchFocus.COMPANY_NEWS: (
            "Find material recent company news, announcements, and filings."
        ),
        WebResearchFocus.EARNINGS: (
            "Find the latest earnings release, guidance, and material analyst context."
        ),
        WebResearchFocus.SECTOR: (
            "Find current sector or industry developments that may affect this ticker."
        ),
        WebResearchFocus.REGULATORY: (
            "Find current regulatory, legal, or policy developments affecting this ticker."
        ),
    }
    return (
        f"Research the public-market ticker {ticker}. {instructions[focus]} "
        "Use current, reputable primary sources when available. Distinguish confirmed "
        "facts from inference, keep the result concise, and cite every material claim."
    )


def _walk_content(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_content(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_content(nested)


def _research_error_codes(message: AIMessage) -> set[str]:
    return {
        str(block.get("error_code"))
        for block in _walk_content(message.content)
        if block.get("type") == "web_search_tool_result_error"
        and block.get("error_code")
    }


def _research_sources(messages: list[AIMessage]) -> list[ResearchSource]:
    sources_by_url: dict[str, ResearchSource] = {}
    for message in messages:
        for block in _walk_content(message.content):
            if block.get("type") not in {
                "web_search_result",
                "web_search_result_location",
            }:
                continue
            url = block.get("url")
            title = block.get("title")
            if not isinstance(url, str) or not isinstance(title, str):
                continue
            existing = sources_by_url.get(url)
            sources_by_url[url] = ResearchSource(
                title=title,
                url=url,
                page_age=block.get("page_age")
                or (existing.page_age if existing else None),
                cited_text=block.get("cited_text")
                or (existing.cited_text if existing else None),
            )
    return list(sources_by_url.values())


def _research_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    text_parts = [
        block["text"].strip()
        for block in message.content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    return "\n".join(text_parts)


def _research_request_count(message: AIMessage) -> int:
    usage = message.response_metadata.get("usage", {})
    if isinstance(usage, dict):
        server_usage = usage.get("server_tool_use", {})
        if isinstance(server_usage, dict):
            count = server_usage.get("web_search_requests")
            if isinstance(count, int) and count >= 0:
                return count
    return sum(
        1
        for block in _walk_content(message.content)
        if block.get("type") == "server_tool_use"
        and block.get("name") == "web_search"
    )


def _web_research_error_message(error_codes: set[str]) -> str:
    if "max_uses_exceeded" in error_codes:
        return "Web research reached its configured search limit."
    if "too_many_requests" in error_codes:
        return "Web research is temporarily rate limited."
    if "unavailable" in error_codes:
        return "Web research is temporarily unavailable."
    return "Web research could not complete the requested search."


def perform_web_research(
    model: BaseChatModel,
    context: AgentContext,
    focus: WebResearchFocus | str,
    *,
    max_uses: int | None = None,
    max_continuations: int | None = None,
) -> WebResearchResult:
    """Run isolated, public-only Anthropic research for one ticker."""
    _, ticker = _bound_ticker_identity(context)
    normalized_focus = WebResearchFocus(focus)
    continuation_limit = (
        settings.agent_web_search_max_continuations
        if max_continuations is None
        else max_continuations
    )
    if (
        isinstance(continuation_limit, bool)
        or not isinstance(continuation_limit, int)
        or not 0 <= continuation_limit <= 2
    ):
        raise ValueError("max_continuations must be an integer between 0 and 2.")

    research_model = bind_web_search(model, max_uses=max_uses)
    conversation: list[BaseMessage] = [
        HumanMessage(content=_public_research_prompt(ticker, normalized_focus))
    ]
    responses: list[AIMessage] = []

    try:
        for attempt in range(continuation_limit + 1):
            response = research_model.invoke(conversation)
            if not isinstance(response, AIMessage):
                raise WebResearchError("Web research returned an invalid response.")
            responses.append(response)

            error_codes = _research_error_codes(response)
            if error_codes:
                raise WebResearchError(_web_research_error_message(error_codes))

            if response.response_metadata.get("stop_reason") != "pause_turn":
                break
            if attempt == continuation_limit:
                raise WebResearchError(
                    "Web research paused beyond its configured continuation limit."
                )
            conversation.append(response)
    except WebResearchError:
        raise
    except Exception:
        raise WebResearchError("Web research is temporarily unavailable.") from None

    summary = _research_text(responses[-1])
    if not summary:
        raise WebResearchError("Web research returned no usable summary.")
    return WebResearchResult(
        summary=summary,
        sources=_research_sources(responses),
        search_requests=sum(_research_request_count(response) for response in responses),
        messages=responses,
    )


def web_research_state_update(result: WebResearchResult) -> dict[str, Any]:
    """Convert research into a partial update for TickerAgentState."""
    return {
        "messages": result.messages,
        "research_sources": result.sources,
        "web_search_requests": result.search_requests,
    }


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
    if (
        not _PUBLIC_TICKER_PATTERN.fullmatch(ticker)
        or ticker != subscription_ticker
    ):
        raise AgentToolConfigurationError(
            "Agent context ticker must be a valid public symbol and match the "
            "subscription ticker."
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
