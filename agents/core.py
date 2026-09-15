from __future__ import annotations

import json
import re
from datetime import datetime
from enum import Enum
from operator import add
from typing import (
    Annotated,
    Any,
    Callable,
    Iterator,
    Mapping,
    NotRequired,
    TypeAlias,
    TypedDict,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from supabase import Client

from config.prompts import web_research_prompt
from config.settings import settings
from data.database import (
    DatabaseError,
    get_latest_ticker_data,
    get_latest_update,
    insert_agent_output,
    list_latest_portfolio_updates,
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
    Confidence,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    PortfolioContext,
    ResearchSource,
)
from security.credentials import CredentialCipher, CredentialSecurityError


TickerAgentOutput: TypeAlias = AgentOutput | HypothesisOutput
StructuredAgentOutput: TypeAlias = TickerAgentOutput | CrossPortfolioOutput
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


class AgentOutputValidationError(RuntimeError):
    """Raised when model analysis cannot become a trusted agent output."""


class AgentExecutionError(RuntimeError):
    """Raised when an agent graph cannot complete safely."""


class AgentPersistenceError(RuntimeError):
    """Raised when a completed agent output cannot be persisted."""


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


class StructuredOutputDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TickerAnalysisDraft(StructuredOutputDraft):
    summary: str = Field(min_length=1, max_length=4000)
    recommendation: str = Field(min_length=1, max_length=2000)
    confidence: Confidence

    @field_validator("summary", "recommendation")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("analysis text must not be blank")
        return normalized


class HypothesisAnalysisDraft(StructuredOutputDraft):
    summary: str | None = Field(default=None, max_length=4000)
    recommendation: str = Field(default="", max_length=2000)
    confidence: Confidence
    flagged: bool
    recommended_next_scan_days: int = Field(ge=1, le=3)

    @model_validator(mode="after")
    def content_must_match_flag(self) -> HypothesisAnalysisDraft:
        summary = self.summary.strip() if self.summary is not None else None
        recommendation = self.recommendation.strip()
        if self.flagged and (not summary or not recommendation):
            raise ValueError(
                "flagged hypotheses require a summary and recommendation"
            )
        if not self.flagged and (summary or recommendation):
            raise ValueError(
                "unflagged hypotheses must not include a summary or recommendation"
            )
        if self.flagged and self.recommended_next_scan_days == 3:
            raise ValueError("flagged hypotheses require a one- or two-day rescan")
        if not self.flagged and self.recommended_next_scan_days != 3:
            raise ValueError("unflagged hypotheses require a three-day rescan")
        self.summary = summary
        self.recommendation = recommendation
        return self


class CrossPortfolioAnalysisDraft(StructuredOutputDraft):
    summary: str = Field(min_length=1, max_length=4000)
    correlations_flagged: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("summary")
    @classmethod
    def summary_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("summary must not be blank")
        return normalized

    @field_validator("correlations_flagged")
    @classmethod
    def correlations_must_not_be_blank(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("correlations must not contain blank entries")
        return normalized


StructuredOutputDraftType: TypeAlias = (
    TickerAnalysisDraft | HypothesisAnalysisDraft | CrossPortfolioAnalysisDraft
)

_DRAFT_SCHEMA_BY_AGENT: dict[AgentType, type[StructuredOutputDraft]] = {
    AgentType.SCHEDULED_REVIEW: TickerAnalysisDraft,
    AgentType.SHARP_MOVE: TickerAnalysisDraft,
    AgentType.MOTIVE: TickerAnalysisDraft,
    AgentType.HYPOTHESIS: HypothesisAnalysisDraft,
    AgentType.CROSS_PORTFOLIO: CrossPortfolioAnalysisDraft,
}

_EVENT_TYPE_BY_AGENT = {
    AgentType.SCHEDULED_REVIEW: EventType.SCHEDULED_UPDATE,
    AgentType.SHARP_MOVE: EventType.SHARP_MOVE,
    AgentType.MOTIVE: EventType.MOTIVE_CHECK,
    AgentType.HYPOTHESIS: EventType.HYPOTHESIS_SCAN,
}


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


class PortfolioSnapshotToolInput(AgentToolInput):
    pass


class TickerAgentState(TypedDict):
    """Shared, credential-free state for all single-ticker agent graphs."""

    context: AgentContext
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    previous_updates: NotRequired[list[TickerAgentOutput]]
    recent_alerts: NotRequired[list[Alert]]
    research_sources: NotRequired[Annotated[list[ResearchSource], add]]
    web_search_requests: NotRequired[Annotated[int, add]]
    output: NotRequired[TickerAgentOutput | None]


class PortfolioAgentState(TypedDict):
    """Credential-free state for the cross-portfolio graph."""

    context: PortfolioContext
    messages: NotRequired[Annotated[list[AnyMessage], add_messages]]
    output: NotRequired[CrossPortfolioOutput | None]


TickerGraphFactory: TypeAlias = Callable[
    [BaseChatModel, list[BaseTool]],
    Runnable[TickerAgentState, Mapping[str, Any]],
]
PortfolioGraphFactory: TypeAlias = Callable[
    [BaseChatModel],
    Runnable[PortfolioAgentState, Mapping[str, Any]],
]


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


def build_portfolio_agent_state(context: PortfolioContext) -> PortfolioAgentState:
    """Create isolated state for one user's cross-portfolio run."""
    return {
        "context": context,
        "messages": [],
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


def get_structured_output_schema(
    agent_type: AgentType | str,
) -> type[StructuredOutputDraft]:
    """Return the model-owned draft schema for an agent."""
    return _DRAFT_SCHEMA_BY_AGENT[AgentType(agent_type)]


def bind_structured_output(
    model: BaseChatModel,
    agent_type: AgentType | str,
) -> Runnable[Any, StructuredOutputDraftType]:
    """Bind native structured output without exposing trusted output fields."""
    schema = get_structured_output_schema(agent_type)
    return model.with_structured_output(
        schema,
        method="json_schema",
        include_raw=False,
    )


def _validate_structured_draft(
    agent_type: AgentType,
    draft: BaseModel | dict[str, Any],
) -> StructuredOutputDraftType:
    schema = get_structured_output_schema(agent_type)
    try:
        return schema.model_validate(draft)
    except (ValidationError, TypeError):
        raise AgentOutputValidationError(
            "The model returned an invalid structured analysis."
        ) from None


def _trusted_research_metadata(
    state: Mapping[str, Any],
) -> tuple[list[ResearchSource], int]:
    raw_sources = state.get("research_sources", [])
    raw_request_count = state.get("web_search_requests", 0)
    if not isinstance(raw_sources, list):
        raise AgentOutputValidationError("Agent research sources are invalid.")
    if (
        isinstance(raw_request_count, bool)
        or not isinstance(raw_request_count, int)
        or raw_request_count < 0
    ):
        raise AgentOutputValidationError("Agent web-search usage is invalid.")

    sources_by_url: dict[str, ResearchSource] = {}
    try:
        for raw_source in raw_sources:
            source = ResearchSource.model_validate(raw_source)
            sources_by_url[str(source.url)] = source
    except (ValidationError, TypeError):
        raise AgentOutputValidationError("Agent research sources are invalid.") from None
    return list(sources_by_url.values()), raw_request_count


def finalize_ticker_output(
    state: TickerAgentState,
    agent_type: AgentType | str,
    draft: BaseModel | dict[str, Any],
) -> TickerAgentOutput:
    """Combine model analysis with trusted ticker-run fields."""
    normalized_agent_type = AgentType(agent_type)
    if normalized_agent_type == AgentType.CROSS_PORTFOLIO:
        raise AgentOutputValidationError(
            "Cross-portfolio output requires PortfolioContext."
        )
    context = state.get("context")
    if not isinstance(context, AgentContext):
        raise AgentOutputValidationError("Ticker agent context is missing or invalid.")
    user_id, ticker = _bound_ticker_identity(context)
    if context.event_type != _EVENT_TYPE_BY_AGENT[normalized_agent_type]:
        raise AgentOutputValidationError(
            "Agent type does not match the trusted event type."
        )

    validated_draft = _validate_structured_draft(normalized_agent_type, draft)
    sources, search_requests = _trusted_research_metadata(state)
    trusted_fields = {
        "ticker": ticker,
        "user_id": user_id,
        "event_type": context.event_type,
        "price_at_update": context.current_price,
        "searched_web": bool(sources or search_requests),
        "sources": sources,
        "web_search_requests": search_requests,
    }

    try:
        if isinstance(validated_draft, HypothesisAnalysisDraft):
            return HypothesisOutput(
                **trusted_fields,
                **validated_draft.model_dump(),
            )
        if not isinstance(validated_draft, TickerAnalysisDraft):
            raise AgentOutputValidationError(
                "Ticker agent received the wrong analysis schema."
            )
        return AgentOutput(
            **trusted_fields,
            **validated_draft.model_dump(),
        )
    except ValidationError:
        raise AgentOutputValidationError(
            "The structured analysis could not form a valid agent output."
        ) from None


def _trusted_portfolio_identity(
    context: PortfolioContext,
) -> tuple[str, list[str]]:
    portfolio_user_id = context.user_id.strip()
    if not portfolio_user_id or not context.positions:
        raise AgentOutputValidationError("Portfolio context is missing or invalid.")
    tickers: list[str] = []
    for position in context.positions:
        user_id, ticker = _bound_ticker_identity(position)
        if user_id != portfolio_user_id:
            raise AgentOutputValidationError(
                "Portfolio positions must belong to the trusted user."
            )
        if ticker not in tickers:
            tickers.append(ticker)
    return portfolio_user_id, tickers


def finalize_cross_portfolio_output(
    context: PortfolioContext,
    draft: BaseModel | dict[str, Any],
) -> CrossPortfolioOutput:
    """Combine portfolio analysis with trusted user and ticker fields."""
    portfolio_user_id, tickers = _trusted_portfolio_identity(context)

    validated_draft = _validate_structured_draft(
        AgentType.CROSS_PORTFOLIO,
        draft,
    )
    if not isinstance(validated_draft, CrossPortfolioAnalysisDraft):
        raise AgentOutputValidationError(
            "Cross-portfolio agent received the wrong analysis schema."
        )
    try:
        return CrossPortfolioOutput(
            user_id=portfolio_user_id,
            tickers_analyzed=tickers,
            **validated_draft.model_dump(),
        )
    except ValidationError:
        raise AgentOutputValidationError(
            "The structured analysis could not form a valid portfolio output."
        ) from None


def _agent_run_config(
    agent_type: AgentType,
    *,
    event_type: EventType | None = None,
    ticker: str | None = None,
    ticker_count: int | None = None,
) -> RunnableConfig:
    metadata: dict[str, Any] = {"agent_type": agent_type.value}
    tags = ["sentient", agent_type.value]
    if event_type is not None:
        metadata["event_type"] = event_type.value
        tags.append(event_type.value)
    if ticker is not None:
        metadata["ticker"] = ticker
    if ticker_count is not None:
        metadata["ticker_count"] = ticker_count
    return {
        "run_name": f"sentient.{agent_type.value}",
        "tags": tags,
        "metadata": metadata,
        "recursion_limit": settings.agent_graph_recursion_limit,
    }


def _timestamp_is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _validate_completed_ticker_output(
    result: Mapping[str, Any],
    context: AgentContext,
    agent_type: AgentType,
) -> TickerAgentOutput:
    output = result.get("output")
    if agent_type == AgentType.HYPOTHESIS:
        if not isinstance(output, HypothesisOutput):
            raise AgentOutputValidationError(
                "The graph did not return a trusted hypothesis output."
            )
    elif not isinstance(output, AgentOutput) or isinstance(output, HypothesisOutput):
        raise AgentOutputValidationError(
            "The graph did not return a trusted ticker output."
        )

    user_id, ticker = _bound_ticker_identity(context)
    sources, search_requests = _trusted_research_metadata(
        {
            "research_sources": result.get("research_sources", []),
            "web_search_requests": result.get("web_search_requests", 0),
        }
    )
    if (
        output.user_id != user_id
        or output.ticker != ticker
        or output.event_type != context.event_type
        or output.event_type != _EVENT_TYPE_BY_AGENT[agent_type]
        or output.price_at_update != context.current_price
        or output.sources != sources
        or output.web_search_requests != search_requests
        or output.searched_web != bool(sources or search_requests)
        or not _timestamp_is_aware(output.timestamp)
    ):
        raise AgentOutputValidationError(
            "The graph output does not match trusted execution context."
        )
    return output


def _validate_completed_portfolio_output(
    result: Mapping[str, Any],
    context: PortfolioContext,
) -> CrossPortfolioOutput:
    output = result.get("output")
    if not isinstance(output, CrossPortfolioOutput):
        raise AgentOutputValidationError(
            "The graph did not return a trusted cross-portfolio output."
        )
    user_id, tickers = _trusted_portfolio_identity(context)
    if (
        output.user_id != user_id
        or output.tickers_analyzed != tickers
        or not _timestamp_is_aware(output.timestamp)
    ):
        raise AgentOutputValidationError(
            "The graph output does not match trusted portfolio context."
        )
    return output


def _persist_completed_output(
    output: StructuredAgentOutput,
    *,
    client: Client | None,
) -> StructuredAgentOutput:
    try:
        return insert_agent_output(output, client=client)
    except Exception:
        raise AgentPersistenceError(
            "The completed agent output could not be persisted."
        ) from None


def execute_ticker_agent(
    context: AgentContext,
    agent_type: AgentType | str,
    graph_factory: TickerGraphFactory,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
    update_limit: int = 5,
    alert_limit: int = 5,
) -> TickerAgentOutput:
    """Execute and persist one ticker agent without retrying the full graph."""
    normalized_agent_type = AgentType(agent_type)
    if normalized_agent_type == AgentType.CROSS_PORTFOLIO:
        raise ValueError("Use execute_cross_portfolio_agent for cross-portfolio runs.")

    state = hydrate_ticker_agent_state(
        context,
        normalized_agent_type,
        update_limit=update_limit,
        alert_limit=alert_limit,
        client=database_client,
    )
    tools = build_ticker_agent_tools(context, client=database_client)
    user_id, ticker = _bound_ticker_identity(context)
    model = create_user_model(
        user_id,
        client=credential_client,
        cipher=cipher,
    )

    try:
        graph = graph_factory(model, tools)
        result = graph.invoke(
            state,
            config=_agent_run_config(
                normalized_agent_type,
                event_type=context.event_type,
                ticker=ticker,
            ),
        )
    except AgentOutputValidationError:
        raise
    except Exception:
        raise AgentExecutionError("The agent graph could not complete.") from None
    if not isinstance(result, Mapping):
        raise AgentOutputValidationError("The agent graph returned invalid state.")

    output = _validate_completed_ticker_output(
        result,
        context,
        normalized_agent_type,
    )
    persisted = _persist_completed_output(output, client=database_client)
    if not isinstance(persisted, (AgentOutput, HypothesisOutput)):
        raise AgentPersistenceError(
            "The database returned an invalid ticker output after persistence."
        )
    try:
        return _validate_completed_ticker_output(
            {**result, "output": persisted},
            context,
            normalized_agent_type,
        )
    except AgentOutputValidationError:
        raise AgentPersistenceError(
            "The database returned an invalid ticker output after persistence."
        ) from None


def execute_cross_portfolio_agent(
    context: PortfolioContext,
    graph_factory: PortfolioGraphFactory,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> CrossPortfolioOutput:
    """Execute and persist one cross-portfolio graph for its trusted user."""
    user_id, tickers = _trusted_portfolio_identity(context)
    state = build_portfolio_agent_state(context)
    model = create_user_model(
        user_id,
        client=credential_client,
        cipher=cipher,
    )

    try:
        graph = graph_factory(model)
        result = graph.invoke(
            state,
            config=_agent_run_config(
                AgentType.CROSS_PORTFOLIO,
                ticker_count=len(tickers),
            ),
        )
    except AgentOutputValidationError:
        raise
    except Exception:
        raise AgentExecutionError("The agent graph could not complete.") from None
    if not isinstance(result, Mapping):
        raise AgentOutputValidationError("The agent graph returned invalid state.")

    output = _validate_completed_portfolio_output(result, context)
    persisted = _persist_completed_output(output, client=database_client)
    if not isinstance(persisted, CrossPortfolioOutput):
        raise AgentPersistenceError(
            "The database returned an invalid portfolio output after persistence."
        )
    try:
        return _validate_completed_portfolio_output(
            {**result, "output": persisted},
            context,
        )
    except AgentOutputValidationError:
        raise AgentPersistenceError(
            "The database returned an invalid portfolio output after persistence."
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
    return web_research_prompt(ticker, focus.value)


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
            raise ToolException(
                "Previous agent output is temporarily unavailable."
            ) from None
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
            raise ToolException(
                "Recent agent outputs are temporarily unavailable."
            ) from None
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

    def portfolio_snapshot() -> str:
        try:
            outputs = list_latest_portfolio_updates(user_id, client=client)
        except Exception:
            raise ToolException(
                "Portfolio updates are temporarily unavailable."
            ) from None
        payload = [
            {
                "ticker": getattr(output, "ticker", None),
                "event_type": getattr(output, "event_type", None),
                "summary": output.summary,
                "recommendation": getattr(output, "recommendation", None),
                "confidence": getattr(output, "confidence", None),
                "timestamp": output.timestamp,
                "price_at_update": getattr(output, "price_at_update", None),
            }
            for output in outputs
            if getattr(output, "ticker", None) != ticker
        ]
        return json.dumps(
            payload,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        )

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
        StructuredTool.from_function(
            func=portfolio_snapshot,
            name="get_portfolio_snapshot",
            description=(
                "Load the newest stored update for the user's other subscribed "
                "tickers without exposing user identity."
            ),
            args_schema=PortfolioSnapshotToolInput,
            handle_tool_error=True,
        ),
    ]
