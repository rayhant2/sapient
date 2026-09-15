from __future__ import annotations

from statistics import fmean, pstdev
from typing import Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator
from supabase import Client

from agents.core import (
    AgentExecutionError,
    TickerAgentState,
    TickerAnalysisDraft,
    WebResearchError,
    WebResearchFocus,
    WebResearchResult,
    bind_structured_output,
    execute_ticker_agent,
    finalize_ticker_output,
    perform_web_research,
    web_research_state_update,
)
from config.prompts import (
    SCHEDULED_REVIEW_SYSTEM_PROMPT,
    scheduled_review_analysis_prompt,
    scheduled_review_research_decision_prompt,
)
from models.schemas import AgentContext, AgentOutput, AgentType, HypothesisOutput
from security.credentials import CredentialCipher


class ScheduledResearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_web: bool
    focus: WebResearchFocus | None = None
    reason: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def focus_must_match_search_decision(self) -> ScheduledResearchDecision:
        self.reason = self.reason.strip()
        if not self.reason:
            raise ValueError("research decision reason must not be blank")
        if self.search_web and self.focus is None:
            raise ValueError("a research focus is required when search is enabled")
        if not self.search_web and self.focus is not None:
            raise ValueError("research focus must be omitted when search is disabled")
        return self


class ScheduledReviewState(TickerAgentState):
    review_payload: NotRequired[dict[str, Any]]
    research_decision: NotRequired[ScheduledResearchDecision]
    research_summary: NotRequired[str]
    research_error: NotRequired[str]


def _safe_change_pct(start: float, end: float) -> float | None:
    if start <= 0:
        return None
    return (end - start) / start


def _review_payload(state: TickerAgentState) -> dict[str, Any]:
    context = state["context"]
    points = sorted(context.datapoints, key=lambda point: point.timestamp)
    if not points:
        raise AgentExecutionError("Scheduled review requires market data.")

    closes = [point.close for point in points]
    returns = [
        change
        for previous, current in zip(closes, closes[1:])
        if (change := _safe_change_pct(previous, current)) is not None
    ]
    volumes = [point.volume for point in points]
    prior_average_volume = fmean(volumes[:-1]) if len(volumes) > 1 else None
    recent_start_index = max(0, len(points) - 5)
    recent_start_close = points[recent_start_index].close

    return {
        "position": {
            "ticker": context.ticker,
            "motive": context.subscription.motive.value,
            "average_price": context.subscription.avg_price,
            "shares": context.subscription.shares,
            "current_price": context.current_price,
            "unrealized_pnl": context.unrealized_pnl,
            "unrealized_pnl_pct": context.unrealized_pnl_pct,
        },
        "market_window": {
            "candle_count": len(points),
            "start": points[0].timestamp.isoformat(),
            "end": points[-1].timestamp.isoformat(),
            "window_change_pct": _safe_change_pct(closes[0], closes[-1]),
            "recent_four_candle_change_pct": _safe_change_pct(
                recent_start_close,
                closes[-1],
            ),
            "high": max(point.high for point in points),
            "low": min(point.low for point in points),
            "mean_candle_return": fmean(returns) if returns else 0.0,
            "candle_return_volatility": pstdev(returns) if len(returns) > 1 else 0.0,
            "average_volume": fmean(volumes),
            "latest_volume_ratio": (
                volumes[-1] / prior_average_volume
                if prior_average_volume is not None and prior_average_volume > 0
                else None
            ),
        },
        "recent_candles": [
            point.model_dump(mode="json") for point in points[-12:]
        ],
        "previous_updates": [
            {
                "timestamp": output.timestamp.isoformat(),
                "event_type": output.event_type.value,
                "summary": output.summary,
                "recommendation": output.recommendation,
                "confidence": output.confidence.value,
            }
            for output in state.get("previous_updates", [])
        ],
        "recent_alerts": [
            {
                "timestamp": alert.timestamp.isoformat(),
                "alert_type": alert.alert_type.value,
                "message": alert.message,
            }
            for alert in state.get("recent_alerts", [])
        ],
    }


def build_scheduled_review_graph(
    model: BaseChatModel,
    _tools: list[BaseTool],
):
    """Compile the routine scheduled-review graph for one user model."""
    decision_model = model.with_structured_output(
        ScheduledResearchDecision,
        method="json_schema",
        include_raw=False,
    )
    output_model = bind_structured_output(model, AgentType.SCHEDULED_REVIEW)

    def prepare_review(state: ScheduledReviewState) -> dict[str, Any]:
        payload = _review_payload(state)
        return {
            "review_payload": payload,
            "messages": [
                HumanMessage(
                    content=scheduled_review_research_decision_prompt(payload)
                )
            ],
        }

    def decide_research(state: ScheduledReviewState) -> dict[str, Any]:
        decision = decision_model.invoke(
            [
                SystemMessage(content=SCHEDULED_REVIEW_SYSTEM_PROMPT),
                state["messages"][-1],
            ]
        )
        if not isinstance(decision, ScheduledResearchDecision):
            raise AgentExecutionError(
                "Scheduled review returned an invalid research decision."
            )
        return {"research_decision": decision}

    def route_research(state: ScheduledReviewState) -> str:
        decision = state.get("research_decision")
        if not isinstance(decision, ScheduledResearchDecision):
            raise AgentExecutionError("Scheduled review research decision is missing.")
        return "research" if decision.search_web else "synthesize"

    def research(state: ScheduledReviewState) -> dict[str, Any]:
        decision = state["research_decision"]
        if decision.focus is None:
            raise AgentExecutionError("Scheduled review research focus is missing.")
        try:
            result = perform_web_research(
                model,
                state["context"],
                decision.focus,
            )
        except WebResearchError:
            return {
                "research_error": "Current web research was unavailable.",
                "research_summary": "Current web research was unavailable.",
            }
        return {
            **web_research_state_update(result),
            "research_summary": result.summary,
        }

    def synthesize(state: ScheduledReviewState) -> dict[str, Any]:
        draft = output_model.invoke(
            [
                SystemMessage(content=SCHEDULED_REVIEW_SYSTEM_PROMPT),
                HumanMessage(
                    content=scheduled_review_analysis_prompt(
                        state["review_payload"],
                        research_summary=state.get("research_summary"),
                    )
                ),
            ]
        )
        return {
            "output": finalize_ticker_output(
                state,
                AgentType.SCHEDULED_REVIEW,
                draft,
            )
        }

    graph = StateGraph(ScheduledReviewState)
    graph.add_node("prepare_review", prepare_review)
    graph.add_node("decide_research", decide_research)
    graph.add_node("research", research)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "prepare_review")
    graph.add_edge("prepare_review", "decide_research")
    graph.add_conditional_edges(
        "decide_research",
        route_research,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_scheduled_review(
    context: AgentContext,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> AgentOutput:
    """Execute the scheduled review through the shared agent boundary."""
    output = execute_ticker_agent(
        context,
        AgentType.SCHEDULED_REVIEW,
        build_scheduled_review_graph,
        database_client=database_client,
        credential_client=credential_client,
        cipher=cipher,
    )
    if isinstance(output, HypothesisOutput):
        raise AgentExecutionError("Scheduled review returned a hypothesis output.")
    return output
