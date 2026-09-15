from __future__ import annotations

from enum import Enum
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
    WebResearchError,
    WebResearchFocus,
    bind_structured_output,
    execute_ticker_agent,
    finalize_ticker_output,
    perform_web_research,
    web_research_state_update,
)
from config.prompts import (
    MOTIVE_REASSESSMENT_SYSTEM_PROMPT,
    motive_assessment_prompt,
    motive_reassessment_analysis_prompt,
)
from models.schemas import AgentContext, AgentOutput, AgentType, HypothesisOutput
from security.credentials import CredentialCipher


class MotiveAlignment(str, Enum):
    ALIGNED = "aligned"
    AT_RISK = "at_risk"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class MotiveAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alignment: MotiveAlignment
    search_web: bool
    focus: WebResearchFocus | None = None
    reason: str = Field(min_length=1, max_length=750)

    @model_validator(mode="after")
    def research_fields_must_agree(self) -> MotiveAssessment:
        self.reason = self.reason.strip()
        if not self.reason:
            raise ValueError("motive assessment reason must not be blank")
        if self.search_web and self.focus is None:
            raise ValueError("research focus is required when search is enabled")
        if not self.search_web and self.focus is not None:
            raise ValueError("research focus must be omitted when search is disabled")
        return self


class MotiveState(TickerAgentState):
    motive_payload: NotRequired[dict[str, Any]]
    assessment: NotRequired[MotiveAssessment]
    research_summary: NotRequired[str]
    research_error: NotRequired[str]


def _change_pct(start: float, end: float) -> float | None:
    if start <= 0:
        return None
    return (end - start) / start


def _motive_payload(state: TickerAgentState) -> dict[str, Any]:
    context = state["context"]
    points = sorted(context.datapoints, key=lambda point: point.timestamp)
    if len(points) < 2:
        raise AgentExecutionError(
            "Motive reassessment requires at least two market candles."
        )

    closes = [point.close for point in points]
    returns = [
        change
        for previous, current in zip(closes, closes[1:])
        if (change := _change_pct(previous, current)) is not None
    ]
    recent_points = points[-26:]
    prior_recent_volumes = [point.volume for point in recent_points[:-4]]
    recent_volumes = [point.volume for point in recent_points[-4:]]
    prior_volume_average = (
        fmean(prior_recent_volumes) if prior_recent_volumes else None
    )
    window_high = max(point.high for point in points)
    window_low = min(point.low for point in points)

    return {
        "motive": {
            "ticker": context.ticker,
            "stated_motive": context.subscription.motive.value,
            "average_price": context.subscription.avg_price,
            "shares": context.subscription.shares,
            "current_price": context.current_price,
            "unrealized_pnl": context.unrealized_pnl,
            "unrealized_pnl_pct": context.unrealized_pnl_pct,
        },
        "available_market_window": {
            "candle_count": len(points),
            "start": points[0].timestamp.isoformat(),
            "end": points[-1].timestamp.isoformat(),
            "window_change_pct": _change_pct(closes[0], closes[-1]),
            "recent_session_change_pct": _change_pct(
                recent_points[0].close,
                recent_points[-1].close,
            ),
            "return_volatility": pstdev(returns) if len(returns) > 1 else 0.0,
            "positive_candle_ratio": (
                sum(change > 0 for change in returns) / len(returns)
                if returns
                else 0.0
            ),
            "window_high": window_high,
            "window_low": window_low,
            "drawdown_from_window_high_pct": _change_pct(
                window_high,
                closes[-1],
            ),
            "rebound_from_window_low_pct": _change_pct(window_low, closes[-1]),
            "recent_volume_ratio": (
                fmean(recent_volumes) / prior_volume_average
                if recent_volumes
                and prior_volume_average is not None
                and prior_volume_average > 0
                else None
            ),
        },
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
        "limitations": [
            "No position entry timestamp is stored.",
            "No free-form entry thesis is stored.",
            "Price analysis is limited to the rolling OHLCV window.",
        ],
    }


def build_motive_graph(model: BaseChatModel, _tools: list[BaseTool]):
    """Compile the weekly motive-reassessment graph."""
    assessment_model = model.with_structured_output(
        MotiveAssessment,
        method="json_schema",
        include_raw=False,
    )
    output_model = bind_structured_output(model, AgentType.MOTIVE)

    def prepare(state: MotiveState) -> dict[str, Any]:
        return {"motive_payload": _motive_payload(state)}

    def assess(state: MotiveState) -> dict[str, Any]:
        assessment = assessment_model.invoke(
            [
                SystemMessage(content=MOTIVE_REASSESSMENT_SYSTEM_PROMPT),
                HumanMessage(content=motive_assessment_prompt(state["motive_payload"])),
            ]
        )
        if not isinstance(assessment, MotiveAssessment):
            raise AgentExecutionError(
                "Motive agent returned an invalid preliminary assessment."
            )
        return {"assessment": assessment}

    def route_research(state: MotiveState) -> str:
        assessment = state.get("assessment")
        if not isinstance(assessment, MotiveAssessment):
            raise AgentExecutionError("Motive assessment is missing.")
        return "research" if assessment.search_web else "synthesize"

    def research(state: MotiveState) -> dict[str, Any]:
        assessment = state["assessment"]
        if assessment.focus is None:
            raise AgentExecutionError("Motive research focus is missing.")
        try:
            result = perform_web_research(
                model,
                state["context"],
                assessment.focus,
            )
        except WebResearchError:
            return {
                "research_summary": "Current motive research was unavailable.",
                "research_error": "Current motive research was unavailable.",
            }
        return {
            **web_research_state_update(result),
            "research_summary": result.summary,
        }

    def synthesize(state: MotiveState) -> dict[str, Any]:
        draft = output_model.invoke(
            [
                SystemMessage(content=MOTIVE_REASSESSMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=motive_reassessment_analysis_prompt(
                        state["motive_payload"],
                        state["assessment"].model_dump(mode="json"),
                        state.get("research_summary"),
                    )
                ),
            ]
        )
        return {
            "output": finalize_ticker_output(
                state,
                AgentType.MOTIVE,
                draft,
            )
        }

    graph = StateGraph(MotiveState)
    graph.add_node("prepare", prepare)
    graph.add_node("assess", assess)
    graph.add_node("research", research)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "assess")
    graph.add_conditional_edges(
        "assess",
        route_research,
        {"research": "research", "synthesize": "synthesize"},
    )
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_motive_reassessment(
    context: AgentContext,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> AgentOutput:
    """Execute and persist one weekly motive reassessment."""
    output = execute_ticker_agent(
        context,
        AgentType.MOTIVE,
        build_motive_graph,
        database_client=database_client,
        credential_client=credential_client,
        cipher=cipher,
    )
    if isinstance(output, HypothesisOutput):
        raise AgentExecutionError("Motive agent returned a hypothesis output.")
    return output
