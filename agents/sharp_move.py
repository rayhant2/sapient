from __future__ import annotations

import json
import math
from statistics import fmean, stdev
from typing import Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    SHARP_MOVE_SYSTEM_PROMPT,
    sharp_move_analysis_prompt,
    sharp_move_sector_decision_prompt,
)
from core.monitor import (
    DEFAULT_ANOMALY_MOVE_FLOOR,
    DEFAULT_MIN_VOLATILITY_OBSERVATIONS,
    DEFAULT_VOLATILITY_WINDOW,
    DEFAULT_VOLATILITY_Z_SCORE,
    DEFAULT_VWAP_WINDOW,
)
from models.schemas import AgentContext, AgentOutput, AgentType, HypothesisOutput
from security.credentials import CredentialCipher


class SectorResearchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_sector: bool
    reason: str = Field(min_length=1, max_length=500)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("sector research reason must not be blank")
        return normalized


class SharpMoveState(TickerAgentState):
    move_payload: NotRequired[dict[str, Any]]
    portfolio_snapshot: NotRequired[list[dict[str, Any]]]
    catalyst_summary: NotRequired[str]
    sector_summary: NotRequired[str]
    sector_decision: NotRequired[SectorResearchDecision]
    research_error: NotRequired[str]


def _change_pct(start: float, end: float) -> float:
    if start <= 0:
        raise AgentExecutionError("Sharp-move analysis requires positive closes.")
    return (end - start) / start


def _move_payload(state: TickerAgentState) -> dict[str, Any]:
    context = state["context"]
    points = sorted(context.datapoints, key=lambda point: point.timestamp)
    if len(points) < 2:
        raise AgentExecutionError(
            "Sharp-move analysis requires at least two market candles."
        )

    returns = [
        _change_pct(previous.close, current.close)
        for previous, current in zip(points, points[1:])
    ]
    latest_return = returns[-1]
    prior_returns = returns[-(DEFAULT_VOLATILITY_WINDOW + 1) : -1]
    baseline_return: float | None = None
    volatility: float | None = None
    z_score: float | str | None = None
    volatility_trigger = False
    if len(prior_returns) >= DEFAULT_MIN_VOLATILITY_OBSERVATIONS:
        baseline_return = fmean(prior_returns)
        volatility = stdev(prior_returns)
        difference = abs(latest_return - baseline_return)
        if math.isclose(volatility, 0.0, abs_tol=1e-12):
            z_score = (
                0.0
                if math.isclose(difference, 0.0, abs_tol=1e-12)
                else "infinite"
            )
            volatility_trigger = z_score == "infinite"
        else:
            z_score = difference / volatility
            volatility_trigger = z_score >= DEFAULT_VOLATILITY_Z_SCORE
        volatility_trigger = (
            volatility_trigger and abs(latest_return) >= DEFAULT_ANOMALY_MOVE_FLOOR
        )

    vwap_points = points[-DEFAULT_VWAP_WINDOW:]
    total_volume = sum(point.volume for point in vwap_points)
    rolling_vwap = (
        sum(point.close * point.volume for point in vwap_points) / total_volume
        if total_volume > 0
        else fmean(point.close for point in vwap_points)
    )
    prior_volumes = [point.volume for point in points[:-1]]
    prior_average_volume = fmean(prior_volumes) if prior_volumes else None
    average_price = context.subscription.avg_price
    previous_close = points[-2].close
    current_close = points[-1].close

    return {
        "position": {
            "ticker": context.ticker,
            "motive": context.subscription.motive.value,
            "average_price": average_price,
            "shares": context.subscription.shares,
            "current_price": context.current_price,
            "unrealized_pnl": context.unrealized_pnl,
            "unrealized_pnl_pct": context.unrealized_pnl_pct,
        },
        "trigger": {
            "candle_timestamp": points[-1].timestamp.isoformat(),
            "previous_close": previous_close,
            "current_close": current_close,
            "move_pct": latest_return,
            "direction": "up" if latest_return > 0 else "down",
            "personal_threshold": context.subscription.sharp_move_threshold,
            "absolute_threshold_triggered": (
                abs(latest_return) >= context.subscription.sharp_move_threshold
            ),
            "baseline_return": baseline_return,
            "return_volatility": volatility,
            "volatility_z_score": z_score,
            "volatility_triggered": volatility_trigger,
            "rolling_vwap": rolling_vwap,
            "distance_from_vwap_pct": _change_pct(rolling_vwap, current_close),
            "crossed_cost_basis": (
                previous_close < average_price <= current_close
                or current_close <= average_price < previous_close
            ),
            "latest_volume": points[-1].volume,
            "latest_volume_ratio": (
                points[-1].volume / prior_average_volume
                if prior_average_volume is not None and prior_average_volume > 0
                else None
            ),
        },
        "window": {
            "candle_count": len(points),
            "start": points[0].timestamp.isoformat(),
            "end": points[-1].timestamp.isoformat(),
            "window_change_pct": _change_pct(points[0].close, current_close),
            "high": max(point.high for point in points),
            "low": min(point.low for point in points),
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
    }


def _load_portfolio_snapshot(tools: list[BaseTool]) -> list[dict[str, Any]]:
    tool = next(
        (item for item in tools if item.name == "get_portfolio_snapshot"),
        None,
    )
    if tool is None:
        return []
    raw_snapshot = tool.invoke({})
    if not isinstance(raw_snapshot, str):
        return []
    try:
        parsed = json.loads(raw_snapshot)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    snapshot: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict) or not isinstance(item.get("ticker"), str):
            continue
        snapshot.append(
            {
                "ticker": item["ticker"][:20],
                "event_type": item.get("event_type"),
                "summary": str(item.get("summary", ""))[:1000],
                "recommendation": str(item.get("recommendation", ""))[:500],
                "confidence": item.get("confidence"),
                "timestamp": item.get("timestamp"),
                "price_at_update": item.get("price_at_update"),
            }
        )
        if len(snapshot) == 20:
            break
    return snapshot


def build_sharp_move_graph(model: BaseChatModel, tools: list[BaseTool]):
    """Compile the immediate sharp-move investigation graph."""
    sector_decision_model = model.with_structured_output(
        SectorResearchDecision,
        method="json_schema",
        include_raw=False,
    )
    output_model = bind_structured_output(model, AgentType.SHARP_MOVE)

    def prepare(state: SharpMoveState) -> dict[str, Any]:
        return {
            "move_payload": _move_payload(state),
            "portfolio_snapshot": _load_portfolio_snapshot(tools),
        }

    def research_catalyst(state: SharpMoveState) -> dict[str, Any]:
        try:
            result = perform_web_research(
                model,
                state["context"],
                WebResearchFocus.PRICE_CATALYST,
            )
        except WebResearchError:
            return {
                "catalyst_summary": "Current catalyst research was unavailable.",
                "research_error": "Current catalyst research was unavailable.",
            }
        return {
            **web_research_state_update(result),
            "catalyst_summary": result.summary,
        }

    def decide_sector_research(state: SharpMoveState) -> dict[str, Any]:
        decision = sector_decision_model.invoke(
            [
                SystemMessage(content=SHARP_MOVE_SYSTEM_PROMPT),
                HumanMessage(
                    content=sharp_move_sector_decision_prompt(
                        state["move_payload"],
                        state["catalyst_summary"],
                        state.get("portfolio_snapshot", []),
                    )
                ),
            ]
        )
        if not isinstance(decision, SectorResearchDecision):
            raise AgentExecutionError(
                "Sharp-move agent returned an invalid sector decision."
            )
        return {"sector_decision": decision}

    def route_sector_research(state: SharpMoveState) -> str:
        decision = state.get("sector_decision")
        if not isinstance(decision, SectorResearchDecision):
            raise AgentExecutionError("Sharp-move sector decision is missing.")
        return "research_sector" if decision.search_sector else "synthesize"

    def research_sector(state: SharpMoveState) -> dict[str, Any]:
        try:
            result = perform_web_research(
                model,
                state["context"],
                WebResearchFocus.SECTOR,
            )
        except WebResearchError:
            return {
                "sector_summary": "Current sector research was unavailable.",
                "research_error": "Current sector research was unavailable.",
            }
        return {
            **web_research_state_update(result),
            "sector_summary": result.summary,
        }

    def synthesize(state: SharpMoveState) -> dict[str, Any]:
        draft = output_model.invoke(
            [
                SystemMessage(content=SHARP_MOVE_SYSTEM_PROMPT),
                HumanMessage(
                    content=sharp_move_analysis_prompt(
                        state["move_payload"],
                        state["catalyst_summary"],
                        state.get("sector_summary"),
                        state.get("portfolio_snapshot", []),
                    )
                ),
            ]
        )
        return {
            "output": finalize_ticker_output(
                state,
                AgentType.SHARP_MOVE,
                draft,
            )
        }

    graph = StateGraph(SharpMoveState)
    graph.add_node("prepare", prepare)
    graph.add_node("research_catalyst", research_catalyst)
    graph.add_node("decide_sector_research", decide_sector_research)
    graph.add_node("research_sector", research_sector)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "research_catalyst")
    graph.add_edge("research_catalyst", "decide_sector_research")
    graph.add_conditional_edges(
        "decide_sector_research",
        route_sector_research,
        {"research_sector": "research_sector", "synthesize": "synthesize"},
    )
    graph.add_edge("research_sector", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_sharp_move(
    context: AgentContext,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> AgentOutput:
    """Execute and persist an immediate sharp-move investigation."""
    output = execute_ticker_agent(
        context,
        AgentType.SHARP_MOVE,
        build_sharp_move_graph,
        database_client=database_client,
        credential_client=credential_client,
        cipher=cipher,
    )
    if isinstance(output, HypothesisOutput):
        raise AgentExecutionError("Sharp-move agent returned a hypothesis output.")
    return output
