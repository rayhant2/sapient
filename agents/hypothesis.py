from __future__ import annotations

from collections.abc import Callable
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
    HypothesisAnalysisDraft,
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
    HYPOTHESIS_SYSTEM_PROMPT,
    hypothesis_analysis_prompt,
    hypothesis_screen_prompt,
)
from models.schemas import AgentContext, AgentType, Confidence, HypothesisOutput
from security.credentials import CredentialCipher


ScheduleNextScan = Callable[[str, str, int], Any]


class HypothesisSchedulingError(RuntimeError):
    """Raised when a completed hypothesis run cannot schedule its next scan."""


class HypothesisScreen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_detected: bool
    pattern: str | None = Field(default=None, max_length=1000)
    focus: WebResearchFocus | None = None
    confidence: Confidence
    reason: str = Field(min_length=1, max_length=750)

    @model_validator(mode="after")
    def candidate_fields_must_agree(self) -> HypothesisScreen:
        self.reason = self.reason.strip()
        self.pattern = self.pattern.strip() if self.pattern is not None else None
        if not self.reason:
            raise ValueError("hypothesis screen reason must not be blank")
        if self.candidate_detected and (not self.pattern or self.focus is None):
            raise ValueError(
                "candidate hypotheses require a pattern and research focus"
            )
        if not self.candidate_detected and (
            self.pattern is not None or self.focus is not None
        ):
            raise ValueError("an empty screen must omit pattern and research focus")
        return self


class HypothesisState(TickerAgentState):
    hypothesis_payload: NotRequired[dict[str, Any]]
    screen: NotRequired[HypothesisScreen]
    research_summary: NotRequired[str]
    research_error: NotRequired[str]


def _change_pct(start: float, end: float) -> float | None:
    if start <= 0:
        return None
    return (end - start) / start


def _average(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _hypothesis_payload(state: TickerAgentState) -> dict[str, Any]:
    context = state["context"]
    points = sorted(context.datapoints, key=lambda point: point.timestamp)
    if len(points) < 12:
        raise AgentExecutionError(
            "Hypothesis scanning requires at least twelve market candles."
        )

    closes = [point.close for point in points]
    returns = [
        change
        for previous, current in zip(closes, closes[1:])
        if (change := _change_pct(previous, current)) is not None
    ]
    recent_returns = returns[-8:]
    prior_returns = returns[-28:-8]
    recent_points = points[-8:]
    prior_points = points[-28:-8]
    recent_ranges = [
        (point.high - point.low) / point.close
        for point in recent_points
        if point.close > 0
    ]
    prior_ranges = [
        (point.high - point.low) / point.close
        for point in prior_points
        if point.close > 0
    ]
    recent_volatility = (
        pstdev(recent_returns) if len(recent_returns) > 1 else 0.0
    )
    prior_volatility = (
        pstdev(prior_returns) if len(prior_returns) > 1 else None
    )
    prior_breakout_window = points[-21:-1]
    prior_high = max(point.high for point in prior_breakout_window)
    prior_low = min(point.low for point in prior_breakout_window)
    recent_volume = _average([point.volume for point in points[-4:]])
    prior_volume = _average([point.volume for point in points[-24:-4]])

    return {
        "ticker": context.ticker,
        "market_window": {
            "candle_count": len(points),
            "start": points[0].timestamp.isoformat(),
            "end": points[-1].timestamp.isoformat(),
            "current_price": context.current_price,
            "window_change_pct": _change_pct(closes[0], closes[-1]),
            "session_change_pct": _change_pct(points[-26].close, closes[-1])
            if len(points) >= 26
            else _change_pct(points[0].close, closes[-1]),
            "recent_eight_candle_change_pct": _change_pct(
                recent_points[0].close,
                recent_points[-1].close,
            ),
            "recent_four_candle_change_pct": _change_pct(
                points[-5].close,
                closes[-1],
            ),
            "preceding_four_candle_change_pct": _change_pct(
                points[-9].close,
                points[-5].close,
            ),
            "recent_return_volatility": recent_volatility,
            "prior_return_volatility": prior_volatility,
            "volatility_ratio": _ratio(recent_volatility, prior_volatility),
            "recent_average_range_pct": _average(recent_ranges),
            "range_compression_ratio": _ratio(
                _average(recent_ranges),
                _average(prior_ranges),
            ),
            "recent_volume_ratio": _ratio(recent_volume, prior_volume),
            "positive_recent_return_ratio": (
                sum(value > 0 for value in recent_returns) / len(recent_returns)
                if recent_returns
                else 0.0
            ),
            "above_prior_twenty_candle_high": closes[-1] > prior_high,
            "below_prior_twenty_candle_low": closes[-1] < prior_low,
            "distance_from_prior_high_pct": _change_pct(prior_high, closes[-1]),
            "distance_from_prior_low_pct": _change_pct(prior_low, closes[-1]),
        },
        "position_context": {
            "motive": context.subscription.motive.value,
            "average_price": context.subscription.avg_price,
            "unrealized_pnl_pct": context.unrealized_pnl_pct,
        },
        "recent_candles": [
            point.model_dump(mode="json") for point in points[-20:]
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


def build_hypothesis_graph(model: BaseChatModel, _tools: list[BaseTool]):
    """Compile the proactive hypothesis scan graph."""
    screen_model = model.with_structured_output(
        HypothesisScreen,
        method="json_schema",
        include_raw=False,
    )
    output_model = bind_structured_output(model, AgentType.HYPOTHESIS)

    def prepare(state: HypothesisState) -> dict[str, Any]:
        return {"hypothesis_payload": _hypothesis_payload(state)}

    def screen(state: HypothesisState) -> dict[str, Any]:
        result = screen_model.invoke(
            [
                SystemMessage(content=HYPOTHESIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=hypothesis_screen_prompt(state["hypothesis_payload"])
                ),
            ]
        )
        if not isinstance(result, HypothesisScreen):
            raise AgentExecutionError("Hypothesis agent returned an invalid screen.")
        return {"screen": result}

    def route_candidate(state: HypothesisState) -> str:
        result = state.get("screen")
        if not isinstance(result, HypothesisScreen):
            raise AgentExecutionError("Hypothesis screen is missing.")
        return "research" if result.candidate_detected else "no_candidate"

    def no_candidate(state: HypothesisState) -> dict[str, Any]:
        screen_result = state["screen"]
        draft = HypothesisAnalysisDraft(
            summary=None,
            recommendation="",
            confidence=screen_result.confidence,
            flagged=False,
            recommended_next_scan_days=3,
        )
        return {
            "output": finalize_ticker_output(
                state,
                AgentType.HYPOTHESIS,
                draft,
            )
        }

    def research(state: HypothesisState) -> dict[str, Any]:
        screen_result = state["screen"]
        if screen_result.focus is None:
            raise AgentExecutionError("Hypothesis research focus is missing.")
        try:
            result = perform_web_research(
                model,
                state["context"],
                screen_result.focus,
            )
        except WebResearchError:
            return {
                "research_summary": "Current hypothesis research was unavailable.",
                "research_error": "Current hypothesis research was unavailable.",
            }
        return {
            **web_research_state_update(result),
            "research_summary": result.summary,
        }

    def synthesize(state: HypothesisState) -> dict[str, Any]:
        draft = output_model.invoke(
            [
                SystemMessage(content=HYPOTHESIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=hypothesis_analysis_prompt(
                        state["hypothesis_payload"],
                        state["screen"].model_dump(mode="json"),
                        state["research_summary"],
                    )
                ),
            ]
        )
        return {
            "output": finalize_ticker_output(
                state,
                AgentType.HYPOTHESIS,
                draft,
            )
        }

    graph = StateGraph(HypothesisState)
    graph.add_node("prepare", prepare)
    graph.add_node("screen", screen)
    graph.add_node("no_candidate", no_candidate)
    graph.add_node("research", research)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "screen")
    graph.add_conditional_edges(
        "screen",
        route_candidate,
        {"research": "research", "no_candidate": "no_candidate"},
    )
    graph.add_edge("research", "synthesize")
    graph.add_edge("no_candidate", END)
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_hypothesis_scan(
    context: AgentContext,
    *,
    schedule_next: ScheduleNextScan,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> HypothesisOutput:
    """Execute, persist, and reschedule one proactive hypothesis scan."""
    output = execute_ticker_agent(
        context,
        AgentType.HYPOTHESIS,
        build_hypothesis_graph,
        database_client=database_client,
        credential_client=credential_client,
        cipher=cipher,
    )
    if not isinstance(output, HypothesisOutput):
        raise AgentExecutionError("Hypothesis agent returned an invalid output.")
    try:
        schedule_next(
            output.user_id,
            output.ticker,
            output.recommended_next_scan_days,
        )
    except Exception:
        raise HypothesisSchedulingError(
            "The next hypothesis scan could not be scheduled."
        ) from None
    return output
