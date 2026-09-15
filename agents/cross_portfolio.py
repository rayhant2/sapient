from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Any, NotRequired

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from supabase import Client

from agents.core import (
    AgentExecutionError,
    PortfolioAgentState,
    bind_structured_output,
    execute_cross_portfolio_agent,
    finalize_cross_portfolio_output,
)
from config.prompts import (
    CROSS_PORTFOLIO_SYSTEM_PROMPT,
    cross_portfolio_analysis_prompt,
)
from models.schemas import (
    AgentContext,
    AgentType,
    CrossPortfolioOutput,
    PortfolioContext,
)
from security.credentials import CredentialCipher


class CrossPortfolioState(PortfolioAgentState):
    portfolio_payload: NotRequired[dict[str, Any]]


def _change_pct(start: float, end: float) -> float | None:
    if start <= 0:
        return None
    return (end - start) / start


def _return_series(position: AgentContext) -> dict[str, float]:
    points = sorted(position.datapoints, key=lambda point: point.timestamp)
    return {
        current.timestamp.isoformat(): change
        for previous, current in zip(points, points[1:])
        if (change := _change_pct(previous.close, current.close)) is not None
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 5:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta)
        * sum(value * value for value in right_delta)
    )
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denominator


def _position_payload(position: AgentContext, total_value: float) -> dict[str, Any]:
    points = sorted(position.datapoints, key=lambda point: point.timestamp)
    closes = [point.close for point in points]
    returns = [
        change
        for previous, current in zip(closes, closes[1:])
        if (change := _change_pct(previous, current)) is not None
    ]
    current_value = position.current_price * position.subscription.shares
    return {
        "ticker": position.ticker,
        "motive": position.subscription.motive.value,
        "current_price": position.current_price,
        "average_price": position.subscription.avg_price,
        "shares": position.subscription.shares,
        "current_value": current_value,
        "portfolio_weight": current_value / total_value if total_value > 0 else 0.0,
        "unrealized_pnl": position.unrealized_pnl,
        "unrealized_pnl_pct": position.unrealized_pnl_pct,
        "available_window_change_pct": (
            _change_pct(closes[0], closes[-1]) if closes else None
        ),
        "return_volatility": pstdev(returns) if len(returns) > 1 else 0.0,
        "candle_count": len(points),
    }


def _portfolio_payload(context: PortfolioContext) -> dict[str, Any]:
    if not context.positions:
        raise AgentExecutionError("Cross-portfolio analysis requires positions.")
    tickers = [position.ticker.upper() for position in context.positions]
    if len(tickers) != len(set(tickers)):
        raise AgentExecutionError(
            "Cross-portfolio analysis requires one position per ticker."
        )
    if any(len(position.datapoints) < 2 for position in context.positions):
        raise AgentExecutionError(
            "Cross-portfolio analysis requires at least two candles per position."
        )
    current_total = sum(
        position.current_price * position.subscription.shares
        for position in context.positions
    )
    positions = [
        _position_payload(position, current_total) for position in context.positions
    ]

    correlations: list[dict[str, Any]] = []
    series = {
        position.ticker: _return_series(position) for position in context.positions
    }
    for index, left_position in enumerate(context.positions):
        for right_position in context.positions[index + 1 :]:
            left_series = series[left_position.ticker]
            right_series = series[right_position.ticker]
            timestamps = sorted(set(left_series) & set(right_series))
            correlation = _pearson(
                [left_series[timestamp] for timestamp in timestamps],
                [right_series[timestamp] for timestamp in timestamps],
            )
            if correlation is not None:
                correlations.append(
                    {
                        "left_ticker": left_position.ticker,
                        "right_ticker": right_position.ticker,
                        "correlation": correlation,
                        "overlapping_returns": len(timestamps),
                    }
                )

    valid_tickers = {position.ticker for position in context.positions}
    cycle_outputs = [
        {
            "ticker": output.ticker,
            "event_type": output.event_type.value,
            "summary": output.summary,
            "recommendation": output.recommendation,
            "confidence": output.confidence.value,
            "timestamp": output.timestamp.isoformat(),
        }
        for output in context.latest_outputs
        if output.user_id == context.user_id and output.ticker in valid_tickers
    ]

    return {
        "portfolio": {
            "ticker_count": len(context.positions),
            "current_tracked_value": current_total,
            "total_unrealized_pnl": context.total_unrealized_pnl,
            "largest_position_weight": max(
                position["portfolio_weight"] for position in positions
            ),
        },
        "positions": positions,
        "overlapping_return_correlations": correlations,
        "current_cycle_outputs": cycle_outputs,
        "limitations": [
            "Correlation is descriptive and does not establish causation.",
            "Sector classifications are unavailable unless stated in agent outputs.",
            "Watching positions may use reference shares and cost values.",
        ],
    }


def build_cross_portfolio_graph(model: BaseChatModel):
    """Compile the cross-portfolio reasoning graph."""
    output_model = bind_structured_output(model, AgentType.CROSS_PORTFOLIO)

    def prepare(state: CrossPortfolioState) -> dict[str, Any]:
        return {"portfolio_payload": _portfolio_payload(state["context"])}

    def synthesize(state: CrossPortfolioState) -> dict[str, Any]:
        draft = output_model.invoke(
            [
                SystemMessage(content=CROSS_PORTFOLIO_SYSTEM_PROMPT),
                HumanMessage(
                    content=cross_portfolio_analysis_prompt(
                        state["portfolio_payload"]
                    )
                ),
            ]
        )
        return {
            "output": finalize_cross_portfolio_output(state["context"], draft)
        }

    graph = StateGraph(CrossPortfolioState)
    graph.add_node("prepare", prepare)
    graph.add_node("synthesize", synthesize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_cross_portfolio(
    context: PortfolioContext,
    *,
    database_client: Client | None = None,
    credential_client: Client | None = None,
    cipher: CredentialCipher | None = None,
) -> CrossPortfolioOutput:
    """Execute and persist one cross-portfolio assessment."""
    return execute_cross_portfolio_agent(
        context,
        build_cross_portfolio_graph,
        database_client=database_client,
        credential_client=credential_client,
        cipher=cipher,
    )
