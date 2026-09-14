from __future__ import annotations

from typing import Annotated, NotRequired, TypeAlias, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from models.schemas import AgentContext, AgentOutput, Alert, HypothesisOutput


TickerAgentOutput: TypeAlias = AgentOutput | HypothesisOutput


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

