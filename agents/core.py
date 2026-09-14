from __future__ import annotations

from typing import Annotated, NotRequired, TypeAlias, TypedDict

from langchain_core.messages import AnyMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph.message import add_messages
from supabase import Client

from config.settings import settings
from data.database import DatabaseError, resolve_user_api_key
from models.schemas import AgentContext, AgentOutput, Alert, HypothesisOutput
from security.credentials import CredentialCipher, CredentialSecurityError


TickerAgentOutput: TypeAlias = AgentOutput | HypothesisOutput


class AgentModelError(RuntimeError):
    """Base exception for per-user model setup failures."""


class MissingUserApiKeyError(AgentModelError):
    """Raised when a user has no stored API key for the requested provider."""


class AgentModelInitializationError(AgentModelError):
    """Raised when a user credential or model client cannot be initialized."""


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
