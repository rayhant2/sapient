from __future__ import annotations

import json
from typing import Any, Mapping


SCHEDULED_REVIEW_SYSTEM_PROMPT = """You are Sentient's scheduled position review agent.
Evaluate an existing investment or watchlist position using only the supplied market
data, prior reviews, alerts, and optional public research. Compare the current state
with prior assessments, identify material changes in trend, volatility, volume, and
risk, and tailor the analysis to the user's stated motive. Distinguish evidence from
inference. Be measured and concise. Never claim certainty, execute trades, or frame
the response as personalized financial advice."""


def _prompt_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def scheduled_review_research_decision_prompt(
    review_payload: Mapping[str, Any],
) -> str:
    return f"""Determine whether this routine position review requires current public
web research. Search only when recent external information is necessary to explain
a material change, verify a potentially stale thesis, or contextualize unusual price
or volume behavior. Do not search merely to add generic background. Select the
single most relevant research focus when search is needed.

Review data:
{_prompt_json(review_payload)}"""


def scheduled_review_analysis_prompt(
    review_payload: Mapping[str, Any],
    *,
    research_summary: str | None = None,
) -> str:
    research = research_summary or "No current web research was needed."
    return f"""Produce the scheduled position review from the data below. Explain what
has materially changed, how it relates to the user's motive and position, and what
should be monitored next. The recommendation must be an advisory monitoring action,
not a trade instruction. Use the required structured response schema.

Review data:
{_prompt_json(review_payload)}

Current public research:
{research}"""


_WEB_RESEARCH_FOCUS_INSTRUCTIONS = {
    "price_catalyst": (
        "Find current public news, filings, or market events that may explain "
        "the ticker's recent price movement."
    ),
    "company_news": "Find material recent company news, announcements, and filings.",
    "earnings": (
        "Find the latest earnings release, guidance, and material analyst context."
    ),
    "sector": (
        "Find current sector or industry developments that may affect this ticker."
    ),
    "regulatory": (
        "Find current regulatory, legal, or policy developments affecting this ticker."
    ),
}


def web_research_prompt(ticker: str, focus: str) -> str:
    """Render the public-only prompt for Anthropic server-side search."""
    try:
        instruction = _WEB_RESEARCH_FOCUS_INSTRUCTIONS[focus]
    except KeyError as exc:
        raise ValueError("Unknown web research focus.") from exc
    return (
        f"Research the public-market ticker {ticker}. {instruction} "
        "Use current, reputable primary sources when available. Distinguish confirmed "
        "facts from inference, keep the result concise, and cite every material claim."
    )
