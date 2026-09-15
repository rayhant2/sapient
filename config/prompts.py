from __future__ import annotations


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

