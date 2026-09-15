from __future__ import annotations

from typing import Any

from supabase import Client as SupabaseClient
from twilio.rest import Client as TwilioClient

from agents.core import StructuredAgentOutput
from config.settings import settings
from data.database import get_user, insert_alert
from models.schemas import (
    AgentOutput,
    Alert,
    AlertType,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    User,
    WhatsAppMessage,
)


MAX_WHATSAPP_BODY_LENGTH = 1500


class WhatsAppConfigurationError(RuntimeError):
    """Raised when Twilio WhatsApp settings are incomplete."""


class WhatsAppDeliveryError(RuntimeError):
    """Raised when an agent output cannot be submitted to Twilio."""


class WhatsAppPersistenceError(RuntimeError):
    """Raised when a sent WhatsApp alert cannot be recorded."""


_ALERT_TYPE_BY_EVENT = {
    EventType.SCHEDULED_UPDATE: AlertType.SCHEDULED,
    EventType.SHARP_MOVE: AlertType.SHARP_MOVE,
    EventType.MOTIVE_CHECK: AlertType.MOTIVE_FLAG,
    EventType.HYPOTHESIS_SCAN: AlertType.HYPOTHESIS,
}

_TITLE_BY_ALERT = {
    AlertType.SCHEDULED: "POSITION REVIEW",
    AlertType.SHARP_MOVE: "SHARP MOVE",
    AlertType.MOTIVE_FLAG: "MOTIVE REVIEW",
    AlertType.HYPOTHESIS: "MARKET HYPOTHESIS",
    AlertType.CROSS_PORTFOLIO: "PORTFOLIO REVIEW",
}


def _truncate(text: str, limit: int = MAX_WHATSAPP_BODY_LENGTH) -> str:
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _source_lines(output: AgentOutput) -> list[str]:
    if not output.sources:
        return []
    lines = ["Sources:"]
    for source in output.sources[:2]:
        lines.append(f"- {source.title}: {source.url}")
    return lines


def _format_ticker_output(output: AgentOutput) -> str:
    alert_type = _ALERT_TYPE_BY_EVENT[output.event_type]
    lines = [
        f"{_TITLE_BY_ALERT[alert_type]} | {output.ticker}",
        "",
        output.summary or "No material hypothesis was confirmed.",
    ]
    if output.recommendation:
        lines.extend(["", f"Next: {output.recommendation}"])
    lines.extend(["", f"Confidence: {output.confidence.value.title()}"])
    if output.price_at_update is not None:
        lines.append(f"Price: ${output.price_at_update:,.2f}")
    lines.extend(_source_lines(output))
    return _truncate("\n".join(lines))


def _format_portfolio_output(output: CrossPortfolioOutput) -> str:
    lines = [
        _TITLE_BY_ALERT[AlertType.CROSS_PORTFOLIO],
        "",
        output.summary,
    ]
    if output.correlations_flagged:
        lines.extend(["", "Notable relationships:"])
        lines.extend(f"- {item}" for item in output.correlations_flagged[:4])
    if output.tickers_analyzed:
        lines.extend(["", f"Tracked: {', '.join(output.tickers_analyzed)}"])
    return _truncate("\n".join(lines))


def format_agent_output(output: StructuredAgentOutput) -> str | None:
    """Format one output, returning None when no user alert is warranted."""
    if isinstance(output, HypothesisOutput) and not output.flagged:
        return None
    if isinstance(output, CrossPortfolioOutput):
        return _format_portfolio_output(output)
    return _format_ticker_output(output)


def _alert_type(output: StructuredAgentOutput) -> AlertType:
    if isinstance(output, CrossPortfolioOutput):
        return AlertType.CROSS_PORTFOLIO
    return _ALERT_TYPE_BY_EVENT[output.event_type]


def _trigger_details(output: StructuredAgentOutput) -> dict[str, Any]:
    if isinstance(output, CrossPortfolioOutput):
        return {
            "agent_type": "cross_portfolio",
            "tickers_analyzed": output.tickers_analyzed,
            "correlations_flagged": output.correlations_flagged,
        }
    details: dict[str, Any] = {
        "event_type": output.event_type.value,
        "confidence": output.confidence.value,
        "searched_web": output.searched_web,
        "source_count": len(output.sources),
    }
    if isinstance(output, HypothesisOutput):
        details.update(
            {
                "flagged": output.flagged,
                "recommended_next_scan_days": output.recommended_next_scan_days,
            }
        )
    return details


def build_whatsapp_message(
    output: StructuredAgentOutput,
    user: User,
) -> WhatsAppMessage | None:
    body = format_agent_output(output)
    if body is None:
        return None
    return WhatsAppMessage(
        to=user.whatsapp_number,
        body=body,
        alert_type=_alert_type(output),
        ticker=getattr(output, "ticker", None),
        user_id=output.user_id,
    )


class WhatsAppNotifier:
    """Runtime output sink that sends through Twilio and logs accepted messages."""

    def __init__(
        self,
        twilio_client: Any,
        from_number: str,
        *,
        database_client: SupabaseClient | None = None,
    ) -> None:
        if not from_number.startswith("whatsapp:+"):
            raise WhatsAppConfigurationError(
                "The Twilio sender must use the whatsapp:+ prefix."
            )
        self._twilio_client = twilio_client
        self._from_number = from_number
        self._database_client = database_client

    def send(self, output: StructuredAgentOutput) -> Alert | None:
        try:
            user = get_user(output.user_id, client=self._database_client)
        except Exception:
            raise WhatsAppDeliveryError(
                "The notification recipient could not be loaded."
            ) from None
        if user is None:
            raise WhatsAppDeliveryError("The notification recipient does not exist.")

        message = build_whatsapp_message(output, user)
        if message is None:
            return None
        try:
            self._twilio_client.messages.create(
                body=message.body,
                from_=self._from_number,
                to=message.to,
            )
        except Exception:
            raise WhatsAppDeliveryError(
                "The WhatsApp message could not be sent."
            ) from None

        alert = Alert(
            user_id=output.user_id,
            ticker=getattr(output, "ticker", None),
            timestamp=output.timestamp,
            alert_type=message.alert_type,
            message=message.body,
            trigger_details=_trigger_details(output),
        )
        try:
            return insert_alert(alert, client=self._database_client)
        except Exception:
            raise WhatsAppPersistenceError(
                "The sent WhatsApp alert could not be recorded."
            ) from None

    def __call__(self, output: StructuredAgentOutput) -> None:
        self.send(output)


def create_whatsapp_notifier(
    *,
    database_client: SupabaseClient | None = None,
) -> WhatsAppNotifier:
    """Create the production notifier from environment-backed settings."""
    if (
        settings.twilio_account_sid is None
        or settings.twilio_auth_token is None
        or settings.twilio_whatsapp_from is None
    ):
        raise WhatsAppConfigurationError(
            "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_WHATSAPP_FROM "
            "must be configured."
        )
    client = TwilioClient(
        settings.twilio_account_sid.get_secret_value(),
        settings.twilio_auth_token.get_secret_value(),
    )
    return WhatsAppNotifier(
        client,
        settings.twilio_whatsapp_from,
        database_client=database_client,
    )
