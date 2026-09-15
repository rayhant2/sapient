import unittest
from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from config import settings as settings_module
from models.schemas import (
    AgentOutput,
    AlertType,
    Confidence,
    CrossPortfolioOutput,
    EventType,
    HypothesisOutput,
    ResearchSource,
    User,
)
from notifications.whatsapp import (
    MAX_WHATSAPP_BODY_LENGTH,
    WhatsAppConfigurationError,
    WhatsAppDeliveryError,
    WhatsAppNotifier,
    build_whatsapp_message,
    create_whatsapp_notifier,
    format_agent_output,
)


def user() -> User:
    return User(
        user_id="user-1",
        whatsapp_number="whatsapp:+15551234567",
    )


def ticker_output(event_type: EventType) -> AgentOutput:
    return AgentOutput(
        ticker="NVDA",
        user_id="user-1",
        event_type=event_type,
        summary="NVDA remains stable.",
        recommendation="Monitor the next session.",
        confidence=Confidence.MEDIUM,
        price_at_update=125.25,
        searched_web=True,
        sources=[
            ResearchSource(
                title="Company filing",
                url="https://example.com/filing",
            )
        ],
        web_search_requests=1,
    )


class WhatsAppFormattingTests(unittest.TestCase):
    def test_formats_each_ticker_alert_type(self):
        expected_titles = {
            EventType.SCHEDULED_UPDATE: "POSITION REVIEW",
            EventType.SHARP_MOVE: "SHARP MOVE",
            EventType.MOTIVE_CHECK: "MOTIVE REVIEW",
            EventType.HYPOTHESIS_SCAN: "MARKET HYPOTHESIS",
        }
        for event_type, title in expected_titles.items():
            with self.subTest(event_type=event_type):
                body = format_agent_output(ticker_output(event_type))
                self.assertIsNotNone(body)
                self.assertIn(f"{title} | NVDA", body)
                self.assertIn("Confidence: Medium", body)
                self.assertIn("https://example.com/filing", body)

    def test_formats_cross_portfolio_output_without_ticker(self):
        output = CrossPortfolioOutput(
            user_id="user-1",
            summary="The portfolio is concentrated.",
            correlations_flagged=["NVDA and AMD moved together."],
            tickers_analyzed=["NVDA", "AMD"],
        )

        message = build_whatsapp_message(output, user())

        self.assertIsNotNone(message)
        self.assertEqual(message.alert_type, AlertType.CROSS_PORTFOLIO)
        self.assertIsNone(message.ticker)
        self.assertIn("PORTFOLIO REVIEW", message.body)

    def test_unflagged_hypothesis_is_silent(self):
        output = HypothesisOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.HYPOTHESIS_SCAN,
            summary=None,
            recommendation="",
            confidence=Confidence.LOW,
            flagged=False,
            recommended_next_scan_days=3,
        )

        self.assertIsNone(format_agent_output(output))
        self.assertIsNone(build_whatsapp_message(output, user()))

    def test_long_messages_are_bounded(self):
        output = ticker_output(EventType.SCHEDULED_UPDATE).model_copy(
            update={"summary": "x" * 3000}
        )

        body = format_agent_output(output)

        self.assertEqual(len(body), MAX_WHATSAPP_BODY_LENGTH)
        self.assertTrue(body.endswith("..."))


class WhatsAppNotifierTests(unittest.TestCase):
    def setUp(self):
        self.twilio = MagicMock(name="twilio-client")
        self.database = MagicMock(name="database-client")
        self.notifier = WhatsAppNotifier(
            self.twilio,
            "whatsapp:+14155238886",
            database_client=self.database,
        )

    def test_sends_then_records_alert(self):
        output = ticker_output(EventType.SHARP_MOVE)

        with (
            patch("notifications.whatsapp.get_user", return_value=user()),
            patch(
                "notifications.whatsapp.insert_alert",
                side_effect=lambda alert, client: alert,
            ) as insert_alert,
        ):
            alert = self.notifier.send(output)

        self.twilio.messages.create.assert_called_once_with(
            body=alert.message,
            from_="whatsapp:+14155238886",
            to="whatsapp:+15551234567",
        )
        insert_alert.assert_called_once()
        self.assertEqual(alert.alert_type, AlertType.SHARP_MOVE)
        self.assertEqual(alert.trigger_details["source_count"], 1)

    def test_failed_send_is_sanitized_and_not_recorded(self):
        self.twilio.messages.create.side_effect = RuntimeError("private Twilio detail")
        with (
            patch("notifications.whatsapp.get_user", return_value=user()),
            patch("notifications.whatsapp.insert_alert") as insert_alert,
        ):
            with self.assertRaisesRegex(
                WhatsAppDeliveryError,
                "could not be sent",
            ) as raised:
                self.notifier.send(ticker_output(EventType.SHARP_MOVE))

        insert_alert.assert_not_called()
        self.assertNotIn("private Twilio detail", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    def test_silent_hypothesis_never_calls_twilio_or_database(self):
        output = HypothesisOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.HYPOTHESIS_SCAN,
            summary=None,
            recommendation="",
            confidence=Confidence.LOW,
            flagged=False,
            recommended_next_scan_days=3,
        )
        with (
            patch("notifications.whatsapp.get_user", return_value=user()),
            patch("notifications.whatsapp.insert_alert") as insert_alert,
        ):
            result = self.notifier.send(output)

        self.assertIsNone(result)
        self.twilio.messages.create.assert_not_called()
        insert_alert.assert_not_called()

    def test_factory_requires_complete_configuration(self):
        with patch.multiple(
            settings_module.settings,
            twilio_account_sid=None,
            twilio_auth_token=None,
            twilio_whatsapp_from=None,
        ):
            with self.assertRaises(WhatsAppConfigurationError):
                create_whatsapp_notifier()

    @patch("notifications.whatsapp.TwilioClient")
    def test_factory_resolves_secrets_without_exposing_them(self, twilio_client):
        with patch.multiple(
            settings_module.settings,
            twilio_account_sid=SecretStr("account-secret"),
            twilio_auth_token=SecretStr("token-secret"),
            twilio_whatsapp_from="whatsapp:+14155238886",
        ):
            notifier = create_whatsapp_notifier(database_client=self.database)

        twilio_client.assert_called_once_with("account-secret", "token-secret")
        self.assertIsInstance(notifier, WhatsAppNotifier)


if __name__ == "__main__":
    unittest.main()
