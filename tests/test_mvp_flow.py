from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import MagicMock, patch

from core.event_bus import EventBus
from core.runtime import AgentRuntime
from core.scheduler import SentientScheduler
from models.schemas import (
    AgentOutput,
    Confidence,
    EventType,
    Motive,
    OHLCVPoint,
    Subscription,
    UpdateInterval,
    User,
)
from notifications.whatsapp import WhatsAppNotifier


START = datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc)


def sharp_move_points() -> list[OHLCVPoint]:
    closes = [100.0, 101.5]
    return [
        OHLCVPoint(
            ticker="NVDA",
            timestamp=START + timedelta(minutes=index * 15),
            open=close,
            high=close * 1.002,
            low=close * 0.998,
            close=close,
            volume=1_000_000 + index,
        )
        for index, close in enumerate(closes)
    ]


class MvpFlowTests(unittest.TestCase):
    def test_market_refresh_reaches_agent_whatsapp_and_alert_log(self):
        points = sharp_move_points()
        subscription = Subscription(
            user_id="user-1",
            ticker="NVDA",
            avg_price=95.0,
            shares=2.0,
            motive=Motive.HOLDING,
            update_interval=UpdateInterval.DAILY,
            sharp_move_threshold=0.01,
        )
        bus = EventBus(datapoint_loader=lambda _ticker, _limit: points)
        bus.refresh_subscription(subscription)

        scheduler_backend = MagicMock()
        scheduler = SentientScheduler(
            bus,
            scheduler=scheduler_backend,
            refresh_ticker=lambda _ticker: points,
            now_provider=lambda: START + timedelta(minutes=15),
            max_retries=0,
        )
        twilio = MagicMock()
        notifier = WhatsAppNotifier(twilio, "whatsapp:+14155238886")
        runtime = AgentRuntime(bus, scheduler, output_sink=notifier)
        runtime.register_handlers()

        agent_output = AgentOutput(
            ticker="NVDA",
            user_id="user-1",
            event_type=EventType.SHARP_MOVE,
            summary="NVDA moved sharply after a material catalyst.",
            recommendation="Review the position against the original thesis.",
            confidence=Confidence.HIGH,
            price_at_update=101.5,
        )
        user = User(
            user_id="user-1",
            whatsapp_number="whatsapp:+15551234567",
        )
        stored_alerts = []

        with (
            patch("core.runtime.run_sharp_move", return_value=agent_output) as agent,
            patch("notifications.whatsapp.get_user", return_value=user),
            patch(
                "notifications.whatsapp.insert_alert",
                side_effect=lambda alert, client=None: stored_alerts.append(alert)
                or alert,
            ),
        ):
            refreshed = scheduler._run_ticker_refresh("NVDA")

        self.assertTrue(refreshed)
        agent.assert_called_once()
        self.assertEqual(agent.call_args.args[0].current_price, 101.5)
        twilio.messages.create.assert_called_once()
        self.assertIn("SHARP MOVE | NVDA", twilio.messages.create.call_args.kwargs["body"])
        self.assertEqual(len(stored_alerts), 1)
        self.assertEqual(stored_alerts[0].user_id, "user-1")


if __name__ == "__main__":
    unittest.main()
