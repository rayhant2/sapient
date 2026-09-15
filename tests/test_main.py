import unittest
from unittest.mock import patch

from main import build_runtime


class MainTests(unittest.TestCase):
    @patch("main.AgentRuntime")
    @patch("main.SentientScheduler")
    @patch("main.EventBus")
    @patch("main.create_whatsapp_notifier")
    def test_build_runtime_uses_whatsapp_notifier_by_default(
        self,
        notifier,
        event_bus,
        scheduler,
        runtime,
    ):
        build_runtime()

        notifier.assert_called_once_with()
        runtime.assert_called_once_with(
            event_bus.return_value,
            scheduler.return_value,
            output_sink=notifier.return_value,
        )

    @patch("main.AgentRuntime")
    @patch("main.SentientScheduler")
    @patch("main.EventBus")
    def test_build_runtime_shares_event_bus_and_forwards_sink(
        self,
        event_bus,
        scheduler,
        runtime,
    ):
        sink = object()

        result = build_runtime(output_sink=sink)

        scheduler.assert_called_once_with(event_bus.return_value)
        runtime.assert_called_once_with(
            event_bus.return_value,
            scheduler.return_value,
            output_sink=sink,
        )
        self.assertIs(result, runtime.return_value)


if __name__ == "__main__":
    unittest.main()
