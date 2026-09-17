import unittest
from unittest.mock import MagicMock, patch

from config.startup import StartupConfigurationError
from main import build_output_sink, build_runtime, main, run_service


class MainTests(unittest.TestCase):
    @patch("main.AgentRuntime")
    @patch("main.SentientScheduler")
    @patch("main.EventBus")
    @patch("main.build_output_sink")
    def test_build_runtime_uses_configured_sink_by_default(
        self,
        output_sink,
        event_bus,
        scheduler,
        runtime,
    ):
        build_runtime()

        output_sink.assert_called_once_with()
        runtime.assert_called_once_with(
            event_bus.return_value,
            scheduler.return_value,
            output_sink=output_sink.return_value,
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

    @patch("main.create_whatsapp_notifier")
    def test_output_sink_skips_twilio_when_disabled(self, notifier):
        with patch("main.settings.whatsapp_enabled", False):
            sink = build_output_sink()

        notifier.assert_not_called()
        self.assertTrue(callable(sink))

    @patch("main.create_whatsapp_notifier")
    def test_output_sink_builds_twilio_when_enabled(self, notifier):
        with patch("main.settings.whatsapp_enabled", True):
            sink = build_output_sink()

        notifier.assert_called_once_with()
        self.assertIs(sink, notifier.return_value)

    def test_run_service_marks_ready_and_shuts_down(self):
        runtime = MagicMock()
        runtime.event_bus.registry = {"NVDA": object()}
        stop_event = MagicMock()
        health_server = MagicMock()
        health_state = MagicMock()

        run_service(runtime, stop_event, health_server, health_state)

        health_server.start.assert_called_once_with()
        runtime.start.assert_called_once_with()
        health_state.mark_ready.assert_called_once_with(active_tickers=1)
        stop_event.wait.assert_called_once_with()
        health_state.mark_stopping.assert_called_once_with()
        runtime.shutdown.assert_called_once_with(wait=True)
        health_state.mark_stopped.assert_called_once_with()
        health_server.shutdown.assert_called_once_with()

    def test_run_service_cleans_up_after_start_failure(self):
        runtime = MagicMock()
        runtime.start.side_effect = RuntimeError("private startup detail")
        health_server = MagicMock()
        health_state = MagicMock()

        with self.assertRaises(RuntimeError):
            run_service(runtime, MagicMock(), health_server, health_state)

        runtime.shutdown.assert_called_once_with(wait=True)
        health_state.mark_stopped.assert_called_once_with()
        health_server.shutdown.assert_called_once_with()

    @patch("main.configure_logging")
    @patch("main.validate_runtime_settings")
    def test_main_exits_cleanly_on_invalid_configuration(
        self,
        validate,
        configure_logging,
    ):
        validate.side_effect = StartupConfigurationError("Missing configuration.")

        with self.assertRaisesRegex(SystemExit, "2"):
            main()

        configure_logging.assert_called_once()


if __name__ == "__main__":
    unittest.main()
