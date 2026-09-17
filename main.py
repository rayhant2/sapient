from __future__ import annotations

import logging
import signal
import threading

from agents.core import StructuredAgentOutput
from config.logging import configure_logging
from config.settings import settings
from config.startup import StartupConfigurationError, validate_runtime_settings
from core.event_bus import EventBus
from core.health import HealthServer, HealthState
from core.runtime import AgentRuntime, OutputSink
from core.scheduler import SentientScheduler
from notifications.whatsapp import create_whatsapp_notifier


logger = logging.getLogger(__name__)


def _disabled_delivery(output: StructuredAgentOutput) -> None:
    logger.info(
        "WhatsApp delivery disabled; retained %s output in update history.",
        type(output).__name__,
    )


def build_output_sink() -> OutputSink:
    if not settings.whatsapp_enabled:
        return _disabled_delivery
    return create_whatsapp_notifier()


def build_runtime(*, output_sink: OutputSink | None = None) -> AgentRuntime:
    """Construct the production runtime around one shared event bus."""
    event_bus = EventBus()
    scheduler = SentientScheduler(event_bus)
    sink = output_sink if output_sink is not None else build_output_sink()
    return AgentRuntime(event_bus, scheduler, output_sink=sink)


def run_service(
    runtime: AgentRuntime,
    stop_event: threading.Event,
    health_server: HealthServer,
    health_state: HealthState,
) -> None:
    health_server.start()
    try:
        runtime.start()
        health_state.mark_ready(active_tickers=len(runtime.event_bus.registry))
        logger.info(
            "Sapient runtime ready with %d active ticker(s).",
            len(runtime.event_bus.registry),
        )
        stop_event.wait()
    except Exception as exc:
        logger.error("Runtime service failed: %s", type(exc).__name__)
        raise
    finally:
        health_state.mark_stopping()
        try:
            runtime.shutdown(wait=True)
        except Exception as exc:
            logger.error("Runtime shutdown failed: %s", type(exc).__name__)
        health_state.mark_stopped()
        health_server.shutdown()
        logger.info("Sapient runtime stopped.")


def main() -> None:
    configure_logging(settings)
    try:
        validate_runtime_settings(settings)
    except StartupConfigurationError as exc:
        logger.error("%s", exc)
        raise SystemExit(2) from None

    stop_event = threading.Event()

    def request_shutdown(_signum=None, _frame=None) -> None:
        logger.info("Shutdown requested.")
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    runtime = build_runtime()
    health_state = HealthState()
    health_server = HealthServer(
        health_state,
        settings.health_host,
        settings.health_port,
    )
    run_service(runtime, stop_event, health_server, health_state)


if __name__ == "__main__":
    main()
