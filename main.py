from __future__ import annotations

import logging
import signal
import threading

from config.settings import settings
from core.event_bus import EventBus
from core.runtime import AgentRuntime, OutputSink
from core.scheduler import SentientScheduler
from notifications.whatsapp import create_whatsapp_notifier


logger = logging.getLogger(__name__)


def build_runtime(*, output_sink: OutputSink | None = None) -> AgentRuntime:
    """Construct the production runtime around one shared event bus."""
    event_bus = EventBus()
    scheduler = SentientScheduler(event_bus)
    sink = output_sink if output_sink is not None else create_whatsapp_notifier()
    return AgentRuntime(event_bus, scheduler, output_sink=sink)


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    stop_event = threading.Event()
    runtime = build_runtime()

    def request_shutdown(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    runtime.start()
    logger.info("Sentient runtime started.")
    try:
        stop_event.wait()
    finally:
        runtime.shutdown(wait=True)
        logger.info("Sentient runtime stopped.")


if __name__ == "__main__":
    main()
