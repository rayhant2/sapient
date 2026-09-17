from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Any, Callable


@dataclass(frozen=True)
class HealthSnapshot:
    status: str
    live: bool
    ready: bool
    started_at: datetime
    active_tickers: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "service": "sapient-runtime",
            "status": self.status,
            "live": self.live,
            "ready": self.ready,
            "started_at": self.started_at.isoformat(),
            "active_tickers": self.active_tickers,
        }


class HealthState:
    """Thread-safe lifecycle state exposed by the worker health server."""

    def __init__(self) -> None:
        self._started_at = datetime.now(timezone.utc)
        self._status = "starting"
        self._live = True
        self._ready = False
        self._active_tickers = 0
        self._lock = threading.Lock()

    def mark_ready(self, *, active_tickers: int) -> None:
        with self._lock:
            self._status = "ready"
            self._ready = True
            self._active_tickers = active_tickers

    def mark_stopping(self) -> None:
        with self._lock:
            self._status = "stopping"
            self._ready = False

    def mark_stopped(self) -> None:
        with self._lock:
            self._status = "stopped"
            self._live = False
            self._ready = False

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(
                status=self._status,
                live=self._live,
                ready=self._ready,
                started_at=self._started_at,
                active_tickers=self._active_tickers,
            )


def _handler_factory(
    snapshot_provider: Callable[[], HealthSnapshot],
) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            snapshot = snapshot_provider()
            if self.path == "/health/live":
                status_code = 200 if snapshot.live else 503
            elif self.path == "/health/ready":
                status_code = 200 if snapshot.ready else 503
            else:
                self.send_error(404)
                return

            body = json.dumps(snapshot.as_dict()).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            return

    return HealthHandler


class HealthServer:
    def __init__(self, state: HealthState, host: str, port: int) -> None:
        self.state = state
        self._server = ThreadingHTTPServer(
            (host, port),
            _handler_factory(state.snapshot),
        )
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_port

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="sapient-health",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        if self._thread is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        self._thread = None
