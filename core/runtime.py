from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import threading
from typing import TypeAlias
from zoneinfo import ZoneInfo

from supabase import Client

from agents.core import StructuredAgentOutput
from agents.cross_portfolio import run_cross_portfolio
from agents.hypothesis import run_hypothesis_scan
from agents.motive import run_motive_reassessment
from agents.scheduled_review import run_scheduled_review
from agents.sharp_move import run_sharp_move
from core.event_bus import EventBus
from core.scheduler import SentientScheduler
from models.schemas import AgentContext, AgentOutput, EventType, UpdateInterval
from security.credentials import CredentialCipher


RUNTIME_TIMEZONE = ZoneInfo("America/New_York")
OutputSink: TypeAlias = Callable[[StructuredAgentOutput], None]
NowProvider: TypeAlias = Callable[[], datetime]


class RuntimeConfigurationError(RuntimeError):
    """Raised when runtime handlers cannot be configured safely."""


@dataclass
class _PortfolioCycle:
    expected_tickers: frozenset[str]
    outputs: dict[str, AgentOutput] = field(default_factory=dict)
    running: bool = False
    completed: bool = False


class AgentRuntime:
    """Connect scheduler events to agents and completed outputs to a sink."""

    def __init__(
        self,
        event_bus: EventBus,
        scheduler: SentientScheduler,
        *,
        output_sink: OutputSink | None = None,
        database_client: Client | None = None,
        credential_client: Client | None = None,
        cipher: CredentialCipher | None = None,
        now_provider: NowProvider | None = None,
    ) -> None:
        if scheduler.event_bus is not event_bus:
            raise RuntimeConfigurationError(
                "Runtime and scheduler must share the same event bus."
            )
        self.event_bus = event_bus
        self.scheduler = scheduler
        self.output_sink = output_sink or (lambda _output: None)
        self.database_client = database_client
        self.credential_client = credential_client
        self.cipher = cipher
        self._now = now_provider or (lambda: datetime.now(timezone.utc))
        self._cycles: dict[tuple[str, date], _PortfolioCycle] = {}
        self._cycle_lock = threading.Lock()
        self._registered = False

    def _agent_kwargs(self) -> dict[str, object]:
        return {
            "database_client": self.database_client,
            "credential_client": self.credential_client,
            "cipher": self.cipher,
        }

    def _handle_sharp_move(self, context: AgentContext) -> None:
        output = run_sharp_move(context, **self._agent_kwargs())
        self.output_sink(output)

    def _handle_motive_check(self, context: AgentContext) -> None:
        output = run_motive_reassessment(context, **self._agent_kwargs())
        self.output_sink(output)

    def _handle_hypothesis_scan(self, context: AgentContext) -> None:
        output = run_hypothesis_scan(
            context,
            schedule_next=self.scheduler.schedule_hypothesis_scan,
            **self._agent_kwargs(),
        )
        self.output_sink(output)

    def _cycle_date(self) -> date:
        return self._now().astimezone(RUNTIME_TIMEZONE).date()

    def _expected_tickers(
        self,
        user_id: str,
        current_context: AgentContext,
        cycle_date: date,
    ) -> frozenset[str]:
        subscriptions = [
            subscription
            for ticker_registry in self.event_bus.registry.values()
            for subscription in ticker_registry.subscribers
            if subscription.user_id == user_id
        ]
        if cycle_date.weekday() == 4:
            expected = {subscription.ticker.upper() for subscription in subscriptions}
        else:
            expected = {
                subscription.ticker.upper()
                for subscription in subscriptions
                if subscription.update_interval == UpdateInterval.DAILY
            }
        current_ticker = current_context.ticker.upper()
        if current_ticker not in expected:
            return frozenset({current_ticker})
        return frozenset(expected)

    def _record_scheduled_output(
        self,
        context: AgentContext,
        output: AgentOutput,
    ) -> None:
        user_id = context.subscription.user_id
        cycle_date = self._cycle_date()
        key = (user_id, cycle_date)
        expected = self._expected_tickers(user_id, context, cycle_date)

        with self._cycle_lock:
            self._prune_cycles(cycle_date)
            cycle = self._cycles.get(key)
            if cycle is None or cycle.expected_tickers != expected:
                cycle = _PortfolioCycle(expected_tickers=expected)
                self._cycles[key] = cycle
            cycle.outputs[context.ticker.upper()] = output
            ready = expected.issubset(cycle.outputs)
            if not ready or cycle.running or cycle.completed:
                return
            cycle.running = True
            cycle_outputs = [cycle.outputs[ticker] for ticker in sorted(expected)]

        try:
            portfolio_context = self.event_bus.build_portfolio_context(
                user_id,
                cycle_outputs,
            )
            portfolio_output = run_cross_portfolio(
                portfolio_context,
                **self._agent_kwargs(),
            )
            self.output_sink(portfolio_output)
        except Exception:
            with self._cycle_lock:
                cycle.running = False
            raise

        with self._cycle_lock:
            cycle.running = False
            cycle.completed = True

    def _prune_cycles(self, current_date: date) -> None:
        stale_keys = [key for key in self._cycles if key[1] != current_date]
        for key in stale_keys:
            del self._cycles[key]

    def _handle_scheduled_update(self, context: AgentContext) -> None:
        output = run_scheduled_review(context, **self._agent_kwargs())
        self.output_sink(output)
        self._record_scheduled_output(context, output)

    def register_handlers(self) -> None:
        """Register each ticker event exactly once for this runtime."""
        if self._registered:
            return
        self.event_bus.register_handler(
            EventType.SCHEDULED_UPDATE,
            self._handle_scheduled_update,
        )
        self.event_bus.register_handler(EventType.SHARP_MOVE, self._handle_sharp_move)
        self.event_bus.register_handler(
            EventType.MOTIVE_CHECK,
            self._handle_motive_check,
        )
        self.event_bus.register_handler(
            EventType.HYPOTHESIS_SCAN,
            self._handle_hypothesis_scan,
        )
        self._registered = True

    def start(self) -> None:
        """Register handlers, load state, seed jobs, and start scheduling."""
        self.register_handlers()
        self.scheduler.start(load_registry=True)

    def shutdown(self, *, wait: bool = True) -> None:
        self.scheduler.shutdown(wait=wait)
