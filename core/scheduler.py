from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import random
import threading
import time
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import settings
from core.event_bus import DispatchResult, EventBus, UnknownTickerError
from data.twelve_data import refresh_ticker_data
from models.schemas import EventType, OHLCVPoint, TickerRegistry, UpdateInterval


logger = logging.getLogger(__name__)

MARKET_TIMEZONE = ZoneInfo("America/New_York")
MARKET_CYCLE_JOB_PREFIX = "market-cycle:"
MARKET_REFRESH_JOB_PREFIX = "market-refresh:"
MARKET_RETRY_JOB_PREFIX = "market-retry:"
SCHEDULED_UPDATE_JOB_PREFIX = "scheduled-update:"
MOTIVE_CHECK_JOB_PREFIX = "motive-check:"
HYPOTHESIS_SCAN_JOB_PREFIX = "hypothesis-scan:"

RefreshTicker = Callable[[str], list[OHLCVPoint]]
NowProvider = Callable[[], datetime]
JitterProvider = Callable[[float, float], float]


class SchedulerError(RuntimeError):
    """Raised when scheduler configuration or coordination fails."""


class EmptyTickerRefreshError(SchedulerError):
    """Raised when a market-data refresh returns no candles."""


@dataclass(frozen=True)
class ScheduledRefresh:
    ticker: str
    run_at: datetime
    priority: int


class _RequestGate:
    """Serializes provider calls and enforces minimum request-start spacing."""

    def __init__(
        self,
        spacing_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._spacing_seconds = spacing_seconds
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._last_started: float | None = None
        self._lock = threading.Lock()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._lock:
            now = self._monotonic()
            if self._last_started is not None:
                wait_seconds = self._spacing_seconds - (now - self._last_started)
                if wait_seconds > 0:
                    self._sleeper(wait_seconds)
            self._last_started = self._monotonic()
            yield


class SentientScheduler:
    def __init__(
        self,
        event_bus: EventBus,
        *,
        scheduler: Any | None = None,
        refresh_ticker: RefreshTicker | None = None,
        now_provider: NowProvider | None = None,
        jitter_provider: JitterProvider | None = None,
        requests_per_minute: int = settings.twelve_data_requests_per_minute,
        rate_limit_buffer_seconds: float = (
            settings.twelve_data_rate_limit_buffer_seconds
        ),
        max_retries: int = settings.twelve_data_max_retries,
        retry_base_delay_seconds: float = (
            settings.twelve_data_retry_base_delay_seconds
        ),
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        if rate_limit_buffer_seconds < 0:
            raise ValueError("rate_limit_buffer_seconds must be non-negative")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_base_delay_seconds < 0:
            raise ValueError("retry_base_delay_seconds must be non-negative")

        self.event_bus = event_bus
        self._scheduler = scheduler or BackgroundScheduler(timezone=MARKET_TIMEZONE)
        self._refresh_ticker = refresh_ticker or refresh_ticker_data
        self._now = now_provider or (lambda: datetime.now(timezone.utc))
        self._jitter = jitter_provider or random.uniform
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._request_spacing_seconds = (
            60.0 + rate_limit_buffer_seconds
        ) / requests_per_minute
        self._request_gate = _RequestGate(self._request_spacing_seconds)
        self._queued_tickers: set[str] = set()
        self._failed_tickers: set[str] = set()
        self._state_lock = threading.Lock()

    @property
    def request_spacing_seconds(self) -> float:
        return self._request_spacing_seconds

    @property
    def failed_tickers(self) -> frozenset[str]:
        return frozenset(self._failed_tickers)

    @staticmethod
    def _ticker_symbol(ticker: str) -> str:
        return ticker.upper()

    @staticmethod
    def _subscription_suffix(user_id: str, ticker: str) -> str:
        return f"{user_id}:{ticker.upper()}"

    def _priority_key(self, ticker_registry: TickerRegistry) -> tuple[Any, ...]:
        symbol = self._ticker_symbol(ticker_registry.ticker)
        has_no_data = (
            ticker_registry.current_price is None
            or ticker_registry.last_fetched is None
        )
        previously_failed = symbol in self._failed_tickers
        last_fetched = (
            ticker_registry.last_fetched.timestamp()
            if ticker_registry.last_fetched is not None
            else float("-inf")
        )
        return (
            0 if has_no_data else 1,
            0 if previously_failed else 1,
            -ticker_registry.subscriber_count,
            last_fetched,
            symbol,
        )

    def prioritized_tickers(self) -> list[str]:
        active = [
            ticker_registry
            for ticker_registry in self.event_bus.registry.values()
            if ticker_registry.is_active
        ]
        return [
            self._ticker_symbol(ticker_registry.ticker)
            for ticker_registry in sorted(active, key=self._priority_key)
        ]

    def run_market_refresh_cycle(
        self,
        *,
        now: datetime | None = None,
    ) -> list[ScheduledRefresh]:
        cycle_started = (now or self._now()).astimezone(timezone.utc)
        cycle_id = cycle_started.strftime("%Y%m%dT%H%M%S%f")
        scheduled: list[ScheduledRefresh] = []

        for priority, ticker in enumerate(self.prioritized_tickers()):
            with self._state_lock:
                if ticker in self._queued_tickers:
                    continue
                self._queued_tickers.add(ticker)

            run_at = cycle_started + timedelta(
                seconds=len(scheduled) * self._request_spacing_seconds
            )
            job_id = f"{MARKET_REFRESH_JOB_PREFIX}{cycle_id}:{ticker}"
            try:
                self._scheduler.add_job(
                    self._run_ticker_refresh,
                    trigger="date",
                    run_date=run_at,
                    args=[ticker, 0],
                    id=job_id,
                    replace_existing=True,
                    max_instances=1,
                    misfire_grace_time=120,
                )
            except Exception:
                with self._state_lock:
                    self._queued_tickers.discard(ticker)
                raise

            scheduled.append(
                ScheduledRefresh(ticker=ticker, run_at=run_at, priority=priority)
            )

        return scheduled

    @staticmethod
    def _exception_text(exc: Exception) -> str:
        messages: list[str] = []
        current: BaseException | None = exc
        while current is not None:
            messages.append(str(current).lower())
            current = current.__cause__ or current.__context__
        return " ".join(messages)

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        message = self._exception_text(exc)
        return any(
            marker in message
            for marker in ("429", "rate limit", "too many requests", "api credits")
        )

    def _is_terminal_error(self, exc: Exception) -> bool:
        message = self._exception_text(exc)
        return any(
            marker in message
            for marker in (
                "invalid api key",
                "apikey is invalid",
                "unauthorized",
                "forbidden",
                "invalid symbol",
            )
        )

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        exponential_delay = self._retry_base_delay_seconds * (2**attempt)
        jitter = self._jitter(0.0, max(1.0, exponential_delay * 0.25))
        delay = exponential_delay + jitter
        if self._is_rate_limit_error(exc):
            return max(65.0, delay)
        return delay

    def _schedule_retry(self, ticker: str, attempt: int, exc: Exception) -> None:
        delay_seconds = self._retry_delay_seconds(exc, attempt)
        run_at = self._now().astimezone(timezone.utc) + timedelta(
            seconds=delay_seconds
        )
        self._scheduler.add_job(
            self._run_ticker_refresh,
            trigger="date",
            run_date=run_at,
            args=[ticker, attempt + 1],
            id=f"{MARKET_RETRY_JOB_PREFIX}{ticker}",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=120,
        )
        logger.warning(
            "Scheduled market-data retry for %s in %.1f seconds after %s",
            ticker,
            delay_seconds,
            type(exc).__name__,
        )

    def _finish_ticker(self, ticker: str, *, failed: bool) -> None:
        with self._state_lock:
            self._queued_tickers.discard(ticker)
            if failed:
                self._failed_tickers.add(ticker)
            else:
                self._failed_tickers.discard(ticker)

    def _run_ticker_refresh(self, ticker: str, attempt: int = 0) -> bool:
        symbol = self._ticker_symbol(ticker)
        try:
            with self._request_gate.slot():
                points = self._refresh_ticker(symbol)
            if not points:
                raise EmptyTickerRefreshError(
                    f"Market-data refresh returned no candles for {symbol}."
                )
        except Exception as exc:
            retryable = not self._is_terminal_error(exc)
            if retryable and attempt < self._max_retries:
                try:
                    self._schedule_retry(symbol, attempt, exc)
                except Exception:
                    self._finish_ticker(symbol, failed=True)
                    raise
                return False

            self._finish_ticker(symbol, failed=True)
            logger.error(
                "Market-data refresh failed for %s after %d attempt(s): %s",
                symbol,
                attempt + 1,
                type(exc).__name__,
            )
            return False

        try:
            self.event_bus.update_ticker_state(
                symbol,
                current_price=points[-1].close,
                last_fetched=self._now().astimezone(timezone.utc),
            )
        except UnknownTickerError:
            logger.info(
                "Ticker %s left the registry while its refresh was running.",
                symbol,
            )

        self._finish_ticker(symbol, failed=False)
        return True

    def _emit_user_event(
        self,
        event_type: EventType,
        ticker: str,
        user_id: str,
    ) -> DispatchResult:
        return self.event_bus.emit(
            event_type,
            ticker,
            target_user_ids={user_id},
        )

    def _add_market_cycle_jobs(self) -> set[str]:
        jobs = {
            f"{MARKET_CYCLE_JOB_PREFIX}open": CronTrigger(
                day_of_week="mon-fri",
                hour=9,
                minute=46,
                timezone=MARKET_TIMEZONE,
            ),
            f"{MARKET_CYCLE_JOB_PREFIX}regular": CronTrigger(
                day_of_week="mon-fri",
                hour="10-15",
                minute="1,16,31,46",
                timezone=MARKET_TIMEZONE,
            ),
            f"{MARKET_CYCLE_JOB_PREFIX}close": CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=1,
                timezone=MARKET_TIMEZONE,
            ),
        }
        for job_id, trigger in jobs.items():
            self._scheduler.add_job(
                self.run_market_refresh_cycle,
                trigger=trigger,
                id=job_id,
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=120,
            )
        return set(jobs)

    def _add_subscription_jobs(self) -> set[str]:
        expected_jobs: set[str] = set()
        for ticker_registry in self.event_bus.registry.values():
            symbol = self._ticker_symbol(ticker_registry.ticker)
            for subscription in ticker_registry.subscribers:
                suffix = self._subscription_suffix(subscription.user_id, symbol)
                update_job_id = f"{SCHEDULED_UPDATE_JOB_PREFIX}{suffix}"
                if subscription.update_interval == UpdateInterval.DAILY:
                    update_trigger = CronTrigger(
                        day_of_week="mon-fri",
                        hour=16,
                        minute=10,
                        timezone=MARKET_TIMEZONE,
                    )
                else:
                    update_trigger = CronTrigger(
                        day_of_week="fri",
                        hour=16,
                        minute=10,
                        timezone=MARKET_TIMEZONE,
                    )

                self._scheduler.add_job(
                    self._emit_user_event,
                    trigger=update_trigger,
                    args=[
                        EventType.SCHEDULED_UPDATE,
                        symbol,
                        subscription.user_id,
                    ],
                    id=update_job_id,
                    replace_existing=True,
                    max_instances=1,
                    coalesce=True,
                    misfire_grace_time=600,
                )
                expected_jobs.add(update_job_id)

                motive_job_id = f"{MOTIVE_CHECK_JOB_PREFIX}{suffix}"
                self._scheduler.add_job(
                    self._emit_user_event,
                    trigger=CronTrigger(
                        day_of_week="fri",
                        hour=16,
                        minute=20,
                        timezone=MARKET_TIMEZONE,
                    ),
                    args=[
                        EventType.MOTIVE_CHECK,
                        symbol,
                        subscription.user_id,
                    ],
                    id=motive_job_id,
                    replace_existing=True,
                    max_instances=1,
                    coalesce=True,
                    misfire_grace_time=600,
                )
                expected_jobs.add(motive_job_id)

        return expected_jobs

    def _remove_obsolete_jobs(self, expected_jobs: set[str]) -> None:
        active_suffixes = {
            self._subscription_suffix(subscription.user_id, ticker_registry.ticker)
            for ticker_registry in self.event_bus.registry.values()
            for subscription in ticker_registry.subscribers
        }
        for job in self._scheduler.get_jobs():
            job_id = job.id
            if job_id.startswith((SCHEDULED_UPDATE_JOB_PREFIX, MOTIVE_CHECK_JOB_PREFIX)):
                if job_id not in expected_jobs:
                    self._scheduler.remove_job(job_id)
            elif job_id.startswith(HYPOTHESIS_SCAN_JOB_PREFIX):
                suffix = job_id.removeprefix(HYPOTHESIS_SCAN_JOB_PREFIX)
                if suffix not in active_suffixes:
                    self._scheduler.remove_job(job_id)

    def sync_jobs(self) -> None:
        active_tickers = any(
            ticker_registry.is_active
            for ticker_registry in self.event_bus.registry.values()
        )
        expected_jobs: set[str] = set()
        if active_tickers:
            expected_jobs.update(self._add_market_cycle_jobs())
            expected_jobs.update(self._add_subscription_jobs())
        else:
            for job in list(self._scheduler.get_jobs()):
                if job.id.startswith(MARKET_CYCLE_JOB_PREFIX):
                    self._scheduler.remove_job(job.id)

        self._remove_obsolete_jobs(expected_jobs)

    def schedule_hypothesis_scan(
        self,
        user_id: str,
        ticker: str,
        days: int,
    ) -> datetime:
        if days <= 0:
            raise ValueError("days must be greater than zero")

        symbol = self._ticker_symbol(ticker)
        ticker_registry = self.event_bus.get_registry(symbol)
        if ticker_registry.get_subscriber(user_id) is None:
            raise SchedulerError(f"User {user_id} is not subscribed to {symbol}.")

        run_at = self._now().astimezone(timezone.utc) + timedelta(days=days)
        job_id = (
            f"{HYPOTHESIS_SCAN_JOB_PREFIX}"
            f"{self._subscription_suffix(user_id, symbol)}"
        )
        self._scheduler.add_job(
            self._emit_user_event,
            trigger="date",
            run_date=run_at,
            args=[EventType.HYPOTHESIS_SCAN, symbol, user_id],
            id=job_id,
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=600,
        )
        return run_at

    def start(self, *, load_registry: bool = True) -> None:
        if load_registry:
            self.event_bus.load_registry()
        self.sync_jobs()
        self._scheduler.start()

    def shutdown(self, *, wait: bool = True) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)
