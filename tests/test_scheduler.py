from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from core.event_bus import EventBus
from core.scheduler import (
    HYPOTHESIS_SCAN_JOB_PREFIX,
    MARKET_CYCLE_JOB_PREFIX,
    MARKET_RETRY_JOB_PREFIX,
    MOTIVE_CHECK_JOB_PREFIX,
    SCHEDULED_UPDATE_JOB_PREFIX,
    SchedulerError,
    SentientScheduler,
    _RequestGate,
)
from models.schemas import (
    Motive,
    OHLCVPoint,
    Subscription,
    UpdateInterval,
)


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, SimpleNamespace] = {}
        self.running = False
        self.started = False
        self.shutdown_wait: bool | None = None

    def add_job(self, func, trigger=None, **kwargs):
        job = SimpleNamespace(
            id=kwargs["id"],
            func=func,
            trigger=trigger,
            kwargs=kwargs,
        )
        self.jobs[job.id] = job
        return job

    def get_jobs(self):
        return list(self.jobs.values())

    def remove_job(self, job_id):
        del self.jobs[job_id]

    def start(self):
        self.running = True
        self.started = True

    def shutdown(self, wait=True):
        self.running = False
        self.shutdown_wait = wait


def subscription(
    user_id: str,
    ticker: str,
    *,
    interval: UpdateInterval = UpdateInterval.DAILY,
) -> Subscription:
    return Subscription(
        user_id=user_id,
        ticker=ticker,
        avg_price=100.0,
        shares=2.0,
        motive=Motive.HOLDING,
        update_interval=interval,
    )


def points(ticker: str = "NVDA", close: float = 110.0) -> list[OHLCVPoint]:
    return [
        OHLCVPoint(
            ticker=ticker,
            timestamp=datetime(2026, 7, 20, 19, 45, tzinfo=timezone.utc),
            open=close - 1,
            high=close + 1,
            low=close - 2,
            close=close,
            volume=1000.0,
        )
    ]


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 20, 14, 0, tzinfo=timezone.utc)

    def test_refresh_cycle_prioritizes_missing_data_then_subscribers(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("aapl-user", "AAPL"))
        for index in range(3):
            bus.refresh_subscription(subscription(f"nvda-user-{index}", "NVDA"))
        bus.refresh_subscription(subscription("msft-user", "MSFT"))
        bus.update_ticker_state(
            "NVDA",
            current_price=110.0,
            last_fetched=self.now - timedelta(minutes=15),
        )
        bus.update_ticker_state(
            "MSFT",
            current_price=210.0,
            last_fetched=self.now - timedelta(minutes=30),
        )
        backend = FakeScheduler()
        scheduler = SentientScheduler(
            bus,
            scheduler=backend,
            requests_per_minute=15,
            rate_limit_buffer_seconds=5,
        )

        scheduled = scheduler.run_market_refresh_cycle(now=self.now)

        self.assertEqual(
            [item.ticker for item in scheduled],
            ["AAPL", "NVDA", "MSFT"],
        )
        self.assertAlmostEqual(scheduler.request_spacing_seconds, 65 / 15)
        self.assertEqual(scheduled[0].run_at, self.now)
        self.assertEqual(
            scheduled[1].run_at,
            self.now + timedelta(seconds=65 / 15),
        )
        self.assertEqual(
            scheduled[2].run_at,
            self.now + timedelta(seconds=2 * 65 / 15),
        )
        self.assertEqual(scheduler.run_market_refresh_cycle(now=self.now), [])

    def test_failed_ticker_is_prioritized_ahead_of_healthy_ticker(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        bus.refresh_subscription(subscription("user-2", "MSFT"))
        for ticker in ("NVDA", "MSFT"):
            bus.update_ticker_state(
                ticker,
                current_price=100.0,
                last_fetched=self.now,
            )
        scheduler = SentientScheduler(bus, scheduler=FakeScheduler())
        scheduler._failed_tickers.add("MSFT")

        self.assertEqual(scheduler.prioritized_tickers(), ["MSFT", "NVDA"])

    def test_successful_refresh_updates_event_bus_and_clears_queue(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        refresh = Mock(return_value=points())
        scheduler = SentientScheduler(
            bus,
            scheduler=FakeScheduler(),
            refresh_ticker=refresh,
            now_provider=lambda: self.now,
        )
        scheduler._queued_tickers.add("NVDA")
        scheduler._failed_tickers.add("NVDA")

        succeeded = scheduler._run_ticker_refresh("nvda")

        self.assertTrue(succeeded)
        refresh.assert_called_once_with("NVDA")
        self.assertEqual(bus.registry["NVDA"].current_price, 110.0)
        self.assertEqual(bus.registry["NVDA"].last_fetched, self.now)
        self.assertNotIn("NVDA", scheduler._queued_tickers)
        self.assertNotIn("NVDA", scheduler.failed_tickers)

    def test_temporary_failure_schedules_short_delayed_retry(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        backend = FakeScheduler()
        scheduler = SentientScheduler(
            bus,
            scheduler=backend,
            refresh_ticker=Mock(side_effect=ConnectionError("network down")),
            now_provider=lambda: self.now,
            jitter_provider=lambda _start, _end: 0.0,
            max_retries=2,
            retry_base_delay_seconds=5,
        )
        scheduler._queued_tickers.add("NVDA")

        succeeded = scheduler._run_ticker_refresh("NVDA")

        self.assertFalse(succeeded)
        retry = backend.jobs[f"{MARKET_RETRY_JOB_PREFIX}NVDA"]
        self.assertEqual(retry.kwargs["run_date"], self.now + timedelta(seconds=5))
        self.assertEqual(retry.kwargs["args"], ["NVDA", 1])
        self.assertIn("NVDA", scheduler._queued_tickers)
        self.assertNotIn("NVDA", scheduler.failed_tickers)

    def test_rate_limit_failure_waits_at_least_65_seconds(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        backend = FakeScheduler()
        scheduler = SentientScheduler(
            bus,
            scheduler=backend,
            refresh_ticker=Mock(side_effect=RuntimeError("HTTP 429 rate limit")),
            now_provider=lambda: self.now,
            jitter_provider=lambda _start, _end: 0.0,
        )

        scheduler._run_ticker_refresh("NVDA")

        retry = backend.jobs[f"{MARKET_RETRY_JOB_PREFIX}NVDA"]
        self.assertEqual(retry.kwargs["run_date"], self.now + timedelta(seconds=65))

    def test_terminal_failure_is_not_retried(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        backend = FakeScheduler()
        scheduler = SentientScheduler(
            bus,
            scheduler=backend,
            refresh_ticker=Mock(side_effect=RuntimeError("Invalid API key")),
        )
        scheduler._queued_tickers.add("NVDA")

        succeeded = scheduler._run_ticker_refresh("NVDA")

        self.assertFalse(succeeded)
        self.assertNotIn(f"{MARKET_RETRY_JOB_PREFIX}NVDA", backend.jobs)
        self.assertIn("NVDA", scheduler.failed_tickers)
        self.assertNotIn("NVDA", scheduler._queued_tickers)

    def test_request_gate_enforces_spacing_across_calls(self):
        clock = [100.0]
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += seconds

        gate = _RequestGate(
            4.0,
            monotonic=lambda: clock[0],
            sleeper=sleep,
        )

        with gate.slot():
            pass
        clock[0] += 1.0
        with gate.slot():
            pass

        self.assertEqual(slept, [3.0])

    def test_sync_jobs_adds_market_and_personal_schedules(self):
        bus = EventBus()
        bus.refresh_subscription(
            subscription("daily-user", "NVDA", interval=UpdateInterval.DAILY)
        )
        bus.refresh_subscription(
            subscription("weekly-user", "NVDA", interval=UpdateInterval.WEEKLY)
        )
        backend = FakeScheduler()
        scheduler = SentientScheduler(bus, scheduler=backend)

        scheduler.sync_jobs()

        market_jobs = [
            job_id
            for job_id in backend.jobs
            if job_id.startswith(MARKET_CYCLE_JOB_PREFIX)
        ]
        self.assertEqual(len(market_jobs), 3)
        self.assertIn(
            f"{SCHEDULED_UPDATE_JOB_PREFIX}daily-user:NVDA",
            backend.jobs,
        )
        self.assertIn(
            f"{SCHEDULED_UPDATE_JOB_PREFIX}weekly-user:NVDA",
            backend.jobs,
        )
        self.assertIn(
            f"{MOTIVE_CHECK_JOB_PREFIX}daily-user:NVDA",
            backend.jobs,
        )
        self.assertIn(
            f"{MOTIVE_CHECK_JOB_PREFIX}weekly-user:NVDA",
            backend.jobs,
        )

    def test_sync_jobs_removes_jobs_for_deleted_subscription(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("active-user", "NVDA"))
        backend = FakeScheduler()
        backend.add_job(
            Mock(),
            trigger="date",
            id=f"{SCHEDULED_UPDATE_JOB_PREFIX}deleted-user:AAPL",
        )
        backend.add_job(
            Mock(),
            trigger="date",
            id=f"{HYPOTHESIS_SCAN_JOB_PREFIX}deleted-user:AAPL",
        )
        scheduler = SentientScheduler(bus, scheduler=backend)

        scheduler.sync_jobs()

        self.assertNotIn(
            f"{SCHEDULED_UPDATE_JOB_PREFIX}deleted-user:AAPL",
            backend.jobs,
        )
        self.assertNotIn(
            f"{HYPOTHESIS_SCAN_JOB_PREFIX}deleted-user:AAPL",
            backend.jobs,
        )

    def test_hypothesis_scan_is_replaceable_and_requires_subscription(self):
        bus = EventBus()
        bus.refresh_subscription(subscription("user-1", "NVDA"))
        backend = FakeScheduler()
        scheduler = SentientScheduler(
            bus,
            scheduler=backend,
            now_provider=lambda: self.now,
        )

        run_at = scheduler.schedule_hypothesis_scan("user-1", "nvda", 2)

        job_id = f"{HYPOTHESIS_SCAN_JOB_PREFIX}user-1:NVDA"
        self.assertEqual(run_at, self.now + timedelta(days=2))
        self.assertEqual(backend.jobs[job_id].kwargs["run_date"], run_at)
        with self.assertRaises(SchedulerError):
            scheduler.schedule_hypothesis_scan("missing-user", "NVDA", 2)

    def test_start_loads_registry_syncs_jobs_and_starts_backend(self):
        bus = EventBus(
            ticker_loader=Mock(return_value=[]),
            subscription_loader=Mock(return_value=[]),
        )
        bus.load_registry = Mock(wraps=bus.load_registry)
        backend = FakeScheduler()
        scheduler = SentientScheduler(bus, scheduler=backend)

        scheduler.start()

        bus.load_registry.assert_called_once_with()
        self.assertTrue(backend.started)
        scheduler.shutdown(wait=False)
        self.assertFalse(backend.shutdown_wait)


if __name__ == "__main__":
    unittest.main()
