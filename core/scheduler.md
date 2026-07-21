# Sentient Scheduler

`scheduler.py` is Sentient's clock and market-data coordinator. It decides when
work is due, while Twelve Data fetches prices, `database.py` persists them, the
event bus routes user work, and future agents perform analysis.

## Market refreshes

One market refresh cycle runs after each completed 15-minute US market candle.
The cycle reads active tickers from the event bus, removes duplicates, and orders
them using this priority:

1. Tickers with no data yet.
2. Tickers whose previous cycle failed.
3. Tickers with the most subscribers.
4. Tickers with the stalest successful fetch.
5. Ticker symbol as a stable tie-breaker.

Requests are evenly spaced using the configured Twelve Data requests-per-minute
quota. A shared request gate also serializes calls and enforces spacing when a
retry happens near a normal request. Multiple users watching one ticker still
produce only one market-data request.

Regular refresh jobs run Monday through Friday in the `America/New_York`
timezone: 9:46 AM, every 15 minutes from 10:01 AM through 3:46 PM, and 4:01 PM.
This places requests just after each regular-session candle closes. Exchange
holidays and early closes are not yet calendar-aware.

## Retries

Temporary failures are retried with exponential delay and small random jitter.
Rate-limit responses wait at least 65 seconds. Authentication errors and invalid
symbols are treated as terminal for the cycle. A failed ticker never blocks the
remaining queue, and exhausted failures are prioritized in the next cycle.

The relevant settings are:

```env
TWELVE_DATA_REQUESTS_PER_MINUTE=8
TWELVE_DATA_RATE_LIMIT_BUFFER_SECONDS=1
TWELVE_DATA_MAX_RETRIES=2
TWELVE_DATA_RETRY_BASE_DELAY_SECONDS=5
```

The requests-per-minute value must match the quota shown in the Twelve Data
account. Retries consume quota and pass through the same request gate.

## User schedules

Daily subscription updates run on weekdays at 4:10 PM ET. Weekly updates run on
Friday at 4:10 PM ET. Motive checks run Friday at 4:20 PM ET. These jobs emit
targeted event-bus events; they do not call agents directly.

Hypothesis scans use one replaceable date job per user and ticker. The future
hypothesis agent calls `schedule_hypothesis_scan()` with its recommended number
of days, replacing the previous scan time.

## Lifecycle

`start()` reloads the event-bus registry, reconciles all jobs, and starts
APScheduler. `sync_jobs()` adds jobs for new subscriptions and removes obsolete
subscription jobs. `shutdown()` stops APScheduler cleanly. Stable job IDs plus
`replace_existing`, `max_instances=1`, and coalescing prevent duplicate or
overlapping recurring jobs.
