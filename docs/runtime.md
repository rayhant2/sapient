# Runtime Integration

`AgentRuntime` is the application boundary connecting deterministic scheduling and
event routing to the five LangGraph agents.

## Startup

1. Construct one shared `EventBus` and `SentientScheduler`.
2. Register handlers for scheduled updates, sharp moves, motive checks, and
   hypothesis scans.
3. Load the ticker/subscription registry from Supabase.
4. Synchronize recurring market and subscription jobs.
5. Seed a three-day hypothesis job for subscriptions that do not already have one.
6. Start APScheduler.

`main.py` owns this lifecycle and handles `SIGINT` and `SIGTERM` for graceful
shutdown.

## Event Routing

- `scheduled_update` runs the scheduled-review agent and records its result in the
  user's current portfolio cycle.
- `sharp_move` runs the sharp-move investigation agent.
- `motive_check` runs the weekly motive-reassessment agent.
- `hypothesis_scan` runs the hypothesis agent and replaces its next scan using the
  returned one-to-three-day cadence.

Each agent resolves the triggered user's credential immediately before model client
creation and persists its own validated output through `data/database.py`.

## Portfolio Cycles

The coordinator groups scheduled-review results by user and New York market date.
On Monday through Thursday, it waits for every daily subscription due in that cycle.
On Friday, it waits for both daily and weekly subscriptions. It then reloads current
market history for the user's full watchlist, builds one `PortfolioContext`, and runs
the cross-portfolio agent once.

Cycle mutation is protected by a lock because APScheduler may execute ticker jobs on
different worker threads. Completed cycles are deduplicated, failed cross-portfolio
runs can be retried, and stale in-memory cycles are removed on the next market date.

## Delivery Boundary

Every completed output is sent to an injected `output_sink` after its database write.
When `WHATSAPP_ENABLED=true`, the production sink formats and sends the output with
Twilio, then records the accepted message in `alerts`. When it is false, delivery is
skipped and the agent output remains available in `updates` and the dashboard. This
allows the worker to run before Twilio onboarding is complete.

## Operations

`main.py` validates required configuration before constructing the runtime, handles
`SIGINT` and `SIGTERM`, and shuts APScheduler down cleanly. Container logs can be
plain text or one-line JSON using `LOG_FORMAT`.

The worker exposes:

- `GET /health/live`: process liveness
- `GET /health/ready`: readiness after registry load and scheduler startup

The readiness response includes only lifecycle state and active ticker count. It
does not expose configuration, user data, credentials, or provider errors.
