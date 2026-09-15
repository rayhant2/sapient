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
The default sink is intentionally a no-op because WhatsApp formatting, sending, and
alert logging belong to the notification layer. This keeps agent execution independent
from Twilio while giving that layer one integration point for all five output types.
