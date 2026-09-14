# Database Schema

Sentient uses Supabase/Postgres as the durable store. The schema lives in
[`supabase/schema.sql`](../supabase/schema.sql) and should be run in the Supabase
SQL editor for a new project.

The core design is: ticker market data is shared, user context is personal.

## Tables

### `users`

One row per Sentient user.

Important fields:

- `user_id`: app-level primary key.
- `whatsapp_number`: unique number used for Twilio delivery.
- `email`: optional unique email.
- `preferences`: JSONB for user-level settings that do not deserve columns yet.

### `tickers`

One row per unique stock symbol across all users.

Important fields:

- `ticker`: primary key, stored uppercase.
- `last_fetched`: last successful market-data fetch.
- `next_fetch_time`: next scheduled fetch time.
- `current_price`: latest known shared price.

This table is shared infrastructure. If 50 users watch NVDA, NVDA appears here
once.

### `ticker_data`

Shared OHLCV candles for each ticker.

Important fields:

- `ticker`: foreign key to `tickers`.
- `timestamp`: candle timestamp.
- `open`, `high`, `low`, `close`, `volume`: validated non-negative OHLCV data.

The primary key is `(ticker, timestamp)` so the same candle cannot be inserted
twice. The app layer will enforce the rolling 150-row limit per ticker in
`data/database.py`.

### `subscriptions`

The personalization layer. This is the heart of the multi-user design.

Important fields:

- `user_id`: foreign key to `users`.
- `ticker`: foreign key to `tickers`.
- `avg_price`: user's average entry price.
- `shares`: user's position size.
- `motive`: one of `holding`, `short-term`, or `watching`.
- `update_interval`: one of `daily` or `weekly`.
- `sharp_move_threshold`: user-specific move threshold, default `0.025`.

The primary key is `(user_id, ticker)`, which means each user can have one
subscription per ticker while many users can watch the same ticker.

### `updates`

History of agent outputs.

Important fields:

- `user_id`: owner of the update.
- `agent_type`: identifies which of the five agents produced the output.
- `ticker`: ticker being discussed; null for cross-portfolio outputs.
- `event_type`: trigger for a ticker-level run; null for cross-portfolio outputs.
- `summary`: agent-written summary; null only for an unflagged hypothesis.
- `recommendation`: agent-written next-step framing; null for cross-portfolio outputs.
- `confidence`: `high`, `medium`, or `low`; null for cross-portfolio outputs.
- `price_at_update`: optional price snapshot.
- `searched_web`: whether the agent used web search.
- `metadata`: specialized fields used to reconstruct typed outputs. Hypothesis rows
  store `flagged` and `recommended_next_scan_days`; cross-portfolio rows store
  `correlations_flagged` and `tickers_analyzed`.

`agent_type` is the persistence discriminator. Database reads use it to rebuild an
`AgentOutput`, `HypothesisOutput`, or `CrossPortfolioOutput` without discarding
specialized fields.

### `alerts`

History of WhatsApp notifications.

Important fields:

- `user_id`: notification recipient.
- `ticker`: nullable because cross-portfolio alerts are not about one ticker.
- `alert_type`: one of the `AlertType` values from `models/schemas.py`.
- `message`: exact notification body or rendered alert text.
- `trigger_details`: JSONB with structured context about why the alert fired.

## Indexes

The schema adds indexes for the app's expected query patterns:

- `subscriptions(ticker)`: event bus fan-out by ticker.
- `subscriptions(user_id)`: dashboard/user portfolio lookups.
- `ticker_data(ticker, timestamp desc)`: latest candles for charts and agent context.
- `tickers(next_fetch_time)`: scheduler lookup for due ticker fetches.
- `updates(user_id, timestamp desc)`: latest user updates.
- `updates(user_id, ticker, timestamp desc)`: latest update for one user/ticker.
- `updates(user_id, agent_type, timestamp desc)`: latest output from one agent.
- `updates(user_id, ticker, agent_type, timestamp desc)`: filtered per-position
  agent history.
- `alerts(user_id, timestamp desc)`: alert history.
- `alerts(user_id, ticker, timestamp desc)`: alert history for one user/ticker.
- `alerts(user_id, ticker, alert_type, timestamp desc)`: filtered notification
  history.

## Memory Retrieval

Agent memory reads stay in `data/database.py` and always require a `user_id`:

- `get_latest_update(...)` returns the newest typed output for an agent, optionally
  scoped to one ticker.
- `list_recent_updates(...)` returns newest-first typed outputs with optional ticker,
  agent, and UTC-aware `since` filters.
- `list_recent_alerts(...)` provides the equivalent bounded alert history.
- `list_latest_portfolio_updates(...)` loads a user's subscriptions, performs one
  bounded update query, and returns the newest output found for each subscribed
  ticker.

Public history queries accept between 1 and 100 rows. Portfolio retrieval scans at
most 20 recent rows per subscribed ticker, capped at 500 total rows, before
deduplicating in memory. Timestamps used for `since` filters must be timezone-aware.

## Constraint Strategy

The schema uses `text check (...)` constraints instead of Postgres enum types.
That keeps early development flexible while still preventing invalid values from
entering the database.

The checked values mirror `models/schemas.py`:

- `motive`: `holding`, `short-term`, `watching`
- `update_interval`: `daily`, `weekly`
- `event_type`: `scheduled_update`, `sharp_move`, `motive_check`, `hypothesis_scan`
- `agent_type`: `scheduled_review`, `sharp_move`, `motive`, `hypothesis`,
  `cross_portfolio`
- `confidence`: `high`, `medium`, `low`
- `alert_type`: `sharp_move`, `motive_flag`, `hypothesis`, `cross_portfolio`, `scheduled`

## Next Step

After this schema is created in Supabase, Step 4 is `data/database.py`: a single
Python interface for all database reads and writes. No other module should talk
to Supabase directly.
