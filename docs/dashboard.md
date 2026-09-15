# MVP dashboard

The Streamlit dashboard is a single-user operating surface for Sapient. It reads
and writes only through `data/database.py`; the UI never queries Supabase directly.

## Views

- **Overview:** total tracked value, unrealized P&L, ticker count, alert count,
  position table, and latest per-ticker analysis.
- **Ticker detail:** current position metrics, 150-candle OHLCV candlestick and
  volume chart, latest recommendation, and research sources.
- **Activity:** searchable table-style history of sent alerts and stored agent
  outputs.
- **Positions:** add or update a subscription and remove tracked tickers.

## Single-user behavior

When `MVP_USER_ID` is empty, the dashboard reads at most two users from the
database. It automatically selects the user when exactly one exists. With multiple
users, set `MVP_USER_ID` explicitly.

No login or session authentication is included in this MVP. Do not expose the
dashboard publicly until authentication and Supabase row-level security are added.

## Run locally

```bash
uv run streamlit run dashboard/app.py
```

Dashboard data is cached for 45 seconds. Use the sidebar refresh control after an
agent run or data refresh when an immediate update is needed.
