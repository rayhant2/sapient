# Scheduled Position Review

The scheduled review is the routine daily or weekly analysis path for a stock a
user already owns or watches. It accepts an `AgentContext` from the event bus and
runs with that user's Anthropic credential through the shared execution boundary.

## Flow

1. Summarize the rolling OHLCV window, recent candles, position economics, prior
   updates, and recent alerts.
2. Ask for a strict structured decision on whether current public research is
   necessary.
3. When necessary, run one focused Anthropic web-research request through the
   isolated public-only research subsystem. A search outage does not prevent the
   routine market-data review from completing.
4. Produce a strict `TickerAnalysisDraft` containing only summary,
   recommendation, and confidence.
5. Add trusted identity, event, price, and research metadata in application code,
   validate the final `AgentOutput`, and persist it through `data/database.py`.

## Review Inputs

- Cost basis, shares, motive, current price, and unrealized P&L
- Full-window price change, recent four-candle change, high, and low
- Mean candle return and realized candle-return volatility
- Average volume and latest-volume ratio
- The latest 12 candles, recent agent updates, and recent alerts

The compact aggregate payload avoids sending all 150 candles to the model on every
routine run. Web research is reserved for material changes, potentially stale
theses, or unusual price and volume behavior. Outputs are monitoring guidance only;
the agent cannot execute trades.
