# Proactive Hypothesis Agent

The hypothesis agent scans each ticker without a predetermined question. It looks for
coherent, testable patterns that may deserve attention before a normal scheduled
review or sharp-move alert.

## Flow

1. Build descriptive features from the rolling OHLCV window, including multi-window
   returns, recent versus prior volatility, range compression, volume change,
   directional consistency, and twenty-candle breakout state.
2. Run a strict candidate screen. Ordinary noise returns an unflagged output without
   using web research.
3. For a candidate pattern, run one focused speculative public search for confirming
   and disconfirming evidence.
4. Let the final structured assessment retain or reject the hypothesis after research.
5. Persist the `HypothesisOutput`, then replace the user's next hypothesis job through
   `SentientScheduler.schedule_hypothesis_scan()`.

## Cadence

- No supported hypothesis: rescan in 3 days.
- Lower-urgency developing pattern: rescan in 2 days.
- Early-stage buildup requiring closer observation: rescan in 1 day.

Immediate intraday movement is handled by `SharpMoveMonitor`, not this daily-scale
cadence. Scheduling is required by the public runner so a successful scan cannot
silently omit its next check. The agent provides monitoring guidance only and cannot
execute trades.
