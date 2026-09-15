# Sharp-Move Investigation Agent

This agent runs immediately after `SharpMoveMonitor` targets a subscription. The
monitor decides deterministically whether a move is significant; the agent explains
what may have caused it and why it matters for that user's position.

## Flow

1. Recompute the trigger evidence from trusted OHLCV data: latest return, personal
   threshold, prior-return volatility, z-score, VWAP distance, volume ratio, and
   whether the candle crossed the user's cost basis.
2. Load a bounded snapshot containing the latest stored update for each of the
   user's other tracked tickers. The tool omits the internal user ID and excludes
   the ticker currently being investigated.
3. Run a focused public catalyst search. If search is unavailable, continue with
   the market evidence and state that the cause is unconfirmed.
4. Use a strict structured decision to determine whether one additional sector
   search would clarify company-specific versus broader movement.
5. Produce a validated `AgentOutput`, add trusted identity and research metadata,
   and persist it through the shared execution boundary.

## Cost Controls

Every sharp move receives at most one catalyst research stage and one conditional
sector stage. Stored portfolio summaries are reused instead of rerunning analysis
for other holdings. Tests mock all model and search calls, so the test suite does
not consume user API credits.

The result provides monitoring and research guidance only. It does not place or
recommend a brokerage order.
