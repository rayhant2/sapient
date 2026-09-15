# Motive Reassessment Agent

The motive agent runs weekly for every subscription. It checks whether observed
conditions align with the user's broad motive: `holding`, `short-term`, or
`watching`.

## Flow

1. Summarize the rolling OHLCV window, P&L context, trend consistency, realized
   volatility, drawdown, rebound, recent volume, prior updates, and alerts.
2. Produce a strict preliminary classification: `aligned`, `at_risk`, or
   `insufficient_evidence`.
3. Optionally run one focused public search when current external evidence is
   necessary to evaluate a material change.
4. Produce a validated `AgentOutput` that explains motive alignment and proposes
   a monitoring or thesis-clarification action.
5. Add trusted identity and research metadata, then persist through the shared
   execution boundary.

## Data Boundary

Subscriptions currently store only a broad motive, average price, and share count.
They do not store an entry timestamp or a detailed investment thesis. The agent
therefore evaluates only the broad motive against the available rolling data and
states that limitation explicitly. It must not invent a holding period, entry date,
or original rationale.

For a `watching` subscription, position-like schema fields are treated as reference
values rather than proof that the user owns the stock. The agent cannot execute or
recommend brokerage orders.
