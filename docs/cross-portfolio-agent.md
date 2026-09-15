# Cross-Portfolio Reasoning Agent

This agent runs after a user's individual position reviews complete for a cycle. It
accepts one `PortfolioContext` containing all tracked positions and the newest
individual outputs from that cycle.

Before model reasoning, application code calculates current tracked-value weights,
P&L, rolling volatility, and pairwise Pearson correlations from overlapping 15-minute
returns. Correlations require at least five shared return timestamps and are omitted
when either series has no variation.

The model uses these metrics to identify concentration, shared movement, and combined
risk without rerunning each ticker analysis. It cannot infer sectors that are absent
from the supplied evidence and must not treat correlation as causation. The resulting
`CrossPortfolioOutput` is validated and persisted through the shared execution
boundary.
