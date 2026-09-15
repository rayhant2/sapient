# Structured Agent Outputs

Agent model responses are bound to strict Pydantic draft schemas using Anthropic's
native JSON-schema output mode. Draft schemas contain only fields the model is
allowed to decide.

## Model-Owned Fields

- Standard ticker agents: summary, recommendation, and confidence.
- Hypothesis agent: summary, recommendation, confidence, flagged status, and the
  next scan interval from one through three days.
- Cross-portfolio agent: summary and flagged correlations.

Unknown fields are rejected. A model cannot provide or override an identity,
ticker, event type, price, citation, search count, or analyzed ticker list.

## Server-Owned Fields

Ticker output finalization copies the user ID, ticker, event type, and current price
from the trusted `AgentContext`. Web-search usage and normalized citations come from
LangGraph state populated by the research subsystem.

Cross-portfolio finalization copies the user ID and ticker list from
`PortfolioContext`. Every position must belong to that same user.

The finalizers also enforce the mapping between agent and event type. For example,
a sharp-move agent cannot finalize output from a scheduled-update context.

## Validation

Flagged hypotheses require a summary and recommendation. Unflagged hypotheses must
omit both, and all hypothesis scan intervals are constrained to one through three
days. Malformed model responses become sanitized `AgentOutputValidationError`
exceptions rather than partially trusted outputs.
