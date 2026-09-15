# Agent Execution

`agents/core.py` owns the shared execution boundary used by all five agent graphs.
Individual agent modules provide graph factories; they do not resolve credentials,
construct model clients, configure traces, or persist outputs themselves.

## Ticker Agent Flow

1. Validate the agent type and trusted user/ticker context.
2. Hydrate bounded updates and alerts from the database.
3. Build database tools locked to that user and ticker.
4. Resolve the user's encrypted Anthropic credential and create a fresh model.
5. Pass the model and tools to the agent's graph factory.
6. Invoke the graph once with safe LangSmith metadata and a recursion limit.
7. Validate the graph output against the original context and research state.
8. Persist the typed output through `data/database.py` and validate the returned row.

The cross-portfolio flow follows the same model, invocation, validation, and
persistence boundaries, but uses `PortfolioContext` and does not receive
ticker-bound tools.

## Reliability and Cost

Provider request timeout and retry behavior are configured on `ChatAnthropic`.
LangGraph loops are capped by `AGENT_GRAPH_RECURSION_LIMIT`, which defaults to 20
and accepts values from 1 through 100.

The execution wrapper does not retry a complete graph. A whole-graph retry could
repeat paid model or web-search calls and produce duplicate output. Failed
persistence also does not repeat inference; it raises `AgentPersistenceError` so
the caller can handle storage recovery separately.

## Trace Safety

LangSmith run metadata contains only agent type, event type, public ticker, or
portfolio ticker count. It excludes user IDs, credentials, position details, and
API keys. The model client remains in the graph closure and is never placed in
LangGraph state.

## Failure Types

- `AgentExecutionError`: graph construction or invocation failed.
- `AgentOutputValidationError`: graph output did not match trusted context.
- `AgentPersistenceError`: a completed output could not be safely stored or read
  back.
- `MissingUserApiKeyError`: the user has no Anthropic credential configured.
