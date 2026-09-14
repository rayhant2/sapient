# Web Research

Sentient uses Anthropic's server-side web search through `ChatAnthropic`. The
backend does not operate a separate scraper or share one system-owned search key.
Each search runs through the model created with the current user's Anthropic API
key.

## Flow

1. An individual agent graph decides whether current external evidence is needed.
2. The graph selects one public research focus: price catalyst, company news,
   earnings, sector conditions, or regulatory developments.
3. `perform_web_research()` builds a fixed prompt containing only the public ticker
   and selected focus.
4. A dedicated model invocation receives only Anthropic's server-side web-search
   tool. Database tools are not mixed into the same call.
5. The response is normalized into a summary, sources, and billed search count.
6. Sources and search usage are attached to agent state and later persisted in the
   update row's JSON metadata.

## Cost Controls

`AGENT_WEB_SEARCH_MAX_USES` limits searches within a request and defaults to two.
The accepted range is one through five. `AGENT_WEB_SEARCH_MAX_CONTINUATIONS`
defaults to one and caps retries when Anthropic returns `pause_turn`.

Web search is opt-in at the graph level. Routine analysis should use existing
market data and memory unless current public information is necessary.

## Privacy

Research prompts never contain user IDs, cost basis, share count, motive, portfolio
composition, contact information, or API keys. The model selects from fixed public
research focuses instead of submitting arbitrary portfolio-derived queries.

Credentials remain inside the per-user model client. They are not added to graph
state, tool arguments, prompts, output metadata, or citations.

## Citations and Continuations

Normalized sources retain only title, URL, page age, and cited text. Provider-owned
encrypted result content is kept only inside the original `AIMessage` required for
an active continuation and is not persisted with the final agent output.

When Anthropic returns `pause_turn`, the exact assistant message is passed back on
the continuation request. Search errors embedded in successful HTTP responses are
detected and converted into safe application errors.
