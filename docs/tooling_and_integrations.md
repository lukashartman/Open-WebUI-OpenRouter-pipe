# Tools, plugins, and integrations

**Scope:** How tool schemas are built, how `function_call` items are executed, and how Open WebUI tool sources (registry + Direct Tool Servers) are attached.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [🔒 Security](security_and_encryption.md)

This pipe supports OpenRouter tool calling either via an internal execution pipeline (pipe-run tools) or via Open WebUI pass-through (OWUI-run tools). Tool sources and integrations:

- Open WebUI tool registry tools (server-side Python tools).
- Open WebUI **Direct Tool Servers** (client-side OpenAPI tools executed in the browser via Socket.IO).
- OpenRouter web-search (attached as an `openrouter:web_search` server tool in the `tools` array — not a `plugins` entry or a function tool).

---

## Tool backends (`TOOL_EXECUTION_MODE`)

This pipe supports two tool execution backends. Choose based on whether you want the pipe to run tools itself, or you want Open WebUI to run them.

### `Pipeline` (default)

The pipe runs the tool loop itself:

- Provider returns `function_call` items.
- The pipe executes those tools (Open WebUI registry tools + Direct Tool Servers where available).
- The pipe appends `function_call_output` items and re-calls the provider until the model stops requesting tools or `MAX_FUNCTION_CALL_LOOPS` is reached (at which point the model gets a synthesis turn).

**You gain:**
- Pipe-level concurrency controls, batching, timeouts, and breaker protections around tool execution.
- Optional persistence/replay of tool results via the pipe artifact store (`PERSIST_TOOL_RESULTS`, `TOOL_OUTPUT_RETENTION_TURNS`), which can reduce repeated tool calls and help with long chats.
- Optional strictification of tool schemas (`ENABLE_STRICT_TOOL_CALLING`) for more predictable function calling.

**You lose / trade off:**
- Tool execution behavior is “owned” by the pipe rather than Open WebUI’s native tool runner (so Open WebUI UX/logs may not exactly match the built-in tool flow).

### `Open-WebUI` (tool bypass / pass-through)

The pipe does **not** execute tools. Instead, it returns tool calls in an OpenAI-compatible `tool_calls` shape and expects Open WebUI to:

- execute tools locally (registry tools and/or Direct Tool Servers), and then
- replay tool outputs back through the pipe as `role:"tool"` messages on the next request.

**You gain:**
- Open WebUI-native tool execution behavior and UI (tool boxes, retries, and tool server flows are handled by OWUI).
- A simpler “adapter-only” path: the pipe focuses on transport translation between Open WebUI and OpenRouter.
- Better compatibility with OpenRouter streaming quirks: OpenRouter `/responses` can emit tool calls with `arguments:""` early; in this mode the pipe will **never** emit `arguments:""` to Open WebUI (it waits for complete args or normalizes to `{}`).

**You lose / trade off:**
- The pipe does not run tool batching, tool timeouts or tool breakers; Open WebUI’s behavior governs execution. The per-user request breaker still applies.
- Tool result persistence in the pipe artifact store is disabled (even if `PERSIST_TOOL_RESULTS=True`). Tool outputs still exist in chat history, but large tool outputs may increase context size/cost versus persistence-based replay.
- In pass-through, the pipe does not strictify or mutate tool schemas; Open WebUI’s schemas are forwarded as-is.

---

## Tool schema assembly (`build_tools`)

Tool *schemas* are assembled by `build_tools(...)` and attached to the outgoing Responses request as `tools`.

### Preconditions

- Tools are only attached when the selected model is recognized as supporting `function_calling`.
- In `TOOL_EXECUTION_MODE="Open-WebUI"`, the pipe does not block tools based on its model capability registry (it forwards tools as Open WebUI provided them).

### Tool sources (in order)

1. **Open WebUI tool registry** (`__tools__` dict)
   - Converted to OpenAI tool specs (`{"type":"function","name",...}`) via `ResponsesBody.transform_owui_tools(...)`.
   - When `TOOL_EXECUTION_MODE="Pipeline"` and `ENABLE_STRICT_TOOL_CALLING=true`, each tool schema is strictified:
     - Object nodes get `additionalProperties: false`.
     - All declared properties are marked required; properties that were not explicitly required become nullable (their type gains `"null"`).
     - `$ref` and `$defs`/`definitions` are preserved: referenced definitions are strictified in place, `$ref` nodes pass through untouched, and single-`$ref` `allOf` wrappers are unwrapped. Multi-branch `allOf` is merged (local `$ref` branches are resolved from the schema's own definitions); unresolvable references leave the `allOf` untouched.
     - Keywords OpenAI strict mode rejects are stripped (`default`, `$schema`, `pattern`, length/numeric/array constraint keywords, `format`).
     - Missing property `type` values are inferred defensively (object/array) so schemas remain valid.
     - If a schema cannot be serialized for strictification, it is sent unmodified (with a warning logged).
     - A small LRU cache (size 128) avoids repeated strictification work for identical schemas.
     - The strictified copy is only what is advertised to the model; the executor keeps the tool's original schema, so argument validation (such as the empty-arguments guard) uses the tool's own `required` list.

2. **Open WebUI Direct Tool Servers** (`__metadata__["tool_servers"]`)
   - These are user-configured OpenAPI tool servers that Open WebUI executes client-side.
   - Open WebUI includes the selected servers in the request body as `tool_servers`; for pipes this arrives under `__metadata__["tool_servers"]`.
   - This pipe:
     - advertises the tools to the model using OpenAPI `operationId` values as tool names; when names collide across sources they are disambiguated with a source prefix (e.g. `direct__`), not overwritten, and
     - executes tool calls via the Socket.IO bridge (`__event_call__`) by emitting `execute:tool` so the browser performs the request.
   - Direct tools are only advertised when `__event_call__` is available; without an active Socket.IO session there is no safe execution path, so the pipe skips them.

3. **Extra tools** (`extra_tools`)
   - A caller-provided list of already OpenAI-format tool specs is appended as-is (non-dict entries are ignored).

### Deduplication

After assembly, tools are deduplicated by `(type, name)` identity. If duplicates exist, the **later** entry wins.

---

## Tool execution lifecycle (Responses API loop)

Tool execution happens in the request loop that follows each Responses API call:

1. The pipe calls the provider (streaming mode for normal chats).
2. When a `response.completed` event arrives, the pipe inspects the response `output` list.
3. Any `output` items with `type == "function_call"` are treated as tool calls to execute locally.
4. The pipe executes the tools and converts each result into `function_call_output` items.
5. The `function_call` items (normalized) and their outputs are appended to the next request’s `input[]`, and the loop continues until either:
   - no more `function_call` items are returned, or
   - `MAX_FUNCTION_CALL_LOOPS` is reached — pending tool calls receive stub responses and the model gets one additional turn to synthesize a final answer.

Notes:

- If a tool name is missing or not present in the tool registry, the pipe returns a structured `function_call_output` indicating the failure.
- The pipe does not “stream” tool outputs mid-request. Tools are executed between Responses calls.
- `MAX_FUNCTION_CALL_LOOPS` only applies when `TOOL_EXECUTION_MODE=”Pipeline”`. In Open-WebUI mode, loop control is managed by Open WebUI.

---

## Adaptive tool output budgeting (Pipeline mode)

This section documents the dynamic context-budget guard used when `TOOL_EXECUTION_MODE="Pipeline"`.

### Problem users observe

In long tool loops, the request can become context-saturated (large replayed artifacts + new tool outputs + reasoning state). A common symptom is:

- tool loops continue, but the model eventually returns no useful assistant text (or an incomplete response) because the prompt budget is exhausted.

### Conceptual fix

The pipe now applies **adaptive, model-aware budgeting** instead of fixed output caps:

- It derives prompt limits from model metadata: `max_prompt_tokens` when the catalog publishes it, otherwise the model's `context_length` less whatever reply allowance the request itself carries, with safe fallbacks. The provider's largest possible completion is not reserved — that is a ceiling the request never asked for, and on most of the catalog it put the budget far below the real window.
- It estimates request/input size and omits oversized `function_call_output` payloads by replacing them with a short model-visible stub that advises the model to retry with a narrower query.
- The model retains full tool access throughout the conversation and can recover from oversized results by retrying with tighter parameters.

This keeps the loop alive, informs the model in-band, and lets the model decide whether to summarize, stop tools, or ask for narrower tool queries.

### User-visible behavior changes

- Some tool outputs may be replaced by an omission stub in the request sent to the model, when the full text would exceed the remaining context budget for that turn.
- The stub exists only in that request. The tool card, the stored chat message and the artifact store all keep the result's full text, and a warning notification names the tools the model did not receive. Because the budget is recomputed every turn, an omitted result is replayed in full once the context has room or the conversation moves to a larger model.
- If tool loops complete without any assistant content growth and no actionable continuation remains, the pipe emits a fallback assistant message instead of staying silent.

### Operator guidance

To reduce omissions and improve reliability:

- Prefer tools that support tight server-side limits (`limit`, `top_k`, date ranges, filters).
- Have tools return concise summaries plus references/IDs instead of full raw blobs.
- For bulky outputs (search results, logs, traces), expose pagination/continuation parameters so the model can request smaller chunks.
- `PERSIST_TOOL_RESULTS` is off by default to keep long conversations lean; enable it (site-wide or per user) when chats need to reuse exact raw tool outputs on later turns instead of re-fetching.

---

## Tool execution cards (`SHOW_TOOL_CARDS`)

When `SHOW_TOOL_CARDS` is enabled, the pipe displays collapsible cards in the chat UI showing tool execution status:

- **In-progress cards**: Appear when a tool starts executing, showing the tool name and arguments.
- **Completed cards**: Replace in-progress cards when execution finishes, showing tool name, arguments, and results.
- **Failed/omitted outputs**: Not rendered as tool cards (they are model-visible only for in-loop recovery).

By default, `SHOW_TOOL_CARDS` is **disabled** for a cleaner chat experience. Tools execute silently without visual indicators.

Enable this valve when you want:
- Debugging visibility into tool execution
- Users to see what tools are running and their outputs
- Transparency about tool arguments and results

This setting is available as both an admin valve and a user valve (users can override the admin default).

**Note:** This feature only applies when `TOOL_EXECUTION_MODE="Pipeline"`. In `Open-WebUI` mode, the pipe doesn't execute tools itself, so it cannot display execution cards.

---

## Concurrency, batching, and timeouts (per request)

Tools are executed via a per-request worker pool backed by a bounded queue:

- Queue size: 50 batches per request (bounded).
- Worker count: `MAX_PARALLEL_TOOLS_PER_REQUEST`.
- Per-request semaphore: limits concurrent tool executions per request.
- Global semaphore: `MAX_PARALLEL_TOOLS_GLOBAL` limits tool executions across all requests.
- Open WebUI's built-in `ask_user` takes no slot from either semaphore, because it waits on a person rather than doing work.

Batching behavior:

- The pipe groups a response's tool calls into batches before any of them runs. Consecutive calls join one batch, up to `TOOL_BATCH_CAP` calls, when they share a tool name, when neither the joining call nor the batch's first call carries a dependency or ordering blocker in its arguments, and when neither of those two names the other's call ID. A call refused before queueing (an unknown tool, invalid arguments or a tripped breaker) does not break a run of consecutive calls.
- A call whose arguments include any of `depends_on`, `_depends_on`, `sequential` or `no_batch` is never batched. These keys only keep the call out of a batch; they do not make it wait for other calls.
- Batching does not require identical arguments and never deduplicates calls. It does not raise concurrency either: every call in a batch except `ask_user` still waits for a per-request slot and a global slot, and all calls in a batch share one batch deadline.
- Each batch is queued separately, so while slots are free, a slow call never holds up a call to another tool, and a response's calls start together as long as there are free slots for all of them. Each call's result is handed back as soon as that call finishes, even while other calls in its batch are still running. The same holds inside internal Fusion, where each model gets as many tool workers as the chat request it answers, and all of those models share that request's slots.

Timeouts:

- Each tool call has a per-call timeout (`TOOL_TIMEOUT_SECONDS`), measured from when the call starts running, not from when it starts waiting for a slot. When it expires the pipe stops waiting, the model reads `Tool '<name>' timed out after <N>s.`, and the timeout counts toward that tool's breaker. An async tool is cancelled; a tool written as a plain function is not, and runs to its end.
- Each tool call runs exactly once. A tool that raises an error is never retried automatically, because tools can have side effects (an MCP tool that sends an email must not fire twice). The failure is reported to the model, which can decide whether to call again.
- If Open WebUI has already closed an MCP tool's client connection, the tool reports "no longer available in this session" rather than a raw error, and this does not count against its breaker.
- Calls grouped into one batch share a batch deadline (`TOOL_BATCH_TIMEOUT_SECONDS`, never shorter than the per-call timeout). Its clock starts with the batch, so time spent waiting for a slot counts. When the deadline passes, finished calls keep their results; every call still running or still waiting is cancelled and reported as `Tool batch '<name>' exceeded <N>s and was cancelled.` Of the cancelled calls, only the running ones count toward the tool's breaker, except any that `TOOL_IDLE_TIMEOUT_SECONDS` had already given up on.
- `TOOL_IDLE_TIMEOUT_SECONDS` (unset by default) caps how long the pipe waits in total for one response's tool results, counted once from when the model asked. Every call whose result has not arrived by then is reported as timed out, however long that call itself has been running. When that time passes, the model reads `Tool '<name>' timed out after <N>s (idle timeout).` Giving up this way does not count toward the tool's breaker. A call that is already running is not stopped: it keeps running and holds its slot until it finishes, another limit ends it, or request cleanup cancels it after `TOOL_SHUTDOWN_TIMEOUT_SECONDS`. The model never receives the late result, though files or embeds the call returns can still appear in the chat. The call's own later error or per-call timeout still counts toward the tool's breaker, and a later success clears the count. A call still waiting for a slot or a worker never starts. The tool workers remain for the whole request, except inside internal Fusion, where a model's workers and its running calls are cancelled as soon as that model's answer ends, without the `TOOL_SHUTDOWN_TIMEOUT_SECONDS` wait.
- Open WebUI's built-in `ask_user` keeps its question open for the time the model asked for, as normalised by Open WebUI. Its per-call limit becomes that time plus 15 seconds, and the batch deadline and the idle limit are raised to at least that long. Since it takes no tool slot, its question does not wait for other requests' tools, though it still needs one of its own request's workers to be free. Its timeout does not count toward the breaker. It runs alone: an `ask_user` call mixed with other calls, or repeated in the same response, is refused with Open WebUI's error text, and the other calls run.

---

## Breakers (stability controls)

Three per-user breakers share `BREAKER_MAX_FAILURES` and `BREAKER_WINDOW_SECONDS`; each internal Fusion run also keeps one count per tool, shared by all of its models, which uses `BREAKER_MAX_FAILURES` but not the window. The per-user breakers count failures in two ways:

- **Per-user request breaker:** counts each failed chat call to OpenRouter within the trailing `BREAKER_WINDOW_SECONDS`, whether the failure is an error reply; a connection that cannot be opened, drops or times out; an error OpenRouter reports after accepting the call (an error event in the stream, or an error body); or a stream that stops before its final event. A request the pipe retries automatically can therefore count more than once. A stream that ends as `response.incomplete` (for example, when the answer reaches its length limit) is a finished call, not a failure. A generation on a picture-only image model or a video model counts once, when it fails after being sent to OpenRouter. A request that ends without an error clears the count, but a request to a picture-only image model or a video model clears it only once its result is delivered, and a request the user stops does not clear it. Housekeeping tasks such as title generation neither count nor clear, and a request refused before anything is sent (such as a missing API key, or a model the pipe will not serve) neither counts nor clears; Open WebUI's merge-responses task counts but never clears. At `BREAKER_MAX_FAILURES`, that user's new requests are refused with "Temporarily disabled due to repeated errors. Please retry later." until the oldest failures age out of the window, or until a request that is still let through ends without an error and clears the count: one already under way when the limit was reached, or an exempt one. A request is exempt, and never refused by this breaker, when its last message is a tool result or a user message directly after a tool result. That is how Open WebUI calls back after running its tools to finish an answer already under way, including a callback whose message carries a tool's images. Inside internal Fusion, each panel, judge or final-answer call that fails at OpenRouter counts; when the run ends, the count, including that run's own failures, is cleared if the run finishes and any panel model answered, and kept if none did or the user stopped the run.
- **Per-user, per-tool breaker:** counts a tool's failures in a row, keyed by tool type and name, so other tools of the same type, including the rest of an MCP server's tools, keep working. A successful call clears the count, and so does a gap longer than `BREAKER_WINDOW_SECONDS` between the tool's last failure and its next call. The gap is measured to the next call, not between failures, so a slow tool that keeps timing out still trips. Errors, per-call timeouts and running calls cancelled by the batch deadline (other than calls the idle limit had already given up on) count; an `ask_user` timeout, a call to an MCP tool whose session already closed, and a call still waiting for a slot do not. Each internal Fusion run keeps one count per tool, shared by all of its models: a tool that fails `BREAKER_MAX_FAILURES` times in a row within the run is skipped from then on. Unlike the user's own count, this one is never cleared by a quiet spell, only by a success; once the tool is skipped, only an already-running call to it can supply that success. Calls to that tool inside the run neither raise nor clear the user's own count for it, and are not skipped because of that count.
- **Per-user DB breaker:** counts failed database reads and writes of stored reasoning, tool results and session logs within the trailing window. A successful read or write clears the count; where Redis buffers writes, a write succeeds once Redis has taken it. At the limit, the pipe skips that user's database reads and writes and shows the warning "DB ops skipped due to repeated errors." Chats still get answers, but that user's reasoning and tool results are neither saved to nor read from the database until failures age out of the window.

While a tool breaker is open, calls to that tool are skipped and the model is told why; outside internal Fusion, a best-effort status message is also sent to the UI. A turn whose tool calls are all skipped this way does not count as a failed request.

---

## OpenRouter web search server tool

The web-search integration is attached as a server tool (not as a `tools` function or legacy `plugins` entry):

- When the **OpenRouter Web Tools** toggle (the filter's `WEB_SEARCH` user valve) is enabled for the request (per chat, or enabled by default via the model’s Default Filters), the pipe appends `{"type": "openrouter:web_search", ...}` to `tools`. Image-output and video-generation models are excluded because they do not receive the Web Tools filter.
- Search parameters (max results, engine, allowed/excluded domains, etc.) are controlled by the OpenRouter Web Tools filter’s admin valves.

Important: Open WebUI also has a separate built-in **Web Search** toggle (Open WebUI-native). OpenRouter Web Tools and Open WebUI Web Search are different systems.
See: [Web Search: OWUI vs OpenRouter](web_search_owui_vs_openrouter_search.md).
See: [OpenRouter Server Tools](openrouter_server_tools.md) for the full server tools reference.

---

## OpenRouter response-healing plugin (intentionally not exposed)

OpenRouter offers a response-healing plugin that can attempt to repair malformed outputs. This pipe does **not** expose that plugin on purpose:

- We prefer failing fast when a model returns malformed JSON or invalid structured output.
- Silent repairs can hide real model issues (bad prompts, low token budgets, provider quirks) and make debugging harder.

If you want auto-healing, integrate it explicitly in your own request layer so it is visible and auditable.

---

## Open WebUI Direct Tool Servers

Direct Tool Servers are configured and executed by Open WebUI, but advertised/executed through this pipe:

- Configure servers in **User Settings → External Tools → Manage Tool Servers** (and ensure the server is enabled/toggled).
- Select tool servers for a chat in the tool picker (Open WebUI sends the selected servers in `tool_servers`).
- When the model calls a direct tool, the pipe emits `execute:tool` via `__event_call__` and the browser performs the OpenAPI request.

Failure handling:
- Direct tool execution is wrapped in `try/except`; tool crashes never crash the pipe/session.
- On failure the tool returns an error payload to the model (and the pipe may emit an OWUI notification best-effort).

---

## MCP note (removed)

This pipe no longer implements “remote MCP server connectivity” (previously surfaced as `REMOTE_MCP_SERVERS_JSON`) because it bypasses Open WebUI’s tool server configuration surface and RBAC/permissions model.

If you want MCP tools in Open WebUI, use an MCP→OpenAPI proxy/aggregator (for example **MCPO** or **MetaMCP**) and add the resulting OpenAPI server through Open WebUI’s tool server UI so access control and future tool server changes remain centralized in OWUI.

For persistence behavior and replay rules of tool artifacts, see:

- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)
- [History Reconstruction & Context Replay](history_reconstruction_and_context.md)
