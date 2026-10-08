# Sentinel Event Schema

Every boundary crossing an instrumented agent performs is recorded as an
**event** — an immutable, sequence-numbered record tied to a capture session.
This document is the contract evaluators, the gate, and the UIX read from
(`S1-T17`). It describes each event type, its payload fields, its `refs`
semantics, and the `schema_version` evolution rules.

## 1. Envelope

Every event is a `sentinel.models.events.Event` (pydantic, `extra="forbid"`):

| Field | Type | Meaning |
|---|---|---|
| `event_id` | `str` (ULID, ADR-0010) | Globally unique, time-ordered id. |
| `session_id` | `str` (ULID) | The owning capture session. |
| `seq` | `int ≥ 0` | Monotonic per session; appended after every existing event. |
| `ts` | `datetime` (UTC, timezone-aware) | When the boundary was crossed. |
| `type` | `str` | One of the taxonomy in §2. |
| `payload` | `dict[str, object]` | Event-specific fields (see §3), JSON-safe. |
| `refs` | `list[RefLink]` | Typed links to *earlier* events (INV-3). |
| `schema_version` | `str`, default `"0.1"` | Payload/semantics version (ADR-0007). |

### 1.1 `refs` kinds

| Kind | Direction | Meaning |
|---|---|---|
| `parent` | child → enclosing | The event ran inside another event (chain step, node, LLM turn). |
| `caused_by` | effect → cause | The later event is the direct result of the earlier one (response ← request, result ← call). |
| `grounds` | claim → evidence | A later claim cites an earlier `tool.result` as its source (reserved for `S3`). |

Referential integrity (INV-3) is enforced at append time: a `ref.event_id`
must name an already-persisted event in the same session, and a
`capture.dropped` marker never carries a ref to the event it replaced.

## 2. Event taxonomy

| Type | Emitted by | Content |
|---|---|---|
| `session.start` / `session.end` | `sentinel.instrument.session` | Session lifecycle bookends. |
| `llm.request` / `llm.response` | langchain, langgraph (inherited), ollama, openai_compat | One LLM (or chat-completion) call. |
| `tool.call` / `tool.result` | langchain, langgraph (inherited) | One tool invocation and its outcome. |
| `agent.step` | langchain (chains), langgraph (nodes), generic (`trace`) | A coarse agent step: which node/function ran, with its state delta. |
| `memory.read` / `memory.write` | `MemoryInstrumentor` | Reads/writes against a `MemoryStore`. |
| `error` | langchain, langgraph, generic | A failed boundary crossing (LLM, tool, node, function, chain). |
| `capture.dropped` | capture writer | The capture pipeline dropped an event (queue overflow or store failure). |

## 3. Payload contracts

Captured text is capped at `MAX_CAPTURED_TEXT` (65 536 chars) with a trailing
`…[truncated]` marker; nested dicts/lists are capped recursively. Non-JSON
objects are stringified so payloads are always serializable.

### 3.1 `session.start` / `session.end`

    {}

### 3.2 `llm.request`

| Field | Producers | Meaning |
|---|---|---|
| `provider` | langchain/graph, ollama, openai_compat | `"langchain"`, `"ollama"`, `"openai_compat"`. |
| `model` | langchain/graph, ollama | Model id or runnable name. |
| `prompts` | langchain/graph | The prompt list passed to the model. |
| `url` | ollama, openai_compat | Endpoint the call went to. |
| `request` | ollama, openai_compat | The full request body (`{model, messages, ...}` for Ollama; the method/url/request in `chat_completion*`). |
| `headers` | openai_compat (`capture_headers=True`) | Request headers. |
| `method` | openai_compat | HTTP method. |

`refs`: `parent` → enclosing `agent.step`, when the call runs inside a chain
or node.

### 3.3 `llm.response`

| Field | Meaning |
|---|---|
| `provider`, `model`, `url`, `method` | Same as the request. |
| `status_code` | HTTP status (transport producers). |
| `latency_ms` | Round trip in milliseconds. |
| `generations` | langchain/graph: the generated texts. |
| `response` | ollama / openai_compat (non-stream): the parsed JSON body. |
| `stream` | `true` for `chat_completion_stream`. |
| `chunk_count` | Number of chunks forwarded (streaming). |
| `transcript` / `truncated` | Reconstructed stream text; `true` when it exceeded the cap. |
| `reasoning` | The model's reasoning trace, when the provider returns one. langchain: a list of traces from the message; openai_compat: a list of traces in `choices` order; ollama: a single string from `message.thinking`. Absent when the provider returned none — never an empty value, so "no reasoning model" stays distinguishable from "a trace we failed to read". |

`reasoning` is lifted out of the raw body rather than left inside `response`
because the provider vocabularies disagree and nest differently: OpenAI's
`reasoning` and DeepSeek's `reasoning_content` sit at `choices[i].message.*`,
Ollama's `thinking` at `message.*`, and LangChain's at
`additional_kwargs.reasoning_content` on the message object. Capturing the body
without naming the field would leave the trace present but unreachable to any
generic reader. `sentinel.eval.session.reasoning_text_of` reads this key first
and still walks the nested shapes, so it also works on logs written by an
instrumentor that did not lift it.

Streamed responses are the exception: `chat_completion_stream` captures a
transcript, and a trace streamed as deltas stays inside that transcript rather
than being lifted. Streaming agents therefore get answer-level checking only.
Deriving structure from deltas is a capture change to the streaming instrumentors.

`refs`: `caused_by` → the matching `llm.request`; `parent` → the enclosing
`agent.step`.

### 3.4 `tool.call` / `tool.result`

| Field | Meaning |
|---|---|
| `provider` | `"langchain"` (langgraph nodes inherit this). |
| `tool` | Tool name. |
| `input` | The tool input string / argument snapshot (capped). |
| `latency_ms` | `tool.result` only. |
| `output` | `tool.result` only: the returned value. |

`refs`: `tool.result` → `caused_by` → its `tool.call`; both may also carry a
`parent` → the enclosing `agent.step`. This is the link the call-graph helper
(`S1-T6`) and the provenance evaluator (`S3-T8`) traverse.

### 3.5 `agent.step`

| Field | Meaning |
|---|---|
| `provider` | `"langchain"`, `"langgraph"`, or `"generic"`. |
| `step` | `"chain"` (langchain run) or `"step"` (langgraph node / `trace` function); `trace(kind=...)` uses the given kind. |
| `name` / `function` | The runnable name (chain/node) or the traced function's `__qualname__`. |
| `node` | langgraph: the node name; when absent the event is the enclosing graph run (`step == "chain"`). |
| `status` | `started` or `ended`. |
| `step_index` | langgraph: the node's `langgraph_step` (0-based) — may be `null`. |
| `input` / `output` | langgraph: bounded state delta at entry/exit. |
| `inputs` / `outputs` | langchain chain: the runnable's full inputs/outputs (capped). `input`/`output` (generic trace) instead. |
| `latency_ms` | `ended` only. |

`refs`: the `ended` event links `parent` → its own `started` event; `started`
events link `parent` → the enclosing step.

### 3.6 `memory.read` / `memory.write`

| Field | Meaning |
|---|---|
| `provider` | `"memory"`. |
| `memory` | The store's `name` (`"in_memory"`, `"postgres"`, …). |
| `key` | Entry key. |
| `value` / `summary` | `memory.write`: the stored value/summary (capped). |
| `query` / `limit` | `memory.read`: the lookup. |
| `hits` / `result` | `memory.read`: match count and the returned `{key, value, summary}` views. |

`refs`: `memory.read` → `caused_by` → every `memory.write` whose returned
entry the read loaded. The write event's `event_id` is passed through to the
store as the entry's `event_id`, so the link survives into the store layer.

### 3.7 `error`

| Field | Meaning |
|---|---|
| `provider` | `"langchain"`, `"langgraph"`, `"generic"`. |
| `scope` | `chain`, `node`, `llm`, `tool`, or `function`. |
| `name` / `node` / `tool` / `model` / `function` | The failing component, per scope. |
| `message` | The exception message (capped). |
| `step_index` | langgraph errors only. |

`refs`: `caused_by` → the component's `started`/request event; `parent` → the
enclosing `agent.step`.

### 3.8 `capture.dropped`

| Field | Meaning |
|---|---|
| `reason` | Why the pipeline dropped the event (queue overflow / store failure). |
| `count` | Events dropped (pipeline markers batch; each carries `1`). |
| `dropped_event_id` | The id of the event that was never persisted. |

`refs`: **none** — the dropped event does not exist in the store (INV-3).

## 4. Producers

| Instrumentor | Events | Notes |
|---|---|---|
| `sentinel.instrument.langchain` | `agent.step` (chains), `llm.*`, `tool.*`, `error` | Via `BaseCallbackHandler`; `raise_error=False` so capture never aborts the run. |
| `sentinel.instrument.langgraph` | `agent.step` (nodes), `llm.*`, `tool.*`, `error` | Node metadata (`langgraph_step`, `langgraph_node`) distinguishes a node step from the enclosing graph run. |
| `sentinel.instrument.generic` (`trace`) | `agent.step`, `error` | Async functions only; sync callables pass through untraced. |
| `sentinel.instrument.memory` (`MemoryInstrumentor`) | `memory.read` / `memory.write` | Wraps any `MemoryStore`. |
| `sentinel.instrument.ollama` | `llm.request` / `llm.response` | `instrument_ollama_call` (triggers `OllamaChatError` on non-2xx). |
| `sentinel.instrument.openai_compat` | `llm.request` / `llm.response` | `chat_completion` / `chat_completion_stream` (triggers `TransportCallError`). |
| `sentinel.instrument.session` | `session.start` / `session.end` | Every session's bookends. |

Every instrumentor honors the registry's `enable()`/`disable()` toggle and
fails open (INV-6): an exception during capture is logged, counted, and
swallowed; the host call proceeds. A capture failure that reaches the writer
instead emits `capture.dropped`.

## 5. `schema_version` evolution rules (ADR-0007)

- The current version is `"0.1"`.
- **Adding** an optional field is non-breaking: bump the *minor* (`0.1 → 0.2`)
  so readers can branch on presence.
- **Removing or renaming** a field, or changing its meaning, is breaking:
  bump the *major* (`1.0`), ship a migration, and keep the old version readable
  until at least one minor release after the transition.
- A payload read under an old `schema_version` is validated on read (invariant
  per ADR-0007) — never migrated in place; the store is append-only.

## 6. Querying

Events are replayed per session with `store.get_session(session_id)` in `seq`
order, and `sentinel.get_call_graph(store, session_id)` resolves the `refs`
into a navigable graph (`S1-T6`) — `tool_results_for(call_id)` and
`llm_calls_for(call_id)` are the read paths the `S3` evaluator builds on.
