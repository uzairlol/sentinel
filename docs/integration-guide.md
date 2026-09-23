# Integration Guide

Sentinel instruments LLM agents at the boundary and writes typed events to an
event store. This guide covers every supported integration path — LangChain,
LangGraph, raw HTTP (Ollama / OpenAI-compatible), the generic `trace`
decorator, and memory stores — plus the query helpers you use to read the
resulting call graph (`S1-T19`).

## 0. Install

    uv add sentinel-sdk
    uv add "sentinel-sdk[langchain]"   # LangChain / LangGraph instrumentors
    uv add "sentinel-sdk[ollama]"      # httpx-based transport helpers
    uv add "sentinel-sdk[postgres]"    # Postgres memory adapter (event store: use SQLite or bring your own)

The core package never imports a framework: instrumentors that need an
optional dependency raise a `...UnavailableError` when it is missing.

## 1. The shape of every integration

```python
from sentinel import SQLiteEventStore, session

store = SQLiteEventStore("agent.db")
async with session(store) as ctx:
    # ... attach instrumentor(s) to ctx and run the agent ...
# every boundary crossing is now in the store
events = await store.get_session(ctx.session_id)
graph = await get_call_graph(store, ctx.session_id)   # S1-T6
```

`session()` opens a `session.start` event and closes `session.end`. `ctx` is
the only handle instrumentors need; it is safe to create several instrumentors
per session (one per framework).

## 2. LangChain

```python
from sentinel import SQLiteEventStore, session
from sentinel.instrument.langchain import LangChainInstrumentor

async with session(SQLiteEventStore("agent.db")) as ctx:
    instrumentor = LangChainInstrumentor(ctx)
    await my_chain.ainvoke(
        {"question": "..."},
        config={"callbacks": [instrumentor.handler()]},
    )
```

Captures, per runnable call:

- **LLM calls** → `llm.request` / `llm.response` (model, prompts, latency,
  generated text).
- **Tools** → `tool.call` / `tool.result`, with the result citing the call
  (`caused_by`).
- **Chains** → `agent.step` bookends so the graph nests correctly.

## 3. LangGraph

Attach the same callback style; `LangGraphInstrumentor` adds LangGraph-aware
capture on top of the LangChain handler:

```python
from sentinel.instrument.langgraph import LangGraphInstrumentor

async with session(SQLiteEventStore("agent.db")) as ctx:
    graph = create_react_agent(model, tools)
    instrumentor = LangGraphInstrumentor(ctx)
    async for _ in graph.astream(
        {"messages": [("user", "hello")]},
        config={"callbacks": [instrumentor.handler()]},
    ):
        pass
```

Each node entry/exit becomes an `agent.step` (`step == "step"`) carrying the
node name, `step_index`, and bounded state delta; the graph's own run is the
enclosing `agent.step` (`step == "chain"`). Node functions calling the model or
a tool inherit the node as their `parent`.

## 4. Raw HTTP — Ollama

```python
import httpx
from sentinel import SQLiteEventStore, instrument_ollama_call, session

async with session(SQLiteEventStore("agent.db")) as ctx, httpx.AsyncClient() as client:
    reply = await instrument_ollama_call(
        client,
        ctx,
        model="llama3.2",
        messages=[{"role": "user", "content": "Say hi."}],
    )
```

Non-2xx responses are captured (status, latency) and then raised as
`sentinel.OllamaChatError`. See `examples/raw_ollama.py`.

## 5. Raw HTTP — OpenAI-compatible

```python
from sentinel.instrument.openai_compat import chat_completion, chat_completion_stream

body = await chat_completion(client, ctx, url="https://.../v1/chat/completions", request={...})
async for chunk in chat_completion_stream(client, ctx, url=..., request={...}):
    ...  # chunks forwarded before any capture work
```

Streams reconstruct a bounded transcript (≤ `MAX_CAPTURED_TEXT`) without
buffering before forwarding; `stream`, `chunk_count`, `transcript`, and
`truncated` ride in the `llm.response` payload.

## 6. Generic `trace` decorator (escape hatch)

For functions that are not LangChain runnables:

```python
from sentinel.instrument import trace


@trace(kind="tool")  # default kind: "function"
async def my_tool(x: int) -> int:
    return x * 2


result = await my_tool(2)  # captures started/ended (and error) events
```

Async functions are instrumented; sync callables pass through untouched. Uses
the ambient `current_session()`, so no callbacks config is needed. Nested
`trace`d calls link parent→child automatically.

## 7. Memory stores

`MemoryInstrumentor` wraps any object matching the `MemoryStore` protocol
(`write(key, value, summary, event_id) -> MemoryEntry`, `read(query, limit)`):

```python
from sentinel.instrument.memory import MemoryInstrumentor
from sentinel.memory import InMemoryMemoryStore, PostgresMemoryStore

memory = MemoryInstrumentor(ctx).instrument(InMemoryMemoryStore())
await memory.write(key="alice", value="likes blue")
hits = await memory.read(query="alice")
```

Writes emit `memory.write`; reads emit `memory.read` whose refs link back to
the writes that produced the hits (`caused_by`) — the load-bearing link for
the `S4` memory-integrity evaluator. Use `for store in registry: ...` to treat
a memory store like any other instrumentor. A Postgres-backed adapter ships as
`sentinel-sdk[postgres]`.

## 8. Registry, toggling, and fail-open

All instrumentors derive from `BaseInstrumentor`; use the registry for
uniform lifecycle control:

```python
from sentinel.instrument.registry import InstrumentorRegistry

registry = InstrumentorRegistry().register(instrumentor)
registry.enable("langchain")
registry.disable("langgraph")  # instrumentor stops emitting, agent keeps running
```

Enabling twice is idempotent. Instrumentors that implement their own
capture loop (langchain) honor the toggle live. Every instrumentor logs,
counts, and swallows capture errors (INV-6); the capped-size and redaction
policies in `sentinel.configure()` are applied before persistence.

## 9. Reading the trace

- `store.get_session(session_id)` — full replay in `seq` order.
- `get_call_graph(store, session_id)` — typed, navigable graph of events and
  their `parent`/`caused_by` links (S1-T6) with `tool_results_for(call_id)`
  and `llm_calls_for(call_id)`.
- `sentinel configure` / `get_config()` — inspect run-time capture settings.

See `docs/event-schema.md` for the full payload contract each event carries —
treat it as the source of truth for anything you build on top.
