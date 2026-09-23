# Examples

Runnable examples live here, one file per integration pattern. Each example is
self-contained: it writes a local SQLite file next to itself and prints a
replay of the captured session. The framework examples require the matching
extra (`sentinel-sdk[langchain]` / `sentinel-sdk[langgraph]`); the raw-Ollama
example needs a running Ollama endpoint.

| File | Shows | Sprint |
|---|---|---|
| `hello_ollama.py` | Capturing one raw Ollama chat call and replaying it | `S0` |
| `langchain_agent.py` | Instrumenting a LangChain chain + tool via callbacks | `S1` |
| `langgraph_agent.py` | Instrumenting a LangGraph node graph as `agent.step` events | `S1` |
| `raw_ollama.py` | Raw HTTP capture with the generic `trace` wrapper | `S1` |

Run from the repo root, e.g. `uv run python examples/langchain_agent.py`.
See `docs/integration-guide.md` for the full instrumentation guide.
