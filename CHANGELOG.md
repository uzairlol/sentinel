# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-09-23

Sprint `S1` — Instrumentation Layer Core. Captured under `main` at `ebeab9a` and
`7141be1`.

### Added

- Event taxonomy & linking: `llm.request`/`llm.response`, `tool.call`/`tool.result`,
  `memory.read`/`memory.write`, `agent.step`, `error`, `capture.dropped`, with
  `parent`/`caused_by`/`grounds` refs enforced with referential integrity
  (INV-3).
- Instrumentor registry with `register()`/`enable()`/`disable()` and a
  `configure()` runtime settings surface (pydantic-settings).
- Framework instrumentors: LangChain (callbacks), LangGraph (node `agent.step`
  events), raw HTTP for Ollama and OpenAI-compatible endpoints (incl. streaming),
  a generic `@sentinel.instrument.trace(kind=...)` decorator, and a memory-store
  protocol with in-memory and Postgres-backed reference adapters.
- Capture pipeline: asynchronous bounded queue, `capture.dropped` on overflow,
  fail-open capture (INV-6), redaction before persistence.
- Call-graph query helper `get_call_graph(store, session_id)` with typed
  tool-result and dependent-LLM-call accessors.
- `docs/event-schema.md`, `docs/integration-guide.md`, and runnable examples
  (`langchain_agent.py`, `langgraph_agent.py`, `raw_ollama.py`).
- Postgres memory-adapter CI job against a service container.

## [Unreleased]

### Added

- Project scaffolding (Sprint `S-1`): package skeleton, tooling (ruff, mypy,
  bandit, pytest, pre-commit), GitHub Actions CI, governance documents,
  ADRs 0001–0010, and this changelog.
