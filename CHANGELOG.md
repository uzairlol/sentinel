# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2026-09-24

Sprint `S2` — Event Store Hardening & Query/Replay.

### Added

- SQLAlchemy 2.0 Alembic-managed Postgres store (`PostgresEventStore`) with
  versioned schema (`0001_initial`), unique `(session_id, seq)` + `event_id`
  constraints, FKs, and query indexes.
- Store protocol (ADR-0002) surface: batched append (idempotent by `event_id`,
  atomic on reference-integrity violation), streaming `iter_session`, call-graph
  and session-listing queries, `health`, gap detection, and keyset-paginated
  retention pruning with tombstone records.
- Least-privilege DB roles (`deploy/roles.sql`): `sentinel_migrator` /
  `sentinel_writer` / `sentinel_reader` (ADR-0011), enforced by
  `tests/integration/test_db_roles.py`.
- Reference compose stack (`deploy/compose.postgres.yml`): Postgres +
  Alembic migration runner + demo capture worker.
- `sentinel` CLI: `sessions list` and `store health` subcommands, `--dsn`
  cross-store selection, `--json`/`--pretty` output.
- Payload truncation with markers + SHA-256 digest (`S1-T16`) and proportional
  sampling with guaranteed critical/error capture (`S1-T15`), both
  `S1`-deferred items.
- CI: expanded Postgres integration job + 1M-event losslessness load job +
  Postgres CLI-replay e2e (`tests/e2e/test_cli_replay_postgres.py`).
- Perf: `perf/bench_writes.py` + `perf/write-benchmark.md` — event write
  p99 **1.45 ms** (batched) / **6.54 ms** (single commit) on local Postgres,
  against the 10 ms `S2` budget (SENTINEL_TDD.md §2.7).

### Changed

- `read_compat.materialize_refs` validates on read: 0.1 flat-string ref rows and
  0.2 typed `RefLink` dicts normalise to a single link shape; malformed entries
  raise instead of coercing.

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
