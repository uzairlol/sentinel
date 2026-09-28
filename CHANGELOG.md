# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Sprint `S3` — Tool-Use Grounding & Provenance. First evaluator module.

### Added

- Universal `Flag` row (ADR-0012): deterministic `flag_id` derived from
  `(session_id, module, module_version, category, dedupe_key)`, typed
  `evidence` refs (`claim` / `evidence` / `context` / `countervailance`),
  severity, confidence, and adjudication provenance. Persisted in SQLite and
  Postgres behind the `EventStore` protocol, with a `0002_flags_s3` migration and
  a `sentinel_reviewer` role whose only mutation is adjudication.
- `EvaluatorWorker` base: watermark/close triggers plus on-demand evaluation,
  idempotent writes, bounded retries with jittered backoff, resumable
  checkpoints, bounded `SessionView`, and call-graph context.
- Claim extraction (`sentinel.eval.provenance_core`): rule/template classifier
  behind a `ClaimExtractor` protocol, maintained deictic *and* attribution
  grounding lexicons, and normalisation of each claim's text, value and named
  source.
- Provenance diff: flags `ungrounded_claim` when no value in the turn supports a
  claim and `contradicted_claim` when the observed value refutes it, with the
  observed value carried in the flag and its evidence marked `countervailance`.
- Fabricated citations: a claim that names a source ("according to the audit
  report", "Acme") while the response cites no tool result is flagged
  `unsourced_citation` with the named source in the flag details. Guarded by a
  session-level precondition — the rule only runs in sessions that are proven to
  record citations at all — and ordered last, so it can reclassify a claim
  without ever creating a flag on silence alone.
- Cherry-picked counts: a claim of the form "2 of 3 passed" is flagged
  `ungrounded_claim` with `cherry_pick` support when the cited output shows a
  larger set, and carries `high` severity whatever its claim kind.
- Legal/safety severity weighting: `SAFETY_LEXICON` applies a severity *floor*
  for the claim's subject (22 safety terms at `high`, 26 legal/regulatory terms at
  `medium`) without overriding the severity the verdict already earned.
- Reasoning-trace reading: `reasoning_text_of` reads `reasoning`,
  `reasoning_content`, `thinking` and `thought` from a response payload or its
  nested message, so the extractor can check claims the model abandoned
  mid-thought. No instrumenter records those keys yet, so the seam currently
  returns `""`; closing that is a capture change in `S1`.
- Review routing (`S3-T15`): flags at or below the module's confidence
  threshold are written `review_only` and queue for `S7` instead of gating.
- 28-case adversarial corpus (12 known-good, 16 known-bad; replayable event
  sequences) covering fabricated citations, contradicted values and cherry-picked
  counts, plus known-good cases that specifically guard the new rules against
  false positives, and the `sentinel eval-fixtures --module provenance` harness,
  which prints a confusion matrix with case/claim FP and FN rates. **Measured:
  FP 0.00%, FN 0.00%** against budgets of ≤ 5% FP / ≤ 10% FN.
- `sentinel.eval.provenance` reusable core, store-free and guarded against
  depending back on `sentinel.instrument`. Built for `S4` to import; `S4` has
  not been written yet, so it has no consumer today.

### Changed

- `MODULE_VERSION` is now `0.2.0` (was `0.1.0`). The new rules change what the
  module finds, so the version bump changes the deterministic `flag_id` and
  forces sessions evaluated under `0.1.0` to be re-evaluated rather than
  silently keeping stale verdicts. `run_corpus` now defaults to the module's
  version instead of a duplicated literal.

- The offline coverage gate no longer counts the Postgres store and its ORM
  models, which it cannot execute without a database. They are no longer
  unmeasured: the `integration-postgres` job runs them against a real database
  and gates them at ≥ 90% via a new `.coveragerc.postgres` (measured 94% store,
  100% models, 97% memory adapter).
- `adjudicate_flag()` is now first-write-wins on both backends. It previously
  let a second reviewer overwrite `adjudicated_by`, destroying the audit record
  the append-only design depends on.
- `Flag` rejects a decision with no author, or an author with no decision, so a
  row can answer "who ruled, and when" on its own once its evidence is pruned.

### Fixed

- **CI was red on `main`, and had been since `S2`.** The first push of the `S2`
  tail and all of `S3` exposed four defects that no local run could catch,
  because every one of them lives in the workflow rather than the code:
  - `deploy/roles.sql` was mounted into the Postgres service container via
    `docker-entrypoint-initdb.d`. Service containers are created *before* any
    step runs, so the workspace was still empty, Docker silently mounted a
    directory where the file was expected, and the container died with
    `could not read from input file: Is a directory` — taking the whole
    integration job with it. The file is now piped in through `psql` after
    checkout, which fails loudly instead of mysteriously.
  - The `load` job's service container starts with an empty schema and that job
    runs only `test_losslessness.py`, so nothing ever created the tables: the
    gate died on `relation "tombstones" does not exist` before measuring
    anything. The job now applies the schema first.
  - The `typecheck` job did not install the `postgres` extra, so `sqlalchemy`
    was absent and `src/sentinel/store/models.py` failed as `Cannot find
    implementation or library`. The store's ORM layer was never actually being
    type-checked in CI. The job now installs the extra, which is what
    `pyproject.toml`'s asyncpg override always claimed.
  - `tests/integration/test_db_roles.py` imported `asyncpg` at module level, so
    the offline test job — which installs no database driver — aborted
    collection of the entire suite, not just that module. Now collected through
    `pytest.importorskip`, matching the other integration modules.
  - The `integration-postgres` job's service container starts with an empty
    schema, and only `test_migration.py` builds one — but pytest runs it after
    the store and memory tests, so those died on `relation "sessions" does not
    exist`. The job now applies the schema before the suite, exactly as the
    `load` job does.
- `_sqlite_path()` stripped leading slashes to turn `sqlite:///abs/path` into
  `/abs/path`, which also turned a **bare** absolute POSIX path into a relative
  one whose parent directory does not exist. The symptom was
  `sqlite3.OperationalError: unable to open database file` on Linux CI only —
  on Windows `tmp_path` begins with a drive letter, so the strip was a no-op
  and every local run passed. `tests/unit/test_store_factory_dsn.py` now pins
  the bare-path, URI-form, relative, Windows and `:memory:` cases.
- `deploy/roles.sql` is now idempotent, resolves the database with
  `current_database()` instead of hardcoding `sentinel` (so one file provisions
  both the compose stack and CI's `sentinel_test`), and skips the reviewer's
  column grant when `flags` does not exist yet instead of erroring.
- Migration `0002_flags_s3` could not run on Postgres at all: it issued a
  column-scoped `ALTER DEFAULT PRIVILEGES`, which PostgreSQL rejects with
  `default privileges cannot be set for columns`, aborting `alembic upgrade`
  before any column was added. **The `integration-postgres` CI job could never
  have passed while this was in place.** The invalid statement is gone, and
  `deploy/roles.sql` no longer carries the same error.
- New migration `0003_memory_entries` creates the memory adapter's table
  instead of the adapter running `CREATE TABLE` on first connect. Runtime DDL
  contradicted the role model in `deploy/roles.sql` (the writer has no
  `CREATE`), and it made the table's existence depend on whether a read had
  happened first, which broke the integration fixture against a fresh database.
- `PostgresMemoryStore` is a pure reader/writer; its table is schema-managed.
- `docs/flag-schema.md` overstated the reviewer's grant as `INSERT, SELECT,
  UPDATE`. The actual grant is `SELECT` plus `UPDATE` limited to
  `adjudication, adjudicated_by, adjudicated_at, auto_resolved`, so a reviewer
  cannot insert or rewrite a finding — only rule on one. Now documented
  correctly and covered by
  `tests/integration/test_db_roles.py::test_reviewer_may_only_adjudicate_flags`,
  which connects as the role and asserts the four writable columns plus the
  five rejections. The `integration-postgres` job now sets
  `SENTINEL_ROLE_REVIEWER_DSN`, without which that test skipped silently and
  the role ADR-0012 depends on would have shipped unverified.

### Known limitations

Carried out of `S3` and recorded in
[`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md). The module detects an agent
contradicting its own tool output. It does **not** detect an agent citing a
source it never consulted: evidence is matched to a claim by value across the
whole turn, with no per-claim source resolution, so fabricated citations and
cherry-picked numbers are silent misses and are absent from the corpus. The
attribution lexicon ("according to the document", "the API shows") is also
unimplemented, as is legal/safety severity weighting. The 0.00% FP/FN figures
are exact for the corpus that exists and do not bound the real-world error rate.

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
