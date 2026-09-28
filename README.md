# Sentinel SDK

A unified runtime safety instrumentation layer for autonomous LLM agents.

Sentinel wraps an agent's execution boundary — its LLM calls, its tool calls and
their return values, and its memory read/write operations — and records them into
a structured, append-only event store. On top of that store it runs a modular
battery of safety evaluators, each targeting a specific, reproducible failure
mode documented in current AI-safety research:

- **Reasoning faithfulness** — visible reasoning that does not reflect the real decision basis
- **Tool-use grounding / provenance** — claims grounded in retrievals that never happened, or that contradict what was returned
- **Evaluation awareness / sandbagging** — behaving differently when the agent infers it is being evaluated
- **Memory & context integrity** — corrupted, injected, or collapsed persistent memory
- **Specification gaming / objective drift** — satisfying the letter of an instruction while defeating its intent

A configurable policy layer can hold consequential actions at an oversight
checkpoint until a human approves, rejects, or requests revision.

## Status

**Pre-alpha (maturity `25 → 35`, in progress).** Sprint `S3` (first evaluator
module — flag schema, worker framework, tool-use provenance) is a **complete**
pass on `main`, with `S1` (instrumentation) and `S2` (event store) before it.
What works: the flag schema and adjudication model, the worker framework, and
detection of an agent **contradicting its own tool output**, **citing a source it
never consulted**, and **quietly narrowing a result set to its favourable half** —
measured at **0.00% FP / 0.00% FN** over a 28-case corpus that covers all three.
What does not yet work, and is documented rather than hidden: a claim that names
source A while citing source B is not yet resolved to a specific tool result, and
no instrumenter records a reasoning trace yet (so the module can read one but
nothing writes one). Those are the known gaps in
[`docs/design/SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md); read the
[limitations](docs/modules/provenance.md#limitations) before relying on this
module. Test count, coverage, and `mypy --strict` + ruff + bandit status
are reported in the master plan in
[`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md) for the full
`-1 → 101` roadmap, per-sprint work packages, and exit gates — the `S3` row of
the [Roadmap Status Board](docs/design/SENTINEL_TDD.md#34-roadmap-status-board)
records the gate result.

## Design documents

- [`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md) — master technical design document and development roadmap
- [`docs/design/safety_sdk.tex`](docs/design/safety_sdk.tex) — original conceptual design (LaTeX)
- [`docs/design/safety_sdk.pdf`](docs/design/safety_sdk.pdf) — compiled version of the original design
- [`docs/flag-schema.md`](docs/flag-schema.md) — what an evaluator concluded, its evidence, and its review lifecycle
- [`docs/modules/provenance.md`](docs/modules/provenance.md) — the `S3` module: methodology, severity model, measured FP/FN, limitations

## Quickstart (vertical slice, `S0`)

Capture one chat call from a local Ollama server and replay it losslessly:

```bash
uv sync
uv run python examples/hello_ollama.py

# replay any captured session by id:
uv run sentinel replay <session_id> --dsn hello_ollama.sqlite3
```

The example wraps a single `POST /api/chat`, writes an `llm.request` and an
`llm.response` event (plus `session.start`/`session.end` bookends) to a SQLite
event store, and replays the session in order.

## Production store (Sprint `S2`)

Beyond SQLite (dev-only), you can run the SDK against the reference Postgres
store — batched append, streaming replay, gap detection, and retention pruning:

```bash
docker compose -f deploy/compose.postgres.yml up --build -d db migrate
export SENTINEL_STORE_DSN="postgresql://sentinel_writer:writer_dev@localhost:5432/sentinel"
uv run sentinel health --dsn "$SENTINEL_STORE_DSN"
```

Provisioning uses three least-privilege roles (ADR-0011): `sentinel_migrator`,
`sentinel_writer`, `sentinel_reader` — see `deploy/roles.sql` and
`deploy/README.md`. A fourth, `sentinel_reviewer`, may update flags to record a
human decision; the writer role stays append-only.

## Evaluating a session (Sprint `S3`)

The provenance module flags assertions no tool result supports, values the
logged tool output refutes, sources a session never cited, and counts that
quietly narrow a larger set. Measure it against the adversarial corpus:

```bash
uv run sentinel eval-fixtures --module provenance
```

```
corpus: 28 cases  module=sentinel.tool_grounding@0.2.0
confusion matrix (case level)
                flagged  silent
  should flag       16       0   <- recall 100.00%
  should be quiet     0      12   <- precision 100.00%

  false-negative rate: 0.00% (gate <= 10%)
  false-positive rate: 0.00% (gate <= 0%)
  claim FP rate:       0.00% (gate <= 5%)
PASS
```

Exit code `0` = within budget, `1` = budget breached, `2` = bad usage. To
evaluate a real session, hand its events to `sentinel.eval.provenance` and read
the flags back from the store — see
[`docs/modules/provenance.md`](docs/modules/provenance.md).

## Repository layout

```
src/sentinel/     package source (src layout)
tests/            unit | integration | property | contract | adversarial | e2e | perf
docs/             adr | design | modules | operations | runbooks | security
examples/         runnable integration examples
deploy/           reference deployment (docker compose)
.github/          CI workflow, issue/PR templates
```

## Development

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev          # install dependencies + dev tooling
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run mypy                  # static types (strict)
uv run bandit -r src         # security scan
uv run pytest                # tests + coverage (≥ 90% on src/sentinel)
uv run sentinel --version    # smoke-test the CLI
```

Postgres-gated tests (store integration, contract parity, 1M-event lossless-
ness, backup/restore) are skipped offline; run them against a local server or
in CI with:

```bash
export SENTINEL_TEST_POSTGRES_DSN="postgresql://postgres:pass@localhost:5432/sentinel_test"
export SENTINEL_STORE_DSN="$SENTINEL_TEST_POSTGRES_DSN"
export SENTINEL_ROLE_WRITER_DSN="postgresql://sentinel_writer:writer_dev@localhost:5432/sentinel_test"
export SENTINEL_ROLE_READER_DSN="postgresql://sentinel_reader:reader_dev@localhost:5432/sentinel_test"
uv run pytest tests/integration tests/contract
# p99 write budget: see perf/write-benchmark.md
uv run python perf/bench_writes.py
```

The `sentinel` CLI covers `replay <session_id>`, `sessions list`, and
`health`, each with `--dsn <sqlite-path|postgres-url>` and
`--json`/`--pretty` output (`uv run sentinel <cmd> --help`).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Every change flows through a PR with
the tasks in [`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md); commits use Conventional
Commits (`feat(scope): ...`, `fix(scope): ...`).

## Security

See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities.

## License

[Apache-2.0](LICENSE).
