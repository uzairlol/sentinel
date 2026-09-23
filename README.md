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

**Pre-alpha (maturity `0`).** Sprint `S0` (vertical slice) gate is green on `main`;
`S1` (instrumentation layer core) is next. See the master plan in
[`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md) for the full
`-1 → 101` roadmap, per-sprint work packages, and exit gates.

## Design documents

- [`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md) — master technical design document and development roadmap
- [`docs/design/safety_sdk.tex`](docs/design/safety_sdk.tex) — original conceptual design (LaTeX)
- [`docs/design/safety_sdk.pdf`](docs/design/safety_sdk.pdf) — compiled version of the original design

## Quickstart (vertical slice, `S0`)

Capture one chat call from a local Ollama server and replay it losslessly:

```bash
uv sync
uv run python examples/hello_ollama.py

# replay any captured session by id:
uv run sentinel replay <session_id> --store hello_ollama.sqlite3
```

The example wraps a single `POST /api/chat`, writes an `llm.request` and an
`llm.response` event (plus `session.start`/`session.end` bookends) to a SQLite
event store, and replays the session in order.

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
uv run pytest                # tests + coverage (≥ 90% on src/sentinel)
uv run sentinel --version    # smoke-test the CLI
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Every change flows through a PR with
the tasks in [`SENTINEL_TDD.md`](docs/design/SENTINEL_TDD.md); commits use Conventional
Commits (`feat(scope): ...`, `fix(scope): ...`).

## Security

See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities.

## License

[Apache-2.0](LICENSE).
