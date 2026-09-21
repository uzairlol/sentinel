# Architecture Decision Records

An Architecture Decision Record (ADR) captures a significant decision, why it
was made, and what the consequences are. Follow the template below. Every
significant decision in this project must have an ADR — see
[`CONTRIBUTING.md`](https://github.com/uzairarif/sentinel/blob/main/CONTRIBUTING.md)
and `SENTINEL_TDD.md` §2.9.

## Format

```markdown
# NNNN — Title

Date: YYYY-MM-DD

## Status
Accepted | Proposed | Superseded-by-NNNN

## Context
Why this decision is being made, and what is at stake.

## Decision
What we decided, in concrete terms.

## Consequences
- Positive: ...
- Negative: ...
- Neutral: ...

## Alternatives considered
What we rejected, and why.
```

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-python-312-async-first-sdk.md) | Python 3.12+ async-first SDK | Accepted |
| [0002](0002-postgres-reference-store.md) | Postgres reference store; SQLite dev-only | Accepted |
| [0003](0003-capture-evaluation-separation.md) | Capture/evaluation separation | Accepted |
| [0004](0004-evaluators-as-stream-workers.md) | Evaluators as independent stream workers | Accepted |
| [0005](0005-auditable-gating-engine.md) | Minimal, auditable gating rules engine | Accepted |
| [0006](0006-self-hosted-core-path.md) | Self-hosted core path; opt-in external models | Accepted |
| [0007](0007-versioned-event-schemas.md) | Versioned event schemas + backward-compatible readers | Accepted |
| [0008](0008-local-ollama-defaults.md) | Local Ollama defaults for embeddings/judge; pluggable | Accepted |
| [0009](0009-small-public-api-surface.md) | Small stable public API surface; SemVer policy | Accepted |
| [0010](0010-ulid-event-ids.md) | ULID event IDs + monotonic sequence | Accepted |
