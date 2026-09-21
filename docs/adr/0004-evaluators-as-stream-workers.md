# 0004 — Evaluators as independent stream workers

Date: 2026-09-21

## Status
Accepted

## Context
Five detection modules share one flag schema and one event stream, but are
otherwise independent. New modules (a sixth, seventh, ... failure mode) should
be addable without touching capture or the core schemas. Different modules need
different computational profiles (cheap structural checks vs expensive LLM/embedding work).

## Decision
Each module runs as an independently deployable `EvaluatorWorker` that subscribes
to the event stream (completed sessions, plus an on-demand evaluate API for the
gating path), computes flags deterministically, and writes them back into the
same store. A worker is idempotent, restart-safe (checkpointed), and versioned so
a module's outputs are reproducible per `(input, module_version)`.

## Consequences
- Positive: modules scale, upgrade, and fail independently.
- Positive: adding module N+1 requires no changes to capture or other modules.
- Negative: more moving parts and queue semantics to operate.
- Neutral: a shared `EvaluatorWorker` base class keeps the surface small.

## Alternatives considered
- All-in-one analyzer — rejected: couples compute profiles and creates upgrade risk.
- External event-bus product (Kafka etc.) — deferred: unnecessary at v1 scale;
  store-based dispatch suffices; revisit if volume demands it.
