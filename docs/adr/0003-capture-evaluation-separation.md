# 0003 — Capture/evaluation separation

Date: 2026-09-21

## Status
Accepted

## Context
Coupling data capture to analysis logic is the most common reason observability
systems become brittle as the underlying agent framework changes. The
instrumentation boundary is the most volatile part of the system.

## Decision
The instrumentation layer performs **zero** evaluation. It only serializes and
writes events. Evaluators never run in the capture path. This invariant (INV-1)
is enforced by an import-boundary test: `sentinel.instrument.*` must not import
from `sentinel.eval.*` or `sentinel.gate.*`, enforced in CI.

## Consequences
- Positive: changes to agent framework wiring never cascade into detection
  logic, and vice versa.
- Positive: capture can be audited and performance-tuned in isolation.
- Positive: capture never inherits evaluator bugs.
- Negative: a degree of indirection; events must carry enough raw signal.
- Neutral: detection inputs are snapshots in the store, which is the contract.

## Alternatives considered
- Evaluators inlining at the capture boundary — rejected: brittleness.
- A single "smart log" package — rejected: conflates concerns.
