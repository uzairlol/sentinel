# 0010 — ULID event IDs + monotonic sequence

Date: 2026-09-21

## Status
Accepted

## Context
Replay, idempotency, and provenance all need stable, sortable, collision-free
event identifiers, plus an in-session causal order that does not depend on clock
precision. Wall-clock timestamps alone are not guaranteed to be unique or be
ordered under concurrency.

## Decision
- Every event has a **ULID** `event_id` — time-ordered, k-sortable, collision-free
  across processes without coordination.
- Every event additionally carries a **`session_id` + monotonic `seq`** integer,
  unique per `(session_id, seq)`. Ordering for replay is by `seq`.
- Timestamps (UTC) are stored for human interpretation and analytics; they are
  not relied on for ordering.
- Idempotency keys for workers/flag writes derive from deterministic inputs
  (session + module + version) per ADR-0004.

## Consequences
- Positive: losslessness verification and gap detection are simple
  (`seq` contiguity per session).
- Positive: concurrent capture from multiple instrumented processes never
  collides and still orders within a session.
- Negative: an extra dependency/implementation for ULID encoding.
- Neutral: ULIDs remain sortable lexicographically, aiding partitioning.

## Alternatives considered
- UUIDv4 — rejected: not sortable, no time component.
- DB-generated sequences — rejected: couples capture to DB latency and availability.
