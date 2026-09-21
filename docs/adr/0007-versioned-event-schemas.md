# 0007 — Versioned event schemas + backward-compatible readers

Date: 2026-09-21

## Status
Accepted

## Context
Events are durable and append-only (INV-2). As the product evolves, event
payloads and flag schemas will change. Unversioned, silently-shipped schema
changes would corrupt replay, provenance, and audit exports.

## Decision
Every event and flag carries an explicit `schema_version`. Readers must support
the previous minor schema version. A breaking schema change requires a migration
plus an ADR; the store persists a `schema_meta` table and can validate-on-read
for older versions. Replaying a session always reflects the schema active at
capture time.

## Consequences
- Positive: safe upgrade path; audit exports remain interpretable after years.
- Positive: failure to support an old schema is caught by contract/schema tests.
- Negative: schema evolution requires discipline and a little extra code.
- Neutral: version numbers are module-scoped, not global.

## Alternatives considered
- Unversioned JSON payloads — rejected: unprovable backward compatibility.
- Full Protobuf/Avro schema registry — rejected: heavy for the current scale.
