# 0011 — Append-only enforcement at the database layer

Date: 2026-09-24

## Status
Accepted (Sprint `S2`)

## Context

The event store is append-only by design (INV-2, ADR-0002): captured evidence
must never be rewritten or destroyed, and any deletion must leave a tombstone.
That invariant is enforced today in application code — every backend only ever
issues `INSERT`/`SELECT` and the only permitted mutation path is the retention
pruner, which tombstone-then-deletes under a documented policy (S2-T12).

Application-level discipline is not a security boundary. A compromised write
credential, a buggy future feature, or a manual `psql` at the wrong terminal can
still issue `UPDATE`/`DELETE`/`TRUNCATE`. We need the database itself to refuse
mutation for the identities the app actually uses, so the append-only property
holds even when nothing else does (S2-T3).

## Decision

Postgres deployments are provisioned with three least-privilege roles
(`deploy/roles.sql`):

| Role | Grants | Used by |
|---|---|---|
| `sentinel_migrator` | DDL owner; runs Alembic, owns every object | migrations only |
| `sentinel_writer` | `INSERT` + `SELECT` (default privileges by migrator); `UPDATE`/`DELETE`/`TRUNCATE`/`REFERENCES`/`TRIGGER` revoked explicitly | capture workers (`SENTINEL_STORE_DSN`) |
| `sentinel_reader` | `SELECT` only | replay, audit, reporting |

Grants for future tables are carried forward with `ALTER DEFAULT PRIVILEGES FOR
ROLE sentinel_migrator`, so the writer keeps `INSERT`/`SELECT` on tables created
by later migrations without re-granting. `CREATE ON SCHEMA public` is revoked
from `PUBLIC` and belongs only to the migrator.

The composition is proven by DB tests (`tests/integration/test_db_roles.py`,
exit criterion #4 of S2): the reader can read but not write, the writer can
append and read back its own rows, and `UPDATE`/`DELETE`/`TRUNCATE` all raise
`permission denied` for the writer.

## Consequences

- Positive: append-only is a database property, not a convention; a stolen
  writer DSN cannot rewrite or purge evidence.
- Positive: three clearly named roles give operators a mental model that maps
  1:1 to deployment phases (migrate / write / read).
- Positive: the reference compose stack (`deploy/compose.postgres.yml`) wires
  the same roles end-to-end: `migrate` runs as migrator, `capture` as writer.
- Negative: application code must keep `INSERT`/`SELECT` within the writer's
  grants; anything exotic (e.g. `COPY` bulk loads) needs a deliberate grant.
- Neutral: a superuser (or `sentinel_migrator`) can still mutate; that is the
  admin boundary and is documented as such.

## Alternatives considered

- **Triggers that reject UPDATE/DELETE** — rejected: complex to maintain across
  migrations, easy to bypass via `TRUNCATE`, and privileges already express the
  intent more simply.
- **A separate audit table vs. tombstones** — rejected as the primary control:
  it punishes after the fact instead of preventing; tombstones (S2-T12) remain
  the path for legitimate, policy-driven deletion.
- **One `sentinel_app` role with conditional grants** — rejected: cannot express
  "can't change existing rows but can create new ones" with a single grant set,
  so the migrator/writer/reader split is kept.
