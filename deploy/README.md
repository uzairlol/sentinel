# Reference deployment (docker compose)

Sprint `S2` lands the Postgres reference stack. The `S0`–`S1` milestones use
SQLite for development; once the Postgres store is in use, deploy through the
files in this directory.

## Available stacks

- `compose.postgres.yml` — Postgres 16 + Alembic migration runner +
  demo capture worker. Provisioned with the three least-privilege roles
  (`deploy/roles.sql`, ADR-0011).
- `Dockerfile.migrate` — image used by both the `migrate` and `capture`
  services (migrator vs. writer credentials respectively).
- `roles.sql` — the `sentinel_migrator` / `sentinel_writer` / `sentinel_reader`
  roles and grants (S2-T3).
- `capture_service.py` — demo capture worker: a `BatchedWriter` streaming
  heartbeat events into the store (S2-T15).

## Quick start

```bash
# 1. bring up the database and apply migrations
docker compose -f deploy/compose.postgres.yml up --build -d db migrate

# 2. watch the capture worker persist events
docker compose -f deploy/compose.postgres.yml up --build -d capture

# 3. inspect from the host (store writer credentials)
SENTINEL_STORE_DSN="postgresql://sentinel_writer:writer_dev@localhost:5432/sentinel" \
  sentinel health --dsn "$SENTINEL_STORE_DSN"
```

## Planned stacks

- `compose.full.yml` — Postgres + evaluator workers + policy/review service (Sprint `S7`)
- `compose.monitoring.yml` — OpenTelemetry collector + Prometheus + Grafana (Sprint `S10`)

See `SENTINEL_TDD.md` §2.10 for the operations standard these compose files must
satisfy (least-privilege DB roles, TLS, backup/restore hooks). The password
defaults here are dev-only; provisioning for anything shared must inject real
secrets (ADR-0011).
