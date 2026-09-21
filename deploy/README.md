# Reference deployment (docker compose)

Sprint `S2` (event store) and Sprint `S7` (policy/gating) will populate this
directory with runnable compose files. The `S0`–`S1` milestones use SQLite for
development, so this tree is intentionally empty until the Postgres store lands.

## Planned stacks

- `compose.postgres.yml` — Postgres-only (store dev/prod baseline)
- `compose.full.yml` — Postgres + evaluator workers + policy/review service (Sprint `S7`)
- `compose.monitoring.yml` — OpenTelemetry collector + Prometheus + Grafana (Sprint `S10`)

See `SENTINEL_TDD.md` §2.10 for the operations standard these compose files must
satisfy (least-privilege DB roles, TLS, backup/restore hooks).
