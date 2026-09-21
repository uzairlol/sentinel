# 0002 — Postgres reference store; SQLite dev-only

Date: 2026-09-21

## Status
Accepted

## Context
The event store must support provenance queries that are relational at their
core: joins across sessions, calls, and flags. Retrieval is not similarity-based
for the primary query patterns. Teams deploying Sentinel are small engineering
teams comfortable operating conventional databases.

## Decision
PostgreSQL 16+ is the reference implementation for production, accessed via
SQLAlchemy 2.0 (asyncio) + asyncpg, with Alembic migrations. SQLite (aiosqlite)
is supported for development and tests only — never for production. The two
backends share one `EventStore` protocol and pass the same contract tests.

## Consequences
- Positive: production-grade durability, WAL archiving/PITR (DR), GIN indexing
  for freedom on JSONB, and a query engine that matches the product's needs.
- Positive: self-hosting fits existing team skills (ADR-0006).
- Negative: requires running Postgres; adds operational surface.
- Neutral: embeddings used by evaluators are maintained internally to each
  evaluator, not stored in the main event store.

## Alternatives considered
- ClickHouse — great for analytics, poor fit for the OLTP append + FK semantics.
- A standalone vector DB — unnecessary; the core queries are relational.
- DuckDB/embedded — fine for dev, not a production multi-writer target.
