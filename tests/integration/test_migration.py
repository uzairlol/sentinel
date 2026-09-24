"""Alembic migration round-trip against real Postgres (``S2-T2``).

Runs ``upgrade base -> head`` then ``downgrade base`` and asserts the schema
state at each stop. Requires ``SENTINEL_TEST_POSTGRES_DSN`` (absent -> skip);
exercised by the CI integration job.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not _DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set"),
    pytest.mark.integration,
]


def _alembic(*args: str) -> None:
    env = dict(os.environ, SENTINEL_STORE_DSN=_DSN or "")
    # fixed headless alembic invocation against the local test database
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "alembic", *args],
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


async def _tables() -> set[str]:
    import asyncpg

    conn = await asyncpg.connect(_DSN or "")
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {row["tablename"] for row in rows}
    finally:
        await conn.close()


async def _schema_version() -> str | None:
    import asyncpg

    conn = await asyncpg.connect(_DSN or "")
    try:
        row = await conn.fetchrow(
            "SELECT value FROM schema_meta WHERE key = 'event_schema_version' LIMIT 1"
        )
        return row["value"] if row else None
    finally:
        await conn.close()


async def test_migration_upgrade_downgrade_round_trip() -> None:
    _alembic("downgrade", "base")
    tables = await _tables()
    assert "events" not in tables
    # Alembic intentionally keeps the (now-empty) version table across a
    # full downgrade; only the schema objects are torn down.
    assert "alembic_version" in tables

    _alembic("upgrade", "head")
    tables = await _tables()
    expected = {"sessions", "events", "event_refs", "flags", "schema_meta", "tombstones"}
    assert expected <= tables
    assert await _schema_version() == "0.2"

    _alembic("downgrade", "base")
    tables = await _tables()
    assert "events" not in tables

    _alembic("upgrade", "head")
    assert expected <= await _tables()
    assert await _schema_version() == "0.2"


async def test_schema_contains_expected_constraints_and_indexes() -> None:
    import asyncpg

    conn = await asyncpg.connect(_DSN or "")
    try:
        constraints = {
            row["conname"]
            for row in await conn.fetch(
                "SELECT conname FROM pg_constraint WHERE connamespace = "
                "(SELECT oid FROM pg_namespace WHERE nspname = 'public')"
            )
        }
        indexes = {
            row["indexname"]
            for row in await conn.fetch(
                "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
            )
        }
    finally:
        await conn.close()

    assert "uq_events_session_seq" in constraints
    assert "events_pkey" in constraints
    assert "ck_events_seq_nonnegative" in constraints
    assert "ck_sessions_status" in constraints
    assert "fk_event_refs_event" in constraints
    assert "fk_event_refs_ref" in constraints
    assert "fk_events_session" in constraints
    assert "fk_flags_event" in constraints
    for index in (
        "uq_events_session_seq",
        "ix_events_session_type_ts",
        "ix_events_ts",
        "ix_event_refs_ref_event_id",
        "ix_sessions_started_at",
    ):
        assert index in indexes
