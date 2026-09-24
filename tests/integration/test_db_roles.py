"""Append-only enforcement at the DB layer (``S2-T3``, exit criteria #4).

Connects as the least-privilege ``sentinel_writer`` and ``sentinel_reader``
roles created by ``deploy/roles.sql`` and proves:

* ``sentinel_reader`` can read but not write;
* ``sentinel_writer`` can append and read back its own rows;
* ``UPDATE``/``DELETE``/``TRUNCATE`` are rejected for the writer, so evidence
  cannot be rewritten or destroyed even with a live app credential.

Requires ``SENTINEL_ROLE_WRITER_DSN`` and ``SENTINEL_ROLE_READER_DSN`` pointing
at a database provisioned with ``deploy/roles.sql``. Skipped when absent so the
suite stays green offline; CI wires it to the Postgres service.

Because sibling migration tests drop/recreate the tables as the *superuser*
(bypassing ``ALTER DEFAULT PRIVILEGES``, which only fires for objects created by
``sentinel_migrator``), a module fixture re-applies the two table grants first —
the same step an operator runs after applying schema changes. The append-only
assertions themselves are not weakened: nothing re-grants ``UPDATE``/``DELETE``,
so a fresh table simply has no mutate privileges for the writer at all.
"""

from __future__ import annotations

import os

import asyncpg
import pytest

from sentinel.models.events import SESSION_START, Event, new_event_id

_WRITER = os.getenv("SENTINEL_ROLE_WRITER_DSN")
_READER = os.getenv("SENTINEL_ROLE_READER_DSN")
_SUPER = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not (_WRITER and _READER), reason="role DSNs not set"),
    pytest.mark.integration,
]


@pytest.fixture(autouse=True)
async def _refresh_table_grants() -> None:
    """Re-apply writer/reader table grants after migrations recreate tables."""
    if not (_WRITER and _READER and _SUPER):
        return
    conn = await asyncpg.connect(_SUPER)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_writer'")
        if exists:
            await conn.execute(
                "GRANT INSERT, SELECT ON ALL TABLES IN SCHEMA public TO sentinel_writer"
            )
            await conn.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_reader")
    finally:
        await conn.close()


def _event(session_id: str, seq: int) -> Event:
    from datetime import UTC, datetime

    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=datetime.now(UTC),
        type=SESSION_START,
        payload={"probe": session_id},
    )


async def test_writer_appends_and_reads_back() -> None:
    session_id = new_event_id()
    e = _event(session_id, 0)
    conn = await asyncpg.connect(_WRITER)
    try:
        await conn.execute(
            "INSERT INTO sessions (session_id, status, started_at, schema_version, meta) "
            "VALUES ($1, 'active', $2::timestamptz, '0.2', '{}'::jsonb)",
            session_id,
            e.ts,
        )
        await conn.execute(
            "INSERT INTO events (event_id, session_id, seq, ts, type, payload, schema_version) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, '0.2')",
            e.event_id,
            session_id,
            0,
            e.ts,
            SESSION_START,
            "{}",
        )
        count = await conn.fetchval("SELECT count(*) FROM events WHERE session_id = $1", session_id)
        assert count == 1
    finally:
        await conn.close()


async def test_reader_can_read_but_not_write() -> None:
    conn = await asyncpg.connect(_READER)
    try:
        seen = await conn.fetchval("SELECT count(*) FROM events")
        assert isinstance(seen, int)
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("INSERT INTO events VALUES (DEFAULT)")  # pragma: no cover
    finally:
        await conn.close()


async def test_writer_cannot_update_delete_or_truncate() -> None:
    session_id = new_event_id()
    e = _event(session_id, 0)
    conn = await asyncpg.connect(_WRITER)
    try:
        await conn.execute(
            "INSERT INTO sessions (session_id, status, started_at, schema_version, meta) "
            "VALUES ($1, 'active', $2::timestamptz, '0.2', '{}'::jsonb)",
            session_id,
            e.ts,
        )
        await conn.execute(
            "INSERT INTO events (event_id, session_id, seq, ts, type, payload, schema_version) "
            "VALUES ($1, $2, $3, $4, $5, '{}'::jsonb, '0.2')",
            e.event_id,
            session_id,
            0,
            e.ts,
            SESSION_START,
        )
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("UPDATE events SET payload = '{}'::jsonb")
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("DELETE FROM events")
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("TRUNCATE events")
    finally:
        await conn.close()
