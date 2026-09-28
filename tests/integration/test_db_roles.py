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

import pytest

# asyncpg lives behind the optional "postgres" extra. Import it through pytest
# so this module stays *collectable* in the offline job, which installs no
# database driver: without this the module-level import aborts collection of
# the whole suite before the skipif below can take effect.
asyncpg = pytest.importorskip("asyncpg")

from sentinel.models.events import SESSION_START, Event, new_event_id  # noqa: E402

_WRITER = os.getenv("SENTINEL_ROLE_WRITER_DSN")
_READER = os.getenv("SENTINEL_ROLE_READER_DSN")
_REVIEWER = os.getenv("SENTINEL_ROLE_REVIEWER_DSN")
_SUPER = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not (_WRITER and _READER), reason="role DSNs not set"),
    pytest.mark.integration,
]


@pytest.fixture(autouse=True)
async def _refresh_table_grants() -> None:
    """Re-apply writer/reader/reviewer table grants after migrations recreate tables."""
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
        has_reviewer = await conn.fetchval(
            "SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reviewer'"
        )
        if has_reviewer and await conn.fetchval("SELECT to_regclass('public.flags')"):
            await conn.execute("GRANT SELECT ON flags TO sentinel_reviewer")
            # the single mutation ADR-0012 grants the reviewer
            await conn.execute(
                "GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved) "
                "ON flags TO sentinel_reviewer"
            )
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


async def test_reviewer_may_only_adjudicate_flags() -> None:
    """``sentinel_reviewer``'s one mutation is a decision on a flag.

    ADR-0012 gives the reviewer role exactly this power: it must be able to
    record ``confirmed``/``rejected`` against an existing finding, and nothing
    else. So the negative assertions matter as much as the positive one -- a
    reviewer that could insert a flag could manufacture an accusation, and one
    that could delete or write events could erase or forge the evidence the
    finding rests on.
    """
    if not (_REVIEWER and _WRITER):
        pytest.skip("reviewer DSN not set")
    flag_id = new_event_id()
    session_id = new_event_id()

    writer = await asyncpg.connect(_WRITER)
    try:
        await writer.execute(
            "INSERT INTO sessions (session_id, status, started_at, schema_version, meta) "
            "VALUES ($1, 'active', now(), '0.2', '{}'::jsonb)",
            session_id,
        )
        await writer.execute(
            "INSERT INTO flags (flag_id, session_id, module, module_version, category, "
            "severity, confidence, summary, evidence, created_at) "
            "VALUES ($1, $2, 'sentinel.tool_grounding', '0.1.0', 'ungrounded_claim', "
            "'medium', 0.9, 'probe', '[]'::jsonb, now())",
            flag_id,
            session_id,
        )
    finally:
        await writer.close()

    conn = await asyncpg.connect(_REVIEWER)
    try:
        # the one thing it is for, across every adjudication column ADR-0012
        # grants it
        await conn.execute(
            "UPDATE flags SET adjudication = 'confirmed', adjudicated_by = 'test-reviewer', "
            "adjudicated_at = now(), auto_resolved = false WHERE flag_id = $1",
            flag_id,
        )
        row = await conn.fetchrow(
            "SELECT adjudication, adjudicated_by, auto_resolved FROM flags WHERE flag_id = $1",
            flag_id,
        )
        assert row["adjudication"] == "confirmed"
        assert row["adjudicated_by"] == "test-reviewer"
        assert row["auto_resolved"] is False

        # ...and nothing else
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute(  # pragma: no cover
                "INSERT INTO flags (flag_id, session_id, module, module_version, category, "
                "severity, confidence, summary, evidence, created_at) "
                "VALUES ($1, $2, 'sentinel.tool_grounding', '0.1.0', 'ungrounded_claim', "
                "'medium', 0.9, 'planted', '[]'::jsonb, now())",
                new_event_id(),
                session_id,
            )
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("DELETE FROM flags")  # pragma: no cover
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute(  # pragma: no cover
                "INSERT INTO events (event_id, session_id, seq, ts, type, payload, schema_version) "
                "VALUES ($1, $2, 0, now(), 'session.start', '{}'::jsonb, '0.2')",
                new_event_id(),
                session_id,
            )
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute("UPDATE events SET payload = '{}'::jsonb")  # pragma: no cover
        # the grant is column-scoped, so a reviewer cannot rewrite the finding
        # it is ruling on -- only the ruling
        with pytest.raises(asyncpg.PostgresError, match=r"permission denied"):
            await conn.execute(  # pragma: no cover
                "UPDATE flags SET summary = 'quietly rewritten' WHERE flag_id = $1", flag_id
            )
    finally:
        await conn.close()
