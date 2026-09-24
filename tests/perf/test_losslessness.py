"""Losslessness load gates (``S2-T16``).

Two variants over the same scenario "emit events round-robin across N
sessions, replay, assert byte-identical digest + zero gaps + exact counts":

* ``sqlite`` — a fast 50k/100 smoke run so every dev gate exercises the
  assertion machinery (no external service).
* ``postgres`` — the production gate: 1,000,000 events across 1,000 sessions.
  Needs ``SENTINEL_TEST_POSTGRES_DSN`` (absent -> skip); runs in the CI
  integration job.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime

import pytest

from sentinel.models.events import LLM_REQUEST, Event, new_event_id
from sentinel.store.factory import build_store
from sentinel.store.protocol import EventStore
from sentinel.store.sqlite import SQLiteEventStore

_PG_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [pytest.mark.perf]


def _make_event(session_id: str, seq: int, i: int) -> Event:
    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=datetime.now(UTC),
        type=LLM_REQUEST,
        payload={"t": "llm.request", "i": i, "pad": f"data-{i % 1000:03d}-"},
    )


def _update_digest(digest: hashlib._Hash, events: list[Event]) -> None:
    for event in events:
        row = (
            event.session_id
            + "\0"
            + str(event.seq)
            + "\0"
            + event.type
            + "\0"
            + json.dumps(event.payload, sort_keys=True, default=str)
            + "\n"
        )
        digest.update(row.encode("utf-8"))


async def _sqlite_digest(store: SQLiteEventStore) -> tuple[int, str]:
    conn = await store._conn()
    digest = hashlib.sha256()
    count = 0
    cursor = await conn.execute(
        "SELECT session_id, seq, type, payload FROM events ORDER BY session_id, seq"
    )
    async for row in cursor:
        count += 1
        payload = json.loads(row["payload"])
        line = (
            row["session_id"]
            + "\0"
            + str(row["seq"])
            + "\0"
            + row["type"]
            + "\0"
            + json.dumps(payload, sort_keys=True, default=str)
            + "\n"
        )
        digest.update(line.encode("utf-8"))
    return count, digest.hexdigest()


async def _pg_digest(_dsn: str) -> tuple[int, str]:
    import asyncpg

    digest = hashlib.sha256()
    count = 0
    conn = await asyncpg.connect(_dsn)
    try:
        stmt = await conn.prepare(
            "SELECT session_id, seq, type, payload FROM events ORDER BY session_id, seq"
        )
        async with conn.transaction():
            async for record in stmt.cursor():
                count += 1
                payload = json.loads(record["payload"])
                line = (
                    record["session_id"]
                    + "\0"
                    + str(record["seq"])
                    + "\0"
                    + record["type"]
                    + "\0"
                    + json.dumps(payload, sort_keys=True, default=str)
                    + "\n"
                )
                digest.update(line.encode("utf-8"))
    finally:
        await conn.close()
    return count, digest.hexdigest()


async def _truncate_pg() -> None:
    import asyncpg

    conn = await asyncpg.connect(_PG_DSN or "")
    try:
        await conn.execute(
            "TRUNCATE tombstones, event_refs, flags, events, sessions RESTART IDENTITY CASCADE"
        )
    finally:
        await conn.close()


async def _run_load(total: int, sessions: int, batch: int) -> tuple[EventStore, str]:
    dsn = "sqlite:///:memory:" if sessions <= 100 else _require_pg()
    if sessions > 100:
        await _truncate_pg()
    store = build_store(dsn)
    per_session = total // sessions
    expected = hashlib.sha256()
    for start in range(0, total, batch):
        events = [
            _make_event(
                session_id=f"{i // per_session:06d}",
                seq=i % per_session,
                i=i,
            )
            for i in range(start, min(start + batch, total))
        ]
        # session-major generation order == the store's (session_id, seq) order,
        # so the expected digest is byte-identical to a full replay
        _update_digest(expected, events)
        await store.append_batch(events)
    return store, expected.hexdigest()


def _require_pg() -> str:
    if not _PG_DSN:
        raise RuntimeError("postgres variant needs SENTINEL_TEST_POSTGRES_DSN")
    return _PG_DSN


async def _assert_lossless_pg(store: EventStore, total: int, sessions: int, expected: str) -> None:
    health = await store.health()
    assert health.events == total
    assert health.sessions == sessions
    assert health.gap_count == 0
    assert health.sessions_with_gaps == 0

    count, actual = await _pg_digest(_PG_DSN or "")
    assert count == total
    assert actual == expected

    # spot-check replay through the store's public surface
    sample = await store.get_session("000000")
    assert len(sample) == total // sessions
    assert [e.seq for e in sample][:3] == [0, 1, 2]
    for sid in ("000000", f"{sessions - 1:06d}"):
        assert await store.detect_gaps(sid) == []


async def _assert_lossless_sqlite(
    store: EventStore, total: int, sessions: int, expected: str
) -> None:
    assert isinstance(store, SQLiteEventStore)
    health = await store.health()
    assert health.events == total
    assert health.sessions == sessions
    assert health.gap_count == 0

    count, actual = await _sqlite_digest(store)
    assert count == total
    assert actual == expected

    for sid in ("000000", f"{sessions - 1:06d}"):
        assert await store.detect_gaps(sid) == []


async def test_losslessness_sqlite_smoke() -> None:
    total, sessions, batch = 50_000, 100, 5_000
    store, expected = await _run_load(total, sessions, batch)
    try:
        await _assert_lossless_sqlite(store, total, sessions, expected)
    finally:
        await store.close()


@pytest.mark.skipif(not _PG_DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set")
async def test_losslessness_postgres_1m() -> None:
    total, sessions, batch = 1_000_000, 1_000, 10_000
    store, expected = await _run_load(total, sessions, batch)
    try:
        await _assert_lossless_pg(store, total, sessions, expected)
    finally:
        await store.close()
