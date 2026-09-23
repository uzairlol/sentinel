"""SQLite-backed event store — dev and tests only, never production.

The reference store is Postgres (docs/adr/0002); this implementation exists so
the vertical slice (``S0``) and every local test can run with zero external
dependencies. It satisfies the same :class:`~sentinel.store.protocol.EventStore`
contract, which ``S2`` double-checks with store-parity contract tests.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

import aiosqlite

from sentinel.models.events import Event, RefLink
from sentinel.store.errors import RefIntegrityError
from sentinel.store.protocol import EventStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    ts             TEXT NOT NULL,
    type           TEXT NOT NULL,
    payload        TEXT NOT NULL,
    refs           TEXT NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_session_seq
    ON events (session_id, seq);
CREATE INDEX IF NOT EXISTS idx_events_session
    ON events (session_id, ts);
"""

_ROW_COLUMNS = ("event_id", "session_id", "seq", "ts", "type", "payload", "refs", "schema_version")


class SQLiteEventStore(EventStore):
    """An append-only event store on a single aiosqlite connection."""

    def __init__(self, path: str = ":memory:") -> None:
        """Create a store on *path*; ``:memory:`` keeps everything in RAM."""
        self._path = path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            self._connection = await aiosqlite.connect(self._path)
            self._connection.row_factory = sqlite3.Row
            await self._connection.executescript(_SCHEMA)
        return self._connection

    async def append(self, event: Event) -> None:
        """Persist *event*.

        Re-appending the same ``event_id`` is a no-op (idempotency, INV-2).
        Appending a *different* event with an already-used ``(session_id, seq)``
        raises :class:`sqlite3.IntegrityError` so replay ordering stays total.
        Every reference in ``event.refs`` must resolve to an event already
        persisted in the same session, otherwise
        :class:`RefIntegrityError` is raised (INV-3).
        """
        conn = await self._conn()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT 1 FROM events WHERE event_id = ?", (event.event_id,)
            )
            if await cursor.fetchone() is not None:
                return
            if event.refs:
                placeholders = ",".join("?" for _ in event.refs)
                cursor = await conn.execute(
                    "SELECT event_id FROM events "  # noqa: S608 -- placeholders are literal "?" tokens, never user input
                    f"WHERE session_id = ? AND event_id IN ({placeholders})",
                    [event.session_id, *(link.event_id for link in event.refs)],
                )
                known = {row["event_id"] for row in await cursor.fetchall()}
                missing = [link for link in event.refs if link.event_id not in known]
                if missing:
                    raise RefIntegrityError(
                        f"event {event.event_id!r} references events not in "
                        f"session {event.session_id!r}: " + ", ".join(m.event_id for m in missing)
                    )
            await conn.execute(
                "INSERT INTO events "
                "(event_id, session_id, seq, ts, type, payload, refs, schema_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.session_id,
                    event.seq,
                    event.ts.isoformat(),
                    event.type,
                    json.dumps(event.payload),
                    json.dumps([link.model_dump() for link in event.refs]),
                    event.schema_version,
                ),
            )
            await conn.commit()

    async def get_session(self, session_id: str) -> list[Event]:
        """Return all events for *session_id* ordered by ``seq`` ascending."""
        conn = await self._conn()
        cursor = await conn.execute(
            "SELECT event_id, session_id, seq, ts, type, payload, refs, schema_version "
            "FROM events WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_event(row) for row in rows]

    async def close(self) -> None:
        """Close the underlying connection and release the file handle."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


def _row_to_event(row: sqlite3.Row) -> Event:
    event_id = row["event_id"]
    session_id = row["session_id"]
    seq = row["seq"]
    ts = row["ts"]
    type_ = row["type"]
    payload = row["payload"]
    refs = row["refs"]
    schema_version = row["schema_version"]
    return Event(
        event_id=event_id,
        session_id=session_id,
        seq=seq,
        ts=_parse_ts(ts),
        type=type_,
        payload=json.loads(payload),
        refs=[RefLink(**link) for link in json.loads(refs)],
        schema_version=schema_version,
    )


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw)
    if ts.utcoffset() is None:
        return ts.replace(tzinfo=UTC)
    return ts
