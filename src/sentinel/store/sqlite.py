"""SQLite-backed event store — dev and tests only, never production.

The reference store is Postgres (docs/adr/0002); this implementation exists so
the vertical slice (``S0``), every local test, and the offline CLI work with
zero external dependencies. Since ``S2`` it satisfies the *same* contract as
:class:`~sentinel.store.postgres.PostgresEventStore` -- append-only semantics,
streaming iteration, session listing, gap detection, health, and retention
pruning -- verified by the parity contract suite
(``tests/contract/test_store_parity.py``).

The SQLite schema shadows Postgres with a ``sessions`` bookkeeping table, an
``event_refs`` link table (foreign keys enforced when ``PRAGMA foreign_keys``
is on), a ``flags`` table for evaluator parity, and ``tombstones`` for the
retention audit trail. The legacy ``events.refs`` JSON column is kept so
pre-``S2`` databases replay unchanged and ``0.1`` flat-string refs still read
(``S2-T18``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import aiosqlite

from sentinel.models.events import (
    SESSION_END,
    SESSION_START,
    Event,
    RefKind,
)
from sentinel.models.flags import Adjudication, EvidenceRef, Flag, Severity
from sentinel.store.errors import RefIntegrityError
from sentinel.store.gaps import SeqGap, seq_gaps
from sentinel.store.protocol import EventStore
from sentinel.store.read_compat import materialize_refs
from sentinel.store.reporting import CallEdge, SessionSummary, StoreHealth
from sentinel.store.retention import PruneReport, RetentionPolicy, merge_cutoffs

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    agent_id       TEXT,
    status         TEXT NOT NULL DEFAULT 'active',
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    schema_version TEXT NOT NULL,
    meta           TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions (started_at);
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
CREATE INDEX IF NOT EXISTS idx_events_ts
    ON events (ts);
CREATE TABLE IF NOT EXISTS event_refs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    ref_event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    UNIQUE (event_id, ref_event_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_event_refs_ref
    ON event_refs (ref_event_id);
CREATE TABLE IF NOT EXISTS flags (
    flag_id        TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id),
    event_id       TEXT REFERENCES events(event_id) ON DELETE SET NULL,
    module         TEXT NOT NULL,
    module_version TEXT NOT NULL,
    category       TEXT NOT NULL,
    severity       TEXT NOT NULL DEFAULT 'medium',
    confidence     REAL NOT NULL,
    summary        TEXT NOT NULL,
    evidence       TEXT NOT NULL DEFAULT '[]',
    details        TEXT NOT NULL DEFAULT '{}',
    review_only    INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL,
    adjudication   TEXT NOT NULL DEFAULT 'pending',
    adjudicated_by TEXT,
    adjudicated_at TEXT,
    auto_resolved  INTEGER,
    schema_version TEXT NOT NULL DEFAULT '0.1'
);
CREATE INDEX IF NOT EXISTS idx_flags_severity_created
    ON flags (severity, created_at);
CREATE INDEX IF NOT EXISTS idx_flags_session_id
    ON flags (session_id);
CREATE TABLE IF NOT EXISTS tombstones (
    event_id      TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    ts            TEXT NOT NULL,
    type          TEXT NOT NULL,
    deleted_at    TEXT NOT NULL,
    reason        TEXT NOT NULL,
    payload_digest TEXT NOT NULL
);
"""

_ROW_COLUMNS = ("event_id", "session_id", "seq", "ts", "type", "payload", "refs", "schema_version")

_FLAG_COLUMNS = (
    "flag_id",
    "session_id",
    "event_id",
    "module",
    "module_version",
    "category",
    "severity",
    "confidence",
    "summary",
    "evidence",
    "details",
    "review_only",
    "created_at",
    "adjudication",
    "adjudicated_by",
    "adjudicated_at",
    "auto_resolved",
    "schema_version",
)

#: Columns added after ``S2`` (the ``S3`` flag schema, ADR-0012). A dev database
#: created before ``S3`` is backfilled in place; production Postgres migrates
#: through Alembic ``0002_flags_s3``.
_FLAG_BACKFILL: dict[str, str] = {
    "details": "TEXT NOT NULL DEFAULT '{}'",
    "review_only": "INTEGER NOT NULL DEFAULT 0",
    "schema_version": "TEXT NOT NULL DEFAULT '0.1'",
}

#: SQL ordering by severity rank, shared by every backend's flag query.
SEVERITY_RANK_SQL = (
    "CASE severity WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 "
    "WHEN 'low' THEN 1 ELSE 0 END"
)

#: Static statements for the flag table. ``INSERT OR IGNORE`` is the SQLite
#: spelling of the Postgres store's ``ON CONFLICT DO NOTHING`` (ADR-0012).
#: Written as literals (no caller value is ever interpolated) and kept in
#: lock-step with ``_FLAG_COLUMNS`` by ``test_flag_statements_match_columns``.
_FLAG_INSERT = (
    "INSERT OR IGNORE INTO flags (flag_id, session_id, event_id, module, module_version, "
    "category, severity, confidence, summary, evidence, details, review_only, created_at, "
    "adjudication, adjudicated_by, adjudicated_at, auto_resolved, schema_version) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_FLAG_SELECT = (
    "SELECT flag_id, session_id, event_id, module, module_version, category, severity, "
    "confidence, summary, evidence, details, review_only, created_at, adjudication, "
    "adjudicated_by, adjudicated_at, auto_resolved, schema_version FROM flags"
)

#: Session lifecycle states, kept in lock-step with the ``flags`` severity set.
_SESSION_ACTIVE = "active"
_SESSION_ENDED = "ended"


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
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.executescript(_SCHEMA)
            await self._backfill_flag_columns()
        return self._connection

    async def _backfill_flag_columns(self) -> None:
        """Add post-``S2`` flag columns to a pre-``S3`` dev database.

        The dev store ships its schema with ``CREATE TABLE IF NOT EXISTS`` so a
        file created during ``S2`` keeps working; new columns are added in
        place. Production Postgres uses Alembic ``0002_flags_s3`` instead.
        """
        if self._connection is None:  # pragma: no cover - guarded by caller
            return
        cursor = await self._connection.execute("PRAGMA table_info(flags)")
        present = {row["name"] for row in await cursor.fetchall()}
        for column, ddl in _FLAG_BACKFILL.items():
            if column not in present:
                await self._connection.execute(
                    f"ALTER TABLE flags ADD COLUMN {column} {ddl}"  # nosec B608
                )
        await self._connection.commit()

    # -- writes -----------------------------------------------------------

    async def append(self, event: Event) -> None:
        """Persist *event* (idempotent by ``event_id``, INV-2)."""
        conn = await self._conn()
        async with self._lock:
            cursor = await conn.execute(
                "SELECT 1 FROM events WHERE event_id = ?", (event.event_id,)
            )
            if await cursor.fetchone() is not None:
                return
            await self._write_events(conn, [event], same_session_check=True)

    async def append_batch(self, events: list[Event]) -> None:
        """Persist *events* in one transaction (``S2-T14`` throughput)."""
        conn = await self._conn()
        async with self._lock:
            await self._write_events(conn, events, same_session_check=True)

    async def _write_events(
        self,
        conn: aiosqlite.Connection,
        events: list[Event],
        *,
        same_session_check: bool,
    ) -> None:
        """Insert *events* plus refs and session bookkeeping in one transaction.

        Re-inserting an already-persisted ``event_id`` is skipped (idempotent);
        a *different* event claiming an existing ``(session_id, seq)`` raises
        :class:`sqlite3.IntegrityError`. Referenced events must already exist
        in the same session (INV-3), enforced here and mirrored by the
        ``event_refs`` foreign keys.
        """
        if not events:
            return
        await conn.execute("BEGIN")
        try:
            existing: set[str] = set()
            cursor = await conn.execute(
                # only "?" placeholders are interpolated; event ids are bound
                "SELECT event_id FROM events WHERE event_id IN ("  # noqa: S608  # nosec B608
                + ",".join("?" for _ in events)
                + ")",
                [e.event_id for e in events],
            )
            for row in await cursor.fetchall():
                existing.add(row["event_id"])
            fresh = [e for e in events if e.event_id not in existing]

            if same_session_check and fresh:
                await self._check_same_session_refs(conn, fresh)

            for event in fresh:
                await self._ensure_session(conn, event)
            for event in fresh:
                await conn.execute(
                    "INSERT INTO events "
                    "(event_id, session_id, seq, ts, type, payload, refs, schema_version) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.session_id,
                        event.seq,
                        _ts_to_str(event.ts),
                        event.type,
                        json.dumps(event.payload),
                        json.dumps([link.model_dump() for link in event.refs]),
                        event.schema_version,
                    ),
                )
                for link in event.refs:
                    await conn.execute(
                        "INSERT OR IGNORE INTO event_refs (event_id, ref_event_id, kind) "
                        "VALUES (?, ?, ?)",
                        (event.event_id, link.event_id, link.kind.value),
                    )
                if event.type == SESSION_END:
                    await conn.execute(
                        "UPDATE sessions SET status = ?, ended_at = ? WHERE session_id = ?",
                        (_SESSION_ENDED, _ts_to_str(event.ts), event.session_id),
                    )
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise

    async def _check_same_session_refs(
        self, conn: aiosqlite.Connection, events: list[Event]
    ) -> None:
        """Raise :class:`RefIntegrityError` for dangling or cross-session refs."""
        if not any(event.refs for event in events):
            return
        known = {(event.session_id, event.event_id) for event in events}
        target_ids = sorted({link.event_id for event in events for link in event.refs})
        cursor = await conn.execute(
            # only "?" placeholders are interpolated; ids are bound
            "SELECT session_id, event_id FROM events WHERE event_id IN ("  # noqa: S608  # nosec B608
            + ",".join("?" for _ in target_ids)
            + ")",
            target_ids,
        )
        for row in await cursor.fetchall():
            known.add((row["session_id"], row["event_id"]))
        missing = [
            link.event_id
            for event in events
            for link in event.refs
            if (event.session_id, link.event_id) not in known
        ]
        if missing:
            raise RefIntegrityError(
                "events reference ids not present in their session: "
                + ", ".join(sorted(set(missing)))
            )

    async def _ensure_session(self, conn: aiosqlite.Connection, event: Event) -> None:
        """Insert the session row on first sight (agent id from ``session.start``)."""
        agent_id: str | None = None
        if event.type == SESSION_START:
            agent_id = event.payload.get("agent_id")
        await conn.execute(
            "INSERT OR IGNORE INTO sessions "
            "(session_id, agent_id, status, started_at, schema_version, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                event.session_id,
                agent_id,
                _SESSION_ACTIVE,
                _ts_to_str(event.ts),
                event.schema_version,
                "{}",
            ),
        )

    # -- reads ------------------------------------------------------------

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

    async def iter_session(
        self,
        session_id: str,
        *,
        after_seq: int | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Event]:
        """Stream *session_id*'s events in ``seq`` order (memory bounded)."""
        conn = await self._conn()
        query = (
            "SELECT event_id, session_id, seq, ts, type, payload, refs, schema_version FROM events "
        )
        where = "WHERE session_id = ?"
        params: list[Any] = [session_id]
        if after_seq is not None:
            where += " AND seq > ?"
            params.append(after_seq)
        query += where + " ORDER BY seq ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor = await conn.execute(query, params)
        async for row in cursor:
            yield _row_to_event(row)

    async def get_call_graph(self, session_id: str) -> list[CallEdge]:
        """Return the session's typed reference edges (``S2-T8``)."""
        conn = await self._conn()
        cursor = await conn.execute(
            "SELECT e.session_id, e.event_id AS from_event_id, e.seq AS from_seq, "
            "e.type AS from_type, r.ref_event_id AS to_event_id, "
            "ref.seq AS to_seq, ref.type AS to_type, r.kind "
            "FROM event_refs r "
            "JOIN events e ON e.event_id = r.event_id "
            "JOIN events ref ON ref.event_id = r.ref_event_id "
            "WHERE e.session_id = ? "
            "ORDER BY e.seq, ref.seq",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            CallEdge(
                session_id=row["session_id"],
                from_event_id=row["from_event_id"],
                from_seq=int(row["from_seq"]),
                from_type=row["from_type"],
                to_event_id=row["to_event_id"],
                to_seq=int(row["to_seq"]),
                to_type=row["to_type"],
                kind=RefKind(row["kind"]),
            )
            for row in rows
        ]

    async def list_sessions(
        self,
        *,
        agent_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        has_flags: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SessionSummary]:
        """List sessions (newest first) with event counts and flag presence."""
        conn = await self._conn()
        where: list[str] = []
        params: list[Any] = []
        if agent_id is not None:
            where.append("s.agent_id = ?")
            params.append(agent_id)
        if since is not None:
            where.append("s.started_at >= ?")
            params.append(_ts_to_str(since))
        if until is not None:
            where.append("s.started_at <= ?")
            params.append(_ts_to_str(until))
        if has_flags:
            where.append("EXISTS (SELECT 1 FROM flags f WHERE f.session_id = s.session_id)")
        query = (
            "SELECT s.session_id, s.agent_id, s.status, s.started_at, s.ended_at, "
            "(SELECT count(*) FROM events e WHERE e.session_id = s.session_id) AS event_count, "
            "EXISTS (SELECT 1 FROM flags f WHERE f.session_id = s.session_id) AS has_flags "
            "FROM sessions s"
        )
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY s.started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                agent_id=row["agent_id"],
                status=row["status"],
                started_at=_parse_ts(row["started_at"]),
                ended_at=_parse_ts(row["ended_at"]) if row["ended_at"] else None,
                event_count=int(row["event_count"]),
                has_flags=bool(row["has_flags"]),
            )
            for row in rows
        ]

    # -- ops --------------------------------------------------------------

    async def detect_gaps(self, session_id: str) -> list[SeqGap]:
        """Return the runs of missing ``seq`` values."""
        conn = await self._conn()
        cursor = await conn.execute(
            "SELECT seq FROM events WHERE session_id = ? ORDER BY seq ASC",
            (session_id,),
        )
        seqs = [int(row["seq"]) for row in await cursor.fetchall()]
        return seq_gaps(seqs)

    async def health(self) -> StoreHealth:
        """Snapshot of store health: counts, span, and gap summary."""
        conn = await self._conn()
        counts: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (await (await conn.execute("SELECT count(*) AS n FROM sessions")).fetchone()) or {},
        )
        events: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (await (await conn.execute("SELECT count(*) AS n FROM events")).fetchone()) or {},
        )
        flags: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (await (await conn.execute("SELECT count(*) AS n FROM flags")).fetchone()) or {},
        )
        tombs: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (await (await conn.execute("SELECT count(*) AS n FROM tombstones")).fetchone()) or {},
        )
        span: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (
                await (
                    await conn.execute("SELECT min(ts) AS lo, max(ts) AS hi FROM events")
                ).fetchone()
            )
            or {},
        )
        gap_rows: Mapping[str, Any] = cast(
            Mapping[str, Any],
            (
                await (
                    await conn.execute(
                        """
                        SELECT count(*) AS gap_count,
                               count(DISTINCT session_id) AS sessions_with_gaps
                        FROM (
                            SELECT session_id, seq,
                                   (SELECT max(e2.seq)
                                    FROM events e2
                                    WHERE e2.session_id = e.session_id
                                      AND e2.seq < e.seq) AS prev
                            FROM events e
                        ) t WHERE prev IS NOT NULL AND seq - prev > 1
                        """
                    )
                ).fetchone()
            )
            or {},
        )
        return StoreHealth(
            sessions=int(counts["n"]),
            events=int(events["n"]),
            flags=int(flags["n"]),
            tombstones=int(tombs["n"]),
            oldest_event=_parse_ts(span["lo"]) if span["lo"] else None,
            newest_event=_parse_ts(span["hi"]) if span["hi"] else None,
            sessions_with_gaps=int(gap_rows["sessions_with_gaps"]),
            gap_count=int(gap_rows["gap_count"]),
        )

    async def prune(self, policy: RetentionPolicy) -> PruneReport:
        """Apply a retention *policy*: tombstone then delete, in batches."""
        now = datetime.now(UTC)
        if not policy.enabled:
            return PruneReport(executed_at=now)
        conn = await self._conn()
        held = policy.legal_hold_sessions
        report = PruneReport(executed_at=now, locked_sessions=len(held))
        async with self._lock:
            for event_type, cutoff in merge_cutoffs(policy, now).items():
                last_eid: str | None = None
                last_ts: str | None = None
                while True:
                    type_clause = ""
                    params: list[Any] = [_ts_to_str(cutoff)]
                    if event_type is not None:
                        type_clause = " AND type = ?"
                        params.append(event_type)
                    else:
                        explicit = [r.event_type for r in policy.rules if r.event_type]
                        if explicit:
                            type_clause = (
                                " AND type NOT IN (" + ",".join("?" for _ in explicit) + ")"
                            )
                            params.extend(explicit)
                    if held:
                        type_clause += (
                            " AND session_id NOT IN (" + ",".join("?" for _ in held) + ")"
                        )
                        params.extend(held)
                    # keyset pagination: a batch that is fully retained (flagged)
                    # must still make progress or we would loop forever
                    if last_eid is not None and last_ts is not None:
                        type_clause += " AND (ts > ? OR (ts = ? AND event_id > ?))"
                        params += [last_ts, last_ts, last_eid]
                    cursor = await conn.execute(
                        "SELECT event_id, session_id, seq, ts, type, payload "  # nosec B608
                        "FROM events WHERE ts < ?"
                        + type_clause
                        + " ORDER BY ts, event_id LIMIT 1000",
                        params,
                    )
                    rows = list(await cursor.fetchall())
                    if not rows:
                        break
                    evict: list[sqlite3.Row] = []
                    retained_evidence = 0
                    for row in rows:
                        flagged = await (
                            await conn.execute(
                                "SELECT 1 FROM flags WHERE event_id = ? LIMIT 1",
                                (row["event_id"],),
                            )
                        ).fetchone()
                        if flagged is not None:
                            retained_evidence += 1
                            continue
                        evict.append(row)
                    if evict:
                        report.pruned_events += len(evict)
                        report.tombstoned += len(evict)
                        await conn.executemany(
                            "INSERT OR REPLACE INTO tombstones "
                            "(event_id, session_id, seq, ts, type, deleted_at, reason, "
                            "payload_digest) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [
                                (
                                    row["event_id"],
                                    row["session_id"],
                                    row["seq"],
                                    row["ts"],
                                    row["type"],
                                    _ts_to_str(now),
                                    "retention",
                                    _payload_digest(row["payload"]),
                                )
                                for row in evict
                            ],
                        )
                        ids = [row["event_id"] for row in evict]
                        await conn.executemany(
                            "DELETE FROM event_refs WHERE event_id = ?", [[i] for i in ids]
                        )
                        await conn.executemany(
                            "DELETE FROM event_refs WHERE ref_event_id = ?", [[i] for i in ids]
                        )
                        await conn.executemany(
                            "DELETE FROM events WHERE event_id = ?", [[i] for i in ids]
                        )
                        await conn.commit()
                    report.retained_evidence += retained_evidence
                    last_eid = rows[-1]["event_id"]
                    last_ts = rows[-1]["ts"]
        return report

    # -- evaluator flags (``S3-T1``) -------------------------------------

    async def put_flag(self, flag: Flag) -> bool:
        """Persist *flag*; ``True`` when newly written, ``False`` when known."""
        return await self.put_flags([flag]) == 1

    async def put_flags(self, flags: Sequence[Flag]) -> int:
        """Persist *flags* in one transaction; returns the number of new rows.

        Re-writing a known ``flag_id`` is a no-op: a repeat evaluator run never
        duplicates a finding and never clobbers a human adjudication
        (ADR-0012). The first writer of an id wins, so the stored row is a pure
        function of the (deterministic) flag content.
        """
        if not flags:
            return 0
        conn = await self._conn()
        values = [tuple(_flag_to_row(flag)[column] for column in _FLAG_COLUMNS) for flag in flags]
        async with self._lock:
            await conn.execute("BEGIN")
            try:
                cursor = await conn.executemany(_FLAG_INSERT, values)
                written = int(cursor.rowcount or 0)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise
        return written

    async def get_flags(
        self,
        *,
        session_id: str | None = None,
        module: str | None = None,
        category: str | None = None,
        min_severity: Severity | None = None,
        min_confidence: float | None = None,
        adjudication: Adjudication | None = None,
        review_only: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Flag]:
        """Query flags newest-first, optionally filtered."""
        conn = await self._conn()
        where: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)
        if module is not None:
            where.append("module = ?")
            params.append(module)
        if category is not None:
            where.append("category = ?")
            params.append(category)
        if min_severity is not None:
            where.append(f"{SEVERITY_RANK_SQL} >= ?")
            params.append(min_severity.rank)
        if min_confidence is not None:
            where.append("confidence >= ?")
            params.append(min_confidence)
        if adjudication is not None:
            where.append("adjudication = ?")
            params.append(adjudication.value)
        if review_only is not None:
            where.append("review_only = ?")
            params.append(1 if review_only else 0)
        # Every fragment below is a module-level constant and every value is
        # bound; nothing from the caller is ever interpolated.
        query = _FLAG_SELECT + ((" WHERE " + " AND ".join(where)) if where else "")
        # newest-first, most severe first within the same instant, id as the
        # tie-break so paging is stable and deterministic
        query += " ORDER BY created_at DESC, " + SEVERITY_RANK_SQL + " DESC, flag_id ASC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        cursor = await conn.execute(query, params)
        return [_row_to_flag(row) for row in await cursor.fetchall()]

    async def adjudicate_flag(
        self,
        flag_id: str,
        adjudication: Adjudication,
        *,
        adjudicated_by: str,
        at: datetime | None = None,
    ) -> bool:
        """Record a human decision on *flag_id*; ``False`` when nothing changed.

        First write wins: a decided flag is not re-decided, so a second
        reviewer cannot overwrite the first decision and the row keeps saying
        what the module found. ``False`` means either "no such flag" or "already
        decided"; re-read to tell them apart.
        """
        conn = await self._conn()
        stamp = _ts_to_str(at or datetime.now(UTC))
        async with self._lock:
            cursor = await conn.execute(
                "UPDATE flags SET adjudication = ?, adjudicated_by = ?, adjudicated_at = ? "
                "WHERE flag_id = ? AND adjudication = ?",
                (
                    adjudication.value,
                    adjudicated_by,
                    stamp,
                    flag_id,
                    Adjudication.PENDING.value,
                ),
            )
            changed = cursor.rowcount
            await conn.commit()
        return changed > 0

    async def close(self) -> None:
        """Close the underlying connection and release the file handle."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None


# -- row mapping -----------------------------------------------------------


def _flag_to_row(flag: Flag) -> dict[str, Any]:
    """Flatten a :class:`Flag` into the ``flags`` table's column shape."""
    return {
        "flag_id": flag.flag_id,
        "session_id": flag.session_id,
        "event_id": flag.event_id,
        "module": flag.module,
        "module_version": flag.module_version,
        "category": flag.category,
        "severity": flag.severity.value,
        "confidence": flag.confidence,
        "summary": flag.summary,
        "evidence": json.dumps([ref.to_row() for ref in flag.evidence]),
        "details": json.dumps(flag.details, sort_keys=True, default=str),
        "review_only": 1 if flag.review_only else 0,
        "created_at": _ts_to_str(flag.created_at),
        "adjudication": flag.adjudication.value,
        "adjudicated_by": flag.adjudicated_by,
        "adjudicated_at": _ts_to_str(flag.adjudicated_at) if flag.adjudicated_at else None,
        "auto_resolved": None if flag.auto_resolved is None else int(flag.auto_resolved),
        "schema_version": flag.schema_version,
    }


def _row_to_flag(row: sqlite3.Row) -> Flag:
    raw = dict(row)
    return Flag(
        flag_id=raw["flag_id"],
        session_id=raw["session_id"],
        event_id=raw["event_id"],
        module=raw["module"],
        module_version=raw["module_version"],
        category=raw["category"],
        severity=Severity(raw["severity"]),
        confidence=float(raw["confidence"]),
        summary=raw["summary"],
        evidence=[EvidenceRef.from_row(item) for item in json.loads(raw["evidence"])],
        details=json.loads(raw["details"] or "{}"),
        review_only=bool(raw["review_only"]),
        created_at=_parse_ts(raw["created_at"]),
        adjudication=Adjudication(raw["adjudication"]),
        adjudicated_by=raw["adjudicated_by"],
        adjudicated_at=_parse_ts(raw["adjudicated_at"]) if raw["adjudicated_at"] else None,
        auto_resolved=None if raw["auto_resolved"] is None else bool(raw["auto_resolved"]),
        schema_version=raw["schema_version"],
    )


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        event_id=row["event_id"],
        session_id=row["session_id"],
        seq=int(row["seq"]),
        ts=_parse_ts(row["ts"]),
        type=row["type"],
        payload=json.loads(row["payload"]),
        refs=materialize_refs(json.loads(row["refs"])),
        schema_version=row["schema_version"],
    )


def _ts_to_str(value: datetime) -> str:
    """Normalise to UTC and render ISO-8601 like the envelope requires."""
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.isoformat()


def _parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw)
    if ts.utcoffset() is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _payload_digest(payload: str) -> str:
    """Canonical SHA-256 of a stored JSON payload for tombstone records."""
    return hashlib.sha256(
        json.dumps(json.loads(payload), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# Kept importable for tests that reference it directly.
_refs_to_links = materialize_refs
