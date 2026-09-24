"""Postgres-backed reference event store (docs/adr/0002, Sprint ``S2``).

Implements the :class:`~sentinel.store.protocol.EventStore` contract via
SQLAlchemy 2.0 async + ``asyncpg`` with Alembic migrations. Properties the
store guarantees:

* **Append-only** â€” events and refs are inserted through ``ON CONFLICT DO
  NOTHING`` on the primary key, so a duplicate ``event_id`` is a no-op never a
  duplicate row (INV-2), while a conflicting ``(session_id, seq)`` still raises.
* **Referential integrity** â€” ``event_refs`` rows carry two foreign keys to
  ``events`` (S2-T4) and the store additionally enforces that a ref's target
  lives in the *same* session (INV-3), mirroring the SQLite store.
* **Lossless under load** â€” :meth:`append_batch` persists in single round
  trips so the 1M-event losslessness gate (``S2-T16``) is achievable.
* **Ops-ready** â€” :meth:`health`, :meth:`detect_gaps`, and :meth:`prune`
  implement the S2 ops surfaces (``S2-T11``/``S2-T12``/``S2-T13``).

The engine is created lazily and the module imports SQLAlchemy/asyncpg only
inside :meth:`__init__`, so the base package keeps working with the SQLite
dev store when the ``postgres`` extra is absent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from sentinel.models.events import (
    SESSION_END,
    SESSION_START,
    Event,
    RefKind,
    RefLink,
)
from sentinel.store.errors import RefIntegrityError
from sentinel.store.gaps import SeqGap, seq_gaps
from sentinel.store.protocol import EventStore
from sentinel.store.reporting import CallEdge, SessionSummary, StoreHealth
from sentinel.store.retention import PruneReport, RetentionPolicy, merge_cutoffs

#: Alias so dynamic SQLAlchemy/asyncpg handles stay explicit yet lint-clean.
_Any = Any

log = logging.getLogger("sentinel.store.postgres")


class PostgresUnavailableError(ImportError):
    """Raised when :class:`PostgresEventStore` needs the ``postgres`` extra."""


class PostgresEventStore(EventStore):
    """Append-only event store backed by PostgreSQL 16+.

    Example::

        store = PostgresEventStore("postgresql://user:pass@localhost:5432/db")
        await store.append(event)
    """

    def __init__(
        self,
        dsn: str,
        *,
        pool_size: int = 10,
        max_overflow: int = 5,
        pool_timeout: float = 30.0,
        connect_timeout: float = 10.0,
        retry_attempts: int = 3,
        retry_jitter_ms: float = 50.0,
    ) -> None:
        """Prepare an async engine for *dsn* (lazy; no connection yet).

        ``pool_size`` / ``max_overflow`` / ``pool_timeout`` tune the
        connection pool (``S2-T14``); transient failures are retried up to
        ``retry_attempts`` times with ``retry_jitter_ms`` of jitter but a
        :class:`RefIntegrityError` or SQL integrity violation is never retried.
        """
        try:
            from sqlalchemy import delete, func, insert, select, text, update
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            from sqlalchemy.ext.asyncio import (
                AsyncEngine,
                AsyncSession,
                async_sessionmaker,
                create_async_engine,
            )
        except ImportError as exc:  # pragma: no cover - exercised by integration
            raise PostgresUnavailableError(
                "PostgresEventStore needs the sentinel-sdk[postgres] extra"
            ) from exc

        self._sa = Any  # placeholder for mypy; never used at runtime
        self._select = select
        self._insert = insert
        self._update = update
        self._delete = delete
        self._func = func
        self._text = text
        self._pg_insert = pg_insert
        self._AsyncEngine: type[AsyncEngine] = AsyncEngine
        self._AsyncSession: type[AsyncSession] = AsyncSession
        self._async_sessionmaker = async_sessionmaker
        self._create_async_engine = create_async_engine

        from sentinel.store import models as _models

        self._models = _models

        self._dsn = _normalize_dsn(dsn)
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._connect_timeout = connect_timeout
        self._retry_attempts = retry_attempts
        self._retry_jitter_ms = retry_jitter_ms
        self._engine: Any | None = None
        self._sessionmaker: Any | None = None

    # -- lifecycle --------------------------------------------------------

    async def _get_engine(self) -> _Any:
        """Return the lazily built async engine."""
        if self._engine is None:
            self._engine = self._create_async_engine(
                self._dsn,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=True,
                connect_args={
                    "timeout": self._connect_timeout,
                    "server_settings": {"application_name": "sentinel"},
                },
            )
            self._sessionmaker = self._async_sessionmaker(
                self._engine, class_=self._AsyncSession, expire_on_commit=False
            )
        return self._engine

    def _session(self) -> _Any:
        """Open an async session, ensuring the engine exists first."""

        async def _open() -> _Any:
            await self._get_engine()
            if self._sessionmaker is None:
                raise RuntimeError("session maker not initialised")
            return self._sessionmaker()

        return _open()

    async def _exec_with_retry(self, op: _Any) -> _Any:
        """Run an async op building its own session, retrying transient errors."""
        attempt = 0
        while True:
            try:
                return await op()
            except Exception as exc:
                transient = _is_transient(exc)
                attempt += 1
                if not transient or attempt >= self._retry_attempts:
                    raise
                # backoff jitter needs spread, not cryptography
                jitter = random.uniform(0.0, self._retry_jitter_ms) / 1000.0  # noqa: S311  # nosec B311
                await asyncio.sleep(0.05 * attempt + jitter)
                log.warning(
                    "store.retry attempt=%d max=%d error=%r",
                    attempt,
                    self._retry_attempts,
                    exc,
                )

    # -- writes -----------------------------------------------------------

    async def append(self, event: Event) -> None:
        """Persist *event* (idempotent by ``event_id``, INV-2)."""

        async def op() -> None:
            async with await self._session() as session:
                await self._append_events(session, [event])

        await self._exec_with_retry(op)

    async def append_batch(self, events: list[Event]) -> None:
        """Persist *events* in a single round trip (``S2-T14`` throughput)."""

        async def op() -> None:
            async with await self._session() as session:
                await self._append_events(session, events)

        await self._exec_with_retry(op)

    async def _append_events(self, session: _Any, events: list[Event]) -> None:
        """Insert events + refs + session bookkeeping in one transaction."""
        models = self._models
        EventRecord = models.EventRecord
        EventRefRecord = models.EventRefRecord
        SessionRecord = models.SessionRecord

        session_ids = {event.session_id for event in events}
        for session_id in session_ids:
            await self._ensure_session(session, events, session_id)

        targets = [(event.session_id, link.event_id) for event in events for link in event.refs]
        if targets:
            await self._check_same_session_refs(session, targets, events)

        if events:
            # asyncpg caps a single multi-row insert at 32767 parameters;
            # chunk so arbitrarily large append batches stay well under it
            for chunk in _chunks(events, 1000):
                await session.execute(
                    self._pg_insert(EventRecord)
                    .values(
                        [
                            {
                                "event_id": event.event_id,
                                "session_id": event.session_id,
                                "seq": event.seq,
                                "ts": event.ts,
                                "type": event.type,
                                "payload": event.payload,
                                "schema_version": event.schema_version,
                            }
                            for event in chunk
                        ]
                    )
                    .on_conflict_do_nothing(index_elements=[EventRecord.event_id])
                )

        ref_rows = [
            {
                "event_id": event.event_id,
                "ref_event_id": link.event_id,
                "kind": link.kind.value,
            }
            for event in events
            for link in event.refs
        ]
        if ref_rows:
            for chunk in _chunks(ref_rows, 1000):
                await session.execute(
                    self._pg_insert(EventRefRecord)
                    .values(chunk)
                    .on_conflict_do_nothing(index_elements=["event_id", "ref_event_id", "kind"])
                )

        for event in events:
            if event.type == SESSION_END:
                await session.execute(
                    self._update(SessionRecord)
                    .where(SessionRecord.session_id == event.session_id)
                    .values(status="ended", ended_at=event.ts)
                )

        await session.commit()

    async def _ensure_session(self, session: _Any, events: list[Event], session_id: str) -> None:
        """Create the session row on first sight, using the earliest event."""
        SessionRecord = self._models.SessionRecord
        earliest = min((e for e in events if e.session_id == session_id), key=lambda e: e.seq)
        agent_id: str | None = None
        if earliest.type == SESSION_START:
            agent_id = earliest.payload.get("agent_id")
        await session.execute(
            self._pg_insert(SessionRecord)
            .values(
                session_id=session_id,
                agent_id=agent_id,
                status="active",
                started_at=earliest.ts,
                schema_version=earliest.schema_version,
                meta={},
            )
            .on_conflict_do_nothing(index_elements=[SessionRecord.session_id])
        )

    async def _check_same_session_refs(
        self,
        session: _Any,
        targets: list[tuple[str, str]],
        events: list[Event],
    ) -> None:
        """Ensure every ref target exists in the *referencing* session (INV-3).

        The referenced events may be earlier commits or events inserted in this
        very batch, so the known set is the batch itself plus the database.
        """
        EventRecord = self._models.EventRecord

        known = {(event.session_id, event.event_id) for event in events}
        target_ids = sorted({eid for _, eid in targets})
        if target_ids:
            rows = await session.execute(
                self._select(EventRecord.session_id, EventRecord.event_id).where(
                    EventRecord.event_id.in_(target_ids)
                )
            )
            known.update((session_id, event_id) for session_id, event_id in rows.all())

        missing: list[str] = []
        for session_id, event_id in targets:
            if (session_id, event_id) not in known:
                missing.append(event_id)
        if missing:
            raise RefIntegrityError(
                "events reference ids not present in their session: "
                + ", ".join(sorted(set(missing)))
            )

    # -- reads ------------------------------------------------------------

    async def get_session(self, session_id: str) -> list[Event]:
        """Return all events for *session_id* in ``seq`` order."""
        return [event async for event in self.iter_session(session_id)]

    async def iter_session(
        self,
        session_id: str,
        *,
        after_seq: int | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Event]:
        """Stream the session's events without materialising the whole log."""
        models = self._models
        EventRecord = models.EventRecord
        EventRefRecord = models.EventRefRecord

        async with await self._session() as session:
            stmt = (
                self._select(EventRecord)
                .where(EventRecord.session_id == session_id)
                .order_by(EventRecord.seq.asc())
            )
            if after_seq is not None:
                stmt = stmt.where(EventRecord.seq > after_seq)
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.stream(stmt)
            async for row in result:
                record = row[0]
                ref_q = await session.execute(
                    self._select(EventRefRecord.ref_event_id, EventRefRecord.kind)
                    .where(EventRefRecord.event_id == record.event_id)
                    .order_by(EventRefRecord.id.asc())
                )
                refs = [
                    RefLink(event_id=ref_event_id, kind=kind) for ref_event_id, kind in ref_q.all()
                ]
                yield _record_to_event(record, refs)

    async def get_call_graph(self, session_id: str) -> list[CallEdge]:
        """Return the session's typed reference edges (``S2-T8``)."""
        async with await self._session() as session:
            rows = (
                await session.execute(
                    self._text(
                        """
                        SELECT
                          e.session_id,
                          e.event_id            AS from_event_id,
                          e.seq                 AS from_seq,
                          e.type                AS from_type,
                          r.ref_event_id        AS to_event_id,
                          ref.seq               AS to_seq,
                          ref.type              AS to_type,
                          r.kind
                        FROM event_refs r
                        JOIN events e    ON e.event_id = r.event_id
                        JOIN events ref  ON ref.event_id = r.ref_event_id
                        WHERE e.session_id = :sid
                        ORDER BY e.seq, ref.seq
                        """
                    ),
                    {"sid": session_id},
                )
            ).all()
        return [
            CallEdge(
                session_id=row.session_id,
                from_event_id=row.from_event_id,
                from_seq=int(row.from_seq),
                from_type=row.from_type,
                to_event_id=row.to_event_id,
                to_seq=int(row.to_seq),
                to_type=row.to_type,
                kind=_ref_kind(row.kind),
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
        models = self._models
        SessionRecord = models.SessionRecord
        EventCount = self._func.count(models.EventRecord.event_id).label("event_count")
        HasFlags = (
            self._select(1).where(models.FlagRecord.session_id == SessionRecord.session_id).exists()
        ).label("has_flags")

        stmt = (
            self._select(
                SessionRecord.session_id,
                SessionRecord.agent_id,
                SessionRecord.status,
                SessionRecord.started_at,
                SessionRecord.ended_at,
                EventCount,
                HasFlags,
            )
            .outerjoin(
                models.EventRecord,
                models.EventRecord.session_id == SessionRecord.session_id,
            )
            .group_by(SessionRecord.session_id)
            .order_by(SessionRecord.started_at.desc())
        )
        if agent_id is not None:
            stmt = stmt.where(SessionRecord.agent_id == agent_id)
        if since is not None:
            stmt = stmt.where(SessionRecord.started_at >= since)
        if until is not None:
            stmt = stmt.where(SessionRecord.started_at <= until)
        if has_flags:
            stmt = stmt.where(
                self._select(1)
                .where(models.FlagRecord.session_id == SessionRecord.session_id)
                .exists()
            )
        stmt = stmt.limit(limit).offset(offset)

        async with await self._session() as session:
            rows = (await session.execute(stmt)).all()
        return [
            SessionSummary(
                session_id=session_id,
                agent_id=agent_id,
                status=status,
                started_at=_as_utc(started_at),
                ended_at=_as_utc(ended_at) if ended_at is not None else None,
                event_count=int(event_count),
                has_flags=bool(has_flags),
            )
            for (
                session_id,
                agent_id,
                status,
                started_at,
                ended_at,
                event_count,
                has_flags,
            ) in rows
        ]

    # -- ops --------------------------------------------------------------

    async def detect_gaps(self, session_id: str) -> list[SeqGap]:
        """Return the runs of missing ``seq`` values for one session."""
        EventRecord = self._models.EventRecord
        seqs: list[int] = []
        async with await self._session() as session:
            result = await session.execute(
                self._select(EventRecord.seq)
                .where(EventRecord.session_id == session_id)
                .order_by(EventRecord.seq.asc())
            )
            for (seq,) in result.all():
                seqs.append(int(seq))
        return seq_gaps(seqs)

    async def health(self) -> StoreHealth:
        """Snapshot of store health: counts, span, and gap summary."""
        async with await self._session() as session:
            counts = (
                await session.execute(
                    self._text(
                        """
                        SELECT
                          (SELECT count(*) FROM sessions)        AS sessions_n,
                          (SELECT count(*) FROM events)          AS events_n,
                          (SELECT count(*) FROM flags)           AS flags_n,
                          (SELECT count(*) FROM tombstones)      AS tombstones_n,
                          (SELECT min(ts) FROM events)           AS oldest_ts,
                          (SELECT max(ts) FROM events)           AS newest_ts
                        """
                    )
                )
            ).one()
            gap_rows = (
                await session.execute(
                    self._text(
                        """
                        WITH ordered AS (
                            SELECT session_id,
                                   seq AS s,
                                   LAG(seq) OVER (PARTITION BY session_id ORDER BY seq) AS prev
                            FROM events
                        )
                        SELECT count(*) AS gap_count,
                               count(DISTINCT session_id) AS sessions_with_gaps
                        FROM ordered
                        WHERE prev IS NOT NULL AND s - prev > 1
                        """
                    )
                )
            ).one()
        return StoreHealth(
            sessions=int(counts.sessions_n),
            events=int(counts.events_n),
            flags=int(counts.flags_n),
            tombstones=int(counts.tombstones_n),
            oldest_event=_as_utc(counts.oldest_ts) if counts.oldest_ts is not None else None,
            newest_event=_as_utc(counts.newest_ts) if counts.newest_ts is not None else None,
            sessions_with_gaps=int(gap_rows.sessions_with_gaps),
            gap_count=int(gap_rows.gap_count),
        )

    async def prune(self, policy: RetentionPolicy) -> PruneReport:
        """Apply a retention *policy*: tombstone then delete, in batches."""
        from sentinel.store.retention import PruneReport as Report

        now = datetime.now(UTC)
        if not policy.enabled:
            return Report(executed_at=now)

        models = self._models
        EventRecord = models.EventRecord
        EventRefRecord = models.EventRefRecord
        TombstoneRecord = models.TombstoneRecord

        report = Report(
            executed_at=now,
            locked_sessions=len(policy.legal_hold_sessions),
        )
        async with await self._session() as session:
            from sqlalchemy import and_, or_

            for event_type, cutoff in merge_cutoffs(policy, now).items():
                last_eid: str | None = None
                last_ts: datetime | None = None
                while True:
                    candidate_stmt = self._select(
                        EventRecord.event_id,
                        EventRecord.session_id,
                        EventRecord.seq,
                        EventRecord.ts,
                        EventRecord.type,
                        EventRecord.payload,
                    ).where(
                        EventRecord.ts < cutoff,
                        EventRecord.session_id.notin_(policy.legal_hold_sessions),
                    )
                    if event_type is not None:
                        candidate_stmt = candidate_stmt.where(EventRecord.type == event_type)
                    else:
                        explicit = [r.event_type for r in policy.rules if r.event_type]
                        if explicit:
                            candidate_stmt = candidate_stmt.where(EventRecord.type.notin_(explicit))
                    # keyset pagination: a batch that is fully retained (flagged)
                    # must still make progress or we would loop forever
                    if last_eid is not None and last_ts is not None:
                        candidate_stmt = candidate_stmt.where(
                            or_(
                                EventRecord.ts > last_ts,
                                and_(
                                    EventRecord.ts == last_ts,
                                    EventRecord.event_id > last_eid,
                                ),
                            )
                        )
                    candidates = (
                        await session.execute(
                            candidate_stmt.order_by(
                                EventRecord.ts.asc(), EventRecord.event_id.asc()
                            ).limit(1000)
                        )
                    ).all()
                    if not candidates:
                        break

                    retained_evidence = 0
                    evict: list[tuple[str, str, int, datetime, str, dict[str, Any]]] = []
                    for (
                        event_id,
                        session_id,
                        seq,
                        ts,
                        type_,
                        payload,
                    ) in candidates:
                        flagged = (
                            await session.scalar(
                                self._select(models.FlagRecord.flag_id).where(
                                    models.FlagRecord.event_id == event_id
                                )
                            )
                        ) is not None
                        if flagged:
                            retained_evidence += 1
                            continue
                        evict.append((event_id, session_id, seq, ts, type_, payload))

                    if evict:
                        report.pruned_events += len(evict)
                        report.tombstoned += len(evict)
                        await session.execute(
                            self._insert(TombstoneRecord),
                            [
                                {
                                    "event_id": eid,
                                    "session_id": sid,
                                    "seq": seq,
                                    "ts": ts,
                                    "type": type_,
                                    "deleted_at": now,
                                    "reason": "retention",
                                    "payload_digest": hashlib.sha256(
                                        _json_bytes(payload)
                                    ).hexdigest(),
                                }
                                for eid, sid, seq, ts, type_, payload in evict
                            ],
                        )
                        evict_ids = [e[0] for e in evict]
                        await session.execute(
                            self._delete(EventRefRecord).where(
                                EventRefRecord.event_id.in_(evict_ids)
                            )
                        )
                        await session.execute(
                            self._delete(EventRefRecord).where(
                                EventRefRecord.ref_event_id.in_(evict_ids)
                            )
                        )
                        await session.execute(
                            self._delete(EventRecord).where(EventRecord.event_id.in_(evict_ids))
                        )
                    report.retained_evidence += retained_evidence
                    last_eid = candidates[-1][0]
                    last_ts = candidates[-1][3]
                await session.commit()
        return report

    async def close(self) -> None:
        """Dispose the engine, releasing every pooled connection."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None


# -- helpers ---------------------------------------------------------------


def _chunks(seq: list[Any], size: int) -> list[list[Any]]:
    """Yield ``seq`` in ``size``-sized slices (for asyncpg parameter limits)."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _normalize_dsn(dsn: str) -> str:
    """Accept ``postgresql://`` and ``postgresql+asyncpg://`` DSNs alike."""
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if dsn.startswith("postgres://"):
        return dsn.replace("postgres://", "postgresql+asyncpg://", 1)
    parsed = urlparse(dsn)
    if parsed.scheme:
        raise ValueError(f"unsupported DSN scheme: {parsed.scheme!r}")
    raise ValueError(f"malformed database DSN: {dsn!r}")


def _as_utc(value: datetime) -> datetime:
    """Normalise any aware datetime to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _ref_kind(value: str) -> RefKind:
    """Parse a stored kind string back into a :class:`RefKind`."""
    return RefKind(value)


def _record_to_event(record: _Any, refs: list[RefLink]) -> Event:
    """Rebuild an :class:`Event` from a model row plus its resolved refs."""
    return Event(
        event_id=record.event_id,
        session_id=record.session_id,
        seq=int(record.seq),
        ts=_as_utc(record.ts),
        type=record.type,
        payload=dict(record.payload) if record.payload else {},
        refs=refs,
        schema_version=record.schema_version,
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Deterministic payload bytes for a tombstone digest."""
    import json

    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _is_transient(exc: BaseException) -> bool:
    """Classify DB errors as transient (retryable) or not.

        Integrity/constraint/subquery violations are *never* transient -- retrying
        an append-only violation would mask a real bug. Network and connection
    errors are.
    """
    from sqlalchemy.exc import DBAPIError, IntegrityError

    if isinstance(exc, IntegrityError):
        return False
    if isinstance(exc, DBAPIError):
        code = getattr(exc.orig, "sqlstate", None) or ""
        return not code.startswith(("23", "22"))
    return isinstance(exc, (OSError, asyncio.TimeoutError))
