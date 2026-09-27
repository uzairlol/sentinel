"""The ``EventStore`` protocol every backend implements (docs/adr/0002).

Sprint ``S0`` defined the minimal surface (append one event, replay a session).
Sprint ``S2`` widens the contract to the production store: batched append for
throughput, streaming iteration for large sessions, session listing/search,
health, gap detection, and retention pruning. Sprint ``S3-T1`` adds the
evaluator *flag* surface, so the flag schema is persisted by the same
append-only contract and read back for the gate and the review queue.

Every backend (SQLite and Postgres) passes the same contract tests
(``tests/contract/test_store_parity.py``), so evaluators the store supports
never depend on which backend is mounted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from sentinel.models.events import Event
from sentinel.models.flags import Adjudication, Flag, Severity
from sentinel.store.gaps import SeqGap
from sentinel.store.reporting import CallEdge, SessionSummary, StoreHealth
from sentinel.store.retention import PruneReport, RetentionPolicy


@runtime_checkable
class EventStore(Protocol):
    """Append-only persistence for :class:`~sentinel.models.events.Event`."""

    async def append(self, event: Event) -> None:
        """Persist *event*.

        Must be idempotent by ``event_id`` and raise on a conflicting
        ``(session_id, seq)``.
        """

    async def append_batch(self, events: list[Event]) -> None:
        """Persist a batch of events in one round trip, best-effort.

        Individual idempotency is preserved (re-appending an already-seen
        ``event_id`` is a no-op), and a reference-integrity violation aborts
        the batch with :class:`RefIntegrityError`.
        """

    async def get_session(self, session_id: str) -> list[Event]:
        """Return all events for *session_id* ordered by ``seq`` ascending."""

    def iter_session(
        self,
        session_id: str,
        *,
        after_seq: int | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[Event]:
        """Stream *session_id*'s events in ``seq`` order without materialising.

        ``after_seq`` resumes after a sequence number (used for replay
        resumes); ``limit`` bounds the number of events returned. Backends
        implement this as an async generator, so callers ``async for`` over it
        directly (never ``await``).
        """

    async def get_call_graph(self, session_id: str) -> list[CallEdge]:
        """Return the session's typed reference edges (``S2-T8``).

        Edges are sorted by the referring event's ``seq`` then the target's
        ``seq``; each edge carries the ``kind`` of the link (parent,
        caused_by, grounds).
        """

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
        """List sessions, optionally filtered, newest first."""

    async def detect_gaps(self, session_id: str) -> list[SeqGap]:
        """Return the runs of missing ``seq`` numbers in *session_id*."""

    async def health(self) -> StoreHealth:
        """Snapshot of store health (counts, newest/oldest, gap summary)."""

    async def prune(self, policy: RetentionPolicy) -> PruneReport:
        """Apply a retention *policy*: tombstone then delete expired events."""

    # -- evaluator flags (``S3-T1``) -------------------------------------

    async def put_flag(self, flag: Flag) -> bool:
        """Persist one evaluator *flag*; idempotent by ``flag_id``.

        Returns ``True`` when the row was newly written and ``False`` when an
        identical ``flag_id`` already exists. A re-run of an evaluator with the
        same module version therefore never duplicates a finding, and never
        overwrites a human adjudication (ADR-0012).
        """

    async def put_flags(self, flags: Sequence[Flag]) -> int:
        """Persist a batch of flags in one round trip; returns rows written."""

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
        """Query flags newest-first, optionally filtered (gate + review reads)."""

    async def adjudicate_flag(
        self,
        flag_id: str,
        adjudication: Adjudication,
        *,
        adjudicated_by: str,
        at: datetime | None = None,
    ) -> bool:
        """Record a human decision on *flag_id*; ``False`` when not found.

        This is the one mutating flag operation and it needs a credential with
        ``UPDATE`` on ``flags`` (the ``sentinel_reviewer`` role), never the
        append-only writer credential (INV-2).
        """

    async def close(self) -> None:
        """Release any underlying resources."""


__all__ = [
    "Adjudication",
    "AsyncIterator",
    "EventStore",
    "Flag",
    "Iterable",
    "Severity",
]
