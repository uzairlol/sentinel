"""The ``EventStore`` protocol every backend implements (docs/adr/0002).

Sprint ``S0`` defined the minimal surface (append one event, replay a session).
Sprint ``S2`` widens the contract to the production store: batched append for
throughput, streaming iteration for large sessions, session listing/search,
health, gap detection, and retention pruning.

Every backend (SQLite and Postgres) passes the same contract tests
(``tests/contract/test_store_parity.py``), so evaluators the store supports
never depend on which backend is mounted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from sentinel.models.events import Event
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

    async def close(self) -> None:
        """Release any underlying resources."""


__all__ = [
    "AsyncIterator",
    "EventStore",
    "Iterable",
]
