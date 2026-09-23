"""The ``EventStore`` protocol every backend implements (docs/adr/0002).

Sprint ``S0-T5`` defines the minimal surface the vertical slice needs: append one
event and replay an entire session in ``seq`` order. ``S2`` widens this to the
full production store (queries, call graph, retention).
"""

from __future__ import annotations

from typing import Protocol

from sentinel.models.events import Event


class EventStore(Protocol):
    """Append-only persistence for :class:`~sentinel.models.events.Event`."""

    async def append(self, event: Event) -> None:
        """Persist *event*.

        Must be idempotent by ``event_id`` and raise on a conflicting
        ``(session_id, seq)``.
        """

    async def get_session(self, session_id: str) -> list[Event]:
        """Return all events for *session_id* ordered by ``seq`` ascending."""

    async def close(self) -> None:
        """Release any underlying resources."""
