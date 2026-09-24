"""Query-layer result types shared by every store backend (``S2-T9``/``S2-T11``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from sentinel.models.events import RefKind
from sentinel.store.gaps import SeqGap


@dataclass(frozen=True)
class SessionSummary:
    """One row of a session listing/search result."""

    session_id: str
    agent_id: str | None
    status: str
    started_at: datetime
    ended_at: datetime | None
    event_count: int
    has_flags: bool = False

    @property
    def duration(self) -> float | None:
        """Session wall-clock duration in seconds, when ended."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class StoreHealth:
    """A point-in-time snapshot of store health (``S2-T11``)."""

    sessions: int
    events: int
    flags: int
    tombstones: int
    oldest_event: datetime | None
    newest_event: datetime | None
    sessions_with_gaps: int = 0
    gap_count: int = 0
    #: Set once a gating worker subscribes (``S7``); ``None`` until then.
    max_worker_lag_seconds: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class CallEdge:
    """A typed reference link between two events (``S2-T8``)."""

    session_id: str
    from_event_id: str
    from_seq: int
    from_type: str
    to_event_id: str
    to_seq: int
    to_type: str
    kind: RefKind


@dataclass
class GapReport:
    """Gap-detection outcome for a single session (``S2-T13``)."""

    session_id: str
    expected_seq_count: int
    present_seq_count: int
    gaps: list[SeqGap] = field(default_factory=list)

    @property
    def gap_free(self) -> bool:
        """True when the session's ``seq`` run is contiguous."""
        return not self.gaps and self.present_seq_count == self.expected_seq_count
