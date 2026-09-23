"""The Sentinel event envelope (schema version 0.1).

Defined in sprint ``S0-T1`` as the *thin* envelope that carries one boundary
crossing into the event store: an ULID ``event_id``, the owning ``session_id``,
a per-session monotonic ``seq``, a UTC ``ts``, a free-form ``type``, an opaque
``payload``, and ``refs`` linking this event to the event(s) that caused it
(INV-3). The schema is versioned so readers can keep supporting older minors
(docs/adr/0007).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ulid import ULID

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The envelope schema version emitted by this package.
SCHEMA_VERSION: Literal["0.1"] = "0.1"

#: Event types introduced by sprint S0 (the full taxonomy lands in S1).
LLM_REQUEST = "llm.request"
LLM_RESPONSE = "llm.response"
SESSION_START = "session.start"
SESSION_END = "session.end"

#: All event types currently known to the SDK.
EVENT_TYPES: frozenset[str] = frozenset({LLM_REQUEST, LLM_RESPONSE, SESSION_START, SESSION_END})

#: Pydantic type for the schema version literal.
SchemaVersion = Literal["0.1"]


def new_event_id() -> str:
    """Return a fresh ULID suitable for an ``event_id`` (docs/adr/0010)."""
    return str(ULID())


class Event(BaseModel):
    """An immutable, sequence-numbered record of one boundary crossing."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    session_id: str
    seq: int
    ts: datetime
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    refs: list[str] = Field(default_factory=list)
    schema_version: SchemaVersion = SCHEMA_VERSION

    @field_validator("event_id")
    @classmethod
    def _require_ulid_event_id(cls, value: str) -> str:
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"event_id must be a valid ULID, got {value!r}") from exc
        return value

    @field_validator("seq")
    @classmethod
    def _require_positive_seq(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"seq must be >= 0, got {value}")
        return value

    @field_validator("ts")
    @classmethod
    def _require_utc_ts(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ts must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("ts must be normalized to UTC (offset 0)")
        return value

    @field_validator("refs")
    @classmethod
    def _require_ulid_refs(cls, value: list[str]) -> list[str]:
        for ref in value:
            try:
                ULID.from_str(ref)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"refs must contain ULIDs, got {ref!r}") from exc
        return value


def make_event(
    *,
    session_id: str,
    seq: int,
    type: str,  # noqa: A002
    payload: dict[str, Any] | None = None,
    refs: Iterable[str] = (),
    ts: datetime | None = None,
) -> Event:
    """Build an ``Event`` with a fresh ``event_id`` and normalized UTC timestamp.

    Handles the bookkeeping the envelope requires (ULID id, UTC time) so call
    sites only supply the semantics.
    """
    event_ts = ts or datetime.now(UTC)
    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=event_ts,
        type=type,
        payload=payload or {},
        refs=list(refs),
    )
