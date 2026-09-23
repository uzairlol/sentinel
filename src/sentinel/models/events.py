"""The Sentinel event envelope (schema version 0.2).

Defined in sprint ``S0-T1`` as the *thin* envelope that carries one boundary
crossing into the event store, and extended in ``S1-T4/T5`` with the full event
taxonomy and typed reference links (INV-3):

* ``event_id`` — an ULID unique across the store (docs/adr/0010).
* ``session_id`` — the owning capture session.
* ``seq`` — a per-session monotonic integer.
* ``ts`` — a normalized UTC timestamp.
* ``type`` — one of the documented event types (see
  ``#S1-T17`` for the full schema catalogue).
* ``payload`` — an opaque, schema-versioned body for the boundary crossing.
* ``refs`` — a list of :class:`RefLink` links to earlier events in the same
  session. Each link has a *kind*: ``parent`` (an enclosing activity),
  ``caused_by`` (this event was produced while handling that event), or
  ``grounds`` (evidence a later claim may cite).

The schema is versioned (docs/adr/0007); ``0.1`` (flat ULID refs) is readable
but no longer produced, ``0.2`` introduced the typed links.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from ulid import ULID

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The envelope schema version emitted by this package.
SCHEMA_VERSION: Literal["0.2"] = "0.2"

#: Event types introduced by sprint S0.
LLM_REQUEST = "llm.request"
LLM_RESPONSE = "llm.response"
SESSION_START = "session.start"
SESSION_END = "session.end"

#: Event types introduced by sprint S1 (full taxonomy, ``S1-T4``).
TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
MEMORY_READ = "memory.read"
MEMORY_WRITE = "memory.write"
AGENT_STEP = "agent.step"
ERROR = "error"
CAPTURE_DROPPED = "capture.dropped"

#: All event types currently known to the SDK. ``Event.type`` is constrained to
#: this set; evolving the taxonomy requires a schema deliberation, not a silent
#: new string.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        LLM_REQUEST,
        LLM_RESPONSE,
        SESSION_START,
        SESSION_END,
        TOOL_CALL,
        TOOL_RESULT,
        MEMORY_READ,
        MEMORY_WRITE,
        AGENT_STEP,
        ERROR,
        CAPTURE_DROPPED,
    }
)

#: Pydantic type for the schema version literal.
SchemaVersion = Literal["0.2"]


class RefKind(StrEnum):
    """How one event links to another (``S1-T5``, INV-3)."""

    PARENT = "parent"
    CAUSED_BY = "caused_by"
    GROUNDS = "grounds"


class RefLink(BaseModel):
    """A typed reference to an earlier event in the same session."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: RefKind

    @field_validator("event_id")
    @classmethod
    def _require_ulid_event_id(cls, value: str) -> str:
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ref event_id must be a valid ULID, got {value!r}") from exc
        return value


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
    refs: list[RefLink] = Field(default_factory=list)
    schema_version: SchemaVersion = SCHEMA_VERSION

    @field_validator("event_id")
    @classmethod
    def _require_ulid_event_id(cls, value: str) -> str:
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"event_id must be a valid ULID, got {value!r}") from exc
        return value

    @field_validator("type")
    @classmethod
    def _require_known_type(cls, value: str) -> str:
        if value not in EVENT_TYPES:
            raise ValueError(f"type must be one of {sorted(EVENT_TYPES)}, got {value!r}")
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
    def _require_no_self_refs(cls, value: list[RefLink], info: ValidationInfo) -> list[RefLink]:
        own_id = info.data.get("event_id")
        for link in value:
            if own_id is not None and link.event_id == own_id:
                raise ValueError("refs must not reference the event itself")
        return value


def make_event(
    *,
    session_id: str,
    seq: int,
    type: str,  # noqa: A002
    payload: dict[str, Any] | None = None,
    refs: Iterable[RefLink | str] = (),
    ts: datetime | None = None,
) -> Event:
    """Build an ``Event`` with a fresh ``event_id`` and normalized UTC timestamp.

    Handles the bookkeeping the envelope requires (ULID id, UTC time) so call
    sites only supply the semantics. Bare string refs are interpreted as
    ``parent`` links for call-site convenience.
    """
    event_ts = ts or datetime.now(UTC)
    ref_links: list[RefLink] = [
        link if isinstance(link, RefLink) else RefLink(event_id=link, kind=RefKind.PARENT)
        for link in refs
    ]
    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=event_ts,
        type=type,
        payload=payload or {},
        refs=ref_links,
    )
