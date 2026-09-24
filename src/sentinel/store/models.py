"""SQLAlchemy 2.0 declarative models for the Postgres event store (``S2-T1``).

Single source of truth for the schema that the Alembic migration
(``alembic/versions/0001_initial.py``) creates:

* ``sessions`` — one row per capture session; created lazily on first event and
  updated by the ``session.start`` / ``session.end`` bookends.
* ``events`` — the append-only event log. ``seq`` is per-session monotonic and
  the ``(session_id, seq)`` pair is unique (``S2-T4``).
* ``event_refs`` — the normalised reference links (INV-3). Every row carries
  two foreign keys back to ``events``, so a link can never dangle.
* ``flags`` — evaluator flags; the ``S3`` evaluators consume this table.
* ``schema_meta`` — schema-version bookkeeping (docs/adr/0007).
* ``tombstones`` — audit records for retention deletions (INV-2): rows are
  never silently removed, they are tombstoned and then physically deleted.

Constraints and indexes from ``S2-T4`` / ``S2-T5`` are declared here so
``Base.metadata`` produced by the migration exactly matches the models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: ULIDs are 26-character Crockford base32 strings.
ID_LENGTH = 26


class Base(DeclarativeBase):
    """Declarative base for every store table."""


class SessionRecord(Base):
    """One capture session, created on demand by the store."""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    agent_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="active")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )

    #: Only ``active`` / ``ended`` are legal session lifecycle states.
    __table_args__ = (
        CheckConstraint("status IN ('active', 'ended')", name="ck_sessions_status"),
        Index("ix_sessions_started_at", "started_at"),
    )


class EventRecord(Base):
    """One append-only boundary-crossing event."""

    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", name="fk_events_session"), nullable=False
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        #: One ``seq`` slot per session: replay ordering is total (S0-T6).
        UniqueConstraint("session_id", "seq", name="uq_events_session_seq"),
        #: seq numbers are never negative (the session counter starts at 0).
        CheckConstraint("seq >= 0", name="ck_events_seq_nonnegative"),
        #: The (session_id, seq) backing index serves session replay.
        Index("ix_events_session_type_ts", "session_id", "type", "ts"),
        #: Retention scans by time (S2-T12).
        Index("ix_events_ts", "ts"),
    )


class EventRefRecord(Base):
    """A typed reference link (INV-3) between two persisted events."""

    __tablename__ = "event_refs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("events.event_id", name="fk_event_refs_event", ondelete="CASCADE"),
        nullable=False,
    )
    ref_event_id: Mapped[str] = mapped_column(
        ForeignKey("events.event_id", name="fk_event_refs_ref", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        #: Re-appending the same link is a no-op, never a duplicate.
        UniqueConstraint("event_id", "ref_event_id", "kind", name="uq_event_refs_reference"),
        #: The call graph resolves reverse edges by walking this index.
        Index("ix_event_refs_ref_event_id", "ref_event_id"),
    )


class FlagRecord(Base):
    """An evaluator flag (schema-forward compatible with the ``S3`` evaluators)."""

    __tablename__ = "flags"

    flag_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", name="fk_flags_session"), nullable=False
    )
    event_id: Mapped[str | None] = mapped_column(
        ForeignKey("events.event_id", name="fk_flags_event", ondelete="SET NULL"),
        nullable=True,
    )
    module: Mapped[str] = mapped_column(String(64), nullable=False)
    module_version: Mapped[str] = mapped_column(String(32), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, server_default="medium")
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    adjudication: Mapped[str] = mapped_column(String(16), nullable=False, server_default="pending")
    adjudicated_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    adjudicated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    auto_resolved: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "severity IN ('info', 'low', 'medium', 'high', 'critical')",
            name="ck_flags_severity",
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_flags_confidence"),
        CheckConstraint(
            "adjudication IN ('pending', 'confirmed', 'rejected')",
            name="ck_flags_adjudication",
        ),
        #: The gate's ordering index (flag.severity, flag.created_at).
        Index("ix_flags_severity_created_at", "severity", "created_at"),
        Index("ix_flags_session_id", "session_id"),
    )


class SchemaMetaRecord(Base):
    """Key/value schema bookkeeping (docs/adr/0007)."""

    __tablename__ = "schema_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TombstoneRecord(Base):
    """Audit record for a physically deleted event (INV-2, ``S2-T12``)."""

    __tablename__ = "tombstones"

    event_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.session_id", name="fk_tombstones_session"), nullable=False
    )
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_digest: Mapped[str] = mapped_column(String(64), nullable=False)
