"""Initial event store schema.

Sprint ``S2-T1``/``S2-T2``. Creates ``sessions``, ``events``, ``event_refs``,
``flags``, ``schema_meta`` and ``tombstones`` with the constraints and indexes
declared in :mod:`sentinel.store.models` (``S2-T4``/``S2-T5``), and seeds the
``schema_meta`` row recording the envelope schema version (docs/adr/0007).

Revision ID: 0001
Revises:
Create Date: 2026-09-24

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create every store table and seed the schema-version row."""
    op.create_table(
        "sessions",
        sa.Column("session_id", sa.String(length=26), primary_key=True),
        sa.Column("agent_id", sa.String(length=256), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.CheckConstraint("status IN ('active', 'ended')", name="ck_sessions_status"),
    )
    op.create_index("ix_sessions_started_at", "sessions", ["started_at"])

    op.create_table(
        "events",
        sa.Column("event_id", sa.String(length=26), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=26),
            sa.ForeignKey("sessions.session_id", name="fk_events_session"),
            nullable=False,
        ),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("schema_version", sa.String(length=16), nullable=False),
        sa.CheckConstraint("seq >= 0", name="ck_events_seq_nonnegative"),
        sa.UniqueConstraint("session_id", "seq", name="uq_events_session_seq"),
    )
    op.create_index("ix_events_session_type_ts", "events", ["session_id", "type", "ts"])
    op.create_index("ix_events_ts", "events", ["ts"])

    op.create_table(
        "event_refs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, primary_key=True),
        sa.Column(
            "event_id",
            sa.String(length=26),
            sa.ForeignKey("events.event_id", name="fk_event_refs_event", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "ref_event_id",
            sa.String(length=26),
            sa.ForeignKey("events.event_id", name="fk_event_refs_ref", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.UniqueConstraint("event_id", "ref_event_id", "kind", name="uq_event_refs_reference"),
    )
    op.create_index("ix_event_refs_ref_event_id", "event_refs", ["ref_event_id"])

    op.create_table(
        "flags",
        sa.Column("flag_id", sa.String(length=26), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=26),
            sa.ForeignKey("sessions.session_id", name="fk_flags_session"),
            nullable=False,
        ),
        sa.Column(
            "event_id",
            sa.String(length=26),
            sa.ForeignKey("events.event_id", name="fk_flags_event", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("module", sa.String(length=64), nullable=False),
        sa.Column("module_version", sa.String(length=32), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("severity", sa.String(length=16), server_default="medium", nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column(
            "evidence",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("adjudication", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column("adjudicated_by", sa.String(length=256), nullable=True),
        sa.Column("adjudicated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("auto_resolved", sa.Boolean(), nullable=True),
        sa.CheckConstraint(
            "severity IN ('info', 'low', 'medium', 'high', 'critical')",
            name="ck_flags_severity",
        ),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_flags_confidence"),
        sa.CheckConstraint(
            "adjudication IN ('pending', 'confirmed', 'rejected')",
            name="ck_flags_adjudication",
        ),
    )
    op.create_index("ix_flags_severity_created_at", "flags", ["severity", "created_at"])
    op.create_index("ix_flags_session_id", "flags", ["session_id"])

    op.create_table(
        "schema_meta",
        sa.Column("key", sa.String(length=64), primary_key=True),
        sa.Column("value", sa.String(length=255), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "tombstones",
        sa.Column("event_id", sa.String(length=26), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(length=26),
            sa.ForeignKey("sessions.session_id", name="fk_tombstones_session"),
            nullable=False,
        ),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(length=64), nullable=False),
        sa.Column("payload_digest", sa.String(length=64), nullable=False),
    )

    op.execute(
        "INSERT INTO schema_meta (key, value, updated_at) "
        "VALUES ('event_schema_version', '0.2', now())"
    )


def downgrade() -> None:
    """Drop every store table (reverse dependency order)."""
    op.drop_table("tombstones")
    op.drop_table("event_refs")
    op.drop_table("flags")
    op.drop_table("events")
    op.drop_table("sessions")
    op.drop_table("schema_meta")
