"""S3 flag schema: structured details, review-only routing, schema version.

Sprint ``S3-T1``. ``0001`` created the ``flags`` table forward-compatible with
the evaluator contract, but the S3 :class:`sentinel.models.flags.Flag` schema
(docs/adr/0012) needs three more columns to be persisted faithfully:

* ``details`` (JSONB) — the module-specific structured payload of a finding
  (claimed vs observed values, the tolerance applied, the threshold crossed).
  ``evidence`` holds *links* to events; ``details`` holds *this module's*
  reading of them, so the gate and review UI never parse prose.
* ``review_only`` (boolean) — findings a human must adjudicate before they may
  gate (``S5-T10``). Stored rather than derived so the routing decision
  survives restarts and is queryable.
* ``schema_version`` (string) — the flag schema version (docs/adr/0007), so a
  reader can branch on an old flag row the way it branches on an old event.

The migration also adds the review-queue index
(``adjudication, created_at``) and, when the ``sentinel_reviewer`` role
exists, its column-scoped ``UPDATE`` grant on the four adjudication columns.
Adjudication is the *only* mutating flag operation in the product, so it is the
only one that needs a role beyond the append-only writer (INV-2, ``S2-T3``).

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-27

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

#: Columns whose value a reviewer is allowed to change. Column-scoped so a
#: reviewer can never rewrite a finding's evidence, severity, or summary.
_REVIEWER_COLUMNS = ("adjudication", "adjudicated_by", "adjudicated_at", "auto_resolved")


def upgrade() -> None:
    """Add the S3 flag columns, the review index, and reviewer grants."""
    op.add_column(
        "flags",
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "flags",
        sa.Column("review_only", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )
    op.add_column(
        "flags",
        sa.Column("schema_version", sa.String(length=16), server_default="0.1", nullable=False),
    )
    op.create_index("ix_flags_adjudication_created_at", "flags", ["adjudication", "created_at"])
    _grant_reviewer()


def downgrade() -> None:
    """Remove the S3 flag additions (the reviewer's grant is left in place).

    Revoking the reviewer grant is deliberately not done here: the role is
    cluster-level (``deploy/roles.sql``) and dropping the columns does not make
    it unsafe — it simply has nothing to update.
    """
    op.drop_index("ix_flags_adjudication_created_at", table_name="flags")
    op.drop_column("flags", "schema_version")
    op.drop_column("flags", "review_only")
    op.drop_column("flags", "details")


def _grant_reviewer() -> None:
    """Grant column-scoped flag UPDATE to ``sentinel_reviewer``, if it exists.

    Guarded so the migration also runs on clusters provisioned without
    ``deploy/roles.sql`` (CI, local dev) instead of aborting the upgrade.
    """
    exists = (
        op.get_bind()
        .execute(text("SELECT 1 FROM pg_roles WHERE rolname = 'sentinel_reviewer'"))
        .scalar()
    )
    if not exists:
        return
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO sentinel_reviewer")
    op.execute(
        "GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved) "
        "ON flags TO sentinel_reviewer"
    )
    op.execute(
        "ALTER DEFAULT PRIVILEGES FOR ROLE sentinel_migrator IN SCHEMA public "
        "GRANT UPDATE (adjudication, adjudicated_by, adjudicated_at, auto_resolved) "
        "ON TABLES TO sentinel_reviewer"
    )
