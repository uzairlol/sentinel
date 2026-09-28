"""Move the memory adapter's table out of runtime DDL into the schema.

Sprint ``S3`` (verification fix). :class:`sentinel.memory.postgres.PostgresMemoryStore`
used to run ``CREATE TABLE IF NOT EXISTS memory_entries`` from the application
on first connect. That is wrong under the role model of ``docs/adr/0011``:
``deploy/roles.sql`` makes ``sentinel_migrator`` the DDL owner and revokes
``CREATE`` on the ``public`` schema from everyone else, so the statement only
ever succeeded when the adapter was handed a superuser credential -- exactly
the credential a production deployment will not have. It also made the table's
existence depend on whether a read had happened first, which broke the
integration fixture's ``TRUNCATE`` against a fresh database.

Creating the table here makes the adapter a pure reader/writer, keeps DDL in
one place, and lets the migrator's grants apply to it.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-28

"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ``memory_entries`` if a pre-0003 adapter has not already done so."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id TEXT PRIMARY KEY,
            entry_key TEXT NOT NULL,
            value TEXT NOT NULL,
            summary TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # IF NOT EXISTS to match the table above: a database that already has the
    # table from the old runtime DDL has no index yet, but making the whole
    # step re-runnable costs nothing and keeps the two halves consistent.
    op.execute("CREATE INDEX IF NOT EXISTS ix_memory_entries_key ON memory_entries (entry_key)")


def downgrade() -> None:
    """Drop the memory adapter's table and its index."""
    op.drop_index("ix_memory_entries_key", table_name="memory_entries")
    op.drop_table("memory_entries")
