"""Async Alembic environment for the Postgres event store.

Sprint ``S2-T2`` requires the migration environment to run against an async
engine (SQLAlchemy + asyncpg). The target DSN resolves from
``SENTINEL_STORE_DSN`` first, then ``SENTINEL_TEST_POSTGRES_DSN``, then the
placeholder in ``alembic.ini`` -- so the same tree works locally and in CI
without editing tracked files.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from sentinel.store.models import Base

#: Alembic's Config object, injected when this environment is loaded.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

#: The declarative models are the single source of truth for the schema.
target_metadata = Base.metadata


def _dsn() -> str:
    """Resolve the migration DSN: env first, placeholder last."""
    dsn = (
        os.getenv("SENTINEL_STORE_DSN")
        or os.getenv("SENTINEL_TEST_POSTGRES_DSN")
        or config.get_main_option("sqlalchemy.url", "")
    )
    if dsn.startswith("postgres:"):
        dsn = dsn.replace("postgres:", "postgresql+asyncpg:", 1)
    if dsn.startswith("postgresql:"):
        dsn = dsn.replace("postgresql:", "postgresql+asyncpg:", 1)
    return dsn


def run_migrations_offline() -> None:
    """Emit migration SQL without a live database (sqloffline path)."""
    context.configure(
        url=_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Attach the migration context to an existing connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations through ``run_sync``."""
    connectable = create_async_engine(_dsn(), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations against a live database via the async engine."""
    import asyncio

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
