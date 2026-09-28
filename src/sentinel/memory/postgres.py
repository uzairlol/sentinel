"""Postgres-backed reference memory adapter (``S1-T11``).

Stores entries in a ``memory_entries`` table via ``asyncpg`` using only
parameterized SQL. The table is created by Alembic migration ``0003``, not at
runtime: DDL belongs to the ``sentinel_migrator`` role under the model of
``docs/adr/0011``, so the adapter is a pure reader/writer. Requires the
``postgres`` extra (``pip install "sentinel-sdk[postgres]"``); the adapter
raises :class:`MemoryPostgresUnavailableError` otherwise. Exercised by the
Postgres integration job in CI (``tests/integration/test_memory_postgres.py``).
"""

from __future__ import annotations

from typing import Any

from sentinel.memory.protocol import MemoryEntry
from sentinel.models.events import new_event_id


class MemoryPostgresUnavailableError(ImportError):
    """Raised when :class:`PostgresMemoryStore` needs the ``postgres`` extra."""


class PostgresMemoryStore:
    """Reference adapter backed by Postgres via ``asyncpg``."""

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        """Prepare an asyncpg connection pool for *dsn* (lazy import)."""
        try:
            import asyncpg
        except ImportError as exc:  # pragma: no cover - exercised by integration
            raise MemoryPostgresUnavailableError(
                "PostgresMemoryStore needs the sentinel-sdk[postgres] extra"
            ) from exc
        self._asyncpg: Any = asyncpg
        self._dsn = dsn
        self._pool: Any | None = None

    async def _open(self) -> Any:  # noqa: ANN401 -- the asyncpg pool type
        if self._pool is None:
            self._pool = await self._asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        return self._pool

    async def write(
        self,
        *,
        key: str,
        value: str,
        summary: str | None = None,
        event_id: str | None = None,
    ) -> MemoryEntry:
        """Insert one entry, keyed by *event_id* (or a fresh id)."""
        pool = await self._open()
        entry_id = event_id or new_event_id()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_entries (id, entry_key, value, summary)"
                " VALUES ($1, $2, $3, $4)",
                entry_id,
                key,
                value,
                summary,
            )
        return MemoryEntry(key=key, value=value, summary=summary, event_id=entry_id)

    async def read(self, *, query: str, limit: int = 8) -> list[MemoryEntry]:
        """Return entries whose ``key`` equals *query*, newest first."""
        pool = await self._open()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, entry_key, value, summary"
                " FROM memory_entries"
                " WHERE entry_key = $1"
                " ORDER BY created_at DESC, id DESC"
                " LIMIT $2",
                query,
                limit,
            )
        return [
            MemoryEntry(key=row[1], value=row[2], summary=row[3], event_id=row[0]) for row in rows
        ]

    async def close(self) -> None:
        """Release the asyncpg pool, if open."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
