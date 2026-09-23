"""Memory-store adapters and protocol (``S1-T11``).

* :class:`MemoryStore` — the instrumentation protocol (KV / vector / custom).
* :class:`InMemoryMemoryStore` — the self-contained reference adapter.
* :class:`PostgresMemoryStore` — the Postgres-backed reference adapter
  (needs the ``postgres`` extra). The instrumenting wrapper lives in
  :mod:`sentinel.instrument.memory`.
"""

from __future__ import annotations

from sentinel.memory.inmemory import InMemoryMemoryStore
from sentinel.memory.postgres import (
    MemoryPostgresUnavailableError,
    PostgresMemoryStore,
)
from sentinel.memory.protocol import MemoryEntry, MemoryStore

__all__ = [
    "InMemoryMemoryStore",
    "MemoryEntry",
    "MemoryPostgresUnavailableError",
    "MemoryStore",
    "PostgresMemoryStore",
]
