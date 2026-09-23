"""Memory-store protocol (``S1-T11``).

The protocol is deliberately thin: any KV, vector, or custom store can be
instrumented by wrapping it with
:class:`~sentinel.instrument.memory.MemoryInstrumentor`, which records
``memory.read`` / ``memory.write`` events without knowing anything about the
backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryEntry:
    """One stored memory fragment, optionally linked to its capture event."""

    key: str
    value: str
    summary: str | None = None
    #: The ``memory.write`` event that persisted :attr:`value`; a later
    #: ``memory.read`` that returns this entry references it back (INV-3).
    event_id: str | None = None


class MemoryStore(Protocol):
    """Read/write storage for agent memory (KV, vector, or custom)."""

    name: str

    async def write(
        self,
        *,
        key: str,
        value: str,
        summary: str | None = None,
        event_id: str | None = None,
    ) -> MemoryEntry:
        """Persist *value* (and optional *summary*) under *key*.

        *event_id* links the entry to the ``memory.write`` event that created
        it; reads returning this entry reference it (INV-3). Adapters fall
        back to a self-generated id for uninstrumented use.
        """

    async def read(self, *, query: str, limit: int = 8) -> list[MemoryEntry]:
        """Return entries matching *query* (best match first), newest first."""

    async def close(self) -> None:
        """Release any underlying resources."""
