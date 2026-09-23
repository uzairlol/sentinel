"""In-memory reference memory adapter (``S1-T11``).

A process-local KV store: reads match entries whose ``key`` equals the query
first; when nothing matches exactly, entries whose ``value`` or ``summary``
contains the query (case-insensitive) are returned. Newest writes win and are
returned most-recent-first.
"""

from __future__ import annotations

from sentinel.memory.protocol import MemoryEntry


class InMemoryMemoryStore:
    """A KV memory store backed by process-local lists."""

    name = "in_memory"

    def __init__(self) -> None:
        """Start with an empty entry list."""
        self._entries: list[MemoryEntry] = []

    async def write(
        self,
        *,
        key: str,
        value: str,
        summary: str | None = None,
        event_id: str | None = None,
    ) -> MemoryEntry:
        """Append *value* (and optional *summary*) under *key*."""
        entry = MemoryEntry(key=key, value=value, summary=summary, event_id=event_id)
        self._entries.append(entry)
        return entry

    async def read(self, *, query: str, limit: int = 8) -> list[MemoryEntry]:
        """Exact ``key`` match first, then content match; newest first."""
        matches = [entry for entry in reversed(self._entries) if entry.key == query]
        if not matches:
            needle = query.casefold()
            matches = [
                entry
                for entry in reversed(self._entries)
                if needle in entry.value.casefold()
                or (entry.summary is not None and needle in entry.summary.casefold())
            ]
        if limit > 0:
            return matches[:limit]
        return matches

    async def close(self) -> None:
        """No resources to release."""
