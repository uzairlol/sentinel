"""Memory read/write instrumentation (``S1-T11``).

Wraps any :class:`~sentinel.memory.MemoryStore` so that ``write`` emits a
``memory.write`` event and ``read`` emits a ``memory.read`` event in the
capture session, with each returned entry referenced back to the write that
created it (``caused_by``, INV-3) — the link the S1/S4 memory-integrity
evaluators rely on to verify what the agent loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import structlog

from sentinel.instrument.langchain import _capped
from sentinel.instrument.registry import BaseInstrumentor
from sentinel.instrument.session import SessionContext
from sentinel.memory import MemoryEntry, MemoryStore
from sentinel.models.events import MEMORY_READ, MEMORY_WRITE, Event, RefKind, RefLink

log = structlog.get_logger("sentinel.memory")


async def _capture(
    ctx: SessionContext,
    *,
    type: str,  # noqa: A002
    payload: dict[str, object],
    refs: Sequence[RefLink],
) -> Event | None:
    """Record one event, failing open (INV-6): capture never raises to the host."""
    try:
        return await ctx.capture(type=type, payload=payload, refs=tuple(refs))
    except Exception:
        log.warning("memory.capture.failed", exc_info=True)
        return None


def _entry_view(entry: MemoryEntry) -> dict[str, object]:
    return {"key": entry.key, "value": entry.value, "summary": entry.summary}


class _InstrumentedMemoryStore:
    """Delegation wrapper that emits capture events around the real store."""

    def __init__(
        self,
        ctx: SessionContext,
        store: MemoryStore,
        *,
        active: Callable[[], bool],
    ) -> None:
        self._ctx = ctx
        self._store = store
        self._active = active
        self.name: str = store.name

    async def write(
        self,
        *,
        key: str,
        value: str,
        summary: str | None = None,
        event_id: str | None = None,
    ) -> MemoryEntry:
        captured_id = event_id
        if self._active():
            event = await _capture(
                self._ctx,
                type=MEMORY_WRITE,
                payload={
                    "provider": "memory",
                    "memory": self._store.name,
                    "key": key,
                    "value": _capped(value),
                    "summary": _capped(summary),
                },
                refs=[],
            )
            if event is not None:
                captured_id = event.event_id
        return await self._store.write(key=key, value=value, summary=summary, event_id=captured_id)

    async def read(self, *, query: str, limit: int = 8) -> list[MemoryEntry]:
        entries = await self._store.read(query=query, limit=limit)
        if self._active():
            await _capture(
                self._ctx,
                type=MEMORY_READ,
                payload={
                    "provider": "memory",
                    "memory": self._store.name,
                    "query": _capped(query),
                    "limit": limit,
                    "hits": len(entries),
                    "result": _capped([_entry_view(entry) for entry in entries]),
                },
                refs=[
                    RefLink(event_id=entry.event_id, kind=RefKind.CAUSED_BY)
                    for entry in entries
                    if entry.event_id is not None
                ],
            )
        return entries

    async def close(self) -> None:
        await self._store.close()


class MemoryInstrumentor(BaseInstrumentor):
    """Wrap a :class:`~sentinel.memory.MemoryStore` to capture its traffic.

    ``write`` becomes a ``memory.write`` event (with the value/summary capped
    at :data:`~sentinel.instrument.langchain.MAX_CAPTURED_TEXT`); ``read``
    becomes a ``memory.read`` event whose payload carries the hits and whose
    references link back to the writes that produced the returned entries
    (``caused_by``, INV-3). Use with the S1-T2 registry::

        from sentinel.memory import InMemoryMemoryStore
        from sentinel.instrument.memory import MemoryInstrumentor

        async with session(store) as ctx:
            memory = MemoryInstrumentor(ctx).instrument(InMemoryMemoryStore())
            entry = await memory.write(key="alice", value="likes blue")

    Capture fails open (INV-6): a closed session or unavailable store never
    raises into the host, and writes/reads still reach the underlying adapter.
    """

    name = "memory"
    event_types: frozenset[str] = frozenset({MEMORY_READ, MEMORY_WRITE})

    def __init__(self, ctx: SessionContext, *, enabled: bool = True) -> None:
        """Wrap *ctx*; start emitting unless ``enabled=False``."""
        self._ctx = ctx
        self.enabled = enabled

    def instrument(self, store: MemoryStore) -> MemoryStore:
        """Return *store* wrapped so reads/writes are captured."""
        return _InstrumentedMemoryStore(self._ctx, store, active=lambda: self.enabled)
