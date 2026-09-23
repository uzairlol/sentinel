"""Tests for memory adapters and instrumentation (``S1-T11``).

Covers the in-memory reference adapter, the MemoryStore protocol shape, read
events referencing the writes they load (``caused_by``, INV-3), the
enable/disable toggle, closed-session fail-open (INV-6), payload capping, the
missing-``postgres``-extra path, and registry integration.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.instrument.langchain import MAX_CAPTURED_TEXT
from sentinel.instrument.memory import MemoryInstrumentor
from sentinel.instrument.registry import InstrumentorRegistry
from sentinel.memory import InMemoryMemoryStore
from sentinel.models.events import MEMORY_READ, MEMORY_WRITE, RefKind

pytestmark = pytest.mark.unit

_HAS_ASYNCPG = importlib.util.find_spec("asyncpg") is not None


@pytest.fixture
async def events_store() -> AsyncIterator[SQLiteEventStore]:
    store = SQLiteEventStore(":memory:")
    try:
        yield store
    finally:
        await store.close()


async def test_inmemory_round_trip() -> None:
    store = InMemoryMemoryStore()
    try:
        await store.write(key="alice", value="likes blue", summary="preference")
        await store.write(key="alice", value="likes green")

        hits = await store.read(query="alice")
        assert [hit.value for hit in hits] == ["likes green", "likes blue"]

        content = await store.read(query="blue")
        assert [hit.value for hit in content] == ["likes blue"]

        assert await store.read(query="zzz") == []

        limited = await store.read(query="alice", limit=1)
        assert [hit.value for hit in limited] == ["likes green"]
    finally:
        await store.close()


async def test_uninstrumented_entries_carry_no_event_link() -> None:
    store = InMemoryMemoryStore()
    try:
        entry = await store.write(key="a", value="b")
        assert entry.event_id is None
        hits = await store.read(query="a")
        assert hits[0].event_id is None
    finally:
        await store.close()


async def test_instrumented_memory_links_read_to_write(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        memory = MemoryInstrumentor(ctx).instrument(InMemoryMemoryStore())
        entry = await memory.write(key="alice", value="likes blue")
        hits = await memory.read(query="alice")

    assert hits[0].value == entry.value
    events = await events_store.get_session(ctx.session_id)
    write_event = next(event for event in events if event.type == MEMORY_WRITE)
    read_event = next(event for event in events if event.type == MEMORY_READ)

    assert entry.event_id == write_event.event_id
    assert write_event.payload["memory"] == "in_memory"
    assert write_event.payload["key"] == "alice"
    assert write_event.payload["value"] == "likes blue"
    assert write_event.payload["summary"] is None

    assert read_event.payload["query"] == "alice"
    assert read_event.payload["hits"] == 1
    assert read_event.payload["result"] == [
        {"key": "alice", "value": "likes blue", "summary": None}
    ]
    caused_by = [ref for ref in read_event.refs if ref.kind == RefKind.CAUSED_BY]
    assert caused_by
    assert caused_by[0].event_id == write_event.event_id


async def test_disabled_instrumentor_emits_nothing(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        memory = MemoryInstrumentor(ctx, enabled=False).instrument(InMemoryMemoryStore())
        await memory.write(key="a", value="b")
        hits = await memory.read(query="a")

    assert [hit.value for hit in hits] == ["b"]
    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


async def test_closed_session_fails_open(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        memory = MemoryInstrumentor(ctx).instrument(InMemoryMemoryStore())
        await ctx.end()
        await memory.write(key="a", value="b")
        hits = await memory.read(query="a")

    assert [hit.value for hit in hits] == ["b"]
    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


async def test_large_value_is_capped(events_store: SQLiteEventStore) -> None:
    blob = "x" * (MAX_CAPTURED_TEXT * 2)
    async with session(events_store) as ctx:
        memory = MemoryInstrumentor(ctx).instrument(InMemoryMemoryStore())
        await memory.write(key="big", value=blob)

    events = await events_store.get_session(ctx.session_id)
    write_event = next(event for event in events if event.type == MEMORY_WRITE)
    captured = str(write_event.payload["value"])
    assert captured.startswith("x" * MAX_CAPTURED_TEXT)
    assert captured.endswith("…[truncated]")


async def test_postgres_adapter_requires_the_extra() -> None:
    if _HAS_ASYNCPG:
        pytest.skip("asyncpg installed; cannot exercise the missing-extra path")
    from sentinel.memory import MemoryPostgresUnavailableError, PostgresMemoryStore

    with pytest.raises(MemoryPostgresUnavailableError):
        PostgresMemoryStore("postgresql://localhost/sentinel_test")


async def test_registers_as_an_instrumentor(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        registry = InstrumentorRegistry()
        instrumentor = MemoryInstrumentor(ctx)
        registry.register(instrumentor)
        assert "memory" in registry
        assert MEMORY_READ in registry.event_types("memory")
        assert MEMORY_WRITE in registry.event_types("memory")
        registry.enable("memory")
        assert registry.is_enabled("memory") is True
