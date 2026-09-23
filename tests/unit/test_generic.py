"""Tests for the generic ``trace`` decorator (``S1-T10``).

Covers ``agent.step`` started/ended pairs with bound-input and result
payloads, custom ``kind`` labels, nesting parent links (INV-3), error
capture with re-raise, payload capping, and the fail-open guarantees of
INV-6 (no active session, closed session) plus the sync-passthrough
limitation (asyncio-only capture).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.instrument import trace
from sentinel.instrument.langchain import MAX_CAPTURED_TEXT
from sentinel.models.events import AGENT_STEP, ERROR, Event, RefKind

pytestmark = pytest.mark.unit


@pytest.fixture
async def events_store() -> AsyncIterator[SQLiteEventStore]:
    store = SQLiteEventStore(":memory:")
    try:
        yield store
    finally:
        await store.close()


@trace()
async def _echo(*, greeting: str, count: int = 1) -> str:
    return (greeting + " ") * count


@trace(kind="inner")
async def _inner(x: int) -> int:
    return x + 1


@trace(kind="outer")
async def _outer(x: int) -> int:
    return await _inner(x)


@trace(kind="risky")
async def _risky(x: int) -> int:
    raise ValueError("boom")


@trace(kind="big")
async def _big(blob: str) -> str:
    return "ok"


@trace(kind="sync")
def _sync_passthrough(x: int) -> int:
    return x * 2


@trace(kind="unused")
async def _standalone(x: int) -> int:
    return x + 10


def _steps(events: list[Event]) -> list[Event]:
    return [event for event in events if event.type == AGENT_STEP]


async def test_async_function_emits_an_agent_step_pair(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        result = await _echo(greeting="hi")

    assert result == "hi "
    events = await events_store.get_session(ctx.session_id)
    steps = _steps(events)
    assert [event.payload["status"] for event in steps] == ["started", "ended"]
    started, ended = steps

    assert started.payload["provider"] == "generic"
    assert started.payload["step"] == "function"
    assert started.payload["function"] == "_echo"
    assert started.payload["input"] == {"greeting": "hi", "count": 1}
    assert started.refs == []

    assert ended.payload["provider"] == "generic"
    assert ended.payload["step"] == "function"
    assert ended.payload["function"] == "_echo"
    assert ended.payload["output"] == "hi "
    assert ended.payload["latency_ms"] >= 0
    assert len(ended.refs) == 1
    assert ended.refs[0].kind == RefKind.PARENT
    assert ended.refs[0].event_id == started.event_id


async def test_custom_kind_and_kwargs_are_captured(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        await _echo(greeting="yo", count=3)

    events = await events_store.get_session(ctx.session_id)
    started = next(e for e in _steps(events) if e.payload["status"] == "started")
    assert started.payload["step"] == "function"
    assert started.payload["input"] == {"greeting": "yo", "count": 3}


async def test_return_value_passes_through(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store):
        assert await _inner(41) == 42
    assert await _standalone(5) == 15


async def test_exception_is_captured_and_re_raised(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        with pytest.raises(ValueError, match="boom"):
            await _risky(0)

    events = await events_store.get_session(ctx.session_id)
    steps = _steps(events)
    assert [event.payload["status"] for event in steps] == ["started"]
    errors = [event for event in events if event.type == ERROR]
    assert len(errors) == 1
    assert errors[0].payload["provider"] == "generic"
    assert errors[0].payload["scope"] == "function"
    assert errors[0].payload["step"] == "risky"
    assert errors[0].payload["function"] == "_risky"
    assert errors[0].payload["message"] == "boom"
    assert errors[0].refs
    assert errors[0].refs[0].kind == RefKind.PARENT
    assert errors[0].refs[0].event_id is not None


async def test_nested_traced_calls_link_parent(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        assert await _outer(1) == 2

    events = await events_store.get_session(ctx.session_id)
    outer_started, inner_started, inner_ended, outer_ended = _steps(events)
    assert inner_started.refs[0].kind == RefKind.PARENT
    assert inner_started.refs[0].event_id == outer_started.event_id
    assert inner_ended.refs[0].event_id == inner_started.event_id
    assert outer_ended.refs[0].event_id == outer_started.event_id


async def test_large_payload_is_capped(events_store: SQLiteEventStore) -> None:
    blob = "x" * (MAX_CAPTURED_TEXT * 2)
    async with session(events_store) as ctx:
        assert await _big(blob) == "ok"

    events = await events_store.get_session(ctx.session_id)
    started = next(e for e in _steps(events) if e.payload["status"] == "started")
    captured = str(started.payload["input"]["blob"])
    assert captured.startswith("x" * MAX_CAPTURED_TEXT)
    assert captured.endswith("…[truncated]")


async def test_no_active_session_passes_through() -> None:
    result = await _standalone(7)
    assert result == 17


async def test_closed_session_fails_open(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        await ctx.end()
        result = await _standalone(2)

    assert result == 12
    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


async def test_sync_callables_pass_through_untraced(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        assert _sync_passthrough(21) == 42

    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


def test_trace_validates_kind() -> None:
    with pytest.raises(TypeError):
        trace(kind="")
