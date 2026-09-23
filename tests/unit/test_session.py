"""Unit tests for capture sessions (``S0-T2``)."""

from __future__ import annotations

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.instrument.session import SessionClosedError
from sentinel.models.events import RefKind


async def test_session_emits_bookends_and_monotonic_seq() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        async with session(store) as ctx:
            assert ctx.seq == 0  # session.start took seq 0
            first = await ctx.capture(type="llm.request", payload={"n": 1})
            second = await ctx.capture(type="llm.response", payload={"n": 2}, refs=[first.event_id])
            assert (first.seq, second.seq) == (1, 2)
            assert [link.event_id for link in second.refs] == [first.event_id]
            assert second.refs[0].kind == RefKind.PARENT

        events = await store.get_session(ctx.session_id)
        assert [e.type for e in events] == [
            "session.start",
            "llm.request",
            "llm.response",
            "session.end",
        ]
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)
    finally:
        await store.close()


async def test_capture_after_session_closed_raises() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        async with session(store) as ctx:
            pass
        with pytest.raises(SessionClosedError):
            await ctx.capture(type="llm.request", payload={})
    finally:
        await store.close()


async def test_end_is_idempotent() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        async with session(store) as ctx:
            pass
        first = await store.get_session(ctx.session_id)
        assert first[-1].type == "session.end"
        assert await ctx.end() is None  # already closed, no duplicate bookend
        second = await store.get_session(ctx.session_id)
        assert len(second) == len(first)
    finally:
        await store.close()


async def test_session_bracketed_even_when_body_raises() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with pytest.raises(RuntimeError):
            async with session(store) as ctx:
                raise RuntimeError("boom")
        events = await store.get_session(ctx.session_id)
        assert [e.type for e in events][-1] == "session.end"
    finally:
        await store.close()
