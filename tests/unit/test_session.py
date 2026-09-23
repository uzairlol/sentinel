"""Unit tests for capture sessions (``S0-T2``)."""

from __future__ import annotations

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.capture.writer import BatchedWriter
from sentinel.instrument.session import SessionClosedError
from sentinel.models.events import RefKind
from sentinel.redact import REDACTED


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


async def test_session_thru_writer_pipeline() -> None:
    store = SQLiteEventStore(":memory:")
    writer = BatchedWriter(store, flush_interval_ms=0)
    try:
        await writer.start()
        async with session(store, writer=writer) as ctx:
            await ctx.capture(type="llm.request", payload={"message": "Bearer abcDEF0123_x"})
            await ctx.capture(
                type="llm.response",
                payload={"ok": True},
                refs=[first_ref := (await ctx.capture(type="tool.call", payload={})).event_id],
            )
        events = await store.get_session(ctx.session_id)
        assert [e.type for e in events] == [
            "session.start",
            "llm.request",
            "tool.call",
            "llm.response",
            "session.end",
        ]
        assert "Bearer" not in str(events[1].payload)
        assert REDACTED in events[1].payload["message"]
        assert events[3].refs[0].event_id == first_ref
    finally:
        await writer.close()
        await store.close()


async def test_session_writer_fail_open_keeps_host_running() -> None:
    store = SQLiteEventStore(":memory:")
    writer = BatchedWriter(store, flush_interval_ms=0)

    async def failing_append(event: object) -> None:
        raise OSError("disk full")

    store.append = failing_append  # type: ignore[method-assign]
    try:
        await writer.start()
        async with session(store, writer=writer) as ctx:
            await ctx.capture(type="llm.request", payload={"n": 1})
            await ctx.capture(type="llm.response", payload={})
        assert writer.failed >= 2
        assert writer.last_error is not None
        assert writer.fatal_error is None  # fail-open; the pipeline kept going
    finally:
        await writer.close()
        await store.close()
