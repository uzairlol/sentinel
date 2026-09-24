"""Tests for the batched capture pipeline (``S1-T12``/``S1-T13``).

Covers bounded-queue overflow -> ``capture.dropped``, fail-open persistence,
fail-closed behaviour, redaction before persistence (``S1-T14``), and FIFO order
preserving ``seq``/``refs`` integrity.
"""

from __future__ import annotations

import asyncio

import pytest

from sentinel import SQLiteEventStore
from sentinel.capture.writer import BatchedWriter, CapturePipelineError
from sentinel.models.events import (
    CAPTURE_DROPPED,
    EVENT_TYPES,
    Event,
    RefKind,
    RefLink,
    make_event,
)
from sentinel.redact import REDACTED


def _request(session_id: str, seq: int, content: str = "hello") -> Event:
    return make_event(
        session_id=session_id,
        seq=seq,
        type="llm.request",
        payload={"message": content},
    )


async def _open_writer(
    store: SQLiteEventStore,
    *,
    batch_max_size: int = 100,
    flush_interval_ms: float = 250.0,
    fail_open: bool = True,
) -> BatchedWriter:
    writer = BatchedWriter(
        store,
        batch_max_size=batch_max_size,
        flush_interval_ms=flush_interval_ms,
        fail_open=fail_open,
    )
    await writer.start()
    return writer


async def test_writer_persists_events_in_submission_order() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = await _open_writer(store, flush_interval_ms=0)
        try:
            events = [_request("session-a", i, content=f"msg-{i}") for i in range(5)]
            for event in events:
                await writer.submit(event)
            await writer.flush()
            persisted = await store.get_session("session-a")
            assert [e.seq for e in persisted] == [0, 1, 2, 3, 4]
            assert [e.payload["message"] for e in persisted] == [f"msg-{i}" for i in range(5)]
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_writer_redacts_before_persisting() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = await _open_writer(store, flush_interval_ms=0)
        try:
            await writer.submit(_request("s", 0, content="Bearer abcDEF0123_x"))
            await writer.flush()
            persisted = await store.get_session("s")
            assert REDACTED in persisted[0].payload["message"]
            assert "abcDEF0123_x" not in str(persisted[0].payload)
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_queue_overflow_counts_drops_and_never_blocks() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = BatchedWriter(store, queue_max_size=1, batch_max_size=1, flush_interval_ms=0)
        await writer.start()
        try:
            stall = asyncio.Event()

            async def _slow(event: Event) -> None:
                await stall.wait()

            original = writer._append
            writer._append = _slow  # type: ignore[method-assign]
            await writer.submit(_request("s", 0))
            await writer.submit(_request("s", 1))
            assert writer.dropped == 1
            assert writer.queued <= 1
            stall.set()
            writer._append = original  # type: ignore[method-assign]
            await writer.flush()
            persisted = await store.get_session("s")
            assert [e.seq for e in persisted] == [0]
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_fail_open_swallows_store_errors() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        original_append = store.append

        async def failing_append(event: Event) -> None:
            raise OSError("disk full")

        store.append = failing_append  # type: ignore[method-assign]
        try:
            writer = await _open_writer(store, fail_open=True, flush_interval_ms=0)
            try:
                await writer.submit(_request("s", 0))
                await writer.flush()
                assert writer.failed >= 1
                assert writer.last_error is not None
            finally:
                await writer.close()
        finally:
            store.append = original_append  # type: ignore[method-assign]
        assert await store.get_session("s") == []
    finally:
        await store.close()


async def test_fail_closed_raises_on_next_submit() -> None:
    store = SQLiteEventStore(":memory:")
    try:

        async def failing_append(event: Event) -> None:
            raise OSError("disk full")

        store.append = failing_append  # type: ignore[method-assign]
        writer = BatchedWriter(store, fail_open=False, flush_interval_ms=0)
        await writer.start()
        await writer.submit(_request("s", 0))
        for _ in range(50):
            if writer.fatal_error is not None:
                break
            await asyncio.sleep(0)
        assert writer.fatal_error is not None
        with pytest.raises(CapturePipelineError):
            await writer.submit(_request("s", 1))
        await writer.close()
    finally:
        await store.close()


async def test_pipeline_preserves_ref_integrity() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = await _open_writer(store, flush_interval_ms=0)
        try:
            request = _request("s", 0)
            response = make_event(
                session_id="s",
                seq=1,
                type="llm.response",
                payload={},
                refs=[RefLink(event_id=request.event_id, kind=RefKind.CAUSED_BY)],
            )
            await writer.submit(request)
            await writer.submit(response)
            await writer.flush()
            persisted = await store.get_session("s")
            assert persisted[1].refs[0].event_id == request.event_id
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_store_error_emits_dropped_marker_with_reason() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        original_append = store.append
        calls = 0

        async def flip_failing_append(event: Event) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("disk full")
            await original_append(event)

        store.append = flip_failing_append  # type: ignore[method-assign]
        try:
            writer = await _open_writer(store, fail_open=True, flush_interval_ms=0)
            try:
                await writer.submit(_request("s", 0))
                await writer.flush()
                assert writer.failed == 1
                persisted = await store.get_session("s")
                marker = next(e for e in persisted if e.type == CAPTURE_DROPPED)
                assert marker.payload["reason"] == "store_error"
                assert marker.payload["count"] == 1
                assert isinstance(marker.payload["dropped_event_id"], str)
            finally:
                await writer.close()
        finally:
            store.append = original_append  # type: ignore[method-assign]
    finally:
        await store.close()


async def test_sampling_drops_marked_slots_instead_of_silent_gaps() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = BatchedWriter(
            store,
            batch_max_size=64,
            flush_interval_ms=0,
            sampler=lambda e: e.type != "llm.request",
        )
        await writer.start()
        try:
            await writer.submit(make_event(session_id="s", seq=0, type="session.start", payload={}))
            for seq in range(1, 6):
                await writer.submit(_request("s", seq))
            await writer.submit(make_event(session_id="s", seq=6, type="session.end", payload={}))
            await writer.flush()
            assert writer.sampled == 5
            persisted = await store.get_session("s")
            assert [e.seq for e in persisted] == [0, 1, 2, 3, 4, 5, 6]
            markers = [e for e in persisted if e.type == CAPTURE_DROPPED]
            assert len(markers) == 5
            assert all(m.payload["reason"] == "sampled" for m in markers)
            assert all("original_type" in m.payload for m in markers)
            assert await store.detect_gaps("s") == []
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_sampling_default_captures_critical_types_regardless_of_rate() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = BatchedWriter(
            store,
            batch_max_size=64,
            flush_interval_ms=0,
            sampler=lambda e: e.type in ("session.start", "session.end", "error", CAPTURE_DROPPED),
        )
        await writer.start()
        try:
            await writer.submit(make_event(session_id="s", seq=0, type="session.start", payload={}))
            await writer.submit(_request("s", 1))
            await writer.submit(make_event(session_id="s", seq=2, type="error", payload={}))
            await writer.submit(make_event(session_id="s", seq=3, type="session.end", payload={}))
            await writer.flush()
            persisted = await store.get_session("s")
            # sampled-out llm.request keeps its seq slot as a capture.dropped marker
            assert [e.type for e in persisted] == [
                "session.start",
                CAPTURE_DROPPED,
                "error",
                "session.end",
            ]
            assert await store.detect_gaps("s") == []
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_oversized_payload_is_capped_with_markers() -> None:
    from sentinel.truncate import TRUNCATED, TRUNCATED_HASH, is_truncated

    store = SQLiteEventStore(":memory:")
    try:
        writer = BatchedWriter(store, batch_max_size=64, flush_interval_ms=0, max_payload_bytes=256)
        await writer.start()
        try:
            await writer.submit(_request("s", 0, content="z" * 4096))
            await writer.flush()
            persisted = await store.get_session("s")
            payload = persisted[0].payload
            assert payload["_truncated"] is True
            assert payload[TRUNCATED] is True
            assert len(payload[TRUNCATED_HASH]) == 64
            assert is_truncated(payload)
            assert "…[truncated" in payload["message"]
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_small_payloads_pass_through_without_truncation() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        writer = BatchedWriter(store, batch_max_size=64, flush_interval_ms=0, max_payload_bytes=256)
        await writer.start()
        try:
            await writer.submit(_request("s", 0, content="tiny"))
            await writer.flush()
            persisted = await store.get_session("s")
            assert persisted[0].payload == {"message": "tiny"}
        finally:
            await writer.close()
    finally:
        await store.close()


async def test_events_validate_against_locked_taxonomy() -> None:
    assert "llm.request" in EVENT_TYPES
    with pytest.raises(ValueError, match="type must be one of"):
        make_event(session_id="s", seq=0, type="llm.unknown")
