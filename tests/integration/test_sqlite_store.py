"""Integration tests for the SQLite event store (``S0-T5``, ``S0-T6``)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentinel import SQLiteEventStore
from sentinel.models.events import Event, new_event_id
from sentinel.store.sqlite import _parse_ts


def _event(session_id: str, seq: int, type_: str = "test.type") -> Event:
    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=datetime.now(UTC),
        type=type_,
        payload={"seq": seq},
    )


async def test_append_and_get_session_ordered() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        session_id = new_event_id()
        for seq in (2, 0, 1, 3):
            await store.append(_event(session_id, seq))
        events = await store.get_session(session_id)
        assert [e.seq for e in events] == [0, 1, 2, 3]
        assert events[0].payload == {"seq": 0}
    finally:
        await store.close()


async def test_get_session_returns_empty_for_unknown_session() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        assert await store.get_session(new_event_id()) == []
    finally:
        await store.close()


async def test_append_is_idempotent_by_event_id() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        event = _event(new_event_id(), 0)
        await store.append(event)
        await store.append(event)
        assert len(await store.get_session(event.session_id)) == 1
    finally:
        await store.close()


async def test_conflicting_session_seq_raises_integrity_error() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        session_id = new_event_id()
        await store.append(_event(session_id, 0, type_="first"))
        with pytest.raises(sqlite3.IntegrityError):
            await store.append(_event(session_id, 0, type_="second"))
    finally:
        await store.close()


async def test_close_then_reopen_resumes(tmp_path: Path) -> None:
    path = str(tmp_path / "events.sqlite3")
    store = SQLiteEventStore(path)
    session_id = new_event_id()
    await store.append(_event(session_id, 0))
    await store.close()
    reopened = SQLiteEventStore(path)
    try:
        assert len(await reopened.get_session(session_id)) == 1
    finally:
        await reopened.close()


def test_parse_ts_naive_value_is_promoted_to_utc() -> None:
    parsed = _parse_ts("2026-09-23T08:50:51")
    assert parsed.utcoffset() == timedelta(0)
