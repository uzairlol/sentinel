"""Integration tests for the Postgres memory adapter (``S1-T11``).

Requires a running Postgres; the DSN comes from ``SENTINEL_TEST_POSTGRES_DSN``
(absent → skip). CI runs this against an ephemeral service container.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from sentinel.memory import MemoryEntry, PostgresMemoryStore

_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not _DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set"),
    pytest.mark.integration,
]


@pytest.fixture
async def store() -> AsyncIterator[PostgresMemoryStore]:
    store = PostgresMemoryStore(_DSN or "")
    try:
        yield store
    finally:
        await store.close()


async def test_write_read_round_trip(store: PostgresMemoryStore) -> None:
    entry = await store.write(key="alice", value="likes blue", summary="preference")
    assert isinstance(entry, MemoryEntry)

    hits = await store.read(query="alice")
    assert [hit.value for hit in hits] == ["likes blue"]
    assert hits[0].summary == "preference"
    assert hits[0].event_id == entry.event_id


async def test_read_missing_key_is_empty(store: PostgresMemoryStore) -> None:
    hits = await store.read(query="no-such-key")
    assert hits == []


async def test_write_with_event_id_round_trips(store: PostgresMemoryStore) -> None:
    await store.write(key="linked", value="payload", event_id="evt-1")
    hits = await store.read(query="linked")
    assert hits[0].event_id == "evt-1"
    assert hits[0].value == "payload"
