"""Postgres-specific store semantics beyond the shared parity contract.

Requires ``SENTINEL_TEST_POSTGRES_DSN`` (absent -> skip); exercised by the CI
integration job. Covers batch atomicity, retry classification, multi-batch ref
resolution, and flagged-evidence retention during prune (``S2-T14``/``S2-T12``).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import cast

import pytest

from sentinel.models.events import (
    LLM_REQUEST,
    LLM_RESPONSE,
    SESSION_START,
    Event,
    RefKind,
    RefLink,
    new_event_id,
)
from sentinel.store.factory import build_store
from sentinel.store.postgres import PostgresEventStore
from sentinel.store.retention import RetentionPolicy
from sentinel.store.sqlite import _parse_ts

_REPO_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not _REPO_DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set"),
    pytest.mark.integration,
]


def _event(session_id: str, seq: int, type_: str = LLM_REQUEST) -> Event:
    from datetime import UTC, datetime

    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=datetime.now(UTC),
        type=type_,
        payload={"seq": seq},
    )


@pytest.fixture
async def store() -> AsyncIterator[PostgresEventStore]:
    result = build_store(_REPO_DSN)
    try:
        yield cast(PostgresEventStore, result)
    finally:
        await result.close()


async def _truncate() -> None:
    import asyncpg

    conn = await asyncpg.connect(_REPO_DSN or "")
    try:
        await conn.execute(
            "TRUNCATE tombstones, event_refs, flags, events, sessions RESTART IDENTITY CASCADE"
        )
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
async def _isolate() -> AsyncIterator[None]:
    await _truncate()
    yield
    await _truncate()


async def test_batch_conflict_raises_atomically(store: PostgresEventStore) -> None:
    session_id = new_event_id()
    await store.append(_event(session_id, 0))
    with pytest.raises(Exception, match=r"(?i)unique|integrity"):
        await store.append_batch([_event(session_id, 1), _event(session_id, 0)])
    # the conflicting second batch wrote nothing (atomicity)
    assert len(await store.get_session(session_id)) == 1


async def test_multi_batch_ref_resolves_across_batches(store: PostgresEventStore) -> None:
    session_id = new_event_id()
    request = _event(session_id, 0)
    response = _event(session_id, 1, type_=LLM_RESPONSE).model_copy(
        update={"refs": [RefLink(event_id=request.event_id, kind=RefKind.CAUSED_BY)]}
    )
    await store.append_batch([request])
    await store.append_batch([response])
    replay = await store.get_session(session_id)
    assert replay[1].refs[0].event_id == request.event_id


async def test_retry_exhaustion_after_transient_errors(store: PostgresEventStore) -> None:
    store._retry_attempts = 2

    async def op() -> None:
        raise TimeoutError("connection reset")

    with pytest.raises(asyncio.TimeoutError):
        await store._exec_with_retry(op)


async def test_integrity_errors_are_never_retried(store: PostgresEventStore) -> None:
    from sqlalchemy.exc import IntegrityError

    try:
        from asyncpg.exceptions import (  # type: ignore[import-untyped]
            UniqueViolationError,
        )

        cause: BaseException = UniqueViolationError()
    except ImportError:  # pragma: no cover - asyncpg always present here
        cause = RuntimeError("duplicate key value violates unique constraint")

    attempts = 0

    async def op() -> None:
        nonlocal attempts
        attempts += 1
        raise IntegrityError("INSERT INTO events ...", {"event_id": "x"}, cause)

    store._retry_attempts = 20
    with pytest.raises(IntegrityError):
        await store._exec_with_retry(op)
    assert attempts == 1  # never retried


async def test_prune_keeps_flagged_evidence(store: PostgresEventStore) -> None:
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import text
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from sentinel.store import models

    session_id = new_event_id()
    old_a = _event(session_id, 0, type_=SESSION_START)
    old_b = _event(session_id, 1)
    await store.append_batch([old_a, old_b])

    # back-date both rows so they are older than today's retention cutoff
    async with await store._session() as session:
        await session.execute(
            text("UPDATE events SET ts = :ts WHERE session_id = :sid"),
            {"ts": datetime.now(UTC) - timedelta(days=60), "sid": session_id},
        )
        # flag one of them as evaluator evidence; it must survive prune
        await session.execute(
            pg_insert(models.FlagRecord)
            .values(
                flag_id=new_event_id(),
                session_id=session_id,
                event_id=old_b.event_id,
                module="gating",
                module_version="0.1",
                category="test",
                severity="low",
                confidence=0.9,
                summary="evidence",
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["flag_id"]),
        )
        await session.commit()

    report = await store.prune(RetentionPolicy(default_ttl=timedelta(days=7)))
    assert report.pruned_events == 1
    assert report.retained_evidence == 1
    assert report.tombstoned == 1

    replay = await store.get_session(session_id)
    assert [e.seq for e in replay] == [1]  # only the flagged event remains

    health = await store.health()
    assert health.tombstones == 1
    assert health.flags == 1


async def test_dsn_normalization_accepts_all_forms() -> None:
    from sentinel.store import postgres as pg_module

    normalized = pg_module._normalize_dsn
    assert normalized("postgresql://u:p@localhost:5432/db").startswith("postgresql+asyncpg://")
    assert normalized("postgres://u:p@localhost:5432/db").startswith("postgresql+asyncpg://")
    assert normalized("postgresql+asyncpg://u:p@localhost:5432/db").startswith(
        "postgresql+asyncpg://"
    )


def test_parse_ts_imported_for_sqlite_parity() -> None:
    assert callable(_parse_ts)
