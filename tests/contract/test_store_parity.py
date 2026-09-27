"""Parity contract between the SQLite and Postgres stores (``S2-T17``).

Every test in this file runs against *both* backends (parametrized via the
``store`` fixture), so the two stores must satisfy the exact same protocol
contract (docs/adr/0002). The Postgres side is skipped offline and exercised by
the CI integration job; migrations run once per session via Alembic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentinel.models.events import (
    LLM_REQUEST,
    LLM_RESPONSE,
    SESSION_START,
    TOOL_CALL,
    Event,
    RefKind,
    RefLink,
    new_event_id,
)
from sentinel.models.flags import Adjudication, EvidenceRef, EvidenceRole, Flag, Severity
from sentinel.store.errors import RefIntegrityError
from sentinel.store.factory import build_store
from sentinel.store.protocol import EventStore
from sentinel.store.retention import RetentionPolicy
from sentinel.store.sqlite import SQLiteEventStore

_PG_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")
_REPO = Path(__file__).resolve().parents[2]
_MIGRATED = False

pytestmark = [pytest.mark.contract]


def _event(session_id: str, seq: int, type_: str = TOOL_CALL) -> Event:
    return Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=seq,
        ts=datetime.now(UTC),
        type=type_,
        payload={"seq": seq},
    )


def _migrate_postgres() -> None:
    global _MIGRATED
    if _MIGRATED:
        return
    env = dict(os.environ, SENTINEL_STORE_DSN=_PG_DSN or "")
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(_REPO),
        env=env,
        check=True,
        capture_output=True,
    )
    _MIGRATED = True


async def _truncate_postgres() -> None:
    import asyncpg

    conn = await asyncpg.connect(_PG_DSN or "")
    try:
        await conn.execute(
            "TRUNCATE tombstones, event_refs, flags, events, sessions RESTART IDENTITY CASCADE"
        )
    finally:
        await conn.close()


async def _write_flag(store: EventStore, session_id: str, flag_id: str) -> None:
    """Insert a flag row through the backend's own connection."""
    from datetime import UTC, datetime

    created_at = datetime.now(UTC)
    if isinstance(store, SQLiteEventStore):
        conn = await store._conn()
        await conn.execute(
            "INSERT INTO flags (flag_id, session_id, module, module_version, "
            "category, severity, confidence, summary, created_at) "
            "VALUES (?, ?, 'gating', '0.1', 'test', 'low', 0.9, 'a flag', ?)",
            (flag_id, session_id, created_at.isoformat()),
        )
    else:
        async with await store._session() as session:  # type: ignore[attr-defined]
            from sqlalchemy import text

            await session.execute(
                text(
                    "INSERT INTO flags (flag_id, session_id, module, module_version, "
                    "category, severity, confidence, summary, created_at) "
                    "VALUES (:f, :s, 'gating', '0.1', 'test', 'low', 0.9, 'a flag', :c)"
                ),
                {"f": flag_id, "s": session_id, "c": created_at},
            )
            await session.commit()


@pytest.fixture(params=["sqlite", "postgres"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[EventStore]:
    backend = request.param
    if backend == "postgres":
        if not _PG_DSN:
            pytest.skip("SENTINEL_TEST_POSTGRES_DSN not set")
        _migrate_postgres()
        await _truncate_postgres()
        result = build_store(_PG_DSN)
        yield result
        await result.close()
        await _truncate_postgres()
        return
    result = SQLiteEventStore(":memory:")
    yield result
    await result.close()


# -- writes ---------------------------------------------------------------


async def test_append_and_replay_round_trip(store: EventStore) -> None:
    session_id = new_event_id()
    for seq in (2, 0, 1, 3):
        await store.append(_event(session_id, seq))
    events = await store.get_session(session_id)
    assert [e.seq for e in events] == [0, 1, 2, 3]
    assert [e.payload for e in events] == [{"seq": 0}, {"seq": 1}, {"seq": 2}, {"seq": 3}]


async def test_append_is_idempotent_by_event_id(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0)
    await store.append(event)
    await store.append(event)
    assert len(await store.get_session(session_id)) == 1


async def test_conflicting_session_seq_raises(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append(_event(session_id, 0, type_=LLM_REQUEST))
    with pytest.raises(Exception, match=r"(?i)unique|integrity"):
        await store.append(_event(session_id, 0, type_=TOOL_CALL))


async def test_append_batch_persists_all_in_order(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append_batch([_event(session_id, i, type_=LLM_REQUEST) for i in range(50)])
    events = await store.get_session(session_id)
    assert len(events) == 50
    assert events[0].seq == 0
    assert events[49].seq == 49


async def test_cross_session_ref_is_rejected(store: EventStore) -> None:
    other = new_event_id()
    await store.append(_event(other, 0, type_=LLM_REQUEST))
    session_id = new_event_id()
    with pytest.raises(RefIntegrityError):
        await store.append(
            _event(session_id, 0, type_=LLM_RESPONSE).model_copy(
                update={
                    "refs": [RefLink(event_id=_event(other, 1).event_id, kind=RefKind.CAUSED_BY)]
                }
            )
        )


async def test_same_session_ref_is_replayed(store: EventStore) -> None:
    session_id = new_event_id()
    request = _event(session_id, 0, type_=LLM_REQUEST)
    await store.append(request)
    response = _event(session_id, 1, type_=LLM_RESPONSE).model_copy(
        update={"refs": [RefLink(event_id=request.event_id, kind=RefKind.CAUSED_BY)]}
    )
    await store.append(response)
    replay = await store.get_session(session_id)
    assert replay[1].refs == [RefLink(event_id=request.event_id, kind=RefKind.CAUSED_BY)]


# -- reads ----------------------------------------------------------------


async def test_iter_session_streams_in_order(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append_batch([_event(session_id, i) for i in range(20)])
    streamed = [e.seq async for e in store.iter_session(session_id)]
    assert streamed == list(range(20))


async def test_iter_session_after_seq_and_limit(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append_batch([_event(session_id, i) for i in range(20)])
    limited = [e.seq async for e in store.iter_session(session_id, after_seq=14, limit=3)]
    assert limited == [15, 16, 17]


async def test_call_graph_returns_typed_edges(store: EventStore) -> None:
    session_id = new_event_id()
    start = _event(session_id, 0, type_=SESSION_START)
    request = _event(session_id, 1, type_=LLM_REQUEST).model_copy(
        update={"refs": [RefLink(event_id=start.event_id, kind=RefKind.PARENT)]}
    )
    response = _event(session_id, 2, type_=LLM_RESPONSE).model_copy(
        update={"refs": [RefLink(event_id=request.event_id, kind=RefKind.CAUSED_BY)]}
    )
    await store.append_batch([start, request, response])
    edges = await store.get_call_graph(session_id)
    assert [(e.from_seq, e.to_seq, e.kind) for e in edges] == [
        (1, 0, RefKind.PARENT),
        (2, 1, RefKind.CAUSED_BY),
    ]
    assert edges[0].to_type == SESSION_START


async def test_list_sessions_filters(store: EventStore) -> None:
    optimistic = new_event_id()
    old = new_event_id()
    start_old = datetime.now(UTC) - timedelta(days=10)
    for sid, agent, started in [
        (optimistic, "agent-a", datetime.now(UTC) - timedelta(hours=1)),
        (old, "agent-b", start_old),
    ]:
        event = _event(sid, 0, type_=SESSION_START)
        event = event.model_copy(
            update={
                "ts": started,
                "payload": {"agent_id": agent},
            }
        )
        await store.append(event)

    all_rows = await store.list_sessions()
    assert len(all_rows) == 2

    by_agent = await store.list_sessions(agent_id="agent-a")
    assert [r.session_id for r in by_agent] == [optimistic]

    since = await store.list_sessions(since=datetime.now(UTC) - timedelta(hours=2))
    assert [r.session_id for r in since] == [optimistic]

    limited = await store.list_sessions(limit=1)
    assert len(limited) == 1


async def test_list_sessions_has_flags(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append(_event(session_id, 0, type_=SESSION_START))
    await _write_flag(store, session_id, new_event_id())
    flagged = await store.list_sessions(has_flags=True)
    assert [r.session_id for r in flagged] == [session_id]
    summary = flagged[0]
    assert summary.has_flags is True
    assert summary.event_count == 1


# -- ops ------------------------------------------------------------------


async def test_detect_gaps_full_and_holey(store: EventStore) -> None:
    full = new_event_id()
    await store.append_batch([_event(full, i) for i in range(5)])
    assert await store.detect_gaps(full) == []

    holey = new_event_id()
    await store.append_batch([_event(holey, s) for s in (0, 1, 4, 7)])
    gaps = await store.detect_gaps(holey)
    assert [(g.first, g.last) for g in gaps] == [(2, 3), (5, 6)]


async def test_health_counts_and_gaps(store: EventStore) -> None:
    s1 = new_event_id()
    s2 = new_event_id()
    await store.append(_event(s1, 0, type_=SESSION_START))
    await store.append(_event(s1, 1, type_=LLM_REQUEST))
    await store.append(_event(s2, 0, type_=SESSION_START))
    health = await store.health()
    assert health.sessions == 2
    assert health.events == 3
    assert health.gap_count == 0
    assert health.sessions_with_gaps == 0
    assert health.oldest_event is not None
    assert health.newest_event is not None


async def test_prune_tombstones_expired_and_preserves_legal_hold(store: EventStore) -> None:
    now = datetime.now(UTC)
    expired = new_event_id()
    held = new_event_id()
    for sid in (expired, held):
        await store.append(
            _event(sid, 0, type_=LLM_REQUEST).model_copy(update={"ts": now - timedelta(days=60)})
        )
        await store.append(
            _event(sid, 1, type_=TOOL_CALL).model_copy(update={"ts": now - timedelta(days=60)})
        )
        await store.append(
            _event(sid, 2, type_=LLM_REQUEST).model_copy(update={"ts": now - timedelta(hours=1)})
        )

    policy = RetentionPolicy(
        default_ttl=timedelta(days=7),
        legal_hold_sessions=frozenset({held}),
    )
    report = await store.prune(policy)

    assert report.pruned_events == 2
    assert report.tombstoned == 2
    assert report.locked_sessions == 1

    assert len(await store.get_session(expired)) == 1  # only the fresh event
    assert len(await store.get_session(held)) == 3  # legal hold untouched

    health = await store.health()
    assert health.events == 4
    assert health.tombstones == 2


async def test_disabled_policy_prunes_nothing(store: EventStore) -> None:
    session_id = new_event_id()
    await store.append(_event(session_id, 0, type_=LLM_REQUEST))
    report = await store.prune(RetentionPolicy())
    assert report.pruned_events == 0
    assert len(await store.get_session(session_id)) == 1


# -- evaluator flags (``S3-T1``) -------------------------------------------


def _flag(session_id: str, event_id: str, **overrides: object) -> Flag:
    base: dict[str, object] = {
        "session_id": session_id,
        "module": "provenance",
        "module_version": "0.1.0",
        "category": "ungrounded_claim",
        "confidence": 0.9,
        "summary": "cites a tool that was never called",
        "evidence": [EvidenceRef(event_id=event_id, role=EvidenceRole.CLAIM, seq=0)],
        "created_at": datetime.now(UTC),
        "dedupe_key": "k1",
    }
    base.update(overrides)
    return Flag.create(**base)  # type: ignore[arg-type]


async def test_flag_round_trip_and_idempotency(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    flag = _flag(session_id, event.event_id, details={"claimed": "42", "observed": None})

    assert await store.put_flag(flag) is True
    assert await store.put_flag(flag) is False
    assert await store.put_flags([flag]) == 0

    stored = await store.get_flags(session_id=session_id)
    assert len(stored) == 1
    assert stored[0] == flag


async def test_flag_batch_and_query_filters(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    flags = [
        _flag(
            session_id,
            event.event_id,
            dedupe_key=f"k{i}",
            severity=severity,
            category=category,
            confidence=confidence,
            created_at=datetime.now(UTC) + timedelta(seconds=i),
        )
        for i, (severity, category, confidence) in enumerate(
            [
                (Severity.LOW, "ungrounded_claim", 0.5),
                (Severity.CRITICAL, "contradicted_claim", 0.99),
                (Severity.MEDIUM, "ungrounded_claim", 0.7),
            ]
        )
    ]
    assert await store.put_flags(flags) == 3
    assert await store.put_flags(flags[:2]) == 0

    everything = await store.get_flags(session_id=session_id)
    assert len(everything) == 3
    # newest first, and severity breaks ties on identical instants
    assert [f.created_at for f in everything] == sorted(
        (f.created_at for f in everything), reverse=True
    )
    assert len(await store.get_flags(min_severity=Severity.MEDIUM, session_id=session_id)) == 2
    assert len(await store.get_flags(category="contradicted_claim")) == 1
    assert len(await store.get_flags(module="provenance")) == 3
    assert len(await store.get_flags(min_confidence=0.9)) == 1
    assert len(await store.get_flags(limit=1)) == 1
    assert await store.get_flags(session_id=new_event_id()) == []


async def test_flag_review_only_filter_is_persisted(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    await store.put_flags(
        [
            _flag(session_id, event.event_id, dedupe_key="a", review_only=True),
            _flag(session_id, event.event_id, dedupe_key="b", review_only=False),
        ]
    )
    assert len(await store.get_flags(review_only=True)) == 1
    assert len(await store.get_flags(review_only=False)) == 1


async def test_adjudication_survives_an_evaluator_rerun(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    flag = _flag(session_id, event.event_id)
    await store.put_flag(flag)

    assert (
        await store.adjudicate_flag(
            flag.flag_id, Adjudication.CONFIRMED, adjudicated_by="reviewer@example"
        )
        is True
    )
    # re-running the evaluator must not resurrect the row as pending
    assert await store.put_flag(flag) is False

    confirmed = await store.get_flags(adjudication=Adjudication.CONFIRMED)
    assert [f.flag_id for f in confirmed] == [flag.flag_id]
    assert confirmed[0].adjudicated_by == "reviewer@example"
    assert confirmed[0].adjudicated_at is not None
    assert await store.get_flags(adjudication=Adjudication.REJECTED) == []


async def test_adjudicating_an_unknown_flag_reports_false(store: EventStore) -> None:
    assert (
        await store.adjudicate_flag(
            new_event_id(), Adjudication.REJECTED, adjudicated_by="reviewer@example"
        )
        is False
    )


async def test_adjudication_is_first_write_wins(store: EventStore) -> None:
    """Both backends refuse to re-decide a flag, and keep the original author."""
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    flag = _flag(session_id, event.event_id)
    await store.put_flag(flag)

    at = datetime(2026, 9, 28, 9, 30, tzinfo=UTC)
    assert (
        await store.adjudicate_flag(
            flag.flag_id, Adjudication.CONFIRMED, adjudicated_by="first@example", at=at
        )
        is True
    )
    assert (
        await store.adjudicate_flag(
            flag.flag_id, Adjudication.REJECTED, adjudicated_by="second@example"
        )
        is False
    )

    stored = await store.get_flags(session_id=session_id)
    assert [f.flag_id for f in stored] == [flag.flag_id]
    assert stored[0].adjudication is Adjudication.CONFIRMED
    assert stored[0].adjudicated_by == "first@example"
    assert stored[0].adjudicated_at == at


async def test_flagged_session_appears_in_flag_filtered_listing(store: EventStore) -> None:
    session_id = new_event_id()
    event = _event(session_id, 0, type_=LLM_RESPONSE)
    await store.append(event)
    assert await store.list_sessions(has_flags=True) == []
    await store.put_flag(_flag(session_id, event.event_id))
    flagged = await store.list_sessions(has_flags=True)
    assert [row.session_id for row in flagged] == [session_id]
    assert flagged[0].has_flags is True
