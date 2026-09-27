"""Evaluator flag persistence (``S3-T1``): contract + dev-store tests.

Two layers:

* the **parity contract** in ``tests/contract/test_store_parity.py`` runs the
  same flag tests against SQLite and Postgres;
* this file covers the SQLite dev store's own behaviour (statement/column
  agreement, backfill of a pre-``S3`` file, adjudication round trip).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentinel.models.events import LLM_RESPONSE, SESSION_START, Event, new_event_id
from sentinel.models.flags import Adjudication, EvidenceRef, EvidenceRole, Flag, Severity
from sentinel.store import sqlite as sqlite_store
from sentinel.store.retention import RetentionPolicy
from sentinel.store.sqlite import SQLiteEventStore

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _flag(session_id: str, **overrides: object) -> Flag:
    base: dict[str, object] = {
        "session_id": session_id,
        "module": "provenance",
        "module_version": "0.1.0",
        "category": "ungrounded_claim",
        "confidence": 0.91,
        "summary": "cites a tool that was never called",
        "evidence": [EvidenceRef(event_id=new_event_id(), role=EvidenceRole.CLAIM, seq=3)],
        "created_at": _TS,
        "dedupe_key": "k1",
    }
    base.update(overrides)
    return Flag.create(**base)  # type: ignore[arg-type]


async def _ready() -> tuple[SQLiteEventStore, str]:
    """A store holding one session, ready to accept flags."""
    store = SQLiteEventStore(":memory:")
    session_id = new_event_id()
    await store.append(
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=0,
            ts=_TS,
            type=SESSION_START,
            payload={"agent_id": "unit"},
        )
    )
    return store, session_id


# -- statement/column agreement --------------------------------------------


def test_flag_statements_match_columns() -> None:
    assert _insert_columns(sqlite_store._FLAG_INSERT) == sqlite_store._FLAG_COLUMNS
    assert _select_columns(sqlite_store._FLAG_SELECT) == sqlite_store._FLAG_COLUMNS
    assert sqlite_store._FLAG_INSERT.count("?") == len(sqlite_store._FLAG_COLUMNS)


def _insert_columns(statement: str) -> tuple[str, ...]:
    body = statement.partition("(")[2]
    return tuple(part.strip() for part in body[: body.index(")")].split(","))


def _select_columns(statement: str) -> tuple[str, ...]:
    body = statement.partition("SELECT ")[2]
    return tuple(part.strip() for part in body[: body.index(" FROM ")].split(","))


# -- pre-S3 database backfill ----------------------------------------------


_LEGACY_SCHEMA = (
    "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, agent_id TEXT, status TEXT NOT NULL "
    "DEFAULT 'active', started_at TEXT NOT NULL, ended_at TEXT, schema_version TEXT NOT NULL, "
    "meta TEXT NOT NULL DEFAULT '{}')",
    "CREATE TABLE events (event_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, seq INTEGER "
    "NOT NULL, ts TEXT NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL, refs TEXT NOT NULL, "
    "schema_version TEXT NOT NULL)",
    "CREATE TABLE flags (flag_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, event_id TEXT, "
    "module TEXT NOT NULL, module_version TEXT NOT NULL, category TEXT NOT NULL, severity TEXT "
    "NOT NULL DEFAULT 'medium', confidence REAL NOT NULL, summary TEXT NOT NULL, evidence TEXT "
    "NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, adjudication TEXT NOT NULL DEFAULT "
    "'pending', adjudicated_by TEXT, adjudicated_at TEXT, auto_resolved INTEGER)",
)


async def test_pre_s3_database_is_backfilled(tmp_path: Path) -> None:
    path = str(tmp_path / "legacy.sqlite3")
    legacy = sqlite3.connect(path)
    for statement in _LEGACY_SCHEMA:
        legacy.execute(statement)
    legacy.commit()
    legacy.close()

    store = SQLiteEventStore(path)
    conn = await store._conn()
    cursor = await conn.execute("PRAGMA table_info(flags)")
    columns = {row["name"] for row in await cursor.fetchall()}
    assert {"details", "review_only", "schema_version"} <= columns

    # the backfilled defaults match the S3 schema
    await conn.execute(
        "INSERT INTO sessions VALUES ('s', NULL, 'active', ?, NULL, '0.2', '{}')",
        (_TS.isoformat(),),
    )
    await conn.execute(
        "INSERT INTO flags (flag_id, session_id, module, module_version, category, confidence, "
        "summary, created_at) VALUES ('f', 's', 'm', '0.1', 'c', 0.5, 'sum', ?)",
        (_TS.isoformat(),),
    )
    await conn.commit()
    cursor = await conn.execute("SELECT details, review_only, schema_version FROM flags")
    row = await cursor.fetchone()
    assert row is not None
    assert dict(row) == {
        "details": "{}",
        "review_only": 0,
        "schema_version": "0.1",
    }
    await store.close()


# -- round trip -------------------------------------------------------------


async def test_put_flag_is_idempotent_and_preserves_adjudication() -> None:
    store, session_id = await _ready()
    flag = _flag(session_id)

    assert await store.put_flag(flag) is True
    assert await store.put_flag(flag) is False
    assert await store.put_flags([flag, flag]) == 0

    await store.adjudicate_flag(flag.flag_id, Adjudication.CONFIRMED, adjudicated_by="alice")
    # a re-run of the evaluator must not resurrect the row as pending
    assert await store.put_flag(flag) is False
    stored = await store.get_flags(session_id=session_id)
    assert len(stored) == 1
    assert stored[0].adjudication is Adjudication.CONFIRMED
    assert stored[0].adjudicated_by == "alice"
    assert stored[0].adjudicated_at is not None
    await store.close()


async def test_flag_round_trips_every_field() -> None:
    store, session_id = await _ready()
    events = await store.get_session(session_id)
    event_id = events[0].event_id
    flag = _flag(
        session_id,
        event_id=event_id,
        severity=Severity.CRITICAL,
        review_only=True,
        details={"claimed": "10", "observed": "12", "tolerance": 0.0},
        evidence=[
            EvidenceRef(event_id=event_id, role=EvidenceRole.CLAIM, seq=0, note="c"),
            EvidenceRef(event_id=new_event_id(), role=EvidenceRole.EVIDENCE, seq=2, note="e"),
        ],
    )
    await store.put_flag(flag)

    stored = (await store.get_flags(session_id=session_id))[0]
    assert stored == flag
    await store.close()


async def test_flag_primary_event_must_exist() -> None:
    """A flag may not point at an event the store never persisted (INV-3)."""
    store, session_id = await _ready()
    with pytest.raises(sqlite3.IntegrityError):
        await store.put_flag(_flag(session_id, event_id=new_event_id()))
    await store.close()


async def test_put_flags_batch_counts_new_rows() -> None:
    store, session_id = await _ready()
    flags = [_flag(session_id, dedupe_key=f"k{i}", confidence=0.5 + i / 100) for i in range(5)]
    assert await store.put_flags(flags) == 5
    assert await store.put_flags(flags[:3]) == 0
    assert await store.put_flags([]) == 0
    assert len(await store.get_flags(session_id=session_id)) == 5
    await store.close()


async def test_get_flags_filters_and_orders() -> None:
    store, session_id = await _ready()
    other = new_event_id()
    await store.append(
        Event(
            event_id=new_event_id(),
            session_id=other,
            seq=0,
            ts=_TS,
            type=SESSION_START,
            payload={},
        )
    )
    await store.put_flags(
        [
            _flag(
                session_id,
                dedupe_key="a",
                severity=Severity.LOW,
                category="ungrounded_claim",
                module="provenance",
                created_at=_TS,
            ),
            _flag(
                session_id,
                dedupe_key="b",
                severity=Severity.CRITICAL,
                category="contradicted_claim",
                module="provenance",
                confidence=0.99,
                created_at=_TS,
            ),
            _flag(
                session_id,
                dedupe_key="c",
                severity=Severity.MEDIUM,
                category="ungrounded_claim",
                module="memory",
                review_only=True,
                created_at=_TS,
            ),
            _flag(other, dedupe_key="d"),
        ]
    )

    newest_first = await store.get_flags(session_id=session_id)
    assert {f.severity for f in newest_first} == {Severity.LOW, Severity.CRITICAL, Severity.MEDIUM}
    assert newest_first[0].severity is Severity.CRITICAL

    assert len(await store.get_flags(min_severity=Severity.MEDIUM, session_id=session_id)) == 2
    assert len(await store.get_flags(category="ungrounded_claim", session_id=session_id)) == 2
    assert len(await store.get_flags(module="memory")) == 1
    assert len(await store.get_flags(min_confidence=0.95)) == 1
    assert len(await store.get_flags(review_only=True)) == 1
    assert len(await store.get_flags(adjudication=Adjudication.PENDING)) == 4
    assert len(await store.get_flags(adjudication=Adjudication.CONFIRMED)) == 0
    assert len(await store.get_flags(limit=2)) == 2
    assert len(await store.get_flags(limit=2, offset=2)) == 2
    assert len(await store.get_flags(limit=2, offset=10)) == 0
    assert len(await store.get_flags(session_id="nope")) == 0
    await store.close()


async def test_adjudicate_unknown_flag_reports_false() -> None:
    store, _ = await _ready()
    assert (
        await store.adjudicate_flag(new_event_id(), Adjudication.REJECTED, adjudicated_by="bob")
        is False
    )
    await store.close()


async def test_adjudicate_records_supplied_instant() -> None:
    store, session_id = await _ready()
    flag = _flag(session_id)
    await store.put_flag(flag)
    at = datetime(2026, 9, 28, 9, 30, tzinfo=UTC)
    assert await store.adjudicate_flag(
        flag.flag_id, Adjudication.REJECTED, adjudicated_by="carol", at=at
    )
    stored = (await store.get_flags(session_id=session_id))[0]
    assert stored.adjudicated_at == at
    await store.close()


async def test_flagged_events_survive_retention_prune() -> None:
    """Evidence referenced by a flag is never pruned (``S2-T12`` guarantee)."""
    store = SQLiteEventStore(":memory:")
    session_id = new_event_id()
    old = datetime.now(UTC) - timedelta(days=60)
    claim = Event(
        event_id=new_event_id(),
        session_id=session_id,
        seq=0,
        ts=old,
        type=LLM_RESPONSE,
        payload={"text": "the API shows 42"},
    )
    await store.append(claim)
    await store.put_flag(
        _flag(session_id, event_id=claim.event_id, created_at=old, dedupe_key="keep")
    )

    report = await store.prune(RetentionPolicy(default_ttl=timedelta(days=7)))
    assert report.pruned_events == 0
    assert report.retained_evidence == 1
    assert len(await store.get_session(session_id)) == 1
    await store.close()
