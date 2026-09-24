"""Backup/restore smoke test (``S2-T19``).

Dumps a seeded database with ``pg_dump``, restores it into a fresh database,
and asserts the replayed sessions are byte-identical to the originals.

Gates:
* ``SENTINEL_TEST_POSTGRES_DSN`` must be set (absent -> skip);
* ``pg_dump`` and ``psql`` must be on ``PATH`` (absent -> skip). The local
  offline dev box typically has neither; the CI integration job installs
  ``postgresql-client`` so the flow is exercised there.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

from sentinel.models.events import LLM_REQUEST, SESSION_START, Event, RefKind, RefLink, new_event_id
from sentinel.store.factory import build_store

_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")
_HAS_TOOLS = bool(shutil.which("pg_dump") and shutil.which("psql"))

pytestmark = [
    pytest.mark.skipif(not _DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set"),
    pytest.mark.skipif(not _HAS_TOOLS, reason="pg_dump/psql not on PATH"),
    pytest.mark.integration,
]

_RESTORE_DB = "sentinel_restore_smoke"


def _dsn_with_db(dbname: str) -> str:
    parts = urlsplit(_DSN or "")
    return urlunsplit((parts.scheme, parts.netloc, f"/{dbname}", "", ""))


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    # arguments are fixed local constants, never untrusted input
    return subprocess.run(args, capture_output=True, text=True, check=False)  # noqa: S603


async def _seed_session(session_id: str) -> list[Event]:
    store = build_store(_DSN)
    try:
        events: list[Event] = []
        start = Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=0,
            ts=datetime.now(UTC),
            type=SESSION_START,
            payload={"agent_id": "restore-agent"},
        )
        events.append(start)
        for seq in range(1, 51):
            payload_event = Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=seq,
                ts=datetime.now(UTC),
                type=LLM_REQUEST,
                payload={"seq": seq, "note": f"payload-{seq}"},
                refs=[RefLink(event_id=start.event_id, kind=RefKind.PARENT)]
                if seq % 10 == 0
                else [],
            )
            events.append(payload_event)
        await store.append_batch(events)
        return await store.get_session(session_id)
    finally:
        await store.close()


def _digest(events: list[Event]) -> str:
    h = hashlib.sha256()
    for event in events:
        line = (
            event.event_id
            + "\0"
            + event.session_id
            + "\0"
            + str(event.seq)
            + "\0"
            + event.type
            + "\0"
            + str(sorted((r.event_id, r.kind.value) for r in event.refs))
            + "\n"
        )
        h.update(line.encode("utf-8"))
    return h.hexdigest()


async def test_pg_dump_restore_replay_equality() -> None:
    session_id = new_event_id()
    original = await _seed_session(session_id)
    assert len(original) == 51

    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "sentinel_dump.sql"

        dump = _run("pg_dump", "--no-owner", "--no-privileges", "-f", str(dump_path), _DSN or "")
        assert dump.returncode == 0, dump.stderr
        assert dump_path.exists()
        assert dump_path.stat().st_size > 0

        drop = _run(
            "psql", "-d", _dsn_with_db("postgres"), "-c", f"DROP DATABASE IF EXISTS {_RESTORE_DB}"
        )
        assert drop.returncode == 0, drop.stderr
        create = _run(
            "psql", "-d", _dsn_with_db("postgres"), "-c", f"CREATE DATABASE {_RESTORE_DB}"
        )
        assert create.returncode == 0, create.stderr

        restore = _run(
            "psql", "-v", "ON_ERROR_STOP=1", "-d", _dsn_with_db(_RESTORE_DB), "-f", str(dump_path)
        )
        assert restore.returncode == 0, restore.stderr

        restored = build_store(_dsn_with_db(_RESTORE_DB))
        try:
            replayed = await restored.get_session(session_id)
        finally:
            await restored.close()

        assert len(replayed) == len(original)
        assert [e.seq for e in replayed] == [e.seq for e in original]
        assert [e.payload for e in replayed] == [e.payload for e in original]
        assert _digest(replayed) == _digest(original)

        refs = [r for e in replayed for r in e.refs]
        assert refs  # same-session links survived the dump/restore
        assert all(r.event_id in {e.event_id for e in original} for r in refs)

    drop = _run(
        "psql", "-d", _dsn_with_db("postgres"), "-c", f"DROP DATABASE IF EXISTS {_RESTORE_DB}"
    )
    assert drop.returncode == 0, drop.stderr
