"""End-to-end: ``sentinel replay`` against the Postgres store (``S2`` exit #6).

Appends a session straight into Postgres and replays it via the CLI
``--dsn``, proving the replay surface works against the production store (not
just the SQLite dev store). Runs in the CI integration job; skipped offline
when ``SENTINEL_TEST_POSTGRES_DSN`` is not set.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest

from sentinel.models.events import LLM_REQUEST, SESSION_START, Event, new_event_id
from sentinel.store.factory import build_store

_DATABASE_DSN = os.getenv("SENTINEL_TEST_POSTGRES_DSN")

pytestmark = [
    pytest.mark.skipif(not _DATABASE_DSN, reason="SENTINEL_TEST_POSTGRES_DSN not set"),
    pytest.mark.e2e,
]


def test_cli_replay_works_against_postgres(capsys: pytest.CaptureFixture[str]) -> None:
    dsn = _DATABASE_DSN
    assert dsn is not None
    session_id = asyncio.run(_seed(dsn))

    from sentinel._cli import main

    code = main(["replay", session_id, "--dsn", dsn])

    output = capsys.readouterr().out
    assert code == 0
    assert f"Session {session_id}" in output
    assert "session.start" in output
    assert "llm.request" in output


async def _seed(dsn: str) -> str:
    store = build_store(dsn)
    try:
        session_id = new_event_id()
        await store.append_batch(
            [
                Event(
                    event_id=new_event_id(),
                    session_id=session_id,
                    seq=0,
                    ts=datetime.now(UTC),
                    type=SESSION_START,
                    payload={"via": "postgres"},
                ),
                Event(
                    event_id=new_event_id(),
                    session_id=session_id,
                    seq=1,
                    ts=datetime.now(UTC),
                    type=LLM_REQUEST,
                    payload={"role": "user", "content": "hi"},
                ),
            ]
        )
        return session_id
    finally:
        await store.close()
