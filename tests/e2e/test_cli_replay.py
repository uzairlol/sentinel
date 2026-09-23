"""End-to-end tests for the ``sentinel replay`` CLI (``S0-T7``)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx

from sentinel import SQLiteEventStore, instrument_ollama_call, session
from sentinel._cli import main


def test_cli_replay_prints_captured_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "events.sqlite3"
    session_id = asyncio.run(_capture(path))

    code = main(["replay", session_id, "--store", str(path)])

    output = capsys.readouterr().out
    assert code == 0
    assert f"Session {session_id}" in output
    assert "llm.request" in output
    assert "llm.response" in output
    assert "session.end" in output


def test_cli_replay_unknown_session_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["replay", "01J0ABCDEFGHJKMNPQRSTVWXY", "--store", str(tmp_path / "nope.sqlite3")])

    error = capsys.readouterr().err
    assert code == 1
    assert "No events found" in error


async def _capture(path: Path) -> str:
    store = SQLiteEventStore(str(path))
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(200, json={"message": {"content": "hi"}, "done": True})
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await instrument_ollama_call(
                    client, ctx, model="llama3.2", messages=[{"role": "user", "content": "hi"}]
                )
                return ctx.session_id
    finally:
        await store.close()
