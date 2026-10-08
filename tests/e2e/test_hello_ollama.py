"""End-to-end vertical slice test (``S0-T9``).

Fake Ollama endpoint via ``respx`` -> one instrumented chat call -> SQLite ->
replay asserts content equality and link integrity. This is the "hello sentinel"
spine that sprint S0 exists to prove.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from sentinel import SQLiteEventStore, instrument_ollama_call, session
from sentinel.eval.session import reasoning_text_of
from sentinel.instrument.ollama import OllamaChatError
from sentinel.models.events import RefKind

FAKE_REPLY = {
    "model": "llama3.2",
    "created_at": "2026-09-23T08:50:51.123Z",
    "message": {"role": "assistant", "content": "hello"},
    "done": True,
}


async def test_instrumented_ollama_call_round_trips() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(200, json=FAKE_REPLY)
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                reply = await instrument_ollama_call(
                    client,
                    ctx,
                    model="llama3.2",
                    messages=[{"role": "user", "content": "Say hello."}],
                )

        assert reply["message"]["content"] == "hello"

        events = await store.get_session(ctx.session_id)
        assert [e.type for e in events] == [
            "session.start",
            "llm.request",
            "llm.response",
            "session.end",
        ]

        request_event = events[1]
        response_event = events[2]
        assert request_event.type == "llm.request"
        assert request_event.payload["model"] == "llama3.2"
        assert request_event.payload["request"]["messages"] == [
            {"role": "user", "content": "Say hello."}
        ]

        assert response_event.type == "llm.response"
        assert [link.event_id for link in response_event.refs] == [request_event.event_id]
        assert response_event.refs[0].kind == RefKind.CAUSED_BY
        assert response_event.payload["status_code"] == 200
        assert response_event.payload["latency_ms"] >= 0
        assert response_event.payload["response"] == FAKE_REPLY

        # Replay is lossless and idempotent: a second read is byte-identical.
        assert await store.get_session(ctx.session_id) == events
    finally:
        await store.close()


async def test_ollama_error_is_captured_then_raised() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(500, text="model overloaded")
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                with pytest.raises(OllamaChatError):
                    await instrument_ollama_call(
                        client,
                        ctx,
                        model="llama3.2",
                        messages=[{"role": "user", "content": "hi"}],
                    )

        events = await store.get_session(ctx.session_id)
        assert events[-2].type == "llm.response"
        assert events[-2].payload["status_code"] == 500
        assert events[-2].refs[0].kind == RefKind.CAUSED_BY
        assert events[-2].refs[0].event_id == events[-3].event_id
    finally:
        await store.close()


async def test_instrumented_call_forwards_options() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(200, json=FAKE_REPLY)
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await instrument_ollama_call(
                    client,
                    ctx,
                    model="llama3.2",
                    messages=[{"role": "user", "content": "hi"}],
                    options={"temperature": 0},
                )

        events = await store.get_session(ctx.session_id)
        assert events[1].payload["request"]["options"] == {"temperature": 0}
    finally:
        await store.close()


async def test_an_ollama_thinking_trace_is_captured() -> None:
    """Ollama returns reasoning models' traces as ``message.thinking``."""
    reply = {
        "model": "deepseek-r1",
        "message": {
            "role": "assistant",
            "content": "The price is $49.",
            "thinking": "Could be $79, but the tool says $49.",
        },
        "done": True,
    }
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(200, json=reply)
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await instrument_ollama_call(
                    client, ctx, model="deepseek-r1", messages=[{"role": "user", "content": "?"}]
                )

        response_event = next(
            e for e in await store.get_session(ctx.session_id) if e.type == "llm.response"
        )
        assert response_event.payload["reasoning"] == "Could be $79, but the tool says $49."
        assert reasoning_text_of(response_event) == "Could be $79, but the tool says $49."
    finally:
        await store.close()


async def test_no_reasoning_key_for_an_ordinary_ollama_reply() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post("http://127.0.0.1:11434/api/chat").mock(
                return_value=httpx.Response(200, json=FAKE_REPLY)
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await instrument_ollama_call(
                    client, ctx, model="llama3.2", messages=[{"role": "user", "content": "hi"}]
                )

        response_event = next(
            e for e in await store.get_session(ctx.session_id) if e.type == "llm.response"
        )
        assert "reasoning" not in response_event.payload
    finally:
        await store.close()
