"""Tests for the OpenAI-compatible transport capture (``S1-T9``).

Covers the non-streaming and streaming paths with a mocked HTTP transport:
event shape, response ``caused_by`` refs, non-2xx capture-then-raise, live chunk
forwarding with a size-capped transcript, and opt-in header capture.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from sentinel import SQLiteEventStore, session
from sentinel.eval.session import reasoning_text_of
from sentinel.instrument.openai_compat import (
    MAX_CAPTURED_TEXT,
    TransportCallError,
    chat_completion,
    chat_completion_stream,
)
from sentinel.models.events import RefKind

URL = "https://api.openai.com/v1/chat/completions"
BODY = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
FAKE_REPLY = {"id": "chatcmpl-1", "choices": [{"message": {"role": "assistant", "content": "yo"}}]}


async def test_non_streaming_captures_request_and_response() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=FAKE_REPLY))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                reply = await chat_completion(client, ctx, url=URL, request=BODY)

        assert reply["id"] == "chatcmpl-1"
        events = await store.get_session(ctx.session_id)
        assert [e.type for e in events] == [
            "session.start",
            "llm.request",
            "llm.response",
            "session.end",
        ]
        req, resp = events[1], events[2]
        assert req.payload["method"] == "POST"
        assert req.payload["url"] == URL
        assert req.payload["request"] == BODY
        assert resp.payload["status_code"] == 200
        assert resp.payload["latency_ms"] >= 0
        assert resp.payload["response"] == FAKE_REPLY
        assert resp.refs[0].kind == RefKind.CAUSED_BY
        assert resp.refs[0].event_id == req.event_id
    finally:
        await store.close()


async def test_non_success_captures_then_raises() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                try:
                    await chat_completion(client, ctx, url=URL, request=BODY)
                    raise AssertionError("expected TransportCallError")
                except TransportCallError:
                    pass

        events = await store.get_session(ctx.session_id)
        resp = events[2]
        assert resp.type == "llm.response"
        assert resp.payload["status_code"] == 500
        assert resp.refs[0].event_id == events[1].event_id
    finally:
        await store.close()


async def test_headers_hidden_by_default_and_opt_in_captured() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=FAKE_REPLY))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await chat_completion(
                    client,
                    ctx,
                    url=URL,
                    request=BODY,
                    headers={"Authorization": "Bearer sk-secret12345", "X-Custom": "v"},
                )

            events = await store.get_session(ctx.session_id)
            assert "headers" not in events[1].payload

            async with session(store) as ctx2, httpx.AsyncClient() as client2:
                await chat_completion(
                    client2,
                    ctx2,
                    url=URL,
                    request=BODY,
                    headers={"Authorization": "Bearer sk-secret12345", "X-Custom": "v"},
                    capture_headers=True,
                )
            events2 = await store.get_session(ctx2.session_id)
            assert events2[1].payload["headers"] == {
                "Authorization": "Bearer sk-secret12345",
                "X-Custom": "v",
            }
    finally:
        await store.close()


async def test_stream_forwards_chunks_and_captures_metadata() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        payload = 'data: {"c":"x"}\n\ndata: [DONE]\n\n'
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, text=payload))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                chunks: list[bytes] = [
                    chunk
                    async for chunk in chat_completion_stream(
                        client, ctx, url=URL, request={**BODY, "stream": True}
                    )
                ]

        assert b"".join(chunks) == payload.encode()
        events = await store.get_session(ctx.session_id)
        resp = events[2]
        assert resp.payload["stream"] is True
        assert resp.payload["truncated"] is False
        assert resp.payload["transcript"] == payload
        assert resp.payload["chunk_count"] == 1
        assert resp.refs[0].event_id == events[1].event_id
    finally:
        await store.close()


async def test_stream_transcript_is_capped_not_buffered() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        big = "a" * (MAX_CAPTURED_TEXT * 2)
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, text=big))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                chunks = [
                    chunk
                    async for chunk in chat_completion_stream(client, ctx, url=URL, request=BODY)
                ]

        assert len(b"".join(chunks)) == MAX_CAPTURED_TEXT * 2
        events = await store.get_session(ctx.session_id)
        resp = events[2]
        assert len(resp.payload["transcript"]) == MAX_CAPTURED_TEXT
        assert resp.payload["truncated"] is True
    finally:
        await store.close()


async def test_stream_non_success_captures_then_raises() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(429, text="rate limited"))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                stream = chat_completion_stream(client, ctx, url=URL, request=BODY)
                with pytest.raises(TransportCallError):
                    await anext(stream)

        events = await store.get_session(ctx.session_id)
        assert events[2].payload["status_code"] == 429
        assert events[2].payload["chunk_count"] == 0
    finally:
        await store.close()


# -- reasoning capture (``S3-T5`` gap)


async def test_a_reasoning_trace_is_lifted_out_of_the_choices() -> None:
    """``reasoning_content`` lives at ``choices[i].message`` — two levels down a
    list. The whole body is stored either way; without naming it, the trace is
    captured and unreachable."""
    reply = {
        "choices": [
            {"message": {"role": "assistant", "content": "$49.", "reasoning_content": "maybe $79"}}
        ]
    }
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=reply))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await chat_completion(client, ctx, url=URL, request=BODY)

        resp = next(e for e in await store.get_session(ctx.session_id) if e.type == "llm.response")
        assert resp.payload["reasoning"] == ["maybe $79"]
        assert reasoning_text_of(resp) == "maybe $79"
    finally:
        await store.close()


async def test_reasoning_is_read_from_delta_for_a_streamed_style_body() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(
                return_value=httpx.Response(
                    200, json={"choices": [{"delta": {"thinking": "step one"}}]}
                )
            )
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await chat_completion(client, ctx, url=URL, request=BODY)

        resp = next(e for e in await store.get_session(ctx.session_id) if e.type == "llm.response")
        assert reasoning_text_of(resp) == "step one"
    finally:
        await store.close()


async def test_no_reasoning_key_when_the_provider_returns_none() -> None:
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=FAKE_REPLY))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await chat_completion(client, ctx, url=URL, request=BODY)

        resp = next(e for e in await store.get_session(ctx.session_id) if e.type == "llm.response")
        assert "reasoning" not in resp.payload
    finally:
        await store.close()


async def test_a_reasoning_trace_is_capped() -> None:
    huge = "x" * (MAX_CAPTURED_TEXT + 1000)
    reply = {"choices": [{"message": {"content": "a", "reasoning_content": huge}}]}
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=reply))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                await chat_completion(client, ctx, url=URL, request=BODY)

        resp = next(e for e in await store.get_session(ctx.session_id) if e.type == "llm.response")
        assert len(resp.payload["reasoning"][0]) == MAX_CAPTURED_TEXT
    finally:
        await store.close()


async def test_a_reasoning_trace_survives_a_malformed_body() -> None:
    """Choices that are not a list of mappings must not raise inside capture.

    Capture never breaks the host call (INV-6), including when a provider
    returns something the rules were not written against.
    """
    store = SQLiteEventStore(":memory:")
    try:
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json={"choices": "nope"}))
            async with session(store) as ctx, httpx.AsyncClient() as client:
                reply = await chat_completion(client, ctx, url=URL, request=BODY)

        assert reply == {"choices": "nope"}
        resp = next(e for e in await store.get_session(ctx.session_id) if e.type == "llm.response")
        assert "reasoning" not in resp.payload
    finally:
        await store.close()
