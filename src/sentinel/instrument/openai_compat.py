"""OpenAI-compatible chat-completions transport capture (``S1-T9``).

Generalizes the ``S0`` Ollama path into a transport-level capture that works
against any OpenAI-compatible ``/chat/completions`` style endpoint. Two entry
points:

* :func:`chat_completion` — non-streaming; posts the request, returns the parsed
  JSON response body.
* :func:`chat_completion_stream` — an async generator that forwards chunks to
  the host exactly as they arrive (never buffering the whole body before
  forwarding) while accumulating a size-capped transcript for the event.

Both emit an ``llm.request`` event and then an ``llm.response`` event that
``caused_by``-refs it (INV-3). Per INV-1 this module only captures the boundary
crossing. Request headers are serialised *only* when ``capture_headers=True`` so
credentials never reach the store by default; the response body is never
captured in full for streams beyond the capped transcript.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from sentinel.instrument.session import SessionContext
from sentinel.models.events import LLM_REQUEST, LLM_RESPONSE, RefKind, RefLink

#: Cap on the stream transcript captured per exchange (``S1-T16`` reserves the
#: full per-event payload cap for ``S2``; this keeps the transport safe today).
MAX_CAPTURED_TEXT = 65_536


class TransportCallError(RuntimeError):
    """Raised when the transport returns a non-2xx response to the call."""


async def chat_completion(
    client: httpx.AsyncClient,
    session_ctx: SessionContext,
    *,
    url: str,
    request: dict[str, Any],
    method: str = "POST",
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | tuple[str, str] | None = None,
    capture_headers: bool = False,
) -> dict[str, Any]:
    """POST an OpenAI-compatible chat completion and capture the exchange.

    Emits ``llm.request`` (method, url, body, optional headers) then
    ``llm.response`` (status, latency ms, parsed JSON body) referencing it.
    Returns the parsed JSON response body. A non-2xx status is captured and then
    raised as :class:`TransportCallError` — the call itself fails, capture never
    does.
    """
    request_payload: dict[str, Any] = {"method": method, "url": url, "request": request}
    if capture_headers:
        request_payload["headers"] = headers or {}

    request_event = await session_ctx.capture(type=LLM_REQUEST, payload=request_payload)

    started = time.perf_counter()
    response = await client.request(method, url, json=request, headers=headers, auth=auth)
    latency_ms = (time.perf_counter() - started) * 1000.0

    body = _parse_json_body(response)
    payload: dict[str, Any] = {
        "method": method,
        "url": url,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "response": body,
    }
    traces = _reasoning_traces(body)
    if traces:
        payload["reasoning"] = traces
    await session_ctx.capture(
        type=LLM_RESPONSE,
        payload=payload,
        refs=[RefLink(event_id=request_event.event_id, kind=RefKind.CAUSED_BY)],
    )

    if response.is_success:
        return body

    raise TransportCallError(f"chat completion at {url} returned {response.status_code}")


async def chat_completion_stream(
    client: httpx.AsyncClient,
    session_ctx: SessionContext,
    *,
    url: str,
    request: dict[str, Any],
    method: str = "POST",
    headers: dict[str, str] | None = None,
    auth: httpx.Auth | tuple[str, str] | None = None,
    capture_headers: bool = False,
) -> AsyncIterator[bytes]:
    """Stream a chat completion, forwarding chunks and capturing live.

    Chunks are yielded to the host in arrival order before any capture work, so
    forwarding has zero buffering latency. The event transcript accumulates at
    most :data:`MAX_CAPTURED_TEXT` bytes. On a non-2xx status the response
    event is captured (with the status) and :class:`TransportCallError` raised.
    """
    request_payload: dict[str, Any] = {"method": method, "url": url, "request": request}
    if capture_headers:
        request_payload["headers"] = headers or {}

    request_event = await session_ctx.capture(type=LLM_REQUEST, payload=request_payload)

    started = time.perf_counter()
    async with client.stream(method, url, json=request, headers=headers, auth=auth) as response:
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not response.is_success:
            await _capture_stream_response(
                session_ctx,
                event_id=request_event.event_id,
                method=method,
                url=url,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
            raise TransportCallError(
                f"chat completion stream at {url} returned {response.status_code}"
            )

        captured = bytearray()
        total_bytes = 0
        chunk_count = 0
        async for chunk in response.aiter_bytes():
            chunk_count += 1
            total_bytes += len(chunk)
            yield chunk
            if len(captured) < MAX_CAPTURED_TEXT:
                captured.extend(chunk[: MAX_CAPTURED_TEXT - len(captured)])

        await _capture_stream_response(
            session_ctx,
            event_id=request_event.event_id,
            method=method,
            url=url,
            status_code=response.status_code,
            latency_ms=latency_ms,
            chunk_count=chunk_count,
            transcript=captured.decode("utf-8", errors="replace"),
            truncated=total_bytes > MAX_CAPTURED_TEXT,
        )


async def _capture_stream_response(
    session_ctx: SessionContext,
    *,
    event_id: str,
    method: str,
    url: str,
    status_code: int,
    latency_ms: float,
    chunk_count: int = 0,
    transcript: str = "",
    truncated: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "method": method,
        "url": url,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "stream": True,
        "chunk_count": chunk_count,
    }
    if transcript:
        payload["transcript"] = transcript
        payload["truncated"] = truncated
    await session_ctx.capture(
        type=LLM_RESPONSE,
        payload=payload,
        refs=[RefLink(event_id=event_id, kind=RefKind.CAUSED_BY)],
    )


def _parse_json_body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"raw": response.text[:MAX_CAPTURED_TEXT]}
    if not isinstance(body, Mapping):
        return {"raw": repr(body)[:MAX_CAPTURED_TEXT]}
    return dict(body)


#: Provider vocabularies for a model's reasoning trace on an OpenAI-shaped
#: message. OpenAI itself calls it ``reasoning``; DeepSeek and several gateways
#: call it ``reasoning_content``; Ollama's OpenAI-compatible endpoint and Qwen
#: call it ``thinking``.
_REASONING_FIELDS = ("reasoning", "reasoning_content", "thinking", "thought")


def _reasoning_traces(body: Mapping[str, Any]) -> list[str]:
    """The reasoning traces in an OpenAI-shaped completion body, in choice order.

    Lifted out of the raw body rather than left for the evaluator to find,
    because the whole body is already stored: without this the trace is present
    but unaddressable, because ``reasoning_content`` lives at
    ``choices[i].message.reasoning_content`` — a list-nested path no generic
    key lookup walks without re-implementing the response schema here.

    Each trace is capped at :data:`MAX_CAPTURED_TEXT`: a reasoning model can
    produce a trace far longer than its answer, and the trace is checked against
    evidence like any other claim but is not worth an unbounded payload.
    """
    choices = body.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return []
    traces: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message") or choice.get("delta")
        if not isinstance(message, Mapping):
            continue
        for name in _REASONING_FIELDS:
            value = message.get(name)
            if isinstance(value, str) and value:
                traces.append(value[:MAX_CAPTURED_TEXT])
                break
    return traces
