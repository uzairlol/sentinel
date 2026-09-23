"""Raw Ollama HTTP instrumentation (``S0-T3``).

Wraps a POST to ``/api/chat`` and emits two events — ``llm.request`` then
``llm.response`` — with the response ``refs``-linked to the request that caused
it (INV-3). Per INV-1 this module performs *zero* analysis: it serializes the
boundary crossing and nothing more.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from sentinel.instrument.session import SessionContext
from sentinel.models.events import LLM_REQUEST, LLM_RESPONSE

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
CHAT_PATH = "/api/chat"


class OllamaChatError(RuntimeError):
    """Raised when Ollama returns a non-2xx response to ``/api/chat``."""


async def instrument_ollama_call(
    client: httpx.AsyncClient,
    session_ctx: SessionContext,
    *,
    model: str,
    messages: list[dict[str, Any]],
    base_url: str = DEFAULT_BASE_URL,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST ``/api/chat`` to *base_url* and capture the call as events.

    Emits an ``llm.request`` event carrying the full request body, then an
    ``llm.response`` event carrying status, latency in ms, and the parsed
    response body. The response event references the request event. Returns the
    parsed JSON response body (``stream=False``).
    """
    url = base_url.rstrip("/") + CHAT_PATH
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        request_body["options"] = options

    request_event = await session_ctx.capture(
        type=LLM_REQUEST,
        payload={"url": url, "model": model, "request": request_body},
    )

    started = time.perf_counter()
    response = await client.post(url, json=request_body)
    latency_ms = (time.perf_counter() - started) * 1000.0

    if response.status_code == 200:
        body: dict[str, Any] = response.json()
        await session_ctx.capture(
            type=LLM_RESPONSE,
            payload={
                "url": url,
                "model": model,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "response": body,
            },
            refs=[request_event.event_id],
        )
        return body

    await session_ctx.capture(
        type=LLM_RESPONSE,
        payload={
            "url": url,
            "model": model,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
        refs=[request_event.event_id],
    )
    raise OllamaChatError(
        f"Ollama /api/chat returned {response.status_code}: {response.text[:200]!r}"
    )
