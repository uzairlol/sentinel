"""Raw HTTP capture with the generic wrapper (``S1-T18``).

Shows the low-level path: a raw OpenAI-compatible ``chat_completion`` call plus
a ``trace``-decorated wrapper around it, captured into SQLite and replayed.

Points at a local Ollama server by default (``http://127.0.0.1:11434``, model
``llama3.2``). Point ``BASE_URL`` at any OpenAI-compatible endpoint instead.
Run from the repo root::

    uv run python examples/raw_ollama.py
"""

from __future__ import annotations

import asyncio

import httpx

from sentinel import SQLiteEventStore, session
from sentinel.instrument import trace
from sentinel.instrument.openai_compat import chat_completion

STORE_PATH = "raw_ollama.sqlite3"
BASE_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "llama3.2"


@trace(kind="tool")
async def ask_ollama(client: httpx.AsyncClient, prompt: str) -> dict:
    """One raw chat completion, wrapped so it lands in the call graph too."""
    body = await chat_completion(
        client,
        current_session(),
        url=BASE_URL,
        request={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return {"reply": body}


def current_session():  # noqa: ANN201
    """Resolve the session context active for the calling task."""
    from sentinel.instrument.session import current_session as cs

    return cs()


async def main() -> None:
    """Wrap one raw chat completion in a trace, then print the session log."""
    store = SQLiteEventStore(STORE_PATH)
    try:
        async with session(store) as ctx, httpx.AsyncClient() as client:
            result = await ask_ollama(client, "Say hello in one word.")
        print(f"ollama replied: {result['reply']['choices'][0]['message']['content']!r}")

        for event in await store.get_session(ctx.session_id):
            kind = event.payload.get("function") or event.payload.get("method") or ""
            extra = f" {kind}" if kind else ""
            print(f"  [{event.seq}] {event.type}{extra}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
