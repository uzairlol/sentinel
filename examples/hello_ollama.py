"""Sentinel quickstart: capture one Ollama chat call into SQLite, then replay it.

Requires a local Ollama server (https://ollama.com) with a model pulled, e.g.::

    ollama pull llama3.2

Run from the repo root::

    uv run python examples/hello_ollama.py
"""

from __future__ import annotations

import asyncio
import json

import httpx

from sentinel import SQLiteEventStore, instrument_ollama_call, session

MODEL = "llama3.2"
STORE_PATH = "hello_ollama.sqlite3"


async def main() -> None:
    """Run the quickstart: capture one chat call, then replay the session."""
    store = SQLiteEventStore(STORE_PATH)
    try:
        async with session(store) as ctx, httpx.AsyncClient() as client:
            reply = await instrument_ollama_call(
                client,
                ctx,
                model=MODEL,
                messages=[{"role": "user", "content": "Say hello in one word."}],
            )
            print(f"ollama replied: {reply['message']['content']!r}")

        print(f"\nReplay of session {ctx.session_id}:")
        for event in await store.get_session(ctx.session_id):
            summary = json.dumps(event.payload, sort_keys=True, separators=(",", ":"))
            print(f"  [{event.seq}] {event.type} {summary[:120]}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
