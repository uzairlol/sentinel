"""Demo capture worker for the reference compose stack (S2-T15).

Opens a :class:`~sentinel.capture.writer.BatchedWriter` bound to the Postgres
store at ``SENTINEL_STORE_DSN`` and emits a heartbeat boundary event every
``SENTINEL_DEMO_INTERVAL`` seconds (default 5). This is the token integration
that proves a long-running capture process can stream events into the append-
only store; a real deployment replaces the synthetic loop with instrumentor
events from the host application.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from datetime import UTC, datetime

from sentinel.capture import BatchedWriter
from sentinel.models.events import SESSION_START, Event, new_event_id
from sentinel.store.factory import build_store


async def main() -> None:
    """Run the heartbeat capture loop until interrupted."""
    dsn = os.getenv("SENTINEL_STORE_DSN") or "sentinel.sqlite3"
    interval = float(os.getenv("SENTINEL_DEMO_INTERVAL", "5"))
    session_id = new_event_id()
    store = build_store(dsn)
    writer = BatchedWriter(store, batch_max_size=1, fail_open=False)
    await writer.start()
    seq = 0
    try:
        with suppress(KeyboardInterrupt, asyncio.CancelledError):
            while True:
                event = Event(
                    event_id=new_event_id(),
                    session_id=session_id,
                    seq=seq,
                    ts=datetime.now(UTC),
                    type=SESSION_START,
                    payload={"heartbeat": True, "seq": seq},
                )
                await writer.submit(event)
                seq += 1
                await asyncio.sleep(interval)
    finally:
        await writer.close()
        await store.close()


if __name__ == "__main__":
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
