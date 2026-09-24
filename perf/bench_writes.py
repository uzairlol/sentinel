"""Event-write latency benchmark (``S2`` exit criterion 3).

Measures per-EVENT append latency against a real Postgres via
``append_batch`` (the production write path) and single ``append``, and
reports the p50/p90/p99 percentiles required by the S2 budget
("Event write (local Postgres) p99 < 10 ms", SENTINEL_TDD.md §2.7).

Run from the repo root with a live store DSN::

    SENTINEL_TEST_POSTGRES_DSN=postgresql://user:pass@host:5432/sentinel_test \
        uv run python perf/bench_writes.py

Without a DSN it falls back to an in-memory batch size probe on SQLite so the
script still runs offline; only the Postgres figures count for the budget.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from datetime import UTC, datetime

from sentinel.models.events import LLM_REQUEST, Event, new_event_id
from sentinel.store.factory import build_store
from sentinel.store.protocol import EventStore

BATCH_SIZE = 50
N_BATCHES = 300
N_SINGLE = 300
WARMUP = 20


def _percentile(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[min(int(q), 99) - 1]


def _report(label: str, per_event_us: list[float]) -> None:
    p50 = _percentile(per_event_us, 50)
    p90 = _percentile(per_event_us, 90)
    p99 = _percentile(per_event_us, 99)
    target = 10_000.0  # 10 ms budget, in microseconds
    print(
        f"{label:<24} p50={p50 / 1000:>8.3f} ms  "
        f"p90={p90 / 1000:>8.3f} ms  "
        f"p99={p99 / 1000:>8.3f} ms  {'OK' if p99 < target else 'OVER BUDGET'}"
    )


async def _run(store: EventStore) -> None:
    session_id = new_event_id()
    seq = 0

    def make_batch() -> list[Event]:
        nonlocal seq
        events = [
            Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=s + seq,
                ts=datetime.now(UTC),
                type=LLM_REQUEST,
                payload={"utf8_probe": "söme ünïcode", "n": s},
            )
            for s in range(BATCH_SIZE)
        ]
        seq += BATCH_SIZE
        return events

    for _ in range(WARMUP):
        await store.append_batch(make_batch())
    batch_events: list[float] = []
    for _ in range(N_BATCHES):
        started = time.perf_counter()
        await store.append_batch(make_batch())
        batch_events.append((time.perf_counter() - started) * 1_000_000 / BATCH_SIZE)

    for _ in range(WARMUP):
        await store.append(make_batch()[0])
    single_events: list[float] = []
    for _ in range(N_SINGLE):
        event = make_batch()[0]
        started = time.perf_counter()
        await store.append(event)
        single_events.append((time.perf_counter() - started) * 1_000_000)

    print(f"store={type(store).__name__:<20} batch={BATCH_SIZE}/flush  samples={len(batch_events)}")
    _report("append_batch (per event)", batch_events)
    _report("append (single)", single_events)


async def main() -> None:
    """Resolve a store DSN (or SQLite fallback) and run both write paths."""
    dsn = os.getenv("SENTINEL_TEST_POSTGRES_DSN") or os.getenv("SENTINEL_STORE_DSN")
    store = build_store(dsn) if dsn else build_store(":memory:")
    try:
        await _run(store)
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
