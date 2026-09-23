"""Capture-overhead micro-benchmark (``S1`` exit criterion 7).

Measures the marginal cost of capturing an ``agent.step`` for an async
function, against the same function uncaptured, using an in-memory store.
Baseline for the ``S2``/``S10`` throughput budgets.

Run from the repo root::

    uv run python perf/bench_capture.py
"""

from __future__ import annotations

import asyncio
import time

from sentinel.instrument import trace
from sentinel.instrument.session import session
from sentinel.store.sqlite import SQLiteEventStore

N = 10_000
_WARMUP = 200

_calls = 0
_touched = 0


async def plain(x: int) -> int:
    """No-op baseline: no instrumention at all."""
    global _calls, _touched
    _calls += 1
    _touched = _touched + x
    return x


@trace(kind="function")
async def traced(x: int) -> int:
    """Same work, but wrapped in Sentinel's trace decorator."""
    global _calls, _touched
    _calls += 1
    _touched = _touched + x
    return x


async def _measure(coro: object, iters: int) -> float:
    for _ in range(_WARMUP):
        await coro(1)  # type: ignore[operator]
    samples: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iters):
            await coro(1)  # type: ignore[operator]
        samples.append((time.perf_counter() - start) / iters)
    return min(samples)


async def main() -> None:
    """Measure best-of-3 captured vs uncaptured call cost and print it."""
    store = SQLiteEventStore(":memory:")
    try:
        async with session(store) as ctx:
            t0 = await _measure(plain, N)
            t1 = await _measure(traced, N)

        overhead = t1 - t0
        ratio = t1 / t0 if t0 else float("nan")
        captured = len(await store.get_session(ctx.session_id))
        print(f"plain  : {t0 * 1e6:9.2f} us/call")
        print(f"traced : {t1 * 1e6:9.2f} us/call")
        print(f"overhead: {overhead * 1e6:8.2f} us/event, ratio x{ratio:.2f}")
        print(f"events captured in session: {captured}")
        print(f"touch checksum: {_touched} over {_calls} calls")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
