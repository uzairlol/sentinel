# Capture Overhead Baseline

`S1` exit criterion 7 — instrumentation overhead measured and recorded as the
baseline for the `S2`/`S10` throughput budgets. Reproduction:

    uv run python perf/bench_capture.py

Method: an async no-op is invoked 10 000 times uncaptured versus wrapped in
`sentinel.instrument.trace` inside a live `session()` that persists every event
to an in-memory SQLite store. Reported is the best of 3 samples after a 200-call
warmup.

## Result — 2026-09-23

| Measure | Value |
|---|---|
| Platform | Windows, Python 3.13 (CI runs 3.12/3.13) |
| Uncaptured call | 0.33 µs/call |
| Captured + persisted | 2249 µs/event |
| Marginal cost | ~2250 µs/event, ×6767 |

Interpretation: the cost is dominated by the full capture → SQLite append path
for a *trivial* function. Real boundary crossings (LLM HTTP round trips, tool
execution) dwarf this; capture is a single `await ctx.capture()` per crossing.
Bit of budget: `S1` targets order-of-magnitude checks only — this baseline
pins the number the `S2` writer (batching, Postgres) and `S10` (indexes,
retention) are measured against. A no-persist capture-only figure and the
batched-writer figure belong to `S2-T12`/`S10-T2` dashboards.
