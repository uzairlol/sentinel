# Event-Write Latency Benchmark

`S2` exit criterion 3 — event write p99 < 10 ms at target concurrency,
"documented numbers" (SENTINEL_TDD.md §2.7). Reproduction:

    SENTINEL_TEST_POSTGRES_DSN=postgresql://user:pass@host:port/sentinel_test \
        uv run python perf/bench_writes.py

Method: against the local Postgres, a fresh session appends one batch of 50
events per iteration (the production capped flush size) after a 20-batch
warmup. Per-event latency = batch wall time ÷ 50. Single-`append` latency is
one row per committed transaction (the worst-case write path). Percentiles
are the 1st/10th/99th q-quantile over 300 samples.

## Result — 2026-09-24

| Measure | p50 | p90 | p99 | Budget (p99) |
|---|---|---|---|---|
| `append_batch` 50/flush, per event | 0.55 ms | 0.78 ms | 1.45 ms | < 10 ms (OK) |
| `append` single-row commit | 5.14 ms | 5.97 ms | 6.54 ms | < 10 ms (OK) |

Platform: Windows, Python 3.11/3.12 (`.venv`), local Postgres 18.4 on the same
host over TCP. The batched path is the one interceptors use; the single-row
commit sub-path holds budget even before batching amortizes the round trip.
Interpretation: the write path is nowhere near the `S2` budget; headroom is
reserved for fsync-heavy hosts and for `S10` tuning (indexes on `flags`,
retention churn). For comparison, SQLite stays dev-only (ADR-0002); its
numbers are not budget-relevant.

## Related

- `perf/overhead-baseline.md` — `S1` capture-overhead baseline.
- `perf/bench_writes.py` — benchmark source (SQLite fallback included, offline-friendly).
