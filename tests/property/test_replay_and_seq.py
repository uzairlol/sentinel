"""Property tests for the S0 core invariants (``S0-T6``/``S0-T8``).

Sequencing and idempotency are the load-bearing properties of the event store:
``seq`` must be strictly monotonic per session, every ``event_id`` unique, and a
second read of a session must be byte-identical to the first.
"""

from __future__ import annotations

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from sentinel import SQLiteEventStore
from sentinel.instrument.session import SessionContext


async def _run_range(payloads: list[int]) -> None:
    store = SQLiteEventStore(":memory:")
    try:
        ctx = SessionContext(store)
        await ctx.start()
        emitted_ids: list[str] = []
        for index, value in enumerate(payloads):
            event = await ctx.capture(type="evt", payload={"i": index, "v": value})
            emitted_ids.append(event.event_id)
        session_id = ctx.session_id
        await ctx.end()

        events = await store.get_session(session_id)
        assert [e.seq for e in events] == list(range(len(payloads) + 2))
        assert [e.event_id for e in events[1:-1]] == emitted_ids
        all_ids = [e.event_id for e in events]
        assert len(set(all_ids)) == len(all_ids)

        again = await store.get_session(session_id)
        assert again == events
    finally:
        await store.close()


@settings(max_examples=50, deadline=None)
@given(payloads=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=40))
def test_seq_is_strict_and_replay_is_idempotent(payloads: list[int]) -> None:
    asyncio.run(_run_range(payloads))
