"""Tests for the call-graph query helper (``S1-T6``).

The exit criterion — *tool result -> tool call -> dependent LLM call* — is
checked against a linked session committed to the store (``append`` enforces
INV-3 referential integrity, so the graph under test is exactly what an
instrumented run produces) and against the real LangChain instrumentor where
the extra is installed.
"""

from __future__ import annotations

import datetime
import importlib.util
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from sentinel import SQLiteEventStore, get_call_graph
from sentinel.models.events import (
    AGENT_STEP,
    LLM_REQUEST,
    MEMORY_READ,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
    RefLink,
    new_event_id,
)
from sentinel.models.flags import Adjudication, Flag
from sentinel.store.gaps import SeqGap
from sentinel.store.reporting import CallEdge, SessionSummary, StoreHealth
from sentinel.store.retention import PruneReport, RetentionPolicy

pytestmark = pytest.mark.unit

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _dedup(events: list[Event]) -> set[str]:
    return {event.event_id for event in events}


async def _build_linked_session(store: SQLiteEventStore) -> tuple[str, Event, Event, Event]:
    """Commit a scripted turn: LLM-A issues a tool call, a result lands, then
    LLM-B runs inside the same chain (the dependent call). Returns
    ``(session_id, call, result, surrounding_llms)``."""
    session_id = "session-query"
    ts = _now()
    events: list[Event] = []
    seq = iter(range(100))

    def put(
        event_type: str, payload: dict[str, object], refs: list[RefLink] | None = None
    ) -> Event:
        event = Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=next(seq),
            ts=ts,
            type=event_type,
            payload=payload,
            refs=refs or [],
        )
        events.append(event)
        return event

    llm_a = put(LLM_REQUEST, {"model": "fake", "prompt": "turn 1"})
    call = put(
        TOOL_CALL,
        {"tool": "add", "input": {"a": 2, "b": 3}},
        [RefLink(event_id=llm_a.event_id, kind=RefKind.PARENT)],
    )
    result = put(
        TOOL_RESULT,
        {"tool": "add", "output": 5},
        [RefLink(event_id=call.event_id, kind=RefKind.CAUSED_BY)],
    )
    step = put(AGENT_STEP, {"step": 1, "status": "started", "node": "n1"})
    put(
        LLM_REQUEST,
        {"model": "fake", "prompt": "turn 2"},
        [RefLink(event_id=llm_a.event_id, kind=RefKind.PARENT)],
    )
    put(
        MEMORY_READ,
        {"query": "alice"},
        [RefLink(event_id=call.event_id, kind=RefKind.CAUSED_BY)],
    )
    put("session.end", {})
    for event in events:
        await store.append(event)
    return session_id, call, result, step


@pytest.fixture
async def events_store() -> AsyncIterator[SQLiteEventStore]:
    store = SQLiteEventStore(":memory:")
    try:
        yield store
    finally:
        await store.close()


async def test_tool_result_links_back_to_its_call(events_store: SQLiteEventStore) -> None:
    session_id, call, result, _ = await _build_linked_session(events_store)
    graph = await get_call_graph(events_store, session_id)

    results = graph.tool_results_for(call.event_id)
    assert _dedup(results) == {result.event_id}
    caused_by = [edge for edge in graph.refs_to(call.event_id) if edge.kind is RefKind.CAUSED_BY]
    assert len(caused_by) == 2  # the tool.result and the dependent memory.read
    edge = next(edge for edge in caused_by if edge.src.type == TOOL_RESULT)
    assert edge.src.event_id == result.event_id
    assert edge.dst.event_id == call.event_id
    assert [event.event_id for event in graph.tool_calls()] == [call.event_id]


async def test_llm_calls_surrounding_a_tool_call(events_store: SQLiteEventStore) -> None:
    session_id, call, _, _ = await _build_linked_session(events_store)
    graph = await get_call_graph(events_store, session_id)

    surrounding = graph.llm_calls_for(call.event_id)
    assert [event.payload["prompt"] for event in surrounding] == ["turn 1", "turn 2"]
    parents = graph.parents(call.event_id)
    assert [event.payload["prompt"] for event in parents] == ["turn 1"]


async def test_every_edge_respects_seq_order(events_store: SQLiteEventStore) -> None:
    session_id, _, _, _ = await _build_linked_session(events_store)
    graph = await get_call_graph(events_store, session_id)

    order = {event.event_id: i for i, event in enumerate(graph.events())}
    assert order  # graph is non-empty
    for edge in graph.edges:
        assert order[edge.src.event_id] > order[edge.dst.event_id]
    assert graph.session_id == session_id


async def test_steps_and_memory_reads_are_queryable(events_store: SQLiteEventStore) -> None:
    session_id, call, _, step = await _build_linked_session(events_store)
    graph = await get_call_graph(events_store, session_id)

    assert [s.payload["node"] for s in graph.steps()] == [step.payload["node"]]
    reads = graph.memory_reads_for(call.event_id)
    assert [event.type for event in reads] == [MEMORY_READ]


async def test_dangling_ref_is_dropped_not_raised(events_store: SQLiteEventStore) -> None:
    """INV-6: the helper degrades gracefully instead of raising."""

    @dataclass
    class _ForgetfulStore:
        store: SQLiteEventStore

        async def get_session(self, session_id: str) -> list[Event]:
            request = Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=1,
                ts=_now(),
                type=LLM_REQUEST,
                payload={},
            )
            shady = Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=2,
                ts=_now(),
                type=TOOL_CALL,
                payload={},
                refs=[RefLink(event_id=new_event_id(), kind=RefKind.PARENT)],
            )
            return [request, shady]

        async def append(self, event: Event) -> None:
            await self.store.append(event)

        async def append_batch(self, events: list[Event]) -> None:
            await self.store.append_batch(events)

        def iter_session(
            self,
            session_id: str,
            *,
            after_seq: int | None = None,
            limit: int | None = None,
        ) -> AsyncIterator[Event]:
            return self.store.iter_session(session_id, after_seq=after_seq, limit=limit)

        async def get_call_graph(self, session_id: str) -> list[CallEdge]:
            return await self.store.get_call_graph(session_id)

        async def list_sessions(
            self,
            *,
            agent_id: str | None = None,
            since: datetime.datetime | None = None,
            until: datetime.datetime | None = None,
            has_flags: bool = False,
            limit: int = 100,
            offset: int = 0,
        ) -> list[SessionSummary]:
            return await self.store.list_sessions(
                agent_id=agent_id,
                since=since,
                until=until,
                has_flags=has_flags,
                limit=limit,
                offset=offset,
            )

        async def detect_gaps(self, session_id: str) -> list[SeqGap]:
            return await self.store.detect_gaps(session_id)

        async def health(self) -> StoreHealth:
            return await self.store.health()

        async def prune(self, policy: RetentionPolicy) -> PruneReport:
            return await self.store.prune(policy)

        async def put_flag(self, flag: Flag) -> bool:
            return await self.store.put_flag(flag)

        async def put_flags(self, flags: Sequence[Flag]) -> int:
            return await self.store.put_flags(flags)

        async def get_flags(self, **kwargs: Any) -> list[Flag]:
            return await self.store.get_flags(**kwargs)

        async def adjudicate_flag(
            self,
            flag_id: str,
            adjudication: Adjudication,
            *,
            adjudicated_by: str,
            at: datetime.datetime | None = None,
        ) -> bool:
            return await self.store.adjudicate_flag(
                flag_id, adjudication, adjudicated_by=adjudicated_by, at=at
            )

        async def close(self) -> None:
            await self.store.close()

    graph = await get_call_graph(_ForgetfulStore(events_store), "session")
    assert len(graph.nodes) == 2
    assert graph.edges == []
    _ = graph.parents  # no crash on the shady node


@pytest.mark.skipif(
    not _HAS_LANGCHAIN,
    reason="langchain-core not installed (sentinel-sdk[langchain])",
)
async def test_real_langchain_tool_run_resolves(events_store: SQLiteEventStore) -> None:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.tools import tool

    from sentinel.instrument.langchain import LangChainInstrumentor
    from sentinel.instrument.session import session

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        callback = instrumentor.handler()
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        chain = ChatPromptTemplate.from_template("turn {n}") | model
        await chain.ainvoke({"n": 1}, config={"callbacks": [callback]})
        await add.ainvoke({"a": 2, "b": 3}, config={"callbacks": [callback]})

    graph = await get_call_graph(events_store, ctx.session_id)
    call = graph.tool_calls()[0]
    [result] = graph.tool_results_for(call.event_id)
    assert result.type == TOOL_RESULT
    assert result.refs[0].kind is RefKind.CAUSED_BY
    assert result.refs[0].event_id == call.event_id
