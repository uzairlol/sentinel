"""Tests for LangGraph node-step capture (``S1-T8``).

Covers node entry/exit as ``agent.step`` events with bounded state deltas and
``step_index``, the graph's run as the enclosing chain step, LLM calls inside a
node inheriting the node step as parent, fail-open on a closed session (INV-6),
the enable/disable toggle, and registry integration. The module skips when
``langgraph`` is not installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator
from typing import Any

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.instrument.langgraph import LangGraphInstrumentor
from sentinel.instrument.registry import InstrumentorRegistry
from sentinel.models.events import (
    AGENT_STEP,
    LLM_REQUEST,
    LLM_RESPONSE,
    Event,
    RefKind,
)

_HAS_LANGGRAPH = (
    importlib.util.find_spec("langgraph") is not None
    and importlib.util.find_spec("langchain_core") is not None
)

_StateGraph: Any
_END: Any
_START: Any

if _HAS_LANGGRAPH:
    from langgraph.graph import END, START, StateGraph

    _StateGraph, _END, _START = StateGraph, END, START
else:
    _StateGraph = None
    _END = None
    _START = None

pytestmark = [
    pytest.mark.skipif(
        not _HAS_LANGGRAPH, reason="langgraph not installed (sentinel-sdk[langgraph])"
    ),
    pytest.mark.unit,
]


@pytest.fixture
async def events_store() -> AsyncIterator[SQLiteEventStore]:
    store = SQLiteEventStore(":memory:")
    try:
        yield store
    finally:
        await store.close()


def _graph() -> Any:
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict, total=False):
        count: int
        last: str

    def node_a(state: State) -> dict[str, object]:
        return {"count": state.get("count", 0) + 1, "last": "a"}

    def node_b(state: State) -> dict[str, object]:
        return {"count": (state.get("count", 0) + 1) * 2, "last": "b"}

    graph = StateGraph(State)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.add_edge(START, "node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    return graph.compile()


def _steps(events: list[Event]) -> list[Event]:
    return [event for event in events if event.type == AGENT_STEP]


async def test_nodes_emit_agent_step_events(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx)
        result = await _graph().ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    assert result["count"] == 4
    assert result["last"] == "b"
    events = await events_store.get_session(ctx.session_id)
    steps = _steps(events)
    node_starts = [
        s for s in steps if s.payload.get("step") == "step" and s.payload["status"] == "started"
    ]
    assert [s.payload["node"] for s in node_starts] == ["node_a", "node_b"]
    assert [s.payload["step_index"] for s in node_starts] == [1, 2]
    assert node_starts[0].payload["input"] == {}
    assert node_starts[1].payload["input"] == {"count": 1, "last": "a"}


async def test_nodes_close_and_link_to_their_start(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx)
        await _graph().ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    steps = _steps(events)
    node_steps = [s for s in steps if s.payload.get("step") == "step"]
    node_a_start = next(
        s for s in node_steps if s.payload["node"] == "node_a" and s.payload["status"] == "started"
    )
    node_a_end = next(
        s for s in node_steps if s.payload["node"] == "node_a" and s.payload["status"] == "ended"
    )
    assert node_a_end.payload["output"] == {"count": 1, "last": "a"}
    assert node_a_end.refs[0].kind == RefKind.PARENT
    assert node_a_end.refs[0].event_id == node_a_start.event_id
    assert node_a_end.payload["latency_ms"] >= 0


async def test_graph_run_is_the_enclosing_chain_step(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx)
        await _graph().ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    steps = _steps(events)
    chain_start = next(
        s for s in steps if s.payload.get("step") == "chain" and s.payload["status"] == "started"
    )
    assert chain_start.payload["status"] == "started"
    node_a_start = next(
        s
        for s in steps
        if s.payload.get("step") == "step"
        and s.payload["node"] == "node_a"
        and s.payload["status"] == "started"
    )
    parents = [ref for ref in node_a_start.refs if ref.kind == RefKind.PARENT]
    assert parents
    assert parents[0].event_id == chain_start.event_id
    assert chain_start.payload["provider"] == "langchain"


async def test_llm_inside_node_inherits_node_step_as_parent(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx)

        from langchain_core.language_models.fake_chat_models import (
            GenericFakeChatModel,
        )
        from langchain_core.messages import AIMessage
        from langchain_core.runnables.config import RunnableConfig
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict

        class State(TypedDict, total=False):
            messages: list[Any]
            ping: str

        model = GenericFakeChatModel(messages=iter([AIMessage(content="hello")]))

        async def agent_node(state: State, config: RunnableConfig) -> dict[str, object]:
            reply = await model.ainvoke(state.get("ping", "hi"), config=config)
            return {"messages": [reply.content]}

        graph = StateGraph(State)
        graph.add_node("agent_node", agent_node)
        graph.add_edge(START, "agent_node")
        graph.add_edge("agent_node", END)
        app = graph.compile()
        await app.ainvoke(
            {"ping": "hi", "messages": []},
            config={"callbacks": [instrumentor.handler()]},
        )

    events = await events_store.get_session(ctx.session_id)
    request_event = next(event for event in events if event.type == LLM_REQUEST)
    response_event = next(event for event in events if event.type == LLM_RESPONSE)
    node_start = next(
        event
        for event in events
        if event.type == AGENT_STEP
        and event.payload.get("node") == "agent_node"
        and event.payload["status"] == "started"
    )
    assert response_event.refs[0].kind == RefKind.CAUSED_BY
    assert response_event.refs[0].event_id == request_event.event_id
    parents = [ref for ref in request_event.refs if ref.kind == RefKind.PARENT]
    assert parents
    assert parents[0].event_id == node_start.event_id


async def test_disabled_instrumentor_emits_nothing(
    events_store: SQLiteEventStore,
) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx, enabled=False)
        await _graph().ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


async def test_closed_session_fails_open(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangGraphInstrumentor(ctx)
        handler = instrumentor.handler()
        await ctx.end()
        result = await _graph().ainvoke({}, config={"callbacks": [handler]})

    assert result["count"] == 4


async def test_registers_as_an_instrumentor(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        registry = InstrumentorRegistry()
        instrumentor = LangGraphInstrumentor(ctx)
        registry.register(instrumentor)
        assert "langgraph" in registry
        assert AGENT_STEP in registry.event_types("langgraph")
        registry.enable("langgraph")
        assert registry.is_enabled("langgraph") is True
