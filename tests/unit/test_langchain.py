"""Tests for LangChain callback capture (``S1-T7``).

Covers LLM/chat-model calls (``llm.request``/``llm.response`` with causal
links), tool invocations (``tool.call``/``tool.result``), chain steps
(``agent.step`` parent links), tool errors, transcript capping, the
enable/disable toggle, and fail-open behavior when the session is closed
(INV-6). The whole module skips when ``langchain-core`` is not installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from types import ModuleType
from typing import Any

import pytest

from sentinel import SQLiteEventStore, session
from sentinel.eval.session import reasoning_text_of
from sentinel.instrument.langchain import (
    MAX_CAPTURED_TEXT,
    LangChainInstrumentor,
    LangChainUnavailableError,
)
from sentinel.instrument.registry import InstrumentorRegistry
from sentinel.models.events import (
    AGENT_STEP,
    ERROR,
    LLM_REQUEST,
    LLM_RESPONSE,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
)

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None

_GenericFakeChatModel: Any
_AIMessage: Any
_ChatPromptTemplate: Any
_tool: Any

if _HAS_LANGCHAIN:
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel as _GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage as _AIMessage
    from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate
    from langchain_core.tools import tool as _tool
else:
    _GenericFakeChatModel = None
    _AIMessage = None
    _ChatPromptTemplate = None
    _tool = None

# The test bodies below are only exercised at runtime when the extra is
# installed; the ``: Any`` annotations keep them type-checked in both branches.
GenericFakeChatModel: Any = _GenericFakeChatModel
AIMessage: Any = _AIMessage
ChatPromptTemplate: Any = _ChatPromptTemplate
tool: Callable[..., Any] = _tool

pytestmark = [
    pytest.mark.skipif(
        not _HAS_LANGCHAIN,
        reason="langchain-core not installed (sentinel-sdk[langchain])",
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


def _model(text: str = "Hello!", *, message: Any = None) -> GenericFakeChatModel:
    """A fake chat model returning one message.

    ``message=`` overrides ``text=`` so a test can hand back a message carrying
    provider fields (``additional_kwargs``, ``reasoning_content``) that
    ``AIMessage(content=...)`` alone cannot express.
    """
    return GenericFakeChatModel(
        messages=iter([message if message is not None else AIMessage(content=text)])
    )


def _llm_events(events: list[Event]) -> tuple[Event, Event]:
    request = next(event for event in events if event.type == LLM_REQUEST)
    response = next(event for event in events if event.type == LLM_RESPONSE)
    return request, response


async def test_llm_call_captures_request_and_response(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("Say hi to {name}") | _model()
        result = await chain.ainvoke(
            {"name": "Ada"}, config={"callbacks": [instrumentor.handler()]}
        )

    assert result.content == "Hello!"
    events = await events_store.get_session(ctx.session_id)
    req, resp = _llm_events(events)
    assert req.payload["provider"] == "langchain"
    assert req.payload["model"] != "unknown"
    assert req.payload["prompts"][0] == "Human: Say hi to Ada"
    parents = [ref for ref in req.refs if ref.kind == RefKind.PARENT]
    assert parents
    parent_event = next(e for e in events if e.event_id == parents[0].event_id)
    assert parent_event.type == AGENT_STEP
    assert parent_event.payload["status"] == "started"
    assert resp.refs[0].kind == RefKind.CAUSED_BY
    assert resp.refs[0].event_id == req.event_id
    assert resp.payload["generations"] == ["Hello!"]
    assert resp.payload["latency_ms"] >= 0


async def test_chains_bracket_agent_step_events(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("Say hi to {name}") | _model()
        await chain.ainvoke({"name": "Bob"}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    steps = [event for event in events if event.type == AGENT_STEP]
    assert any(s.payload["status"] == "started" for s in steps)
    ended = next(s for s in steps if s.payload["status"] == "ended")
    assert [ref.kind for ref in ended.refs] == [RefKind.PARENT]
    start_event = next(e for e in events if e.event_id == ended.refs[0].event_id)
    assert start_event.payload["status"] == "started"
    assert start_event.payload["step"] == "chain"


async def test_tool_call_captures_call_and_result(events_store: SQLiteEventStore) -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        result = await add.ainvoke({"a": 2, "b": 3}, config={"callbacks": [instrumentor.handler()]})

    assert result == 5
    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == [
        "session.start",
        TOOL_CALL,
        TOOL_RESULT,
        "session.end",
    ]
    call, result_event = events[1], events[2]
    assert call.payload["tool"] == "add"
    assert "2" in call.payload["input"]
    assert "3" in call.payload["input"]
    assert result_event.payload["output"] == 5
    assert result_event.payload["latency_ms"] >= 0
    assert result_event.refs[0].kind == RefKind.CAUSED_BY
    assert result_event.refs[0].event_id == call.event_id


async def test_tool_error_is_captured(events_store: SQLiteEventStore) -> None:
    @tool
    def explode() -> None:
        """Always raises."""
        raise ValueError("kaboom")

    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        with pytest.raises(ValueError, match="kaboom"):
            await explode.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    error_event = next(event for event in events if event.type == ERROR)
    assert error_event.payload["provider"] == "langchain"
    assert error_event.payload["scope"] == "tool"
    assert error_event.payload["tool"] == "explode"
    assert "kaboom" in error_event.payload["message"]
    call_event = next(event for event in events if event.type == TOOL_CALL)
    assert error_event.refs[0].kind == RefKind.CAUSED_BY
    assert error_event.refs[0].event_id == call_event.event_id


async def test_outputs_are_capped(events_store: SQLiteEventStore) -> None:
    @tool
    def big() -> str:
        """Return oversized text."""
        return "x" * (MAX_CAPTURED_TEXT * 2)

    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        await big.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    result_event = next(event for event in events if event.type == TOOL_RESULT)
    assert result_event.payload["output"].startswith("x" * MAX_CAPTURED_TEXT)
    assert "[truncated]" in result_event.payload["output"]


async def test_disabled_instrumentor_emits_nothing(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx, enabled=False)
        await _model().ainvoke("hi", config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    assert [event.type for event in events] == ["session.start", "session.end"]


async def test_closed_session_fails_open(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        handler = instrumentor.handler()
        await ctx.end()
        result = await _model("still works").ainvoke("go", config={"callbacks": [handler]})

    assert result.content == "still works"


async def test_registers_as_an_instrumentor(events_store: SQLiteEventStore) -> None:
    async with session(events_store) as ctx:
        registry = InstrumentorRegistry()
        instrumentor = LangChainInstrumentor(ctx)
        registry.register(instrumentor)
        assert "langchain" in registry
        assert LLM_REQUEST in registry.event_types("langchain")
        registry.enable("langchain")
        assert registry.is_enabled("langchain") is True
        registry.disable("langchain")
        assert registry.is_enabled("langchain") is False


def test_missing_langchain_raises_unavailable_error() -> None:
    import builtins

    import sentinel.instrument.langchain as module

    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] == "langchain_core":
            raise ImportError("langchain_core not installed")
        return real_import(name, globals_, locals_, fromlist, level)

    module._HANDLER_CLASS = None
    try:
        builtins.__import__ = fake_import  # type: ignore[assignment]
        with pytest.raises(LangChainUnavailableError):
            LangChainInstrumentor(None).handler()  # type: ignore[arg-type]
    finally:
        builtins.__import__ = real_import
        module._HANDLER_CLASS = None


# -- reasoning capture (``S3-T5`` gap: a trace the reader must be able to find)


async def test_a_reasoning_trace_is_captured_from_additional_kwargs(
    events_store: SQLiteEventStore,
) -> None:
    """A figure stated while thinking is a claim, and it is captured.

    LangChain puts a provider's thinking trace beside ``.content`` rather than
    inside it, so capturing the text alone drops every number the model
    considered — including the ones it revised away, which are the ones no
    reader ever sees.
    """
    message = AIMessage(
        content="The price is $49.",
        additional_kwargs={"reasoning_content": "Maybe $79? No, $49 is in the tool."},
    )
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("price?") | _model(message=message)
        await chain.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    _, resp = _llm_events(events)
    assert resp.payload["reasoning"] == ["Maybe $79? No, $49 is in the tool."]
    assert reasoning_text_of(resp) == "Maybe $79? No, $49 is in the tool."


async def test_a_reasoning_trace_is_captured_from_a_direct_field(
    events_store: SQLiteEventStore,
) -> None:
    message = AIMessage(content="done", reasoning_content="thinking hard")
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("x") | _model(message=message)
        await chain.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    _, resp = _llm_events(events)
    assert reasoning_text_of(resp) == "thinking hard"


async def test_no_reasoning_key_is_written_when_there_is_no_trace(
    events_store: SQLiteEventStore,
) -> None:
    """Absent, not empty: an empty trace is indistinguishable from a
    non-reasoning model once it reaches a flag's evidence."""
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("Say hi") | _model()
        await chain.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    _, resp = _llm_events(events)
    assert "reasoning" not in resp.payload
    assert reasoning_text_of(resp) == ""


async def test_a_reasoning_trace_is_capped_like_any_transcript(
    events_store: SQLiteEventStore,
) -> None:
    long_trace = "x" * (MAX_CAPTURED_TEXT + 500)
    message = AIMessage(content="done", reasoning_content=long_trace)
    async with session(events_store) as ctx:
        instrumentor = LangChainInstrumentor(ctx)
        chain = ChatPromptTemplate.from_template("x") | _model(message=message)
        await chain.ainvoke({}, config={"callbacks": [instrumentor.handler()]})

    events = await events_store.get_session(ctx.session_id)
    _, resp = _llm_events(events)
    trace = resp.payload["reasoning"][0]
    assert len(trace) < len(long_trace)
    assert trace.endswith("…[truncated]")
