"""A read-only view of one session, shared by every evaluator module.

Loading a session means two things the modules all need: the ordered event log
and the resolved call graph (``sentinel.query.CallGraph``, INV-3). Building the
graph once per session and handing the same immutable view to every module keeps
the evaluators pure — a module's output depends on the events and nothing else
(``S3-T4`` determinism) — and keeps the graph build O(events).

This module imports ``sentinel.store`` and ``sentinel.query`` but never
``sentinel.instrument``: INV-1 means evaluation reads the log, it does not
capture it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import structlog

from sentinel.models.events import (
    AGENT_STEP,
    LLM_REQUEST,
    LLM_RESPONSE,
    MEMORY_READ,
    MEMORY_WRITE,
    SESSION_END,
    SESSION_START,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
)
from sentinel.query import CallGraph, build_call_graph
from sentinel.store.protocol import EventStore

log = structlog.get_logger("sentinel.eval.session")

#: Default cap on how many events one evaluation will consider. A pathological
#: session must not be able to pin a worker forever; the truncation is logged
#: so an operator can see the module is no longer seeing the whole session.
DEFAULT_MAX_EVENTS = 5_000


@dataclass(frozen=True)
class SessionView:
    """One session's events plus its resolved call graph.

    Attributes:
        session_id: The session the view was loaded from.
        events: Events in ``seq`` order, possibly truncated by ``max_events``.
        graph: The resolved call graph over those events.
        truncated: Whether ``max_events`` cut the session short.
    """

    session_id: str
    events: tuple[Event, ...]
    graph: CallGraph
    truncated: bool = False
    _by_type: dict[str, tuple[Event, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    async def load(
        cls,
        store: EventStore,
        session_id: str,
        *,
        max_events: int | None = None,
    ) -> SessionView:
        """Load *session_id* and resolve its call graph.

        ``max_events`` bounds the work a single evaluation may do; the default
        is :data:`DEFAULT_MAX_EVENTS`. Truncation is explicit (``truncated``)
        and logged, never silent.
        """
        cap = DEFAULT_MAX_EVENTS if max_events is None else max_events
        events = await store.get_session(session_id)
        truncated = len(events) > cap
        if truncated:
            log.warning(
                "session.truncated",
                session_id=session_id,
                events=len(events),
                max_events=cap,
            )
            events = events[:cap]
        return cls(
            session_id=session_id,
            events=tuple(events),
            graph=build_call_graph(session_id, events),
            truncated=truncated,
        )

    # -- lifecycle --------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        """Whether the session has its ``session.end`` bookend.

        The worker's watermark trigger only evaluates *complete* sessions
        (``S3-T3``): a half-finished log can still grow, and a module that reads
        a growing log would produce a different answer on the next pass.
        """
        return any(event.type == SESSION_END for event in self.events)

    @property
    def agent_id(self) -> str | None:
        """The agent the session belongs to, as recorded at ``session.start``."""
        start = self.of_type(SESSION_START)
        if not start:
            return None
        raw = start[0].payload.get("agent_id")
        return str(raw) if raw is not None else None

    @property
    def started_at(self) -> datetime | None:
        """The session's ``session.start`` timestamp, or the first event's."""
        start = self.of_type(SESSION_START)
        return start[0].ts if start else (self.events[0].ts if self.events else None)

    @property
    def ended_at(self) -> datetime | None:
        """The ``session.end`` timestamp when present."""
        end = self.of_type(SESSION_END)
        return end[0].ts if end else None

    # -- typed accessors --------------------------------------------------

    def of_type(self, event_type: str) -> tuple[Event, ...]:
        """Every event of *event_type*, in ``seq`` order."""
        cached = self._by_type.get(event_type)
        if cached is None:
            cached = tuple(event for event in self.events if event.type == event_type)
            self._by_type[event_type] = cached
        return cached

    def llm_requests(self) -> tuple[Event, ...]:
        """``llm.request`` events in order."""
        return self.of_type(LLM_REQUEST)

    def llm_responses(self) -> tuple[Event, ...]:
        """``llm.response`` events in order (the claim-bearing events)."""
        return self.of_type(LLM_RESPONSE)

    def tool_calls(self) -> tuple[Event, ...]:
        """``tool.call`` events in order."""
        return self.of_type(TOOL_CALL)

    def tool_results(self) -> tuple[Event, ...]:
        """``tool.result`` events in order."""
        return self.of_type(TOOL_RESULT)

    def memory_reads(self) -> tuple[Event, ...]:
        """``memory.read`` events in order (``S4``)."""
        return self.of_type(MEMORY_READ)

    def memory_writes(self) -> tuple[Event, ...]:
        """``memory.write`` events in order (``S4``)."""
        return self.of_type(MEMORY_WRITE)

    def steps(self) -> tuple[Event, ...]:
        """``agent.step`` events in order (``S5``/``S6`` objective tracking)."""
        return self.of_type(AGENT_STEP)

    def tool_name(self, event: Event) -> str:
        """The tool name recorded on a ``tool.call``/``tool.result`` event."""
        raw = event.payload.get("tool")
        return str(raw) if raw is not None else ""

    def tool_input(self, event: Event) -> str:
        """The tool input recorded on a ``tool.call`` event (may be empty)."""
        raw = event.payload.get("input")
        return str(raw) if raw is not None else ""

    def tool_output(self, event: Event) -> str:
        """The value a ``tool.result`` returned, as text (may be empty)."""
        raw = event.payload.get("output")
        return as_text(raw)

    def response_text(self, event: Event) -> str:
        """The generated text of an ``llm.response``, whatever its shape.

        Instrumentors differ in where they put the text (``generations`` for
        LangChain, ``response.message.content`` for Ollama, ``transcript`` for
        streamed OpenAI-compatible calls), so every producer is tried in turn
        and the longest candidate wins.
        """
        return as_text(response_payload(event))


def response_payload(event: Event) -> object:
    """The raw field holding an ``llm.response``'s generated text."""
    payload = event.payload
    generations = payload.get("generations")
    if isinstance(generations, list) and generations:
        first = generations[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list) and first:
            return first[0]
        if isinstance(first, dict):
            return first.get("text")
    response = payload.get("response")
    if isinstance(response, dict):
        message = response.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return message["content"]
        if response.get("content") is not None:
            return response["content"]
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and isinstance(first.get("message"), dict):
                return first["message"].get("content")
    if response is not None:
        return response
    return payload.get("transcript")


def response_text_of(event: Event) -> str:
    """Module-level alias of :meth:`SessionView.response_text`."""
    return as_text(response_payload(event))


def as_text(value: object) -> str:
    """Render a captured value as text without inventing structure."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={as_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(item) for item in value)
    return str(value)


__all__ = [
    "DEFAULT_MAX_EVENTS",
    "SessionView",
    "as_text",
    "response_payload",
    "response_text_of",
]
