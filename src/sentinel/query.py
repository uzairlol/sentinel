"""Call-graph query helper (``S1-T6``).

Lifts a session's flat event log into a navigable graph: nodes are events,
edges are the ``refs`` links the instrumentors wrote (INV-3), and the typed
accessors expose the ``tool result -> tool call -> dependent LLM call``
relationships the provenance evaluators (``S3-T8``) need.

The helper is read-only: it never writes to the store and never raises on
dangling refs (INV-6) -- it drops the dangling edge and logs it instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sentinel.models.events import (
    AGENT_STEP,
    LLM_REQUEST,
    MEMORY_READ,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
)
from sentinel.store.protocol import EventStore

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger("sentinel.query")


@dataclass(frozen=True)
class Edge:
    """A typed, directed link from ``src`` to the earlier event ``dst``."""

    src: Event
    dst: Event
    kind: RefKind


@dataclass
class CallGraph:
    """A session's events resolved into a navigable directed graph."""

    session_id: str
    nodes: dict[str, Event] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    _incoming: dict[str, list[Edge]] = field(default_factory=dict)
    _outgoing: dict[str, list[Edge]] = field(default_factory=dict)

    # -- graph access ---------------------------------------------------

    def events(self) -> list[Event]:
        """The session's events in ``seq`` order."""
        return sorted(self.nodes.values(), key=lambda event: event.seq)

    def refs_from(self, event_id: str) -> list[Edge]:
        """Refs the event emits (outgoing edges: this event -> earlier)."""
        return list(self._outgoing.get(event_id, ()))

    def refs_to(self, event_id: str) -> list[Edge]:
        """Refs pointing at the event (incoming edges: later -> this)."""
        return list(self._incoming.get(event_id, ()))

    def parents(self, event_id: str) -> list[Event]:
        """Ancestors reached by climbing ``parent`` refs, outermost first."""
        chain: list[Event] = []
        seen: set[str] = set()
        node = self.nodes.get(event_id)
        while node is not None and node.event_id not in seen:
            seen.add(node.event_id)
            upstream = [
                edge.dst
                for edge in self._outgoing.get(node.event_id, ())
                if edge.kind is RefKind.PARENT
            ]
            if not upstream:
                break
            chain.append(upstream[0])
            node = upstream[0]
        return chain[::-1]

    # -- typed accessors ------------------------------------------------

    def tool_calls(self) -> list[Event]:
        """Every ``tool.call`` event in execution order."""
        return [event for event in self.events() if event.type == TOOL_CALL]

    def tool_results_for(self, call_id: str) -> list[Event]:
        """The ``tool.result`` events that cite *call_id* as their cause."""
        return [
            edge.src
            for edge in self._incoming.get(call_id, ())
            if edge.kind is RefKind.CAUSED_BY and edge.src.type == TOOL_RESULT
        ]

    def llm_calls_for(self, call_id: str) -> list[Event]:
        """The LLM calls that surround a tool call.

        Returns the ``llm.request`` ancestors of *call_id* (the LLM turn
        that issued the tool call) plus any ``llm.request`` emitted inside
        that same chain after the tool result landed (the *dependent* LLM
        calls that consume the result).
        """
        chain_ids = {event.event_id for event in self.parents(call_id)}
        after_seq = max((event.seq for event in self.tool_results_for(call_id)), default=-1)
        related = [
            event
            for event in self.events()
            if event.type == LLM_REQUEST
            and (
                event.event_id in chain_ids
                or (
                    event.seq > after_seq
                    and any(
                        link.kind is RefKind.PARENT and link.event_id in chain_ids
                        for link in event.refs
                    )
                )
            )
        ]
        by_id = {event.event_id: event for event in related}
        return sorted(by_id.values(), key=lambda event: event.seq)

    def memory_reads_for(self, call_id: str) -> list[Event]:
        """The ``memory.read`` events that live under *call_id*'s chain."""
        chain_ids = {event.event_id for event in self.parents(call_id)} | {call_id}
        return [
            event
            for event in self.events()
            if event.type == MEMORY_READ and any(link.event_id in chain_ids for link in event.refs)
        ]

    def steps(self) -> list[Event]:
        """Every ``agent.step`` event in execution order (LangGraph nodes)."""
        return [event for event in self.events() if event.type == AGENT_STEP]


async def get_call_graph(store: EventStore, session_id: str) -> CallGraph:
    """Build and return a :class:`CallGraph` for *session_id*.

    Never raises: a dangling ref (already prevented by INV-3) is dropped
    and logged rather than bubbling up into the caller.
    """
    return build_call_graph(session_id, await store.get_session(session_id))


def build_call_graph(session_id: str, events: Iterable[Event]) -> CallGraph:
    """Resolve *events* into a :class:`CallGraph` without touching the store.

    The pure half of :func:`get_call_graph`, so a caller that has already
    loaded (or deliberately truncated) an event list does not have to read it
    twice. Refs that point outside *events* are dropped and logged — a
    truncated or mid-flight session simply has fewer edges.
    """
    graph = CallGraph(session_id=session_id)
    for event in events:
        graph.nodes[event.event_id] = event
    for event in events:
        for ref in event.refs:
            dst = graph.nodes.get(ref.event_id)
            if dst is None:
                logger.warning("call-graph: dangling ref %s on %s", ref.event_id, event.event_id)
                continue
            edge = Edge(src=event, dst=dst, kind=ref.kind)
            graph.edges.append(edge)
            graph._outgoing.setdefault(event.event_id, []).append(edge)
            graph._incoming.setdefault(ref.event_id, []).append(edge)
    return graph
