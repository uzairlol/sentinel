"""LangGraph node capture example (``S1-T18``).

Instrument a small LangGraph ``StateGraph`` with Sentinel so each node entry
and exit becomes an ``agent.step`` event, then replay the session.

Requires the ``sentinel-sdk[langgraph]`` extra. Run from the repo root::

    uv run python examples/langgraph_agent.py
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langchain_core.callbacks import (
    BaseCallbackHandler,  # noqa: F401  (keeps langchain-core a hard dep of this example)
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from sentinel import SQLiteEventStore, get_call_graph, session
from sentinel.instrument.langgraph import LangGraphInstrumentor

STORE_PATH = "langgraph_agent.sqlite3"


class State(TypedDict):
    """Graph channel state: the question in, the answer out."""

    question: str
    answer: str


async def main() -> None:
    """Compile a two-node graph, instrument it, and replay the agent.step log."""
    store = SQLiteEventStore(STORE_PATH)
    try:
        from langgraph.graph import END, START, StateGraph

        model = GenericFakeChatModel(messages=iter([AIMessage(content="I heard you.")]))

        def extract(state: State) -> dict[str, Any]:
            """Step one: echo the question back."""
            sentinel_step = {}  # placeholder: link events to this node
            _ = sentinel_step
            return {"answer": state["question"]}

        def respond(state: State) -> dict[str, Any]:
            """Step two: produce the final answer."""
            _ = model  # truthful example: real nodes call the model
            return {"answer": f"Final answer for {state['question']}"}

        builder = StateGraph(State)
        builder.add_node("extract", extract)
        builder.add_node("respond", respond)
        builder.add_edge(START, "extract")
        builder.add_edge("extract", "respond")
        builder.add_edge("respond", END)
        graph = builder.compile()

        async with session(store) as ctx:
            instrumentor = LangGraphInstrumentor(ctx)
            result = await graph.ainvoke(
                {"question": "what is 2+2?"},
                config={"callbacks": [instrumentor.handler()]},
            )
        print(f"answer: {result['answer']}")

        graph_view = await get_call_graph(store, ctx.session_id)
        for event in graph_view.steps():
            print(
                f"  [{event.seq}] agent.step node={event.payload.get('node'):<8} "
                f"{event.payload.get('status'):<7} step={event.payload.get('step')}"
            )
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
