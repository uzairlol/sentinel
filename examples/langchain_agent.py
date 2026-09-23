"""LangChain agent capture example (``S1-T18``).

Instrument an LLM chain and a tool with Sentinel, then replay the session and
print the call graph. Uses langchain-core's fake model so it runs with no API
key.

Requires the ``sentinel-sdk[langchain]`` extra. Run from the repo root::

    uv run python examples/langchain_agent.py
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from sentinel import SQLiteEventStore, get_call_graph, session
from sentinel.instrument.langchain import LangChainInstrumentor

STORE_PATH = "langchain_agent.sqlite3"


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    """Instrument a chain + tool run, then replay the session with the graph."""
    store = SQLiteEventStore(STORE_PATH)
    try:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        chain = ChatPromptTemplate.from_template("Say hi to {name}") | model

        async with session(store) as ctx:
            instrumentor = LangChainInstrumentor(ctx)
            callbacks = [instrumentor.handler()]
            await chain.ainvoke({"name": "Ada"}, config={"callbacks": callbacks})
            await add.ainvoke({"a": 2, "b": 3}, config={"callbacks": callbacks})

        graph = await get_call_graph(store, ctx.session_id)
        for event in graph.events():
            print(f"  [{event.seq}] {event.type:<16} {_summary(event.payload)}")
        call = graph.tool_calls()[0]
        results = graph.tool_results_for(call.event_id)
        print(
            f"\ntool.call {call.event_id} -> {len(results)} result(s), "
            f"surrounded by {len(graph.llm_calls_for(call.event_id))} LLM call(s)"
        )
    finally:
        await store.close()


def _summary(payload: dict) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))[:100]


if __name__ == "__main__":
    asyncio.run(main())
