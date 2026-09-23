"""Public instrumentation surface (``S1-T1``).

Everything reachable through :mod:`sentinel.instrument` is stable:

* ``session`` / ``SessionContext`` — capture sessions and monotonic sequencing.
* ``instrument_ollama_call`` — raw Ollama ``/api/chat`` capture (``S0``).
* ``InstrumentorRegistry`` / ``BaseInstrumentor`` — the instrumentor
  registration and toggle surface (``S1-T2``).
* Framework instrumentors land under stable submodules as they ship:
  ``sentinel.instrument.langchain``, ``langgraph``, ``openai_compat``,
  ``memory``, ``generic`` (``S1``).

Per INV-1 this package only captures; it contains no analysis or evaluation.
"""

from __future__ import annotations

from sentinel.instrument.generic import trace
from sentinel.instrument.langchain import (
    MAX_CAPTURED_TEXT,
    LangChainInstrumentor,
    LangChainUnavailableError,
)
from sentinel.instrument.langgraph import LangGraphInstrumentor
from sentinel.instrument.ollama import (
    DEFAULT_BASE_URL,
    OllamaChatError,
    instrument_ollama_call,
)
from sentinel.instrument.openai_compat import (
    TransportCallError,
    chat_completion,
    chat_completion_stream,
)
from sentinel.instrument.registry import (
    BaseInstrumentor,
    Instrumentor,
    InstrumentorNotFoundError,
    InstrumentorRegistry,
)
from sentinel.instrument.session import (
    SessionClosedError,
    SessionContext,
    session,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "MAX_CAPTURED_TEXT",
    "BaseInstrumentor",
    "Instrumentor",
    "InstrumentorNotFoundError",
    "InstrumentorRegistry",
    "LangChainInstrumentor",
    "LangChainUnavailableError",
    "LangGraphInstrumentor",
    "OllamaChatError",
    "SessionClosedError",
    "SessionContext",
    "TransportCallError",
    "chat_completion",
    "chat_completion_stream",
    "instrument_ollama_call",
    "session",
    "trace",
]
