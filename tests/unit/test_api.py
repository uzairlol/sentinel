"""Tests for the frozen public API surface (``S1-T1``).

Anything exported from :mod:`sentinel` is semver-stable; the contract is pinned
so a refactor cannot silently move the public names. Internal helper modules
(``sentinel._cli``, ``sentinel.config``) stay out of ``__all__`` and may change.
"""

from __future__ import annotations

import importlib

import sentinel
import sentinel.instrument


def test_top_level_public_names_are_stable() -> None:
    expected = {
        "DEFAULT_BASE_URL",
        "BaseInstrumentor",
        "Event",
        "Instrumentor",
        "InstrumentorNotFoundError",
        "InstrumentorRegistry",
        "OllamaChatError",
        "SQLiteEventStore",
        "SessionClosedError",
        "SessionContext",
        "__version__",
        "configure",
        "get_config",
        "instrument_ollama_call",
        "new_event_id",
        "session",
    }
    # A frozen *subset*: every pinned name stays importable forever; new names
    # may be added as S1 submodules ship.
    assert expected <= set(sentinel.__all__)


def test_instrument_public_names_are_stable() -> None:
    expected = {
        "DEFAULT_BASE_URL",
        "BaseInstrumentor",
        "Instrumentor",
        "InstrumentorNotFoundError",
        "InstrumentorRegistry",
        "OllamaChatError",
        "SessionClosedError",
        "SessionContext",
        "instrument_ollama_call",
        "session",
    }
    assert expected <= set(sentinel.instrument.__all__)
    # S1-T9 additions: the openai_compat transport surface.
    assert {
        "TransportCallError",
        "chat_completion",
        "chat_completion_stream",
    } <= set(sentinel.instrument.__all__)
    # S1-T7 additions: the LangChain capture surface.
    assert {
        "LangChainInstrumentor",
        "LangChainUnavailableError",
        "MAX_CAPTURED_TEXT",
    } <= set(sentinel.instrument.__all__)
    # S1-T8 additions: the LangGraph capture surface.
    assert {"LangGraphInstrumentor"} <= set(sentinel.instrument.__all__)


def test_internal_modules_are_not_part_of_the_contract() -> None:
    assert importlib.import_module("sentinel._cli") is not None
    assert "config" not in sentinel.__all__
    assert "_cli" not in sentinel.__all__


def test_public_callables_exist() -> None:
    assert callable(sentinel.configure)
    assert callable(sentinel.get_config)
    assert callable(sentinel.session)
    assert callable(sentinel.InstrumentorRegistry)
