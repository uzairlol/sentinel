"""Tests for the shared session view (``S3-T2``) and the worker framework.

The view is the only thing an evaluator sees, so its contract matters twice
over: a claim extracted from a truncated log is a wrong answer, not a partial
one, and the graph has to be the same object the analyzers walk. The worker
tests are about the promises a detection module inherits for free — completed
sessions only, idempotent re-runs, bounded retries, batched writes — because a
module that has to re-implement those is a module that will get them wrong.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sentinel.eval.session import (
    DEFAULT_MAX_EVENTS,
    SessionView,
    as_text,
    response_text_of,
)
from sentinel.eval.worker import (
    DeterministicEvaluationError,
    EvaluationError,
    EvaluatorWorker,
    InMemoryCheckpointStore,
    SqliteCheckpointStore,
    WorkerConfig,
)
from sentinel.models.events import (
    LLM_REQUEST,
    LLM_RESPONSE,
    SESSION_END,
    SESSION_START,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
    RefLink,
    new_event_id,
)
from sentinel.models.flags import EvidenceRef, EvidenceRole, Flag
from sentinel.query import build_call_graph
from sentinel.store.sqlite import SQLiteEventStore

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def _view(events: list[Event]) -> SessionView:
    session_id = events[0].session_id
    return SessionView(
        session_id=session_id,
        events=tuple(events),
        graph=build_call_graph(session_id, events),
    )


def _session_events(session_id: str, *, count: int = 6, close: bool = True) -> list[Event]:
    """A flat session of *count* events, optionally bookended at the end."""
    events = [
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=0,
            ts=_TS,
            type=SESSION_START,
            payload={"agent_id": "unit"},
        ),
        *(
            Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=seq,
                ts=_TS + timedelta(milliseconds=seq),
                type=LLM_REQUEST,
                payload={"provider": "unit"},
            )
            for seq in range(1, count)
        ),
    ]
    if close:
        events.append(
            Event(
                event_id=new_event_id(),
                session_id=session_id,
                seq=len(events),
                ts=_TS + timedelta(seconds=1),
                type=SESSION_END,
                payload={"agent_id": "unit"},
            )
        )
    return events


def _turn_events(session_id: str) -> list[Event]:
    """One realistic turn: request, tool call, tool result, response."""
    call_id = new_event_id()
    return [
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=0,
            ts=_TS,
            type=SESSION_START,
            payload={"agent_id": "unit"},
        ),
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=1,
            ts=_TS + timedelta(milliseconds=1),
            type=LLM_REQUEST,
            payload={"provider": "unit"},
        ),
        Event(
            event_id=call_id,
            session_id=session_id,
            seq=2,
            ts=_TS + timedelta(milliseconds=2),
            type=TOOL_CALL,
            payload={"tool": "billing.lookup", "args": {"plan": "pro"}},
        ),
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=3,
            ts=_TS + timedelta(milliseconds=3),
            type=TOOL_RESULT,
            payload={"output": "Plan pro costs $29 per month."},
            refs=[RefLink(event_id=call_id, kind=RefKind.CAUSED_BY)],
        ),
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=4,
            ts=_TS + timedelta(milliseconds=4),
            type=LLM_RESPONSE,
            payload={"generations": ["The pro plan is $29 per month."]},
            refs=[RefLink(event_id=call_id, kind=RefKind.GROUNDS)],
        ),
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=5,
            ts=_TS + timedelta(seconds=1),
            type=SESSION_END,
            payload={"agent_id": "unit"},
        ),
    ]


async def _store_with(events: list[Event]) -> SQLiteEventStore:
    store = SQLiteEventStore(":memory:")
    for event in events:
        await store.append(event)
    return store


# -- session view ----------------------------------------------------------


class TestSessionView:
    async def test_load_orders_events_and_builds_the_graph(self) -> None:
        session_id = new_event_id()
        events = _session_events(session_id)
        store = await _store_with(events)

        view = await SessionView.load(store, session_id)

        assert [event.seq for event in view.events] == list(range(len(events)))
        assert view.graph.nodes.keys() == {event.event_id for event in events}
        assert view.truncated is False

    async def test_an_open_session_is_not_closed(self) -> None:
        session_id = new_event_id()
        store = await _store_with(_session_events(session_id, close=False))
        view = await SessionView.load(store, session_id)
        assert view.is_closed is False

    async def test_a_bookended_session_is_closed(self) -> None:
        session_id = new_event_id()
        store = await _store_with(_session_events(session_id))
        view = await SessionView.load(store, session_id)
        assert view.is_closed is True

    async def test_truncation_is_explicit_never_silent(self) -> None:
        """A capped view is a partial answer, so it has to say so."""
        session_id = new_event_id()
        store = await _store_with(_session_events(session_id, count=8))

        capped = await SessionView.load(store, session_id, max_events=3)

        assert len(capped.events) == 3
        assert capped.truncated is True
        assert (await SessionView.load(store, session_id)).truncated is False

    def test_the_default_cap_is_bounded(self) -> None:
        assert 0 < DEFAULT_MAX_EVENTS <= 50_000

    def test_typed_accessors_filter_by_type(self) -> None:
        view = _view(_turn_events(new_event_id()))
        assert len(view.llm_responses()) == 1
        assert len(view.llm_requests()) == 1
        assert len(view.tool_calls()) == 1
        assert len(view.tool_results()) == 1
        assert view.tool_name(view.tool_calls()[0]) == "billing.lookup"
        assert view.tool_output(view.tool_results()[0]).strip().startswith("Plan pro")

    def test_lifecycle_helpers_report_the_bookends(self) -> None:
        view = _view(_turn_events(new_event_id()))
        assert view.agent_id == "unit"
        assert view.started_at is not None
        assert view.ended_at is not None
        assert view.ended_at > view.started_at

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"generations": ["hello"]}, "hello"),
            ({"generations": [["hello"]]}, "hello"),
            ({"generations": [{"text": "hello"}]}, "hello"),
            ({"response": {"message": {"content": "hello"}}}, "hello"),
            ({"response": {"content": "hello"}}, "hello"),
            ({"transcript": ["hello"]}, "hello"),
            ({"generations": []}, ""),
            ({}, ""),
        ],
    )
    def test_response_text_reads_every_producer_shape(
        self, payload: dict[str, object], expected: str
    ) -> None:
        """Instrumentors disagree on the shape; the view has to normalize them."""
        event = Event(
            event_id=new_event_id(),
            session_id=new_event_id(),
            seq=0,
            ts=_TS,
            type=LLM_RESPONSE,
            payload=payload,
        )
        assert response_text_of(event) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, ""),
            ("text", "text"),
            (7, "7"),
            ({"a": 1, "b": "two"}, "a=1, b=two"),
            (["one", "two"], "one two"),
        ],
    )
    def test_as_text_renders_without_inventing_structure(
        self, value: object, expected: str
    ) -> None:
        assert as_text(value) == expected


# -- checkpoints -----------------------------------------------------------


class TestCheckpointStore:
    async def test_the_in_memory_store_round_trips(self) -> None:
        store = InMemoryCheckpointStore()
        assert await store.is_done("m@1", "s1") is False
        await store.mark_done("m@1", "s1", flag_ids=("f1",))
        assert await store.is_done("m@1", "s1") is True
        assert await store.flag_ids("m@1", "s1") == ("f1",)

    async def test_namespaces_are_isolated(self) -> None:
        """A new module version re-evaluates; the old checkpoint must not leak."""
        store = InMemoryCheckpointStore()
        await store.mark_done("m@1", "s1")
        assert await store.is_done("m@2", "s1") is False

    async def test_clear_drops_one_namespace_or_all_of_them(self) -> None:
        store = InMemoryCheckpointStore()
        await store.mark_done("m@1", "s1")
        await store.mark_done("m@2", "s2")
        await store.clear("m@1")
        assert await store.is_done("m@1", "s1") is False
        assert await store.is_done("m@2", "s2") is True
        await store.clear()
        assert await store.is_done("m@2", "s2") is False

    async def test_failures_accumulate_and_clear_on_success(self) -> None:
        store = InMemoryCheckpointStore()
        assert await store.record_failure("m@1", "s1", "boom") == 1
        assert await store.record_failure("m@1", "s1", "boom") == 2
        assert await store.attempts("m@1", "s1") == 2
        await store.mark_done("m@1", "s1")
        assert await store.attempts("m@1", "s1") == 0

    async def test_the_sqlite_store_survives_a_restart(self, tmp_path: Path) -> None:
        """The durable store is the whole point of ``S3-T2``: a restarted worker
        must not redo finished work."""
        path = str(tmp_path / "checkpoints.sqlite3")
        first = SqliteCheckpointStore(path)
        await first.mark_done("m@1", "s1", flag_ids=("f1",), evaluated_at=_TS)
        await first.close()

        second = SqliteCheckpointStore(path)
        assert await second.is_done("m@1", "s1") is True
        assert await second.flag_ids("m@1", "s1") == ("f1",)
        await second.close()

    async def test_the_sqlite_store_counts_failures(self, tmp_path: Path) -> None:
        store = SqliteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
        assert await store.record_failure("m@1", "s1", "boom") == 1
        assert await store.record_failure("m@1", "s1", "boom") == 2
        await store.close()


# -- worker ----------------------------------------------------------------


class _CountingWorker(EvaluatorWorker):
    """A module that flags once per response, and can be told to misbehave."""

    def __init__(self, store: SQLiteEventStore, *, config: WorkerConfig, **kwargs: object):
        super().__init__(store, config=config, **kwargs)  # type: ignore[arg-type]
        self.calls = 0
        self.confidence = 0.9
        #: The exception raised while :attr:`fail_times` is above zero.
        self.fail_with: Exception | None = None
        self.fail_times = 0

    async def evaluate(self, session: SessionView) -> list[Flag]:
        self.calls += 1
        if self.fail_with is not None and self.fail_times:
            self.fail_times -= 1
            raise self.fail_with
        return [
            Flag.create(
                session_id=session.session_id,
                module=self.module,
                module_version=self.module_version,
                category="ungrounded_claim",
                confidence=self.confidence,
                summary=f"response {index} needs review",
                evidence=[
                    EvidenceRef(event_id=event.event_id, role=EvidenceRole.CLAIM, seq=event.seq)
                ],
                created_at=event.ts,
                event_id=event.event_id,
                dedupe_key=f"c{index}",
            )
            for index, event in enumerate(session.llm_responses())
        ]


def _responses(session_id: str, count: int, *, start_seq: int = 2) -> list[Event]:
    """*count* response events at contiguous seqs, for the batching tests."""
    return [
        Event(
            event_id=new_event_id(),
            session_id=session_id,
            seq=start_seq + index,
            ts=_TS + timedelta(milliseconds=start_seq + index),
            type=LLM_RESPONSE,
            payload={"generations": [f"claim {index}"]},
        )
        for index in range(count)
    ]


async def _worker(tmp_path: Path, **overrides: object) -> tuple[_CountingWorker, str]:
    session_id = new_event_id()
    store = await _store_with(_session_events(session_id))
    config = WorkerConfig(
        module="unit",
        module_version="0.1.0",
        backoff_base_s=0.0,
        backoff_max_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )
    worker = _CountingWorker(store, config=config)
    return worker, session_id


class TestWorker:
    async def test_the_namespace_is_module_at_version(self, tmp_path: Path) -> None:
        worker, _ = await _worker(tmp_path)
        assert worker.namespace == "unit@0.1.0"

    async def test_a_pass_evaluates_completed_sessions_only(self, tmp_path: Path) -> None:
        worker, _ = await _worker(tmp_path)
        run = await worker.run_once()
        assert run.sessions_evaluated == 1
        assert run.sessions_failed == 0
        assert run.total_sessions == 1

    async def test_a_second_pass_skips_what_it_finished(self, tmp_path: Path) -> None:
        worker, _ = await _worker(tmp_path)
        await worker.run_once()
        run = await worker.run_once()
        assert run.sessions_evaluated == 0
        assert worker.calls == 1

    async def test_force_re_evaluates_and_does_not_duplicate(self, tmp_path: Path) -> None:
        """Flag writes are idempotent by ``flag_id``, so a forced re-run is safe."""
        session_id = new_event_id()
        store = await _store_with(
            _session_events(session_id, count=2, close=False) + _responses(session_id, 1)
        )
        config = WorkerConfig(module="unit", module_version="0.1.0", backoff_base_s=0.0)
        worker = _CountingWorker(store, config=config)

        first = await worker.evaluate_session(session_id, force=True)
        second = await worker.evaluate_session(session_id, force=True)
        stored = await store.get_flags(session_id=session_id)

        assert [flag.flag_id for flag in first] == [flag.flag_id for flag in second]
        assert len(stored) == len(first)

    async def test_an_already_evaluated_session_returns_no_flags(self, tmp_path: Path) -> None:
        worker, session_id = await _worker(tmp_path)
        await worker.evaluate_session(session_id)
        assert await worker.evaluate_session(session_id) == []

    async def test_a_transient_failure_is_retried(self, tmp_path: Path) -> None:
        worker, session_id = await _worker(tmp_path, max_attempts=3)
        worker.fail_with = RuntimeError("store hiccup")
        worker.fail_times = 99
        with pytest.raises(EvaluationError):
            await worker.evaluate_session(session_id)
        assert worker.calls == 3
        assert await worker.checkpoints.attempts(worker.namespace, session_id) == 3

    async def test_a_transient_failure_that_clears_succeeds(self, tmp_path: Path) -> None:
        worker, session_id = await _worker(tmp_path, max_attempts=3)
        worker.fail_with = RuntimeError("store hiccup")
        worker.fail_times = 2

        assert await worker.evaluate_session(session_id) == []
        assert worker.calls == 3

    async def test_a_deterministic_failure_is_never_retried(self, tmp_path: Path) -> None:
        """A module bug that would recur identically only burns budget."""
        worker, session_id = await _worker(tmp_path, max_attempts=5)
        worker.fail_with = DeterministicEvaluationError("malformed payload")
        worker.fail_times = 99
        with pytest.raises(DeterministicEvaluationError):
            await worker.evaluate_session(session_id)
        assert worker.calls == 1

    async def test_one_bad_session_does_not_stall_the_pass(self, tmp_path: Path) -> None:
        session_id = new_event_id()
        store = await _store_with(_session_events(session_id))
        config = WorkerConfig(module="unit", module_version="0.1.0", backoff_base_s=0.0)
        worker = _CountingWorker(store, config=config)
        worker.fail_with = RuntimeError("boom")
        worker.fail_times = 99

        run = await worker.run_once()

        assert run.sessions_evaluated == 0
        assert run.sessions_failed == 1
        assert run.errors
        assert session_id in run.errors[0]

    async def test_flags_are_written_in_batches(self, tmp_path: Path) -> None:
        """Batching is what keeps a 500-flag session to a few round trips."""
        session_id = new_event_id()
        store = await _store_with(
            _session_events(session_id, count=2, close=False) + _responses(session_id, 5)
        )
        config = WorkerConfig(
            module="unit", module_version="0.1.0", flag_batch_size=2, backoff_base_s=0.0
        )
        worker = _CountingWorker(store, config=config)

        await worker.evaluate_session(session_id)

        assert len(await store.get_flags(session_id=session_id)) == 5

    async def test_low_confidence_flags_are_routed_to_review(self, tmp_path: Path) -> None:
        session_id = new_event_id()
        store = await _store_with(
            _session_events(session_id, count=2, close=False) + _responses(session_id, 1)
        )
        config = WorkerConfig(
            module="unit", module_version="0.1.0", review_confidence_threshold=0.95
        )
        worker = _CountingWorker(store, config=config)

        await worker.evaluate_session(session_id)

        flags = await store.get_flags(session_id=session_id)
        assert flags
        assert all(flag.review_only for flag in flags)

    async def test_a_new_module_version_re_evaluates(self, tmp_path: Path) -> None:
        session_id = new_event_id()
        store = await _store_with(_session_events(session_id))
        first = _CountingWorker(store, config=WorkerConfig(module="unit", module_version="0.1.0"))
        await first.evaluate_session(session_id)
        second = _CountingWorker(store, config=WorkerConfig(module="unit", module_version="0.2.0"))
        second.calls = 0
        pending = await second.pending_sessions()
        assert session_id in pending

    async def test_run_forever_stops_when_asked(self, tmp_path: Path) -> None:
        worker, _ = await _worker(tmp_path)
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run_forever(interval_s=0.01, stop=stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        assert worker.calls >= 1
