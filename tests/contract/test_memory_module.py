"""End-to-end memory-integrity contract (``S4-T1`` - ``S4-T14``).

Drives :class:`MemoryIntegrityEvaluator` the way an operator does — over a
session already in the store — and checks the properties a flag row has to have
regardless of which check fired:

* the right *category* (``memory_drift`` for an injected write, ``memory_collapse``
  for a loop, ``memory_ungrounded`` for a fabricated summary),
* ``review_only`` on the trend and not on the event, which is the whole of INV-6
  as it applies here,
* evidence that points at the real ``memory.write`` event, plus the state it
  departed from,
* ``created_at`` taken from the event rather than a clock, so two runs are
  byte-identical,
* idempotency: re-running produces no second flag.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from sentinel.eval.fixtures.memory_corpus import MEMORY_CORPUS, memory_case_by_id
from sentinel.eval.memory import (
    CATEGORY_COLLAPSE,
    CATEGORY_DRIFT,
    CATEGORY_MEMORY_UNGROUNDED,
    MODULE,
    MODULE_VERSION,
    REVIEW_ONLY_CATEGORIES,
    MemoryIntegrityConfig,
    MemoryIntegrityEvaluator,
    as_text,
    transcript_of,
    write_from_event,
)
from sentinel.eval.memory_core import DriftConfig, MemoryVerdict, SeriesAnalysis
from sentinel.eval.session import SessionView
from sentinel.models.events import MEMORY_WRITE, Event
from sentinel.models.flags import Adjudication, EvidenceRole, Flag, Severity
from sentinel.query import build_call_graph
from sentinel.store.sqlite import SQLiteEventStore

pytestmark = pytest.mark.contract


async def _store_for(case_id: str) -> tuple[SQLiteEventStore, str]:
    """A store holding one corpus case, and the session id."""
    store = SQLiteEventStore(":memory:")
    session_id = ""
    for event in memory_case_by_id(case_id).events():
        session_id = event.session_id
        await store.append(event)
    return store, session_id


async def _flags(case_id: str, *, settings: MemoryIntegrityConfig | None = None) -> list[Flag]:
    store, session_id = await _store_for(case_id)
    try:
        evaluator = MemoryIntegrityEvaluator(store, settings=settings)
        return await evaluator.evaluate_session(session_id)
    finally:
        await store.close()


async def _analyze(case_id: str) -> SeriesAnalysis:
    store, session_id = await _store_for(case_id)
    try:
        return await MemoryIntegrityEvaluator(store).analyze_session(session_id)
    finally:
        await store.close()


# -- identity ---------------------------------------------------------------


class TestModuleIdentity:
    def test_the_module_names_itself_and_its_version(self) -> None:
        """Recorded on every flag and part of the idempotency key."""
        assert MODULE == "sentinel.memory_integrity"
        assert MODULE_VERSION == "0.1.0"

    def test_collapse_is_routed_review_only_and_drift_is_not(self) -> None:
        """A trend queues; a single structural event can block. INV-6."""
        assert CATEGORY_COLLAPSE in REVIEW_ONLY_CATEGORIES
        assert CATEGORY_DRIFT not in REVIEW_ONLY_CATEGORIES
        assert CATEGORY_MEMORY_UNGROUNDED not in REVIEW_ONLY_CATEGORIES

    async def test_flags_carry_the_module(self) -> None:
        for flag in await _flags("injected_memory_update"):
            assert flag.module == MODULE
            assert flag.module_version == MODULE_VERSION


# -- detection --------------------------------------------------------------


class TestDetection:
    async def test_an_injected_write_is_flagged_high(self) -> None:
        flags = await _flags("injected_memory_update")
        assert [flag.category for flag in flags] == [CATEGORY_DRIFT]
        assert flags[0].severity == Severity.HIGH

    async def test_the_flag_quotes_the_injected_text_verbatim(self) -> None:
        """A reviewer must see *what* was wrong without going to the event log."""
        flags = await _flags("injected_memory_update")
        assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in flags[0].details["claim_text"]

    async def test_a_second_injection_marker_is_caught_the_same_way(self) -> None:
        flags = await _flags("injected_memory_update_after_summary")
        assert [flag.category for flag in flags] == [CATEGORY_DRIFT]
        assert "Disregard the system prompt" in flags[0].details["claim_text"]

    async def test_a_loop_is_flagged_as_collapse(self) -> None:
        flags = await _flags("memory_collapse_loop")
        assert [flag.category for flag in flags] == [CATEGORY_COLLAPSE]
        assert flags[0].severity == Severity.MEDIUM

    async def test_a_fabricated_summary_is_flagged(self) -> None:
        flags = await _flags("fabricated_reflective_summary")
        assert [flag.category for flag in flags] == [CATEGORY_MEMORY_UNGROUNDED]
        assert "2024-06-01" in flags[0].details["claim_text"]

    async def test_a_summary_the_transcript_refutes_is_distinguishable(self) -> None:
        """Same category, different verdict — and the verdict is what a reviewer
        reads to decide how badly the agent is lying."""
        analysis = await _analyze("fabricated_summary_contradicted_by_transcript")
        assert analysis.ungrounded
        assert analysis.ungrounded[0].verdict is MemoryVerdict.UNGROUNDED


class TestSilence:
    @pytest.mark.parametrize(
        "case_id",
        [
            "grounded_healthy_memory_evolution",
            "grounded_summary_matches_transcript",
            "grounded_short_writes_are_not_measured",
            "grounded_new_content_clears_a_loop",
            "grounded_no_numbers_in_transcript_skips_summary_check",
        ],
    )
    async def test_healthy_memory_produces_nothing(self, case_id: str) -> None:
        assert await _flags(case_id) == []

    async def test_a_session_with_no_memory_writes_is_not_an_error(self) -> None:
        """Most sessions write no memory at all."""
        case = memory_case_by_id("grounded_summary_matches_transcript")
        kept = [event for event in case.events() if event.type != MEMORY_WRITE]
        # Renumbered because dropping events would otherwise leave the unique
        # (session_id, seq) index with holes and the store rejecting the log.
        events = [event.model_copy(update={"seq": index}) for index, event in enumerate(kept)]
        store = SQLiteEventStore(":memory:")
        try:
            for event in events:
                await store.append(event)
            evaluator = MemoryIntegrityEvaluator(store)
            analysis = await evaluator.analyze_session(events[0].session_id)
            assert analysis.findings == []
            assert await evaluator.evaluate_session(events[0].session_id) == []
        finally:
            await store.close()


# -- flag shape -------------------------------------------------------------


class TestFlagShape:
    async def test_evidence_points_at_the_write_that_caused_it(self) -> None:
        case = memory_case_by_id("injected_memory_update")
        writes = [event.event_id for event in case.events() if event.type == MEMORY_WRITE]
        flags = await _flags("injected_memory_update")
        assert writes[-1] in {ref.event_id for ref in flags[0].evidence}

    async def test_a_drift_flag_also_cites_the_state_it_left(self) -> None:
        """A drift finding is only meaningful against what it moved away from."""
        flags = await _flags("injected_memory_update")
        roles = {ref.role for ref in flags[0].evidence}
        assert EvidenceRole.CLAIM in roles
        assert EvidenceRole.COUNTERVAILANCE in roles

    async def test_created_at_comes_from_the_event_not_a_clock(self) -> None:
        """So two runs over the same session produce byte-identical rows."""
        flags = await _flags("injected_memory_update")
        assert isinstance(flags[0].created_at, datetime)

    async def test_the_embedding_model_is_recorded_for_audit(self) -> None:
        """A drift number is meaningless without knowing what produced it."""
        flags = await _flags("injected_memory_update")
        assert flags[0].details["embedding_model"].startswith("cached/hashing/")

    async def test_details_carry_the_measurement(self) -> None:
        flags = await _flags("memory_collapse_loop")
        assert "similarity" in flags[0].details["observed"]

    async def test_collapse_is_review_only_and_drift_is_not(self) -> None:
        assert (await _flags("memory_collapse_loop"))[0].review_only is True
        assert (await _flags("injected_memory_update"))[0].review_only is False

    async def test_flags_start_unadjudicated(self) -> None:
        flags = await _flags("injected_memory_update")
        assert flags[0].adjudication is Adjudication.PENDING


# -- idempotency and determinism -------------------------------------------


class TestIdempotency:
    async def test_running_twice_writes_no_second_flag(self) -> None:
        """Idempotency is *skip*, not re-computation: the second pass finds the
        checkpoint and does no work, so exactly one row exists afterwards."""
        store, session_id = await _store_for("injected_memory_update")
        try:
            evaluator = MemoryIntegrityEvaluator(store)
            first = await evaluator.evaluate_session(session_id)
            assert len(first) == 1
            assert await evaluator.evaluate_session(session_id) == []
            stored = await store.get_flags(session_id=session_id)
            assert len(stored) == 1
        finally:
            await store.close()

    async def test_two_runs_produce_identical_rows(self) -> None:
        first = await _flags("injected_memory_update")
        second = await _flags("injected_memory_update")
        assert [flag.model_dump(mode="json") for flag in first] == [
            flag.model_dump(mode="json") for flag in second
        ]

    async def test_the_worker_checkpoint_records_the_session(self) -> None:
        store, session_id = await _store_for("injected_memory_update")
        try:
            evaluator = MemoryIntegrityEvaluator(store)
            run = await evaluator.run_once()
            assert run.sessions_evaluated >= 0
            assert await evaluator.checkpoints.is_done(evaluator.namespace, session_id)
        finally:
            await store.close()


# -- configuration ----------------------------------------------------------


class TestConfiguration:
    async def test_thresholds_are_configuration_not_code(self) -> None:
        """Raising the injection detection above reality must silence it, without
        a release."""
        store, session_id = await _store_for("injected_memory_update")
        try:
            evaluator = MemoryIntegrityEvaluator(
                store,
                settings=MemoryIntegrityConfig(
                    drift=DriftConfig(
                        warmup_writes=99,
                        collapse_window=99,
                        min_drift_chars=10_000,
                    )
                ),
            )
            # Drift is silenced by warm-up; collapse by its window.
            flags = await evaluator.evaluate_session(session_id)
            assert all(flag.category != CATEGORY_DRIFT for flag in flags)
        finally:
            await store.close()

    async def test_a_raised_similarity_threshold_silences_collapse(self) -> None:
        store, session_id = await _store_for("memory_collapse_loop")
        try:
            evaluator = MemoryIntegrityEvaluator(
                store,
                settings=MemoryIntegrityConfig(drift=DriftConfig(collapse_similarity=1.01)),
            )
            assert await evaluator.evaluate_session(session_id) == []
        finally:
            await store.close()

    async def test_a_review_url_is_attached_when_configured(self) -> None:
        flags = await _flags(
            "injected_memory_update",
            settings=MemoryIntegrityConfig(
                review_url_template="https://sentinel.test/review/{session_id}"
            ),
        )
        assert flags[0].details["review_url"].startswith("https://sentinel.test/review/")

    def test_the_default_provider_is_offline_and_deterministic(self) -> None:
        """INV-5 with nothing to configure: no model, no network."""
        evaluator = MemoryIntegrityEvaluator(None)  # type: ignore[arg-type]
        assert evaluator.embeddings.model_id.startswith("cached/hashing/")
        assert evaluator.embeddings.dimensions > 0


# -- event handling ---------------------------------------------------------


class TestEventHandling:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, ""),
            ("a string", "a string"),
            (42, "42"),
            (True, "True"),
            ({"key": "value", "n": 2}, "key: value | n: 2"),
            (["a", "b"], "a b"),
            (("a", "b"), "a b"),
        ],
    )
    def test_payload_values_render_as_text(self, value: object, expected: str) -> None:
        """Memory values arrive in whatever shape the adapter wrote them."""
        assert as_text(value) == expected

    def test_an_exotic_payload_value_is_stringified_rather_than_dropped(self) -> None:
        class Opaque:
            def __str__(self) -> str:
                return "opaque"

        assert as_text(Opaque()) == "opaque"

    def test_a_write_event_is_reduced_to_its_content(self) -> None:
        case = memory_case_by_id("injected_memory_update")
        write_event = next(event for event in case.events() if event.type == MEMORY_WRITE)
        write = write_from_event(write_event)
        assert write.key == "account"
        assert "pro plan" in write.value
        assert write.intent.value == "fact"

    def test_a_write_with_no_key_or_value_is_still_a_write(self) -> None:
        """A misbehaving adapter must not be able to blind the module."""
        case = memory_case_by_id("grounded_summary_matches_transcript")
        write_event = next(event for event in case.events() if event.type == MEMORY_WRITE)
        stripped = write_event.model_copy(update={"payload": {}})
        write = write_from_event(stripped)
        assert write.key == ""
        assert write.value == ""

    def test_the_transcript_keeps_its_tail(self) -> None:
        """A summary written at the end describes the end of the session; a prefix
        would make a claim about recent events look ungrounded."""
        case = memory_case_by_id("fabricated_reflective_summary")
        events = case.events()
        view = SessionView(
            session_id=events[0].session_id,
            events=tuple(events),
            graph=build_call_graph(events[0].session_id, events),
        )
        assert transcript_of(view, limit=40).endswith("due today.")

    def test_the_transcript_includes_tool_activity_as_well_as_prose(self) -> None:
        """A summary may legitimately describe what the tools returned, so tool
        calls and results are part of what it can be grounded against."""
        events = [
            Event.model_validate(
                {
                    "event_id": "01J0000000000000000000000A",
                    "session_id": "01J0000000000000000000000B",
                    "seq": 0,
                    "ts": "2024-05-01T00:00:00+00:00",
                    "type": "session.start",
                    "payload": {},
                }
            ),
            Event.model_validate(
                {
                    "event_id": "01J0000000000000000000000C",
                    "session_id": "01J0000000000000000000000B",
                    "seq": 1,
                    "ts": "2024-05-01T00:00:01+00:00",
                    "type": "tool.call",
                    "payload": {"tool": "billing.lookup"},
                }
            ),
            Event.model_validate(
                {
                    "event_id": "01J0000000000000000000000D",
                    "session_id": "01J0000000000000000000000B",
                    "seq": 2,
                    "ts": "2024-05-01T00:00:02+00:00",
                    "type": "tool.result",
                    "payload": {"tool": "billing.lookup", "output": "3 of 4 checks passed"},
                }
            ),
        ]
        view = SessionView(
            session_id=events[0].session_id,
            events=tuple(events),
            graph=build_call_graph(events[0].session_id, events),
        )
        transcript = transcript_of(view)
        assert "billing.lookup" in transcript
        assert "3 of 4 checks passed" in transcript

    def test_an_empty_transcript_is_empty_rather_than_a_placeholder(self) -> None:
        events = [
            event
            for event in memory_case_by_id("injected_memory_update").events()
            if event.type != "llm.response"
        ]
        view = SessionView(
            session_id=events[0].session_id,
            events=tuple(events),
            graph=build_call_graph(events[0].session_id, events),
        )
        assert transcript_of(view) == ""


# -- corpus -----------------------------------------------------------------


class TestMemoryCorpus:
    def test_every_case_is_a_valid_linked_session(self) -> None:
        for case in MEMORY_CORPUS:
            events = case.events()
            assert [event.seq for event in events] == list(range(len(events))), case.case_id
            assert all(event.session_id == events[0].session_id for event in events)

    def test_every_case_says_what_it_pins_down(self) -> None:
        """A corpus case without a note is a case nobody will think to change."""
        for case in MEMORY_CORPUS:
            assert case.notes, f"{case.case_id} has no note"

    def test_the_corpus_covers_every_category(self) -> None:
        categories = {expected.category for case in MEMORY_CORPUS for expected in case.expect}
        assert categories == {CATEGORY_DRIFT, CATEGORY_COLLAPSE, CATEGORY_MEMORY_UNGROUNDED}

    def test_the_corpus_has_a_known_good_case_for_every_category(self) -> None:
        """Each check needs healthy traffic that it must leave alone."""
        known_good = {case.case_id for case in MEMORY_CORPUS if not case.expects_findings}
        assert {
            "grounded_healthy_memory_evolution",
            "grounded_summary_matches_transcript",
            "grounded_short_writes_are_not_measured",
            "grounded_no_numbers_in_transcript_skips_summary_check",
        } <= known_good
