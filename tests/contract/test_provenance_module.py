"""End-to-end provenance contract (``S3-T5`` - ``S3-T11``, ``S3-T15``).

These tests drive :class:`ProvenanceEvaluator` the way an operator does — over a
session already in the store — and check the properties a flag row has to have
regardless of which rule fired:

* the right *category* (``ungrounded_claim`` for silence, ``contradicted_claim``
  for an active refutation),
* evidence that points at real events, with the claim first,
* ``created_at`` taken from the response event rather than a clock, so two runs
  are byte-identical,
* a review link that a queue can render, and ``review_only`` routing for the
  softer verdicts.
"""

from __future__ import annotations

import pytest

from sentinel.eval.fixtures.provenance_corpus import case_by_id
from sentinel.eval.provenance import (
    CATEGORY_CONTRADICTED,
    CATEGORY_UNGROUNDED,
    MODULE,
    MODULE_VERSION,
    ProvenanceEvaluator,
    ProvenanceResult,
    gather_evidence,
)
from sentinel.eval.session import SessionView
from sentinel.models.events import LLM_RESPONSE
from sentinel.models.flags import Adjudication, EvidenceRole, Flag
from sentinel.query import build_call_graph
from sentinel.store.sqlite import SQLiteEventStore

pytestmark = pytest.mark.contract


async def _store(case_id: str) -> tuple[SQLiteEventStore, str]:
    case = case_by_id(case_id)
    events = case.cited_events()
    store = SQLiteEventStore(":memory:")
    for event in events:
        await store.append(event)
    return store, events[0].session_id


async def _flags(case_id: str, **kwargs: object) -> list[Flag]:
    store, session_id = await _store(case_id)
    evaluator = ProvenanceEvaluator(store, **kwargs)  # type: ignore[arg-type]
    return await evaluator.evaluate_session(session_id)


async def _analyze(case_id: str) -> ProvenanceResult:
    store, session_id = await _store(case_id)
    return await ProvenanceEvaluator(store).analyze_session(session_id)


class TestCategories:
    async def test_silence_is_an_ungrounded_claim(self) -> None:
        flags = await _flags("ungrounded_price_invented")
        assert [flag.category for flag in flags] == [CATEGORY_UNGROUNDED]

    async def test_a_refutation_is_a_contradicted_claim(self) -> None:
        flags = await _flags("contradicted_price")
        assert [flag.category for flag in flags] == [CATEGORY_CONTRADICTED]

    async def test_a_grounded_response_produces_no_flags(self) -> None:
        assert await _flags("grounded_explicit_price") == []

    async def test_a_no_tool_response_is_ungrounded(self) -> None:
        flags = await _flags("ungrounded_no_tool_output")
        assert flags
        assert flags[0].category == CATEGORY_UNGROUNDED


class TestFlagShape:
    async def test_every_flag_is_attributed_to_the_module(self) -> None:
        for flag in await _flags("contradicted_price"):
            assert flag.module == MODULE
            assert flag.module_version == MODULE_VERSION

    async def test_the_claim_event_is_the_first_evidence(self) -> None:
        """A reviewer opens the claim, then the source that bears on it."""
        flags = await _flags("contradicted_price")
        assert flags
        assert flags[0].evidence[0].role is EvidenceRole.CLAIM

    async def test_the_evidence_events_exist_in_the_log(self) -> None:
        store, session_id = await _store("contradicted_price")
        evaluator = ProvenanceEvaluator(store)
        await evaluator.evaluate_session(session_id)

        logged = {event.event_id for event in await store.get_session(session_id)}
        for flag in await store.get_flags(session_id=session_id):
            assert {ref.event_id for ref in flag.evidence} <= logged

    async def test_the_flag_carries_the_claim_and_the_verdict(self) -> None:
        flags = await _flags("contradicted_price")
        assert flags
        details = flags[0].details
        assert "$29" in details["claim_text"]
        assert details["verdict"] == "conflicted"
        assert details["support_kind"]

    async def test_confidence_and_severity_are_in_range(self) -> None:
        flags = await _flags("contradicted_price")
        assert flags
        assert 0.0 <= flags[0].confidence <= 1.0
        assert flags[0].severity in {"low", "medium", "high", "critical"}

    async def test_a_contradiction_outranks_an_ungrounded_claim(self) -> None:
        """Money first: the corpus case that is wrong is also the one that hurts."""
        flags = await _flags("contradicted_price")
        assert flags[0].severity in {"high", "critical"}


class TestDeterminism:
    async def test_two_runs_produce_identical_flags(self) -> None:
        """``S3-T4``: same log, same version, byte-identical rows."""
        store, session_id = await _store("contradicted_price")
        first = await ProvenanceEvaluator(store).evaluate_session(session_id, force=True)
        second = await ProvenanceEvaluator(store).evaluate_session(session_id, force=True)
        assert [flag.model_dump_json() for flag in first] == [
            flag.model_dump_json() for flag in second
        ]

    async def test_created_at_is_the_response_timestamp(self) -> None:
        """Never a clock read: the flag points at when the claim was made."""
        store, session_id = await _store("contradicted_price")
        await ProvenanceEvaluator(store).evaluate_session(session_id)
        response = next(
            event for event in await store.get_session(session_id) if event.type == LLM_RESPONSE
        )
        flags = await store.get_flags(session_id=session_id)
        assert all(flag.created_at == response.ts for flag in flags)

    async def test_a_new_module_version_produces_new_flag_ids(self) -> None:
        """The idempotency key has to change with the rules, or a fix never lands."""
        store, session_id = await _store("contradicted_price")
        await ProvenanceEvaluator(store).evaluate_session(session_id)
        bumped = ProvenanceEvaluator(
            store,
            config=ProvenanceEvaluator(store).config.__class__(
                module=MODULE, module_version="0.2.0", review_confidence_threshold=0.0
            ),
        )
        await bumped.evaluate_session(session_id, force=True)
        stored = await store.get_flags(session_id=session_id)
        assert {flag.module_version for flag in stored} == {MODULE_VERSION, "0.2.0"}


class TestReview:
    async def test_a_review_link_rides_in_details(self) -> None:
        flags = await _flags(
            "contradicted_price", review_url_template="https://review.test/{session_id}"
        )
        assert flags
        assert flags[0].details["review_url"] == f"https://review.test/{flags[0].session_id}"

    async def test_no_template_means_no_link(self) -> None:
        flags = await _flags("contradicted_price")
        assert flags
        assert "review_url" not in flags[0].details

    async def test_soft_verdicts_are_routed_to_review(self) -> None:
        """``S3-T15``: an ungrounded claim is a hypothesis, so it queues."""
        flags = await _flags("ungrounded_price_invented", review_confidence_threshold=1.0)
        assert flags
        assert all(flag.review_only for flag in flags)

    async def test_review_only_flags_do_not_gate(self) -> None:
        store, session_id = await _store("ungrounded_price_invented")
        evaluator = ProvenanceEvaluator(store, review_confidence_threshold=1.0)
        await evaluator.evaluate_session(session_id)
        gated = await store.get_flags(session_id=session_id, review_only=False)
        queued = await store.get_flags(session_id=session_id, review_only=True)
        assert gated == []
        assert len(queued) == 1


class TestAdjudication:
    async def test_a_reviewer_can_adjudicate_a_flag(self) -> None:
        """``S3-T15``: the queue closes the loop, and the verdict is first-write-wins."""
        store, session_id = await _store("contradicted_price")
        evaluator = ProvenanceEvaluator(store)
        flags = await evaluator.evaluate_session(session_id)

        await store.adjudicate_flag(
            flags[0].flag_id, Adjudication.CONFIRMED, adjudicated_by="sentinel_reviewer"
        )

        stored = await store.get_flags(session_id=session_id, adjudication=Adjudication.CONFIRMED)
        assert len(stored) == 1
        assert stored[0].adjudication == Adjudication.CONFIRMED


class TestEvidenceGathering:
    async def test_a_cited_result_is_explicit_evidence(self) -> None:
        case = case_by_id("grounded_explicit_price")
        events = case.cited_events()
        graph = build_call_graph(events[0].session_id, events)
        response = next(event for event in events if event.type == LLM_RESPONSE)

        context, refs = gather_evidence(graph, response)

        assert context.explicit
        assert context.explicit[0].startswith("Plan pro")
        assert any(ref.role is EvidenceRole.EVIDENCE for ref in refs)

    async def test_an_uncited_result_is_still_context(self) -> None:
        """Available-but-unused evidence must not become a false positive."""
        case = case_by_id("grounded_explicit_price")
        events = case.events()
        graph = build_call_graph(events[0].session_id, events)
        response = next(event for event in events if event.type == LLM_RESPONSE)

        context, refs = gather_evidence(graph, response)

        assert context.explicit == ()
        assert context.context
        assert all(ref.role is EvidenceRole.CONTEXT for ref in refs)

    async def test_a_response_with_no_turn_has_no_evidence(self) -> None:
        case = case_by_id("ungrounded_no_tool_output")
        events = case.events()
        graph = build_call_graph(events[0].session_id, events)
        response = next(event for event in events if event.type == LLM_RESPONSE)

        context, refs = gather_evidence(graph, response)

        assert context.has_evidence is False
        assert refs == ()

    async def test_evidence_is_capped_per_response(self) -> None:
        from sentinel.eval.provenance import MAX_EVIDENCE_RESULTS

        case = case_by_id("grounded_explicit_price")
        events = case.cited_events()
        graph = build_call_graph(events[0].session_id, events)
        response = next(event for event in events if event.type == LLM_RESPONSE)
        _, refs = gather_evidence(graph, response)
        assert len(refs) <= MAX_EVIDENCE_RESULTS + 1


class TestAnalysis:
    async def test_the_result_reports_what_it_saw(self) -> None:
        result = await _analyze("ungrounded_price_invented")
        assert result.responses_seen == 1
        assert result.responses_without_evidence == 0
        assert result.flagged_claims >= 1

    async def test_a_response_with_no_tool_reports_no_evidence(self) -> None:
        result = await _analyze("ungrounded_no_tool_output")
        assert result.responses_without_evidence == 1

    async def test_supported_claims_are_reported_as_grounded(self) -> None:
        result = await _analyze("grounded_explicit_price")
        assert result.supported
        assert result.findings == []

    async def test_the_view_is_the_only_input(self) -> None:
        """``analyze`` is pure: the same view twice gives the same answer."""
        store, session_id = await _store("contradicted_price")
        evaluator = ProvenanceEvaluator(store)
        view = await SessionView.load(store, session_id)
        first = await evaluator.analyze(view)
        second = await evaluator.analyze(view)
        assert [item.claim.text for item in first.findings] == [
            item.claim.text for item in second.findings
        ]
