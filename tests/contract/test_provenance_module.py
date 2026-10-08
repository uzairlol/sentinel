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

from collections.abc import Mapping
from decimal import Decimal

import pytest

from sentinel.eval.fixtures.provenance_corpus import case_by_id
from sentinel.eval.provenance import (
    CATEGORY_CONTRADICTED,
    CATEGORY_UNGROUNDED,
    CATEGORY_UNSOURCED,
    MODULE,
    MODULE_VERSION,
    ProvenanceEvaluator,
    ProvenanceResult,
    _response_spans,
    category_for,
    gather_evidence,
)
from sentinel.eval.provenance_core import (
    Claim,
    ClaimKind,
    DiffContext,
    GroundingLexicon,
    RuleBasedClaimExtractor,
    SupportKind,
    Value,
    ValueKind,
    Verdict,
    diff_claim,
    severity_for,
    split_sentences,
)
from sentinel.eval.session import SessionView
from sentinel.models.events import LLM_RESPONSE, Event
from sentinel.models.flags import Adjudication, EvidenceRole, Flag, Severity
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
        assert flags
        assert flags[0].severity in {"high", "critical"}


class TestContradictionEvidence:
    """S3-T8: a contradiction has to carry the value that refutes it."""

    @pytest.mark.parametrize(
        "case_id",
        [
            "contradicted_price",
            "contradicted_weekday",
            "contradicted_date",
            "contradicted_bound_violation",
            "contradicted_negation",
            "contradicted_membership",
            "contradicted_membership_extra",
            "cherry_picked_count",
        ],
    )
    async def test_every_contradiction_names_what_it_saw(self, case_id: str) -> None:
        flags = await _flags(case_id)
        assert flags
        assert flags[0].details["observed_value"]

    @pytest.mark.parametrize(
        "case_id",
        [
            "contradicted_price",
            "contradicted_weekday",
            "contradicted_date",
            "contradicted_bound_violation",
            "contradicted_negation",
            "contradicted_membership",
            "contradicted_membership_extra",
        ],
    )
    async def test_the_refuting_ref_is_labelled_as_such(self, case_id: str) -> None:
        """A reviewer must see the disagreement in the flag, not go find it."""
        flags = await _flags(case_id)
        assert flags
        roles = {ref.role for ref in flags[0].evidence}
        assert EvidenceRole.COUNTERVAILANCE in roles
        assert EvidenceRole.CLAIM in roles

    async def test_the_observed_value_is_readable_english(self) -> None:
        """Not a dataclass repr: this string is quoted to a person."""
        flags = await _flags("contradicted_price")
        assert flags
        assert flags[0].details["observed_value"] == "49 usd"

    async def test_an_ungrounded_claim_names_no_observed_value(self) -> None:
        """There is nothing observed to quote when the agent never looked."""
        flags = await _flags("ungrounded_price_invented")
        assert flags
        assert "observed_value" not in flags[0].details


class TestReasoningTraces:
    """S3-T5: a claim made while thinking is a claim.

    The extractor reads the answer and any captured reasoning trace as two spans
    of one turn, grounded by the same evidence. No instrumenter records a trace
    today, so these tests build one by hand — which is the point: closing the
    ``S1`` capture gap must be a capture change, not a module change.
    """

    @staticmethod
    async def _store_with_response(payload: Mapping[str, object]) -> SQLiteEventStore:
        """A cited price session whose response carries *payload*.

        Returns the store; the session id is the one every corpus case uses, so
        the caller does not have to thread it back out.
        """
        case = case_by_id("grounded_explicit_price")
        events = case.cited_events()
        store = SQLiteEventStore(":memory:")
        for event in events:
            await store.append(
                event.model_copy(update={"payload": payload})
                if event.type == LLM_RESPONSE
                else event
            )
        return store

    @staticmethod
    def _response(payload: Mapping[str, object]) -> Event:
        # Valid ULIDs (Crockford base32, no I/L/O/U) so the event validates
        # without importing the corpus's hashing helper into a unit test.
        return Event.model_validate(
            {
                "event_id": "01J0000000000000000000000A",
                "session_id": "01J0000000000000000000000B",
                "seq": 3,
                "ts": "2024-05-01T00:00:00+00:00",
                "type": LLM_RESPONSE,
                "payload": payload,
            }
        )

    @pytest.mark.parametrize(
        "payload",
        [
            {"generations": ["The answer is 49."], "reasoning": "Maybe it is 79."},
            {"generations": ["The answer is 49."], "thinking": "Maybe it is 79."},
            {
                "response": {
                    "message": {
                        "content": "The answer is 49.",
                        "reasoning_content": "Maybe it is 79.",
                    }
                }
            },
        ],
    )
    def test_a_trace_is_read_wherever_the_provider_put_it(
        self, payload: Mapping[str, object]
    ) -> None:
        assert "Maybe it is 79." in _response_spans(self._response(payload))

    def test_no_trace_means_the_answer_alone(self) -> None:
        spans = _response_spans(self._response({"generations": ["The answer is 49."]}))
        assert spans == "The answer is 49."

    def test_the_spans_do_not_merge_into_a_phantom_claim(self) -> None:
        """The boundary is a blank line, so a truncated trace cannot glue itself
        onto the answer and produce a claim quoting text from neither span."""
        spans = _response_spans(
            self._response({"generations": ["The price is 49 per"], "reasoning": "month, probably"})
        )
        fragments = split_sentences(spans)
        assert fragments == ["The price is 49 per", "month, probably"]

    async def test_a_claim_only_in_the_trace_is_still_grounded_or_flagged(
        self,
    ) -> None:
        """A figure abandoned mid-thought gets the same check as a stated one."""
        payload = {
            "generations": ["The pro plan costs $49 per month."],
            "reasoning": "Maybe the plan costs 79 per month after all.",
        }
        store = await self._store_with_response(payload)
        try:
            result = await ProvenanceEvaluator(store).analyze_session(
                case_by_id("grounded_explicit_price").events()[0].session_id
            )
        finally:
            await store.close()
        claims = [analysis.claim.text for analysis in result.analyses]
        assert any("79" in claim for claim in claims), claims
        # The abandoned figure is judged against the *same* evidence as the
        # answer: $79 is nowhere in the result, so it is unsupported rather than
        # grounded. A trace-only claim must not be invisible to the diff.
        invented = next(analysis for analysis in result.analyses if "79" in analysis.claim.text)
        assert invented.diff.verdict is Verdict.UNKNOWN
        # and the answer's own grounded claim survives beside it, so reading the
        # trace adds a finding rather than replacing one
        assert any(
            analysis.diff.verdict in (Verdict.SUPPORTED, Verdict.IMPLIED)
            for analysis in result.analyses
        ), [a.claim.text for a in result.analyses]


class TestOrderIndependence:
    """The verdict must not depend on the order the evidence arrived.

    Real logs interleave tool calls differently per framework, so a rule that
    only holds for one interleaving is a rule that will be wrong in production
    rather than merely untested. The property under test is narrow and
    deliberate: **the order of the results in the turn, and repeated runs, may
    not change a verdict.**

    Citation state is *not* varied here, because it is a semantic input the
    unsourced rule is designed to read. Varying it would assert that
    fabrication and honest citation are indistinguishable, which is the
    opposite of what the rule is for.
    """

    async def _verdicts(self, case_id: str) -> list[tuple[str, str]]:
        events = case_by_id(case_id).events()
        store = SQLiteEventStore(":memory:")
        try:
            for event in events:
                await store.append(event)
            result = await ProvenanceEvaluator(store).analyze_session(events[0].session_id)
        finally:
            # Explicit close: the aiosqlite worker thread outlives the test's
            # event loop otherwise, and pytest reports the fallout as an
            # unhandled thread exception that looks like a module bug.
            await store.close()
        return [(category_for(a.diff), a.diff.verdict.value) for a in result.findings]

    @pytest.mark.parametrize(
        "case_id",
        [
            "grounded_explicit_price",
            "contradicted_price",
            "ungrounded_price_invented",
            "fabricated_citation_no_source",
            "cherry_picked_count",
            "grounded_count_reported_whole",
            "grounded_count_verbatim",
        ],
    )
    async def test_two_runs_agree_for_every_rule(self, case_id: str) -> None:
        """Each rule reads the same evidence twice with the same result.

        The new rules thread session-level state (``citations_recorded``) and
        scan a whole turn, which is exactly where order-dependence creeps in.
        """
        assert await self._verdicts(case_id) == await self._verdicts(case_id)

    @pytest.mark.parametrize(
        "case_id",
        [
            "grounded_count_reported_whole",
            "grounded_count_verbatim",
            "grounded_attributed_and_cited",
        ],
    )
    async def test_the_grounded_cases_stay_silent(self, case_id: str) -> None:
        """The guards, restated as properties rather than as one case each.

        A count the agent reported in full, a count copied verbatim, and a
        claim that named a source and cited it: none of these may ever produce
        a finding. They are the three ways the new rules could fire on honest
        output, so they are worth asserting together.
        """
        assert await self._verdicts(case_id) == []

    async def test_a_repeated_run_is_byte_identical(self) -> None:
        store, session_id = await _store("cherry_picked_count")
        first = await ProvenanceEvaluator(store).evaluate_session(session_id, force=True)
        second = await ProvenanceEvaluator(store).evaluate_session(session_id, force=True)
        assert [flag.model_dump_json() for flag in first] == [
            flag.model_dump_json() for flag in second
        ]

    async def test_the_count_rule_reads_the_whole_turn_not_the_first_result(self) -> None:
        """A count split across two results must still be checked as a set.

        ``_enumerated_count`` scans every result the turn carried, so an agent
        cannot escape the rule by receiving its checks in separate tool calls.
        """
        claims = RuleBasedClaimExtractor().extract_sync("2 of 3 checks passed.")
        context, _refs = (
            DiffContext(explicit=("check_1 pass, check_2 pass.", "check_3 fail, check_4 fail.")),
            (),
        )
        diff = diff_claim(claims[0], context)
        assert diff.verdict is Verdict.CONFLICTED
        assert diff.support_kind is SupportKind.CHERRY_PICK


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
        events = case.uncited_events()
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


class TestUnsourcedCitations:
    """S3-T8: the claim named a source and cited none.

    This is the case the sprint exists for. The whole feature rests on one
    distinction, and these tests pin both halves of it: a fabricated citation
    is reported *only* when the session proves citations are being recorded, and
    it is never reported when the source really was cited.
    """

    async def test_a_named_source_that_was_never_cited_is_unsourced(self) -> None:
        flags = await _flags("fabricated_citation_no_source")
        assert [flag.category for flag in flags] == [CATEGORY_UNSOURCED]

    async def test_the_flag_names_the_source_the_agent_claimed(self) -> None:
        """A reviewer needs to know *which* document was never opened."""
        flags = await _flags("fabricated_citation_no_source")
        assert flags
        assert "compliance report" in flags[0].details["claim_text"]
        # and as a field of its own, because a reviewer should not have to
        # re-read the sentence to learn what the agent made up
        assert flags[0].details["claimed_source"] == "the compliance report"

    async def test_an_uncited_claim_beside_real_tool_output_is_unsourced(self) -> None:
        """The harder variant: a tool ran, and the agent cited something else."""
        flags = await _flags("fabricated_citation_uncited_among_tools")
        assert [flag.category for flag in flags] == [CATEGORY_UNSOURCED]

    async def test_a_claim_that_cites_its_source_is_never_unsourced(self) -> None:
        """The false-positive guard, and the reason the rule is safe to ship."""
        assert await _flags("grounded_attributed_and_cited") == []

    async def test_unsourced_is_distinct_from_plain_silence(self) -> None:
        """Different remedy: 'never opened' is not 'no support for this number'."""
        unsourced = await _flags("fabricated_citation_no_source")
        ungrounded = await _flags("ungrounded_price_invented")
        assert unsourced[0].category != ungrounded[0].category
        assert unsourced[0].details["verdict"] == "unsourced"

    async def test_the_verdict_survives_the_flag_round_trip(self) -> None:
        store, session_id = await _store("fabricated_citation_no_source")
        await ProvenanceEvaluator(store).evaluate_session(session_id)
        stored = await store.get_flags(session_id=session_id)
        assert [flag.category for flag in stored] == [CATEGORY_UNSOURCED]

    async def test_a_session_that_never_cites_does_not_claim_fabrication(self) -> None:
        """No citations recorded anywhere means the instrumentation may not emit
        them at all, which is indistinguishable from silence. Staying quiet is
        the honest answer; guessing would flag every honest answer-from-memory.
        """
        case = case_by_id("ungrounded_no_tool_output")
        events = case.events()
        store = SQLiteEventStore(":memory:")
        for event in events:
            await store.append(event)
        flags = await ProvenanceEvaluator(store).evaluate_session(events[0].session_id)
        assert [flag.category for flag in flags] == [CATEGORY_UNGROUNDED]

    async def test_confidence_sits_between_silence_and_contradiction(self) -> None:
        flags = await _flags("fabricated_citation_no_source")
        ungrounded = await _flags("ungrounded_no_tool_output")
        assert ungrounded[0].confidence < flags[0].confidence


class TestSafetySeverity:
    """S3-T9: a wrong safety limit must not rank like a wrong meeting time."""

    @staticmethod
    def _severity_of(response: str, evidence: str) -> Severity:
        claim = RuleBasedClaimExtractor().extract_sync(response)[0]
        context = DiffContext(explicit=(evidence,), citations_recorded=True)
        return severity_for(claim, diff_claim(claim, context))

    @pytest.mark.parametrize(
        ("response", "evidence", "floor"),
        [
            ("The medication is safe for everyone.", "label: high risk", Severity.HIGH),
            (
                "The procedure is approved by the regulator.",
                "filing_status: not approved",
                Severity.HIGH,
            ),
            (
                "The vendor holds ISO 27001 certification.",
                "certs: none on file",
                Severity.MEDIUM,
            ),
        ],
    )
    def test_a_consequential_domain_raises_the_severity(
        self, response: str, evidence: str, floor: Severity
    ) -> None:
        assert self._severity_of(response, evidence).rank >= floor.rank

    def test_an_ordinary_claim_is_not_escalated(self) -> None:
        """The floor must be inert on everyday claims, or it escalates noise."""
        assert (
            self._severity_of("The price is 49 per month.", "The price is 79.") is not Severity.HIGH
        )

    def test_a_grounded_safety_claim_is_still_info(self) -> None:
        """The floor is a floor: it never invents severity for supported claims."""
        claim = RuleBasedClaimExtractor().extract_sync("The medication dosage is 5 mg.")[0]
        context = DiffContext(explicit=("medication dosage = 5 mg",), citations_recorded=True)
        diff = diff_claim(claim, context)
        assert diff.verdict is Verdict.SUPPORTED
        assert severity_for(claim, diff) is Severity.INFO

    def test_the_lexicon_cannot_be_evaded_by_inflection(self) -> None:
        """``compliant``/``compliance``/``medications`` must all land."""
        for response in (
            "The vendor is HIPAA compliant.",
            "The vendor has a compliance report.",
            "The medication list is short.",
        ):
            assert self._severity_of(response, "audit: gaps found") is not Severity.INFO


class TestCherryPicking:
    """S3-T12: a count that omits the items the source enumerated."""

    async def test_an_undercounted_denominator_is_a_contradiction(self) -> None:
        flags = await _flags("cherry_picked_count")
        assert [flag.category for flag in flags] == [CATEGORY_CONTRADICTED]
        assert flags[0].details["verdict"] == "conflicted"

    async def test_the_support_kind_names_the_technique(self) -> None:
        flags = await _flags("cherry_picked_count")
        assert flags[0].details["support_kind"] == "cherry_pick"

    async def test_the_omission_is_quantified(self) -> None:
        """ "2 of 3" against four items: the reviewer is told one was dropped."""
        flags = await _flags("cherry_picked_count")
        assert "4 items" in flags[0].details["observed_value"]

    async def test_it_outranks_a_contradicted_number(self) -> None:
        """A miscounted report is a misrepresentation, so it reads as high."""
        flags = await _flags("cherry_picked_count")
        assert flags[0].severity in {"high", "critical"}

    async def test_a_count_over_the_whole_set_is_not_flagged(self) -> None:
        assert await _flags("grounded_count_reported_whole") == []

    async def test_a_verbatim_count_is_not_flagged(self) -> None:
        """``12 of 20 seats`` with no enumerable set: copied, not cherry-picked."""
        assert await _flags("grounded_count_verbatim") == []

    async def test_a_ratio_inside_a_mixed_response_does_not_disturb_it(self) -> None:
        """The existing mixed case carries a count; it must keep one finding."""
        flags = await _flags("grounded_multi_claim_mixed")
        assert [flag.category for flag in flags] == [CATEGORY_UNGROUNDED]


class TestRatioExtraction:
    """The ratio rule only fires on a count it can reason about."""

    def test_a_count_with_a_noun_is_a_ratio_claim(self) -> None:
        extractor = RuleBasedClaimExtractor()
        claim = extractor.extract_sync("2 of 3 checks passed.")[0]
        assert claim.kind is ClaimKind.RATIO
        assert claim.value is not None
        assert claim.value.ratio == (Decimal(2), Decimal(3))

    @pytest.mark.parametrize(
        "text",
        [
            "The release is version 2 of 3.",
            "Seats used: 12 of 20.",
            "Version 4 of 12 shipped.",
        ],
    )
    def test_counts_without_a_countable_noun_are_left_alone(self, text: str) -> None:
        """ "2 of 3" alone is a version or a score, not a claim about a set.

        Requiring the noun is what keeps the cherry-picking rule silent on prose
        it cannot reason about, which is where its false positives would live.
        """
        extractor = RuleBasedClaimExtractor()
        assert all(claim.kind is not ClaimKind.RATIO for claim in extractor.extract_sync(text))

    @pytest.mark.parametrize(
        "text",
        ["3 of 2 checks passed.", "0 of 0 checks passed."],
    )
    def test_an_impossible_count_is_not_a_ratio_claim(self, text: str) -> None:
        """A numerator above the denominator, or an empty set, is a parsing
        accident, not a count somebody could have cherry-picked."""
        extractor = RuleBasedClaimExtractor()
        assert all(claim.kind is not ClaimKind.RATIO for claim in extractor.extract_sync(text))


class TestAttributionExtraction:
    """Which source a claim names, and which shapes are not citations."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("According to the audit report, latency fell.", "the audit report"),
            ("The filing states revenue grew.", "The filing"),
            ("The API returned 3 rows.", "The API"),
            ("Based on the compliance summary, two checks failed.", "the compliance summary"),
            ("According to Acme, the plan costs $49.", "Acme"),
        ],
    )
    def test_a_named_source_is_captured(self, text: str, expected: str) -> None:
        assert GroundingLexicon().attribution(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "The plan costs $49 per month.",
            "In 2023 the migration report showed growth.",
            "The user says the export is slow.",
            "The test shows a failure.",
        ],
    )
    def test_prose_that_is_not_a_citation_names_no_source(self, text: str) -> None:
        """ "In 2023" is a date and "the test shows" is a test result.

        Both would otherwise read as citations, and both are the kind of thing a
        perfectly honest response says constantly.
        """
        assert GroundingLexicon().attribution(text) == ""


class TestPluggableExtraction:
    """``ClaimExtractor`` is a real seam, not a promise in a docstring.

    ``S3-T5`` asks for rules first with a classifier behind an interface, so
    swapping in a model-backed extractor has to work end to end without touching
    the evaluator. This is the only test that injects a non-default one.
    """

    class _FixedExtractor:
        """Stands in for a model-backed extractor: one claim, always."""

        def __init__(self, claim: Claim) -> None:
            self._claim = claim
            self.seen: list[str] = []

        async def extract(self, text: str) -> list[Claim]:
            self.seen.append(text)
            return [self._claim]

    async def test_an_injected_extractor_replaces_the_rules(self) -> None:
        store, session_id = await _store("grounded_explicit_price")
        claim = Claim(
            claim_id="c-injected-1",
            text="the annual fee is 400 usd",
            kind=ClaimKind.NUMERIC,
            cue="injected",
            value=Value(
                kind=ValueKind.NUMBER,
                canonical="400 usd",
                number=Decimal(400),
                unit="usd",
            ),
        )
        extractor = self._FixedExtractor(claim)
        evaluator = ProvenanceEvaluator(store, extractor=extractor)

        flags = await evaluator.evaluate_session(session_id)

        # the default rules find nothing wrong here, so a flag can only come
        # from the injected claim
        assert [flag.category for flag in flags] == [CATEGORY_UNGROUNDED]
        assert extractor.seen, "the evaluator must use the injected extractor"

    async def test_the_default_extractor_is_used_when_none_is_injected(self) -> None:
        store, _ = await _store("grounded_explicit_price")
        evaluator = ProvenanceEvaluator(store)
        assert isinstance(evaluator._extractor, RuleBasedClaimExtractor)


class TestReasoningCaptureIsWiredEndToEnd:
    """The ``S3-T5`` capture gap, closed: a real transport, a real store, a real
    evaluation.

    The trace-reading seam is covered above with hand-built payloads, and each
    instrumentor has its own test proving it writes the key. Neither is the claim
    that matters, which is that the two halves agree: the key the transport
    writes is the key the evaluator reads. This test captures a response through
    the real OpenAI-compatible transport, takes the payload it produced, and
    evaluates a session that uses it — so a rename on either side fails here
    rather than silently disabling reasoning capture in production.
    """

    URL = "https://api.openai.com/v1/chat/completions"

    async def _capture_payload(self) -> Mapping[str, object]:
        """A real ``llm.response`` payload, captured through the transport."""
        import httpx
        import respx

        from sentinel import session
        from sentinel.instrument.openai_compat import chat_completion

        reply = {
            "id": "chatcmpl-trace",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The pro plan costs $49 per month.",
                        # A figure the model considered and dropped on the way to
                        # the answer it gave.
                        "reasoning_content": "The upgrade costs $79 per month, I think.",
                    }
                }
            ],
        }
        store = SQLiteEventStore(":memory:")
        try:
            with respx.mock:
                respx.post(self.URL).mock(return_value=httpx.Response(200, json=reply))
                async with session(store) as ctx, httpx.AsyncClient() as client:
                    await chat_completion(
                        client, ctx, url=self.URL, request={"model": "gpt-4o", "messages": []}
                    )
            response = next(
                event
                for event in await store.get_session(ctx.session_id)
                if event.type == LLM_RESPONSE
            )
            return response.payload
        finally:
            await store.close()

    async def test_a_trace_the_transport_wrote_is_the_one_the_evaluator_reads(self) -> None:
        captured = await self._capture_payload()
        assert captured["reasoning"] == ["The upgrade costs $79 per month, I think."]

        # The corpus case supplies the turn's shape — a tool call, its result,
        # and a response citing it. The answer text is the corpus's; the
        # ``reasoning`` key is the transport's, verbatim.
        case = case_by_id("grounded_explicit_price")
        store = SQLiteEventStore(":memory:")
        try:
            for event in case.events():
                if event.type == LLM_RESPONSE:
                    event = event.model_copy(
                        update={"payload": {**event.payload, "reasoning": captured["reasoning"]}}
                    )
                await store.append(event)
            result = await ProvenanceEvaluator(store).analyze_session(case.events()[0].session_id)
        finally:
            await store.close()

        # The answer's own figure is grounded against the cited tool result.
        assert any(
            analysis.diff.verdict in (Verdict.SUPPORTED, Verdict.IMPLIED)
            for analysis in result.analyses
        ), [a.claim.text for a in result.analyses]
        # The figure that only ever appeared in the trace is flagged: it is a
        # claim the model made, and $49-per-month evidence does not support it.
        trace_claims = [a for a in result.analyses if "79" in a.claim.text]
        assert trace_claims, [a.claim.text for a in result.analyses]
        assert trace_claims[0].diff.verdict is Verdict.UNKNOWN
        assert trace_claims[0].is_finding is True
