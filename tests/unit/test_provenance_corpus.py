"""The corpus itself: structure, and the ``S3-T14`` gate.

Two different jobs live here. The first is hygiene — every case has to be a
*valid* session with contiguous ``seq``, resolvable refs, and unique ids, or the
numbers below are measuring a broken fixture rather than the rules. The second is
the gate: run every case through the real analyzer and assert the published
rates, which is the sprint's exit criterion and the number quoted in the docs.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sentinel.eval.fixtures.provenance_corpus import (
    BASE_TS,
    CORPUS,
    CorpusCase,
    case_by_id,
)
from sentinel.eval.harness import (
    MAX_CLAIM_FP_RATE,
    MAX_FALSE_NEGATIVE_RATE,
    MAX_FALSE_POSITIVE_RATE,
    run_case,
    run_corpus,
)
from sentinel.models.events import TOOL_CALL, TOOL_RESULT

pytestmark = [pytest.mark.adversarial, pytest.mark.unit]


class TestCorpusShape:
    def test_the_corpus_is_not_trivially_small(self) -> None:
        assert len(CORPUS) >= 20

    def test_case_ids_are_unique(self) -> None:
        ids = [case.case_id for case in CORPUS]
        assert len(set(ids)) == len(ids)

    def test_every_case_explains_itself(self) -> None:
        for case in CORPUS:
            assert case.notes, f"{case.case_id} has no note saying what it pins down"

    def test_the_corpus_covers_both_directions(self) -> None:
        """A corpus of only known-bad cases cannot measure precision."""
        assert any(case.expects_findings for case in CORPUS)
        assert any(not case.expects_findings for case in CORPUS)
        categories = {expected.category for case in CORPUS for expected in case.expect}
        assert categories == {
            "contradicted_claim",
            "ungrounded_claim",
            "unsourced_citation",
        }

    @pytest.mark.parametrize("case", CORPUS, ids=lambda case: case.case_id)
    def test_a_case_is_a_valid_linked_session(self, case: CorpusCase) -> None:
        events = case.events()
        assert [event.seq for event in events] == list(range(len(events)))
        assert events[0].ts == BASE_TS
        seen: list[str] = []
        for event in events:
            for ref in event.refs:
                assert ref.event_id in seen, f"{case.case_id}: ref to a later/unknown event"
            seen.append(event.event_id)

    @pytest.mark.parametrize("case", CORPUS, ids=lambda case: case.case_id)
    def test_a_case_is_reproducible(self, case: CorpusCase) -> None:
        """Same case in, same events out: a corpus that drifts cannot be quoted."""
        first = [event.model_dump_json() for event in case.events()]
        second = [event.model_dump_json() for event in case.events()]
        assert first == second

    def test_a_case_without_a_tool_has_no_tool_events(self) -> None:
        case = case_by_id("ungrounded_no_tool_output")
        assert not case.tool
        types = {event.type for event in case.events()}
        assert TOOL_CALL not in types
        assert TOOL_RESULT not in types

    def test_a_case_with_a_tool_cites_its_result(self) -> None:
        case = case_by_id("grounded_explicit_price")
        cited = case.cited_events()
        response = next(event for event in cited if event.type == "llm.response")
        assert any(ref.kind.value == "grounds" for ref in response.refs)

    def test_the_uncited_variant_has_no_grounding_ref(self) -> None:
        case = case_by_id("grounded_explicit_price")
        response = next(event for event in case.events() if event.type == "llm.response")
        assert not any(ref.kind.value == "grounds" for ref in response.refs)


class TestGate:
    def test_every_case_behaves_as_written(self) -> None:
        """Per-case output first: this is the failure that names the case."""
        failures = [outcome for outcome in run_corpus().outcomes if not outcome.passed]
        assert not failures, run_corpus().render()

    def test_the_corpus_passes_the_published_gate(self) -> None:
        report = run_corpus()
        assert report.failures == []
        assert report.passed is True

    def test_the_rates_are_inside_the_thresholds(self) -> None:
        report = run_corpus()
        assert report.false_negative_rate <= MAX_FALSE_NEGATIVE_RATE
        assert report.false_positive_rate <= MAX_FALSE_POSITIVE_RATE
        assert report.claim_false_positive_rate <= MAX_CLAIM_FP_RATE

    def test_the_rates_are_stable_across_runs(self) -> None:
        """Determinism is part of the contract (``S3-T4``): the same corpus and
        the same module version must report the same numbers, every time."""
        first = run_corpus().to_dict()
        second = run_corpus().to_dict()
        assert first == second

    def test_the_report_renders_the_gate_verdict(self) -> None:
        report = run_corpus()
        rendered = report.render()
        assert rendered.startswith("corpus:")
        assert rendered.rstrip().endswith("PASS")

    def test_the_report_is_json_serialisable_for_ci(self) -> None:
        payload = json.loads(run_corpus().to_json())
        assert payload["cases"] == len(CORPUS)
        assert payload["passed"] is True
        assert payload["gates"]["max_false_positive_rate"] == MAX_FALSE_POSITIVE_RATE

    def test_a_missed_detection_is_reported_as_missing(self) -> None:
        """A case expecting a flag the rules cannot produce still has to be
        measured, not skipped — that is what keeps the FN rate honest."""
        case = case_by_id("contradicted_price")
        broken = CorpusCase(
            case_id="broken",
            prompt=case.prompt,
            tool=case.tool,
            tool_input=case.tool_input,
            tool_output="The pro plan is free.",
            response=case.response,
            expect=case.expect,
        )
        outcome = run_case(broken)
        assert outcome.missing, outcome

    def test_uncited_evidence_still_grounds_a_claim(self) -> None:
        """The claim was right either way; only the citation differed."""
        case = case_by_id("grounded_explicit_price")
        assert run_case(case, cited=True).passed
        assert run_case(case, cited=False).passed

    def test_the_citation_default_comes_from_the_case(self) -> None:
        """``cited=None`` has to mean "whatever this case is", not "cited"."""
        cited_case = case_by_id("grounded_explicit_price")
        uncited_case = replace(cited_case, case_id="uncited", cited=False)

        assert cited_case.cited
        assert run_case(cited_case).passed
        assert run_case(uncited_case).passed
        # And the explicit override still wins over the case setting.
        assert run_case(uncited_case, cited=True).passed
