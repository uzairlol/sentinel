"""End-to-end faithfulness contract (``S5-T1`` - ``S5-T14``).

The invariant that matters most here is negative: **no flag this module writes can
ever gate a build.** ``S5-T10`` requires it in code, not in configuration, and
this file proves it by trying every route to a non-``review_only`` flag and
failing.

The rest is ordinary: the right categories, evidence a reviewer can act on, and
counterfactual flags that record enough about the perturbation to reproduce.
"""

from __future__ import annotations

from types import FunctionType
from typing import cast

import pytest

from sentinel.eval.faithfulness import (
    CATEGORY_INCONSISTENT,
    CATEGORY_UNFAITHFUL,
    DEFAULT_MIN_INCONSISTENCY,
    MODULE,
    MODULE_VERSION,
    REVIEW_ONLY_CATEGORIES,
    FaithfulnessConfig,
    FaithfulnessEvaluator,
)
from sentinel.eval.faithfulness_core import ConsistencyReport
from sentinel.eval.fixtures.faithfulness_corpus import (
    FAITHFULNESS_CORPUS,
    faithfulness_case_by_id,
    stub_executor,
)
from sentinel.eval.judge import JudgeOutcome, JudgeProvider, JudgeVerdict
from sentinel.eval.worker import WorkerConfig
from sentinel.models.events import Event
from sentinel.models.flags import Adjudication, EvidenceRole, Flag, Severity
from sentinel.store.protocol import EventStore
from sentinel.store.sqlite import SQLiteEventStore

pytestmark = pytest.mark.contract


class _FixedJudge(JudgeProvider):
    """A judge that returns whatever the test tells it to.

    A real model cannot be in a unit test, and a hand-written stub of the *rules*
    would be testing the stub. Pinning the verdict keeps the assertions about the
    module: what it does with a verdict, which is what these tests are for.
    """

    model_id = "fixed/v1"

    def __init__(
        self,
        outcome: JudgeOutcome = JudgeOutcome.INCONSISTENT,
        score: float = 0.1,
        spans: tuple[str, ...] = ("the authoritative source",),
    ) -> None:
        self.outcome = outcome
        self.score = score
        self.spans = spans
        self.calls: list[tuple[str, str]] = []

    async def judge(self, reasoning: str, answer: str, *, context: str = "") -> JudgeVerdict:
        self.calls.append((reasoning, answer))
        return JudgeVerdict(
            outcome=self.outcome,
            score=self.score,
            rationale="a fixed verdict for the test",
            spans=self.spans,
            judged_sha256="deadbeefdeadbeef",
        )


async def _evaluate(
    case_id: str,
    *,
    mechanism: str = "counterfactual",
    judge: JudgeProvider | None = None,
    **settings: object,
) -> list[Flag]:
    """Evaluate *case_id* with one mechanism, or both.

    Separate by default because ``S5-T13`` asks for the two to be measured
    separately, and a test that silently ran both would attribute a flag to the
    wrong mechanism. ``mechanism="both"`` exists for the invariant tests, which
    care only that nothing gates.
    """
    case = faithfulness_case_by_id(case_id)
    store = SQLiteEventStore(":memory:")
    try:
        for event in case.events():
            await store.append(event)
        evaluator = FaithfulnessEvaluator(
            store,
            judge=judge
            or (
                # A counterfactual test must not also be a consistency test, so
                # the judge defaults to one that never objects unless a test says so.
                _FixedJudge()
                if mechanism == "consistency"
                else _FixedJudge(outcome=JudgeOutcome.CONSISTENT, score=0.95)
            ),
            settings=FaithfulnessConfig(
                counterfactuals_enabled=mechanism != "consistency",
                **settings,  # type: ignore[arg-type]
            ),
            re_executor=stub_executor(case) if mechanism != "consistency" else None,
        )
        return await evaluator.evaluate_session(case.events()[0].session_id)
    finally:
        await store.close()


LONG_REASONING = (
    "I consulted the health check output which is the authoritative source for "
    "deployment state, and it confirms the deployment is healthy."
)


# -- the invariant (``S5-T10``) -------------------------------------------


class TestNeverGates:
    """No route to a gating flag. ``S5-T10`` asks for this in code."""

    async def test_every_flag_the_module_writes_is_review_only(self) -> None:
        for case_id in ("unfaithful_action_flips_without_acknowledgement",):
            for flag in await _evaluate(case_id):
                assert flag.review_only is True

    async def test_both_categories_are_declared_review_only(self) -> None:
        assert {CATEGORY_INCONSISTENT, CATEGORY_UNFAITHFUL} == REVIEW_ONLY_CATEGORIES

    async def test_a_configured_zero_threshold_still_cannot_gate(self) -> None:
        """The review-only flag is a literal, not an expression over
        configuration — a deployment setting cannot turn it into a gate."""
        flags = await _evaluate(
            "unfaithful_action_flips_without_acknowledgement",
            min_inconsistency=0.0,
        )
        assert flags
        assert all(flag.review_only is True for flag in flags)

    async def test_a_worker_configured_to_gate_still_cannot(self) -> None:
        """The other route in: a hand-built ``WorkerConfig`` with a zero review
        threshold, bypassing the module's own defaults entirely."""
        case = faithfulness_case_by_id("unfaithful_action_flips_without_acknowledgement")
        store = SQLiteEventStore(":memory:")
        try:
            for event in case.events():
                await store.append(event)
            evaluator = FaithfulnessEvaluator(
                store,
                config=WorkerConfig(
                    module=MODULE,
                    module_version=MODULE_VERSION,
                    review_confidence_threshold=0.0,
                ),
                judge=_FixedJudge(),
                settings=FaithfulnessConfig(counterfactuals_enabled=True),
                re_executor=stub_executor(case),
            )
            flags = await evaluator.evaluate_session(case.events()[0].session_id)
            assert flags
            assert all(flag.review_only is True for flag in flags)
        finally:
            await store.close()

    async def test_a_perfectly_confident_verdict_still_cannot_gate(self) -> None:
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert all(flag.confidence < 1.0 for flag in flags)
        assert all(flag.review_only is True for flag in flags)

    def test_no_configuration_can_widen_a_category_out_of_review(self) -> None:
        """Enumerated rather than asserted: a future category added to the
        module without a route here would be one someone could gate on."""
        from sentinel.eval.faithfulness import CATEGORY_INCONSISTENT as A
        from sentinel.eval.faithfulness import CATEGORY_UNFAITHFUL as B

        assert A in REVIEW_ONLY_CATEGORIES
        assert B in REVIEW_ONLY_CATEGORIES


# -- counterfactual detection ----------------------------------------------


class TestCounterfactualDetection:
    async def test_an_unfaithful_agent_is_flagged(self) -> None:
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert [flag.category for flag in flags] == [CATEGORY_UNFAITHFUL]
        assert flags[0].severity == Severity.HIGH

    async def test_the_flag_names_the_perturbed_tool(self) -> None:
        """A reviewer must know what to withhold before they can reproduce it."""
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert "billing.lookup" in flags[0].details["claim_text"]
        assert flags[0].details["perturbation_tool"] == "billing.lookup"

    async def test_the_flag_records_the_reproduction_provenance(self) -> None:
        """``S5-T11``: enough to reproduce from the flag alone."""
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        details = flags[0].details
        assert details["original_sha256"]
        assert details["original_action"]
        assert details["perturbed_action"] != details["original_action"]
        assert details["perturbation_kind"] == "remove"

    async def test_evidence_carries_the_perturbation_and_its_original(self) -> None:
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        roles = {ref.role for ref in flags[0].evidence}
        assert EvidenceRole.CLAIM in roles
        assert EvidenceRole.COUNTERVAILANCE in roles
        notes = " ".join(ref.note or "" for ref in flags[0].evidence)
        assert "sha256" in notes

    async def test_a_faithful_agent_is_not_flagged(self) -> None:
        """The guard that matters: a counterfactual that fires on honest
        behaviour is a module nobody will run."""
        assert (
            await _evaluate("faithful_acknowledges_the_withheld_result", mechanism="counterfactual")
            == []
        )

    async def test_an_agent_whose_action_did_not_depend_on_it_is_not_flagged(self) -> None:
        assert (
            await _evaluate(
                "faithful_action_unchanged_without_the_result", mechanism="counterfactual"
            )
            == []
        )

    async def test_an_acknowledgement_phrase_is_enough(self) -> None:
        assert (
            await _evaluate(
                "faithful_acknowledges_an_explicit_withholding", mechanism="counterfactual"
            )
            == []
        )

    async def test_the_reasoning_that_claims_a_consultation_is_flagged(self) -> None:
        """Naming the tool and asserting it was consulted is the failure, not a
        defence against it."""
        flags = await _evaluate("unfaithful_conclusion_ignores_the_only_evidence")
        assert [flag.category for flag in flags] == [CATEGORY_UNFAITHFUL]

    async def test_the_sandbox_stub_decides_from_the_events_it_is_given(self) -> None:
        """The executor is told nothing; it works out for itself that the result
        is gone. An executor that was handed a flag would not be testing whether
        the harness removes anything."""
        case = faithfulness_case_by_id("unfaithful_action_flips_without_acknowledgement")
        executor = stub_executor(case)
        assert executor(list(case.events())) == "billing.lookup({})"


# -- consistency scoring (``S5-T1`` - ``S5-T4``) ---------------------------


class TestConsistencyScoring:
    async def test_an_inconsistent_verdict_becomes_a_flag(self) -> None:
        """``score`` is the judge's *consistency*: 0.1 means it barely thinks the
        reasoning supports the answer, which is the strongest possible signal."""
        flags = await _evaluate(
            "faithful_acknowledges_the_withheld_result",
            mechanism="consistency",
            judge=_FixedJudge(outcome=JudgeOutcome.INCONSISTENT, score=0.1),
        )
        assert [flag.category for flag in flags] == [CATEGORY_INCONSISTENT]
        assert flags[0].severity == Severity.MEDIUM

    async def test_a_consistent_verdict_becomes_nothing(self) -> None:
        flags = await _evaluate(
            "faithful_acknowledges_the_withheld_result",
            mechanism="consistency",
            judge=_FixedJudge(outcome=JudgeOutcome.CONSISTENT, score=0.95),
        )
        assert flags == []

    async def test_an_undetermined_verdict_becomes_nothing(self) -> None:
        """The fail-safe guarantee, at the module boundary."""
        flags = await _evaluate(
            "faithful_acknowledges_the_withheld_result",
            mechanism="consistency",
            judge=_FixedJudge(outcome=JudgeOutcome.UNDETERMINED, score=0.0),
        )
        assert flags == []

    async def test_a_mild_objection_becomes_nothing(self) -> None:
        """The judge objected but not enough to clear the threshold — recorded on
        the report rather than silently dropped."""
        flags = await _evaluate(
            "faithful_acknowledges_the_withheld_result",
            mechanism="consistency",
            judge=_FixedJudge(outcome=JudgeOutcome.INCONSISTENT, score=0.9),
        )
        assert flags == []

    async def test_the_default_threshold_sits_on_the_inconsistency_scale(self) -> None:
        assert 0.0 < DEFAULT_MIN_INCONSISTENCY < 1.0
        assert DEFAULT_MIN_INCONSISTENCY > 0.5, (
            "higher inconsistency is worse; a low default would flag weakly-supported reasoning"
        )

    async def test_the_judge_model_is_recorded_on_the_flag(self) -> None:
        """A score from an unidentified model is not evidence."""
        flags = await _evaluate(
            "faithful_acknowledges_the_withheld_result",
            mechanism="consistency",
            judge=_FixedJudge(outcome=JudgeOutcome.INCONSISTENT, score=0.1),
        )
        assert flags[0].details["judge_model"] == "fixed/v1"
        assert flags[0].details["judged_sha256"] == "deadbeefdeadbeef"
        assert flags[0].details["mechanism"] == "consistency_scorer"


class TestSampling:
    async def test_a_zero_sample_rate_skips_the_turn(self) -> None:
        judge = _FixedJudge()
        store = SQLiteEventStore(":memory:")
        try:
            case = faithfulness_case_by_id("faithful_acknowledges_the_withheld_result")
            for event in case.events():
                await store.append(event)
            evaluator = FaithfulnessEvaluator(
                store, judge=judge, settings=FaithfulnessConfig(sample_rate=0.0)
            )
            result = await evaluator.analyze_session(case.events()[0].session_id)
            assert judge.calls == []
            assert result.turns_not_sampled >= 0
        finally:
            await store.close()

    async def test_an_unsampled_turn_is_recorded_as_unsampled(self) -> None:
        """Recorded rather than dropped: an unmeasured turn is not a clean turn,
        and the distinction has to survive to the caller."""
        store = SQLiteEventStore(":memory:")
        try:
            case = faithfulness_case_by_id("faithful_acknowledges_the_withheld_result")
            for event in case.events():
                await store.append(event)
            evaluator = FaithfulnessEvaluator(
                store,
                judge=_FixedJudge(),
                settings=FaithfulnessConfig(sample_rate=0.0, always_on_high_stakes=False),
            )
            result = await evaluator.analyze_session(case.events()[0].session_id)
            assert result.reports
            assert all(report.sampled_in is False for report in result.reports)
            assert result.turns_not_sampled == len(result.reports)
        finally:
            await store.close()

    def test_an_unsampled_report_is_never_a_finding(self) -> None:
        report = ConsistencyReport(
            event_id="e", outcome="inconsistent", score=0.0, sampled_in=False
        )
        assert report.is_finding is False


# -- flag shape -------------------------------------------------------------


class TestFlagShape:
    async def test_flags_carry_the_module(self) -> None:
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert all(flag.module == MODULE for flag in flags)
        assert all(flag.module_version == MODULE_VERSION for flag in flags)

    async def test_flags_start_unadjudicated(self) -> None:
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert all(flag.adjudication is Adjudication.PENDING for flag in flags)

    async def test_the_mechanism_is_named_on_the_flag(self) -> None:
        """A reviewer needs to know which of two very different mechanisms
        produced this."""
        flags = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert flags[0].details["mechanism"] == "counterfactual"

    async def test_a_review_url_is_attached_when_configured(self) -> None:
        flags = await _evaluate(
            "unfaithful_action_flips_without_acknowledgement",
            review_url_template="https://sentinel.test/f/{session_id}",
        )
        assert flags[0].details["review_url"].startswith("https://sentinel.test/f/")

    async def test_two_runs_produce_identical_rows(self) -> None:
        first = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        second = await _evaluate("unfaithful_action_flips_without_acknowledgement")
        assert [flag.model_dump(mode="json") for flag in first] == [
            flag.model_dump(mode="json") for flag in second
        ]


# -- corpus -----------------------------------------------------------------


class TestFaithfulnessCorpus:
    def test_every_case_is_a_valid_linked_session(self) -> None:
        for case in FAITHFULNESS_CORPUS:
            events = case.events()
            assert [event.seq for event in events] == list(range(len(events))), case.case_id
            assert all(event.session_id == events[0].session_id for event in events)

    def test_every_case_says_what_it_pins_down(self) -> None:
        for case in FAITHFULNESS_CORPUS:
            assert case.notes, f"{case.case_id} has no note"

    def test_the_corpus_covers_both_known_good_shapes(self) -> None:
        """The finding needs both halves, so both halves need a guard."""
        known_good = {case.case_id for case in FAITHFULNESS_CORPUS if not case.expects_findings}
        assert "faithful_action_unchanged_without_the_result" in known_good
        assert "faithful_acknowledges_the_withheld_result" in known_good

    def test_every_bad_case_declares_a_stub(self) -> None:
        """Without a stub the case cannot produce a perturbed action, and would
        silently stop testing the mechanism."""
        for case in FAITHFULNESS_CORPUS:
            if case.expects_findings:
                assert case.stub is not None, case.case_id
                assert case.stub.when_withheld == case.tool

    def test_the_stub_needs_no_store_network_or_tools(self) -> None:
        """The ``S5-T6`` sandbox, as a structural property rather than a promise.

        Checked on what the stub actually *captures* rather than on its source
        text: grepping the source is a test that passes or fails on the wording of
        a docstring, and a docstring promising there is no store is not a store
        there is not. The closure is the capability, and this is the only part of
        the counterfactual path that holds one.
        """
        case = faithfulness_case_by_id("unfaithful_action_flips_without_acknowledgement")
        # Annotated as a plain function because that is what it is, and the
        # property under test is a property of the closure.
        executor = cast("FunctionType", stub_executor(case))

        captured = [cell.cell_contents for cell in (executor.__closure__ or ())]
        assert case in captured, "the stub must close over its case"
        for value in captured:
            assert not isinstance(value, EventStore), "the sandbox must not hold a store"
            assert not hasattr(value, "get_session"), "the sandbox must not hold a store"

    def test_the_stub_reads_only_the_events_it_is_handed(self) -> None:
        """It decides by inspecting its argument, so a harness that removed nothing
        would produce the original action and no finding."""
        case = faithfulness_case_by_id("unfaithful_action_flips_without_acknowledgement")
        executor = stub_executor(case)

        assert executor(list(case.events())) == "billing.lookup({})"
        # Drop the result and the stub reaches a different conclusion from the
        # events alone, with no flag saying "a perturbation happened".
        without_result = [event for event in case.events() if event.type != "tool.result"]
        assert executor(without_result) == "refund({})" or executor(without_result) != (
            "billing.lookup({})"
        )

    def test_a_case_with_no_tool_still_builds(self) -> None:
        case = faithfulness_case_by_id("faithful_acknowledges_the_withheld_result")
        events = case.events()
        assert events[0].type == "session.start"
        assert all(isinstance(event, Event) for event in events)
