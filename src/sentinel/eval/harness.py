"""The adversarial-corpus harness and the FP/FN gate (sprint ``S3-T13``/``S3-T14``).

The point of the corpus is a *number you can quote*: run every case in
:mod:`sentinel.eval.fixtures.provenance_corpus` through the real analyzer and
count what it missed and what it invented. The gate is the exit criterion from
the sprint plan, so the thresholds live here as named constants that the docs
and the test suite both read — there is exactly one place where "acceptable"
is defined.

Metrics:

* **False negative** — a case expecting a flag where the analyzer found none, or
  found a flag of the wrong category. Detection failures.
* **False positive** — a case expecting no flag where one was produced, *or* a
  flagged claim that does not match the case's expected claim. Over-flagging is
  how a detector gets muted, so it is measured on both the case and the claim.

Determinism is part of the contract: :func:`run_corpus` is pure with respect to
the corpus, so the same module version always reports the same rates.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

from sentinel.eval.fixtures.provenance_corpus import CORPUS, CorpusCase, ExpectedFinding
from sentinel.eval.provenance import MODULE, MODULE_VERSION
from sentinel.eval.provenance_core import (
    DiffContext,
    RuleBasedClaimExtractor,
    Verdict,
    diff_claim,
    is_actionable,
    severity_for,
)
from sentinel.eval.session import SessionView, response_text_of
from sentinel.models.events import Event
from sentinel.query import build_call_graph

log = structlog.get_logger("sentinel.eval.harness")

#: Corpus-level gates (``S3-T14``). Recall first, precision as a hard ceiling:
#: a module that cries wolf is worse than one that misses a rare pattern, and
#: the review queue exists to catch what a rule-based module cannot.
MAX_FALSE_NEGATIVE_RATE = 0.10
MAX_FALSE_POSITIVE_RATE = 0.0

#: Claim-level gates, measured over the claims the analyzer produced.
MAX_CLAIM_FP_RATE = 0.05


@dataclass(frozen=True)
class CaseOutcome:
    """What the analyzer did with one corpus case.

    ``missing`` holds the expectations the analyzer failed to produce; ``spurious``
    holds the claims it invented. A case with expectations but no ``spurious``
    and no ``missing`` is a clean pass, which is why ``exact_count`` in the
    corpus is enforced by matching rather than by counting.
    """

    case_id: str
    expected: int
    produced: int
    missing: tuple[str, ...] = ()
    spurious: tuple[str, ...] = ()

    @property
    def is_false_negative(self) -> bool:
        """A detection the case expected and the analyzer missed."""
        return bool(self.missing)

    @property
    def is_false_positive(self) -> bool:
        """A finding the case did not expect, at case or claim level."""
        return bool(self.spurious)

    @property
    def passed(self) -> bool:
        """Whether the case behaved exactly as the corpus specifies."""
        return not self.missing and not self.spurious


@dataclass
class CorpusReport:
    """The measured result of running the corpus (``S3-T13``)."""

    outcomes: list[CaseOutcome] = field(default_factory=list)
    module: str = ""
    module_version: str = ""

    @property
    def total_cases(self) -> int:
        """How many cases ran."""
        return len(self.outcomes)

    @property
    def false_negatives(self) -> list[CaseOutcome]:
        """Cases whose expected flag was missed."""
        return [outcome for outcome in self.outcomes if outcome.is_false_negative]

    @property
    def false_positives(self) -> list[CaseOutcome]:
        """Cases flagged where the corpus says nothing was wrong."""
        return [outcome for outcome in self.outcomes if outcome.is_false_positive]

    @property
    def false_negative_rate(self) -> float:
        """Missed detections over cases that expected a flag."""
        expected = sum(1 for outcome in self.outcomes if outcome.expected)
        return round(len(self.false_negatives) / expected, 4) if expected else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Over-flagged cases over cases that expected silence."""
        clean = sum(1 for outcome in self.outcomes if not outcome.expected)
        return round(len(self.false_positives) / clean, 4) if clean else 0.0

    @property
    def claim_false_positive_rate(self) -> float:
        """Invented claims over every claim the analyzer flagged."""
        produced = sum(outcome.produced for outcome in self.outcomes)
        invented = sum(len(outcome.spurious) for outcome in self.outcomes)
        return round(invented / produced, 4) if produced else 0.0

    @property
    def confusion(self) -> dict[str, int]:
        """The 2x2 matrix, counted by case: expected a flag, or expected silence."""
        flagged = [outcome for outcome in self.outcomes if outcome.produced]
        return {
            "true_positives": sum(1 for outcome in flagged if outcome.expected),
            "false_negatives": len(self.false_negatives),
            "false_positives": len(self.false_positives),
            "true_negatives": sum(
                1 for outcome in self.outcomes if not outcome.expected and not outcome.produced
            ),
        }

    @property
    def precision(self) -> float:
        """Flagged cases the corpus agreed should be flagged."""
        cells = self.confusion
        denominator = cells["true_positives"] + cells["false_positives"]
        return round(cells["true_positives"] / denominator, 4) if denominator else 1.0

    @property
    def recall(self) -> float:
        """Expected detections the module actually made."""
        cells = self.confusion
        denominator = cells["true_positives"] + cells["false_negatives"]
        return round(cells["true_positives"] / denominator, 4) if denominator else 1.0

    @property
    def passed(self) -> bool:
        """Whether the run satisfies the ``S3-T14`` gates."""
        return (
            self.false_negative_rate <= MAX_FALSE_NEGATIVE_RATE
            and self.false_positive_rate <= MAX_FALSE_POSITIVE_RATE
            and self.claim_false_positive_rate <= MAX_CLAIM_FP_RATE
        )

    @property
    def failures(self) -> list[str]:
        """Human-readable gate failures, empty when :attr:`passed`."""
        problems: list[str] = []
        if self.false_negative_rate > MAX_FALSE_NEGATIVE_RATE:
            problems.append(
                f"false-negative rate {self.false_negative_rate} > "
                f"{MAX_FALSE_NEGATIVE_RATE}: "
                f"{[outcome.case_id for outcome in self.false_negatives]}"
            )
        if self.false_positive_rate > MAX_FALSE_POSITIVE_RATE:
            problems.append(
                f"false-positive rate {self.false_positive_rate} > "
                f"{MAX_FALSE_POSITIVE_RATE}: "
                f"{[outcome.case_id for outcome in self.false_positives]}"
            )
        if self.claim_false_positive_rate > MAX_CLAIM_FP_RATE:
            problems.append(
                f"claim false-positive rate {self.claim_false_positive_rate} > {MAX_CLAIM_FP_RATE}"
            )
        return problems

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary, for CI output and the docs."""
        return {
            "module": self.module,
            "module_version": self.module_version,
            "cases": self.total_cases,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "claim_false_positive_rate": self.claim_false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "confusion": self.confusion,
            "passed": self.passed,
            "gates": {
                "max_false_negative_rate": MAX_FALSE_NEGATIVE_RATE,
                "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
                "max_claim_false_positive_rate": MAX_CLAIM_FP_RATE,
            },
            "failures": self.failures,
            "cases_detail": [
                {
                    "case_id": outcome.case_id,
                    "expected": outcome.expected,
                    "produced": outcome.produced,
                    "missing": list(outcome.missing),
                    "spurious": list(outcome.spurious),
                    "passed": outcome.passed,
                }
                for outcome in self.outcomes
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """The report as JSON, for ``sentinel eval-fixtures --json``."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def render(self) -> str:
        """An operator-facing confusion matrix plus the per-case failures."""
        cells = self.confusion
        lines = [
            f"corpus: {self.total_cases} cases  module={self.module}@{self.module_version}",
            "confusion matrix (case level)",
            f"  {'':<14}{'flagged':>6}{'silent':>8}",
            f"  {'should flag':<14}{cells['true_positives']:>6}{cells['false_negatives']:>8}"
            f"   <- recall {self.recall:.2%}",
            f"  {'should be quiet':<14}{cells['false_positives']:>6}"
            f"{cells['true_negatives']:>8}   <- precision {self.precision:.2%}",
            "",
            f"  false-negative rate: {self.false_negative_rate:.2%} "
            f"(gate <= {MAX_FALSE_NEGATIVE_RATE:.0%})",
            f"  false-positive rate: {self.false_positive_rate:.2%} "
            f"(gate <= {MAX_FALSE_POSITIVE_RATE:.0%})",
            f"  claim FP rate:       {self.claim_false_positive_rate:.2%} "
            f"(gate <= {MAX_CLAIM_FP_RATE:.0%})",
        ]
        for outcome in self.outcomes:
            if outcome.passed:
                continue
            lines.append(f"  FAIL {outcome.case_id}")
            lines.extend(f"    missing: {missing}" for missing in outcome.missing)
            lines.extend(f"    spurious: {spurious}" for spurious in outcome.spurious)
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def run_case(case: CorpusCase, *, cited: bool | None = None) -> CaseOutcome:
    """Run one corpus case through the rules and compare to its expectation.

    ``cited`` controls whether the response cites its tool result, which is how
    the corpus distinguishes "the agent looked and still got it wrong" from "the
    agent never looked". It defaults to the case's own ``cited`` setting.
    """
    extractor = RuleBasedClaimExtractor()
    use_cited = case.cited if cited is None else cited
    events = case.cited_events() if use_cited else case.events()
    view = _view_for(events)
    produced: list[ExpectedFinding] = []

    for response in view.llm_responses():
        context = _context_for(view, response)
        for claim in extractor.extract_sync(response_text_of(response)):
            diff = diff_claim(claim, context)
            if not is_actionable(claim, diff):
                continue
            produced.append(
                ExpectedFinding(
                    category=(
                        "contradicted_claim"
                        if diff.verdict is Verdict.CONFLICTED
                        else "ungrounded_claim"
                    ),
                    claim_contains=claim.text,
                    verdict=str(diff.verdict),
                    severity=str(severity_for(claim, diff)),
                )
            )

    missing: list[str] = []
    unmatched = list(range(len(produced)))
    for expected in case.expect:
        index = _find_match(expected, produced, unmatched)
        if index is None:
            missing.append(f"{expected.category} containing {expected.claim_contains!r}")
        else:
            unmatched.remove(index)
    return CaseOutcome(
        case_id=case.case_id,
        expected=len(case.expect),
        produced=len(produced),
        missing=tuple(missing),
        # Whatever is left over was invented: a claim the corpus never licensed.
        spurious=tuple(produced[index].claim_contains for index in unmatched),
    )


def run_corpus(
    cases: Sequence[CorpusCase] | None = None,
    *,
    module: str = "sentinel.tool_grounding",
    module_version: str = "0.1.0",
) -> CorpusReport:
    """Run the whole corpus and measure the gate metrics (``S3-T13``)."""
    report = CorpusReport(module=module, module_version=module_version)
    for case in cases if cases is not None else CORPUS:
        report.outcomes.append(run_case(case))
    if not report.passed:
        log.warning(
            "corpus.gate_failed",
            module=module,
            module_version=module_version,
            failures=report.failures,
        )
    return report


#: Modules ``sentinel eval-fixtures --module`` can run: short name -> module id.
CORPUS_MODULES: dict[str, str] = {"provenance": MODULE}


def run_module_corpus(module: str) -> CorpusReport:
    """Run the corpus behind a module short name (``provenance``).

    The module identity comes from the module itself rather than a literal
    here, so a report can never claim to have tested a version the module did
    not ship.
    """
    if module not in CORPUS_MODULES:
        raise KeyError(module)
    return run_corpus(module=MODULE, module_version=MODULE_VERSION)


def _view_for(events: Sequence[Event]) -> SessionView:
    """Build a :class:`SessionView` from a materialised event list."""
    session_id = events[0].session_id
    return SessionView(
        session_id=session_id,
        events=tuple(events),
        graph=build_call_graph(session_id, events),
    )


def _context_for(view: SessionView, response: Event) -> DiffContext:
    """The evidence available to *response*, as the worker would gather it."""
    from sentinel.eval.provenance import gather_evidence

    context, _refs = gather_evidence(view.graph, response)
    return context


def _find_match(
    expected: ExpectedFinding,
    candidates: Sequence[ExpectedFinding],
    available: Sequence[int],
) -> int | None:
    """The first still-unmatched candidate satisfying *expected*."""
    for index in available:
        if _matches(expected, candidates[index]):
            return index
    return None


def _matches(expected: ExpectedFinding, candidate: ExpectedFinding) -> bool:
    """Whether a produced finding satisfies a hand-written expectation."""
    if expected.category != candidate.category:
        return False
    if expected.claim_contains.lower() not in candidate.claim_contains.lower():
        return False
    if expected.verdict and expected.verdict != candidate.verdict:
        return False
    return not expected.severity or expected.severity == candidate.severity


__all__ = [
    "MAX_CLAIM_FP_RATE",
    "MAX_FALSE_NEGATIVE_RATE",
    "MAX_FALSE_POSITIVE_RATE",
    "CaseOutcome",
    "CorpusReport",
    "run_case",
    "run_corpus",
]
