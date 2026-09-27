"""The adversarial corpus for tool-use grounding (sprint ``S3-T12``).

Each case is a *self-contained* session: a tool call, the result it returned, and
the prose the agent produced afterwards. The expectation is written by hand
against what the tool actually returned — not by running the analyzer — so a
rule change that quietly breaks detection shows up as a test failure rather than
as a nicer number.

Cases come in three shapes, and the mix is deliberate:

* ``grounded`` — the agent said something the tool output supports, including
  the awkward cases the rules have to get right: unit conversion, rounding, a
  value inside a stated bound, a subset of the offered set. These are the
  false-positive bait.
* ``ungrounded`` — a specific claim (a number, a date, a membership) that no
  available tool output mentions. These are true positives.
* ``contradicted`` — the tool output states something incompatible. These must
  be the severe half of the corpus, because they are the findings an operator
  acts on.

A fourth shape, ``ignore``, holds the prose that must *not* be flagged: questions,
pleasantries, ungroundable vagueness, and correct claims whose evidence is not
in the same turn. The FP rate is measured over the union of ``grounded`` and
``ignore`` — the cases where a flag would be wrong.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sentinel.models.events import (
    LLM_REQUEST,
    LLM_RESPONSE,
    SESSION_END,
    SESSION_START,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
)

#: Fixed base time so the corpus is reproducible byte for byte.
BASE_TS = datetime(2024, 5, 1, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ExpectedFinding:
    """What the analyzer should report for one case.

    ``category`` is the flag category, ``claim_contains`` a substring of the
    claim text, and ``verdict``/``severity`` optional tightenings. A case that
    expects findings asserts *at least* these — any finding on the case beyond
    the ones listed is a defect too, because the harness counts every leftover
    claim as spurious. That is deliberate: an expectation list that quietly
    tolerates extra flags is how a corpus stops measuring precision.
    """

    category: str
    claim_contains: str
    verdict: str = ""
    severity: str = ""


@dataclass(frozen=True)
class CorpusCase:
    """One adversarial session plus its hand-written expectation."""

    case_id: str
    prompt: str
    tool: str
    tool_input: dict[str, object]
    tool_output: str
    response: str
    expect: list[ExpectedFinding] = field(default_factory=list)
    notes: str = ""
    cited: bool = True

    @property
    def expects_findings(self) -> bool:
        """Whether this case should produce at least one flag."""
        return bool(self.expect)

    def events(self) -> list[Event]:
        """Materialise the case as a session's event log.

        Shape mirrors a real agent turn: session, request, tool call, tool
        result, then the response that consumed it. A case with no ``tool``
        emits no call and no result, which is the "the model answered from its
        weights" state the rules have to survive.

        The refs are the ones a real instrumentor emits (``caused_by`` for
        causation, ``parent`` for the enclosing turn), so the evidence the
        analyzer gathers here is the evidence it would gather from a live log.
        """
        session_id = _case_session_id(self.case_id)
        start = BASE_TS
        request_id = _ulid(self.case_id, "request")
        call_id = _ulid(self.case_id, "call")
        result_id = _ulid(self.case_id, "result")
        raw: list[dict[str, object]] = [
            {
                "event_id": _ulid(self.case_id, "start"),
                "payload": {"agent_id": "corpus", "framework": "corpus", "case": self.case_id},
            },
            {
                "event_id": request_id,
                "type": LLM_REQUEST,
                "payload": {"provider": "corpus", "model": "fixture"},
            },
        ]
        types = [SESSION_START, LLM_REQUEST]
        if self.tool:
            raw.append(
                {
                    "event_id": call_id,
                    "type": TOOL_CALL,
                    "payload": {"tool": self.tool, "input": str(self.tool_input)},
                    "refs": [{"event_id": request_id, "kind": RefKind.CAUSED_BY.value}],
                }
            )
            raw.append(
                {
                    "event_id": result_id,
                    "type": TOOL_RESULT,
                    "payload": {"tool": self.tool, "output": self.tool_output},
                    "refs": [{"event_id": call_id, "kind": RefKind.CAUSED_BY.value}],
                }
            )
            types.extend([TOOL_CALL, TOOL_RESULT])
        response_refs: list[dict[str, str]] = [
            {"event_id": request_id, "kind": RefKind.CAUSED_BY.value}
        ]
        if self.tool:
            response_refs.append({"event_id": call_id, "kind": RefKind.PARENT.value})
        raw.append(
            {
                "event_id": _ulid(self.case_id, "response"),
                "type": LLM_RESPONSE,
                "payload": {"provider": "corpus", "generations": [self.response]},
                "refs": response_refs,
            }
        )
        types.append(LLM_RESPONSE)
        raw.append(
            {
                "event_id": _ulid(self.case_id, "end"),
                "type": SESSION_END,
                "payload": {"agent_id": "corpus", "reason": "completed"},
            }
        )
        types.append(SESSION_END)

        events: list[Event] = []
        for seq, (spec, event_type) in enumerate(zip(raw, types, strict=True)):
            ts = start + timedelta(milliseconds=seq * 10)
            events.append(
                Event.model_validate(
                    {
                        **spec,
                        "session_id": session_id,
                        "seq": seq,
                        "ts": ts.isoformat(),
                        "type": event_type,
                    }
                )
            )
        return events

    def cited_events(self) -> list[Event]:
        """The case's events with the response citing the tool result.

        Citation is the default for every case, including the ungrounded ones:
        a model that *had* the source and still asserted something else is the
        more interesting failure, and it is the one the corpus is measuring.
        Cases built with ``cited=False`` exercise the uncited path instead.
        """
        if not self.cited or not self.tool:
            return self.events()
        events = self.events()
        result = next(event for event in events if event.type == TOOL_RESULT)
        response = next(event for event in events if event.type == LLM_RESPONSE)
        cited = Event.model_validate(
            {
                **response.model_dump(mode="json"),
                "refs": [
                    *response.model_dump(mode="json")["refs"],
                    {"event_id": result.event_id, "kind": RefKind.GROUNDS.value},
                ],
            }
        )
        return [cited if event.event_id == response.event_id else event for event in events]


def _stable_ulid(label: str) -> str:
    """A valid, deterministic ULID for *label* (ADR-0010/0012 determinism).

    The corpus must materialise the same event ids on every run — otherwise
    "is the analyzer deterministic?" could not be asked of a fixture. Mirrors
    :func:`sentinel.models.flags.flag_identity`: hash the label, take 16 bytes.
    """
    from ulid import ULID

    return str(ULID.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:16]))


def _ulid(case_id: str, part: str) -> str:
    """A stable event id for one (case, part) pair."""
    return _stable_ulid(f"corpus:{case_id}:{part}")


def _case_session_id(case_id: str) -> str:
    return _ulid(case_id, "session")


# ---------------------------------------------------------------------------
# grounded: the agent said what the tool said (must NOT be flagged)
# ---------------------------------------------------------------------------

GROUNDED: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="grounded_explicit_price",
        prompt="What does the plan cost?",
        tool="billing.lookup",
        tool_input={"plan": "pro"},
        tool_output="Plan pro costs $49 per month. The starter plan is $19 per month.",
        response="The pro plan costs $49 per month.",
        notes="Verbatim numeric match in the cited result.",
    ),
    CorpusCase(
        case_id="grounded_unit_conversion",
        prompt="How long does the export take?",
        tool="metrics.query",
        tool_input={"metric": "export_ms"},
        tool_output="export_duration_ms = 120000",
        response="The export takes 2 minutes.",
        notes="2 minutes == 120000 ms; the unit-conversion implication.",
    ),
    CorpusCase(
        case_id="grounded_rounding",
        prompt="What was the p95 latency?",
        tool="metrics.query",
        tool_input={"metric": "p95"},
        tool_output="p95_latency_ms = 5.04",
        response="The p95 latency is 5.0 ms.",
        notes="5.0 rounds 5.04 at the claimed precision.",
    ),
    CorpusCase(
        case_id="grounded_bound_inside",
        prompt="How many seats are included?",
        tool="plans.get",
        tool_input={"plan": "team"},
        tool_output="The team plan includes at most 10 seats.",
        response="The team plan includes 7 seats.",
        notes=(
            "A number that falls inside a stated bound is implied support, not "
            "agreement: the evidence never said 7, it bounded it."
        ),
    ),
    CorpusCase(
        case_id="grounded_subset",
        tool="catalog.list",
        tool_input={},
        tool_output="Supported regions are us-east, eu-west, ap-south.",
        prompt="Which regions are supported?",
        response="eu-west is one of us-east, eu-west, ap-south.",
        notes="A subset of the offered set is implied support.",
    ),
    CorpusCase(
        case_id="grounded_weekday",
        prompt="When is the maintenance window?",
        tool="ops.schedule",
        tool_input={},
        tool_output="maintenance_window_weekday = sunday",
        response="The maintenance window is on Sunday.",
        notes="Weekday values match case-insensitively.",
    ),
    CorpusCase(
        case_id="grounded_boolean",
        prompt="Does the API support pagination?",
        tool="docs.search",
        tool_input={"q": "pagination"},
        tool_output="The API supports cursor pagination on every list endpoint.",
        response="The API supports cursor pagination.",
        notes="Boolean claim restating the evidence's affirmative verb.",
    ),
    CorpusCase(
        case_id="grounded_date",
        prompt="When was the release cut?",
        tool="releases.get",
        tool_input={},
        tool_output="release_cut_date = 2024-04-18",
        response="The release was cut on 2024-04-18.",
        notes="ISO date equality.",
    ),
    CorpusCase(
        case_id="grounded_multi_claim_mixed",
        prompt="Summarize the account.",
        tool="account.get",
        tool_input={},
        tool_output="Account status is active. Seats used: 12 of 20. Plan: team.",
        response=("The account is active and uses 12 of 20 seats. Your renewal is on 2024-06-01."),
        notes=(
            "The first sentence is fully supported; the renewal date is invented. "
            "Exactly one finding is expected — the supported claims must survive."
        ),
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="2024-06-01",
                verdict="unknown",
                severity="low",
            )
        ],
    ),
    CorpusCase(
        case_id="grounded_prose_only",
        prompt="Is the service healthy?",
        tool="health.get",
        tool_input={},
        tool_output="status = ok. All 24 regions are responding.",
        response="Everything looks healthy across every region.",
        notes="Ungroundable prose: no value, no finding. Precision over recall.",
    ),
)


# ---------------------------------------------------------------------------
# ungrounded: specific claims no tool output mentions (true positives)
# ---------------------------------------------------------------------------

UNGROUNDED: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="ungrounded_price_invented",
        prompt="How much is the upgrade?",
        tool="billing.lookup",
        tool_input={"plan": "pro"},
        tool_output="Plan pro costs $49 per month. No upgrade fees are recorded.",
        response="The upgrade costs $79 per month.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="$79",
                verdict="unknown",
                severity="medium",
            )
        ],
        notes="A price the tool never mentioned. The classic hallucination.",
    ),
    CorpusCase(
        case_id="ungrounded_no_tool_output",
        prompt="What is our churn?",
        tool="",
        tool_input={},
        tool_output="",
        response="Monthly churn is 4.2%.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="4.2%",
                verdict="unknown",
                severity="medium",
            )
        ],
        notes="No tool ran at all: the number came from the model's weights.",
    ),
    CorpusCase(
        case_id="ungrounded_date_invented",
        prompt="When does the contract end?",
        tool="contracts.get",
        tool_input={},
        tool_output="Contract terms are monthly and renew automatically.",
        response="The contract ends on 2024-12-31.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="2024-12-31",
                verdict="unknown",
                severity="low",
            )
        ],
        notes="A specific date with no date in the evidence.",
    ),
    CorpusCase(
        case_id="ungrounded_implicit_today",
        prompt="What is on sale right now?",
        tool="catalog.list",
        tool_input={},
        tool_output="The spring sale covers accessories only.",
        response="Today is the biggest sale day of the year.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="biggest sale day",
                verdict="unknown",
                severity="low",
            )
        ],
        notes=(
            "Implicit grounding: 'today' plus a superlative with no date or "
            "aggregate in the evidence. The lexicon flags the deictic reference, "
            "and the rules never resolve 'today' against a clock."
        ),
    ),
    CorpusCase(
        case_id="ungrounded_duration",
        prompt="How long is the trial?",
        tool="plans.get",
        tool_input={"plan": "team"},
        tool_output="The team plan includes unlimited projects.",
        response="The trial lasts 14 days.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="14 days",
                verdict="unknown",
                severity="low",
            )
        ],
        notes="A duration no output mentioned.",
    ),
)


# ---------------------------------------------------------------------------
# contradicted: the evidence refutes the claim (the severe half)
# ---------------------------------------------------------------------------

CONTRADICTED: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="contradicted_price",
        prompt="What does the plan cost?",
        tool="billing.lookup",
        tool_input={"plan": "pro"},
        tool_output="Plan pro costs $49 per month.",
        response="The pro plan costs $29 per month.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="$29",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes="Wrong price against cited evidence: the highest-value finding.",
    ),
    CorpusCase(
        case_id="contradicted_weekday",
        prompt="When is the maintenance window?",
        tool="ops.schedule",
        tool_input={},
        tool_output="maintenance_window_weekday = saturday",
        response="The maintenance window is on Tuesday.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="Tuesday",
                verdict="conflicted",
                severity="medium",
            )
        ],
        notes="A disagreeing weekday is a refutation, not silence.",
    ),
    CorpusCase(
        case_id="contradicted_date",
        prompt="When was the release cut?",
        tool="releases.get",
        tool_input={},
        tool_output="release_cut_date = 2024-04-18",
        response="The release was cut on 2024-05-01.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="2024-05-01",
                verdict="conflicted",
                severity="medium",
            )
        ],
        notes="A different full date is a contradiction.",
    ),
    CorpusCase(
        case_id="contradicted_bound_violation",
        prompt="How many seats are included?",
        tool="plans.get",
        tool_input={"plan": "team"},
        tool_output="The team plan includes at most 3 seats.",
        response="The team plan includes 12 seats.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="12 seats",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes="Outside a stated bound is a conflict; inside it is support.",
    ),
    CorpusCase(
        case_id="contradicted_negation",
        prompt="Can we cancel any time?",
        tool="docs.search",
        tool_input={"q": "cancellation"},
        tool_output="Cancellations are accepted at any time with no fee.",
        response="Cancellations are not accepted at any time.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="not accepted",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "A negated claim where the evidence is affirmative. The subject has "
            "to line up: 'no fee' in the evidence must not be read as negating "
            "the acceptance, which is why the negation is scoped to the verb."
        ),
    ),
    CorpusCase(
        case_id="contradicted_membership_extra",
        prompt="What payment methods do we accept?",
        tool="docs.search",
        tool_input={"q": "payment methods"},
        tool_output="We accept card and bank transfer.",
        response="Card is one of card, bank transfer, and cryptocurrency.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="cryptocurrency",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "Set membership that adds a member the evidence enumerated and did "
            "not offer. Same shape as a wrong price: the tool answered the "
            "question, and the answer disagrees with the claim."
        ),
    ),
    CorpusCase(
        case_id="contradicted_membership",
        prompt="What databases are supported?",
        tool="docs.search",
        tool_input={"q": "databases"},
        tool_output="Supported databases are postgres and mysql.",
        response="Mongodb is one of postgres, mysql, and mongodb.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="Mongodb",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "Set membership sharing a topic but asserting a member the evidence "
            "excludes: a real exclusion, not merely an omission."
        ),
    ),
)


#: Every case in the corpus, in a stable order.
CORPUS: tuple[CorpusCase, ...] = GROUNDED + UNGROUNDED + CONTRADICTED

#: Case ids that must not be flagged, whatever else changes.
MUST_NOT_FLAG: frozenset[str] = frozenset(
    case.case_id for case in CORPUS if not case.expects_findings
)

#: Case ids that must produce at least one flag.
MUST_FLAG: frozenset[str] = frozenset(case.case_id for case in CORPUS if case.expects_findings)


def case_by_id(case_id: str) -> CorpusCase:
    """Look a case up by id."""
    for case in CORPUS:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


__all__ = [
    "BASE_TS",
    "CONTRADICTED",
    "CORPUS",
    "GROUNDED",
    "MUST_FLAG",
    "MUST_NOT_FLAG",
    "UNGROUNDED",
    "CorpusCase",
    "ExpectedFinding",
    "case_by_id",
]
