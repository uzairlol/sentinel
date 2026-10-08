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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
class ToolUse:
    """A tool the turn called, and whether the response cited its result.

    Separate from :class:`CorpusCase`'s primary tool because the misattributed
    citation case needs *two* results and a citation that points at only one of
    them. Whether each result was cited has to be stated per result — that
    asymmetry is the entire case.
    """

    tool: str
    tool_output: str
    tool_input: Mapping[str, object] = field(default_factory=dict)
    cited: bool = False


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
    #: Materialise an earlier turn that cites its own tool result before the
    #: turn under test. That is what makes a fabricated citation *provable*:
    #: the module only reports ``unsourced_citation`` when the session proves
    #: the instrumentation records citations at all, and a one-turn session
    #: cannot prove that.
    prior_citing_turn: bool = False
    #: Further tools the same turn called, in call order. Each carries its own
    #: ``cited`` flag, so a case can have the agent read two documents and
    #: point at one of them.
    extra_tools: tuple[ToolUse, ...] = ()

    @property
    def all_tools(self) -> tuple[tuple[str, Mapping[str, object], str, bool], ...]:
        """Every tool in call order: ``(name, input, output, cited)``.

        The primary tool's citation is :attr:`cited`; an extra tool carries its
        own. Read through this rather than through :attr:`tool` directly so the
        two cannot drift apart when a case is built programmatically.
        """
        primary = (self.tool, self.tool_input, self.tool_output, self.cited) if self.tool else ()
        return (
            *((primary,) if primary else ()),
            *((use.tool, use.tool_input, use.tool_output, use.cited) for use in self.extra_tools),
        )

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
        # Every tool the turn called, with the ids the response will point at.
        # The primary tool keeps the unsuffixed ids so cases that never grow an
        # ``extra_tools`` entry materialise exactly the events they always did.
        calls: list[tuple[str, str, bool]] = []
        for index, (tool, tool_input, tool_output, cited) in enumerate(self.all_tools):
            suffix = "" if index == 0 else f"_extra{index - 1}"
            call_id = _ulid(self.case_id, f"call{suffix}")
            result_id = _ulid(self.case_id, f"result{suffix}")
            raw.append(
                {
                    "event_id": call_id,
                    "type": TOOL_CALL,
                    "payload": {"tool": tool, "input": str(tool_input)},
                    "refs": [{"event_id": request_id, "kind": RefKind.CAUSED_BY.value}],
                }
            )
            raw.append(
                {
                    "event_id": result_id,
                    "type": TOOL_RESULT,
                    "payload": {"tool": tool, "output": tool_output},
                    "refs": [{"event_id": call_id, "kind": RefKind.CAUSED_BY.value}],
                }
            )
            types.extend([TOOL_CALL, TOOL_RESULT])
            calls.append((call_id, result_id, cited))

        # The response is a child of the request and of every tool call in the
        # turn, and cites whichever results the case says it cited.
        response_refs: list[dict[str, str]] = [
            {"event_id": request_id, "kind": RefKind.CAUSED_BY.value}
        ]
        response_refs.extend(
            {"event_id": call_id, "kind": RefKind.PARENT.value} for call_id, _, _ in calls
        )
        response_refs.extend(
            {"event_id": result_id, "kind": RefKind.GROUNDS.value}
            for _, result_id, cited in calls
            if cited
        )
        raw.append(
            {
                "event_id": _ulid(self.case_id, "response"),
                "type": LLM_RESPONSE,
                "payload": {"provider": "corpus", "generations": [self.response]},
                "refs": response_refs,
            }
        )
        types.append(LLM_RESPONSE)
        if self.prior_citing_turn:
            # Spliced in after ``session.start`` (index 0) and before the rest,
            # so the earlier turn precedes this one in the log without
            # displacing the session header. The main turn's ids are derived
            # above and stay stable either way.
            prior_raw, prior_types = self._prior_turn()
            raw = [raw[0], *prior_raw, *raw[1:]]
            types = [types[0], *prior_types, *types[1:]]
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

    def _prior_turn(self) -> tuple[list[dict[str, object]], list[str]]:
        """An earlier turn that cites its own result: the citation-tracking proof.

        Returns the ``(specs, types)`` half-list to prepend. Kept separate from
        :meth:`events` because it is only materialised on request, and the
        ids have to be derived rather than passed in.
        """
        request_id = _ulid(self.case_id, "prior_request")
        call_id = _ulid(self.case_id, "prior_call")
        result_id = _ulid(self.case_id, "prior_result")
        caused = [{"event_id": request_id, "kind": RefKind.CAUSED_BY.value}]
        specs: list[dict[str, object]] = [
            {
                "event_id": request_id,
                "type": LLM_REQUEST,
                "payload": {"provider": "corpus", "model": "fixture"},
            },
            {
                "event_id": call_id,
                "type": TOOL_CALL,
                "payload": {"tool": "session.lookup", "input": {"case": self.case_id}},
                "refs": list(caused),
            },
            {
                "event_id": result_id,
                "type": TOOL_RESULT,
                "payload": {"tool": "session.lookup", "output": "turn_index = 1"},
                "refs": [{"event_id": call_id, "kind": RefKind.CAUSED_BY.value}],
            },
            {
                "event_id": _ulid(self.case_id, "prior_response"),
                "type": LLM_RESPONSE,
                "payload": {"provider": "corpus", "generations": ["This is turn one."]},
                "refs": [
                    *caused,
                    {"event_id": call_id, "kind": RefKind.PARENT.value},
                    {"event_id": result_id, "kind": RefKind.GROUNDS.value},
                ],
            },
        ]
        types = [LLM_REQUEST, TOOL_CALL, TOOL_RESULT, LLM_RESPONSE]
        return specs, types

    def cited_events(self) -> list[Event]:
        """The case's events with the response citing its tool results.

        Citation is the default for every case, including the ungrounded ones: a
        model that *had* the source and still asserted something else is the more
        interesting failure, and it is the one the corpus is measuring. Cases
        built with ``cited=False`` exercise the uncited path instead.

        Equivalent to :meth:`events` since citation moved onto the tools
        themselves (:attr:`ToolUse.cited`), which is what allows a case to cite
        one of two results. Kept as a name because it is what the call sites mean.
        """
        return self.events()

    def uncited_events(self) -> list[Event]:
        """The same turn with every citation removed.

        The counterpart to :meth:`events`, for tests that need to prove
        available-but-uncited evidence is treated as context rather than as a
        citation. Built by rewriting the flags rather than by dropping refs by
        hand, so it stays correct when a case grows a second tool.
        """
        stripped = replace(
            self,
            cited=False,
            extra_tools=tuple(replace(use, cited=False) for use in self.extra_tools),
        )
        return stripped.events()


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
    CorpusCase(
        case_id="grounded_count_reported_whole",
        prompt="Did the compliance checks pass?",
        tool="compliance.run",
        tool_input={},
        tool_output="check_1 pass, check_2 pass, check_3 pass, check_4 fail.",
        response="3 of 4 checks passed.",
        notes=(
            "The count a competent agent reports. The source enumerates four "
            "siblings and the claim counts four — the cherry-picking rule has to "
            "stay silent on an honest denominator, or it flags every summary."
        ),
    ),
    CorpusCase(
        case_id="grounded_count_verbatim",
        prompt="How many seats are in use?",
        tool="account.get",
        tool_input={},
        tool_output="Seats used: 12 of 20.",
        response="The account is using 12 of 20 seats.",
        notes=(
            "A count-shaped phrase with no countable set behind it. The numbers "
            "came from the tool verbatim, so there is nothing to check and "
            "nothing to flag."
        ),
    ),
    CorpusCase(
        case_id="grounded_attributed_and_cited",
        prompt="What does the audit say about the retention window?",
        tool="docs.search",
        tool_input={"q": "retention"},
        tool_output="The retention window is 400 days.",
        response="According to the audit report, the retention window is 400 days.",
        prior_citing_turn=True,
        notes=(
            "Attribution plus a real citation. The claim names a source *and* "
            "cites one, so the unsourced rule must not fire — this is the case "
            "that keeps the feature from flagging honest citation as fabrication."
        ),
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
    CorpusCase(
        case_id="fabricated_citation_no_source",
        prompt="Did the incident rate improve last quarter?",
        tool="",
        tool_input={},
        tool_output="",
        response="According to the compliance report, the incident rate fell 12%.",
        cited=False,
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="unsourced_citation",
                claim_contains="compliance report",
                verdict="unsourced",
            )
        ],
        notes=(
            "The gap this module was named for. A named document, no tool call, "
            "no citation — and a session that proves citations are recorded, so "
            "the absence means something."
        ),
    ),
    CorpusCase(
        case_id="fabricated_citation_uncited_among_tools",
        prompt="Which regions are supported?",
        tool="catalog.list",
        tool_input={},
        tool_output="Supported regions are us-east, eu-west, ap-south.",
        response="The vendor documentation confirms that coverage is 9 regions.",
        cited=False,
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="unsourced_citation",
                claim_contains="vendor documentation",
                verdict="unsourced",
            )
        ],
        notes=(
            "A tool *was* called and its output was available, and the response "
            "still named a source it did not cite. The uncited-and-unsupported "
            "shape: the harder variant, because a reader has no way to notice "
            "the tool ran."
        ),
    ),
    CorpusCase(
        case_id="cherry_picked_count",
        prompt="How did the pre-flight checks go?",
        tool="preflight.run",
        tool_input={},
        tool_output="check_1 pass, check_2 pass, check_3 fail, check_4 fail.",
        response="2 of 3 checks passed.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="2 of 3",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "Four checks ran; the agent reported three of them, and the one it "
            "left out was a failure. The denominator is the tell — it is smaller "
            "than what the source enumerates, which is the arithmetic signature "
            "of selecting the favourable subset."
        ),
    ),
    CorpusCase(
        case_id="cherry_picked_universal_skipped",
        prompt="Did the pre-flight checks all pass?",
        tool="preflight.run",
        tool_input={},
        tool_output="check_1 pass, check_2 pass, check_3 skipped, check_4 pass.",
        response="All checks passed.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="All checks passed",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "The shape the counted rule cannot see. 'All checks passed' states no "
            "denominator, so there is no number to compare with the enumeration "
            "— the favourable subset has simply been declared to be the whole. "
            "A *skipped* check is the sharpest version: in a summary line it is "
            "indistinguishable from a passing one, which is exactly how this "
            "sentence gets written."
        ),
    ),
    CorpusCase(
        case_id="cherry_picked_universal_failed",
        prompt="How did the nightly jobs go?",
        tool="ci.report",
        tool_input={},
        tool_output="job_1: succeeded, job_2: failed, job_3: succeeded.",
        response="Every job succeeded.",
        expect=[
            ExpectedFinding(
                category="contradicted_claim",
                claim_contains="Every job succeeded",
                verdict="conflicted",
                severity="high",
            )
        ],
        notes=(
            "The plainer version of the same misrepresentation, with the "
            "key/value status shape rather than the bare-adjective one. Worth "
            "having both: the two are read by different patterns, and a rule "
            "that only understood one would pass this corpus while failing on "
            "half of real CI output."
        ),
    ),
)


# ---------------------------------------------------------------------------
# misattributed citation: named one source, cited another
# ---------------------------------------------------------------------------

MISATTRIBUTED: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="misattributed_citation_named_uncited",
        prompt="Summarise the quarter for the board deck.",
        tool="weather.forecast",
        tool_input={"city": "berlin"},
        tool_output="Conditions: light rain, 12C.",
        extra_tools=(
            ToolUse(
                tool="compliance.report",
                tool_input={"quarter": "q3"},
                tool_output="The Q3 compliance report records 0 reported incidents.",
                cited=False,
            ),
        ),
        response=("According to the compliance report, the retention window is 900 days."),
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="misattributed_citation",
                claim_contains="900 days",
                verdict="misattributed",
                severity="medium",
            )
        ],
        notes=(
            "The case that per-claim resolution exists for. Two results ran; the "
            "response cited the weather feed and pointed at the compliance "
            "report, which it had in hand but did not cite. A turn-level rule "
            "sees a citation and stops looking; this one resolves the named "
            "phrase against the log and finds the citation points somewhere "
            "else."
        ),
    ),
    CorpusCase(
        case_id="misattributed_citation_named_absent",
        prompt="What is our refund position?",
        tool="billing.lookup",
        tool_input={"plan": "pro"},
        tool_output="Plan pro costs $49 per month. Refunds are handled by support.",
        response=("According to the vendor security assessment, the refund window is 14 days."),
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="misattributed_citation",
                claim_contains="14 days",
                verdict="misattributed",
                severity="medium",
            )
        ],
        notes=(
            "The cited source was real and the named one never ran at all. "
            "Distinct from `fabricated_citation_no_source`, where the turn cited "
            "nothing: here the agent had a genuine source in hand and pointed "
            "the reader at a document that does not exist in the log."
        ),
    ),
)


# ---------------------------------------------------------------------------
# misattribution guards: named sources that must NOT be called misattributed
# ---------------------------------------------------------------------------

MISATTRIBUTION_GUARDS: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="grounded_named_source_is_the_cited_one",
        prompt="What does the compliance report say about retention?",
        tool="compliance.report",
        tool_input={},
        tool_output="The retention window is 400 days.",
        response="According to the compliance report, the retention window is 400 days.",
        prior_citing_turn=True,
        notes=(
            "The named phrase resolves to the very result the response cited. "
            "This is the case that stops the resolution rule from flagging "
            "honest citation because a tool name happens to contain a generic "
            "word."
        ),
    ),
    CorpusCase(
        case_id="grounded_generic_source_name_is_not_misattributed",
        prompt="What does the report promise about availability?",
        tool="billing.lookup",
        tool_input={"plan": "pro"},
        tool_output="Plan pro costs $49 per month.",
        response="According to the report, the uptime commitment is 99.99%.",
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="99.99%",
                verdict="unknown",
            )
        ],
        notes=(
            "'The report' names no document in particular — every source is "
            "called something like that. The rule refuses to resolve a phrase "
            "with no distinctive word, so this is reported as an ungrounded "
            "claim, which is what it is. The expectation is load-bearing: it "
            "fails loudly if the resolution rule ever guesses its way to a "
            "`misattributed_citation` here."
        ),
    ),
    CorpusCase(
        case_id="grounded_ambiguous_source_name_is_not_misattributed",
        prompt="How did the year close?",
        tool="revenue.q3",
        tool_input={"year": 2024},
        tool_output="Q3 closed two days late.",
        extra_tools=(ToolUse(tool="revenue.q2", tool_output="Q2 closed on time."),),
        response="According to the revenue table, the Q4 close is scheduled for 2024-09-30.",
        prior_citing_turn=True,
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="2024-09-30",
                verdict="unknown",
            )
        ],
        notes=(
            "Two results match 'revenue', so 'the revenue table' does not "
            "identify one of them. Ambiguity is silence, not evidence of "
            "fabrication — the same reason a cautious reviewer would not act on "
            "this. The evidence carries no numbers or dates on purpose: the case "
            "is about which source the claim names, and a value the other rules "
            "could quarrel with would make it about something else."
        ),
    ),
)


# ---------------------------------------------------------------------------
# universal-claim guards: "all passed" that must NOT be cherry-picking
# ---------------------------------------------------------------------------

UNIVERSAL_GUARDS: tuple[CorpusCase, ...] = (
    CorpusCase(
        case_id="grounded_universal_all_passed",
        prompt="Did the pre-flight checks pass?",
        tool="preflight.run",
        tool_input={},
        tool_output="check_1 pass, check_2 pass, check_3 pass, check_4 pass.",
        response="All checks passed.",
        notes=(
            "The earned version. The rule has to be able to *support* a universal "
            "claim, or an operator learns that 'all passed' always produces a flag "
            "and stops reading them."
        ),
    ),
    CorpusCase(
        case_id="grounded_universal_too_few_items_to_count",
        prompt="How did the smoke tests go?",
        tool="smoke.run",
        tool_input={},
        tool_output="smoke_1 pass, smoke_2 skipped.",
        response="All smoke tests passed.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="All smoke tests passed",
                verdict="unknown",
            )
        ],
        notes=(
            "Two items is below the enumeration floor, so the rule cannot tell "
            "whether the set was complete. It reports the claim as ungrounded "
            "rather than as cherry-picking — the weaker and more honest category "
            "— and at UNKNOWN's confidence it is review-only and never gates. The "
            "expectation is load-bearing: it fails if the rule ever starts "
            "asserting a conflict it has not established."
        ),
    ),
    CorpusCase(
        case_id="grounded_universal_prose_status_words",
        prompt="What does the status say?",
        tool="status.lookup",
        tool_input={},
        tool_output="There is no further work. The migration is done and nothing is missing.",
        response="All checks passed.",
        expect=[
            ExpectedFinding(
                category="ungrounded_claim",
                claim_contains="All checks passed",
                verdict="unknown",
            )
        ],
        notes=(
            "Prose containing status words — 'no', 'done', 'missing' — must not "
            "read as an item-status enumeration. Without the identifier/separator "
            "requirement this case would be caught as cherry-picking on the "
            "strength of three English words, which is precisely the failure the "
            "floor exists to prevent."
        ),
    ),
)


#: Every case in the corpus, in a stable order.
CORPUS: tuple[CorpusCase, ...] = (
    GROUNDED + UNGROUNDED + CONTRADICTED + MISATTRIBUTED + MISATTRIBUTION_GUARDS + UNIVERSAL_GUARDS
)

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
