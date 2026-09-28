"""The tool-use grounding worker (sprint ``S3-T5`` - ``S3-T11``).

:class:`ProvenanceEvaluator` walks a finished session and asks one question of
every ``llm.response``: *did the tool output support what the agent said?* The
analysis itself lives in :mod:`sentinel.eval.provenance_core`; this module
supplies the parts that need a store — evidence gathering over the call graph
and flag construction.

Evidence gathering is where provenance lives (``S3-T8``):

* A result the response **cited** (``RefKind.GROUNDS``) is *explicit* evidence,
  and an explicit match is what turns a claim into
  :attr:`~sentinel.eval.provenance_core.Verdict.SUPPORTED`.
* Results the agent could have used — the tool calls that ran earlier in the
  same turn — are *context*. Being wrong about what counts as available is the
  expensive false positive, so they are included.
* With no results at all, every specific claim is ungrounded by construction:
  the agent answered from its weights.

Every flag is deterministic: ids come from
:func:`~sentinel.models.flags.flag_identity` over the response event, and
``created_at`` is the response's own timestamp, so two runs of one module
version produce byte-identical rows (``S3-T4``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from sentinel.eval.provenance_core import (
    DEFAULT_LEXICON,
    Claim,
    ClaimExtractor,
    DiffContext,
    DiffResult,
    GroundingLexicon,
    RuleBasedClaimExtractor,
    Verdict,
    confidence_for,
    diff_claim,
    is_actionable,
    normalize_text,
    severity_for,
)
from sentinel.eval.session import SessionView, response_text_of
from sentinel.eval.worker import (
    CheckpointStore,
    EvaluatorWorker,
    WorkerConfig,
)
from sentinel.models.events import (
    LLM_REQUEST,
    TOOL_CALL,
    TOOL_RESULT,
    Event,
    RefKind,
)
from sentinel.models.flags import (
    EvidenceRef,
    EvidenceRole,
    Flag,
    Severity,
    register_category,
)
from sentinel.query import CallGraph
from sentinel.store.protocol import EventStore

log = structlog.get_logger("sentinel.eval.provenance")

#: Module name recorded on every provenance flag, and part of the idempotency
#: key: any change to the rules ships as a version bump, which re-evaluates.
MODULE = "sentinel.tool_grounding"
MODULE_VERSION = "0.1.0"

#: Flag categories, per the universal flag schema.
CATEGORY_UNGROUNDED = "ungrounded_claim"
CATEGORY_CONTRADICTED = "contradicted_claim"

#: Publish the taxonomy this module ships, so `known_categories()` and any tool
#: that enumerates it agree with the code rather than with a doc.
register_category(CATEGORY_UNGROUNDED, CATEGORY_CONTRADICTED)

#: Max tool results treated as evidence for one response. Bounded so a session
#: with thousands of calls cannot make a single diff pathological.
MAX_EVIDENCE_RESULTS = 25

#: Cap on the evidence snippet stored on an ``EvidenceRef`` note.
MAX_QUOTE_CHARS = 200


@dataclass(frozen=True)
class ResponseAnalysis:
    """What the evaluator concluded about one claim of one response."""

    event_id: str
    claim: Claim
    diff: DiffResult
    severity: Severity
    confidence: float
    evidence: tuple[EvidenceRef, ...] = ()

    @property
    def is_finding(self) -> bool:
        """Whether this analysis becomes a flag (see :func:`is_actionable`)."""
        return is_actionable(self.claim, self.diff)


@dataclass
class ProvenanceResult:
    """Everything one session's analysis produced."""

    session_id: str
    analyses: list[ResponseAnalysis] = field(default_factory=list)
    responses_seen: int = 0
    responses_without_evidence: int = 0

    @property
    def findings(self) -> list[ResponseAnalysis]:
        """Only the analyses worth flagging."""
        return [item for item in self.analyses if item.is_finding]

    @property
    def contradictions(self) -> list[ResponseAnalysis]:
        """Findings the evidence actively refutes."""
        return [item for item in self.analyses if item.diff.verdict is Verdict.CONFLICTED]

    @property
    def supported(self) -> list[ResponseAnalysis]:
        """Analyses the evidence backed (explicitly or by implication)."""
        return [item for item in self.analyses if item.diff.is_grounded]

    @property
    def flagged_claims(self) -> int:
        """How many claims produced a flag."""
        return len(self.findings)


@dataclass(frozen=True)
class ClaimFinding:
    """A finding, ready to be rendered as a :class:`Flag`.

    ``ts`` is the *response event's* timestamp, carried on the finding so the
    worker's flag builder never needs a clock (``S3-T4``).
    """

    session_id: str
    category: str
    claim: Claim
    diff: DiffResult
    severity: Severity
    confidence: float
    event_id: str
    ts: datetime
    evidence: tuple[EvidenceRef, ...] = ()
    review_url: str | None = None

    @property
    def dedupe_key(self) -> str:
        """Identity of this finding within the module/category namespace."""
        return f"{self.event_id}:{self.claim.claim_id}"

    @property
    def summary(self) -> str:
        """A one-line, human-readable explanation of the finding."""
        verb = "contradicted" if self.diff.verdict is Verdict.CONFLICTED else "not grounded"
        return f"Claim {verb} by tool output: {self.claim.text.strip()} ({self.diff.detail})"

    def to_flag(
        self,
        *,
        module: str = MODULE,
        module_version: str = MODULE_VERSION,
        review_only: bool = False,
    ) -> Flag:
        """Build the deterministic flag for this finding (ADR-0012)."""
        details: dict[str, object] = {
            "claim_text": self.claim.text,
            "claim_kind": str(self.claim.kind),
            "claim_cue": self.claim.cue,
            "claim_value": str(self.claim.value) if self.claim.value else "",
            "verdict": str(self.diff.verdict),
            "support_kind": str(self.diff.support_kind),
            "detail": self.diff.detail,
        }
        if self.diff.matched_value:
            # S3-T8: a contradiction has to carry the value that refutes it,
            # otherwise the reviewer has to go and re-derive it by hand.
            details["observed_value"] = self.diff.matched_value
        if self.review_url:
            # A review link is deployment-shaped, so it rides in details rather
            # than the universal schema (docs/adr/0012-flag-schema.md).
            details["review_url"] = self.review_url
        return Flag.create(
            session_id=self.session_id,
            module=module,
            module_version=module_version,
            category=self.category,
            severity=self.severity,
            confidence=self.confidence,
            summary=self.summary,
            evidence=list(self.evidence),
            details=details,
            event_id=self.event_id,
            dedupe_key=self.dedupe_key,
            created_at=self.ts,
            review_only=review_only,
        )


class ProvenanceEvaluator(EvaluatorWorker):
    """Flags agent claims that the tool output does not support (``S3-T5``).

    Pass ``extractor=`` to swap in another :class:`ClaimExtractor`. That is a
    module-version change in practice — a different extractor yields different
    claims — which the idempotency key will correctly treat as a re-evaluation.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        config: WorkerConfig | None = None,
        checkpoints: CheckpointStore | None = None,
        extractor: ClaimExtractor | None = None,
        lexicon: GroundingLexicon | None = None,
        review_url_template: str | None = None,
        review_confidence_threshold: float | None = None,
    ) -> None:
        """Create a provenance worker over *store* with rule-first extraction.

        ``review_confidence_threshold`` is the common knob for a deployment
        running a review queue: anything at or below it is written with
        ``review_only`` and never gates a build. It defaults to 0.0, which
        routes *every* non-contradictory finding to a human — ungrounded claims
        are hypotheses, not verdicts, and this module does not pretend
        otherwise.
        """
        self._lexicon = lexicon or DEFAULT_LEXICON
        self._extractor = extractor or RuleBasedClaimExtractor(lexicon=self._lexicon)
        self._review_url_template = review_url_template
        defaults = WorkerConfig(
            module=MODULE,
            module_version=MODULE_VERSION,
            review_confidence_threshold=(
                0.0 if review_confidence_threshold is None else review_confidence_threshold
            ),
        )
        super().__init__(
            store,
            config=config or defaults,
            checkpoints=checkpoints,
        )

    # -- module contract --------------------------------------------------

    async def evaluate(self, session: SessionView) -> list[Flag]:
        """Return the provenance flags *session* deserves."""
        threshold = self.config.review_confidence_threshold
        return [
            finding.to_flag(
                module=self.module,
                module_version=self.module_version,
                # The base class would apply this to the flag; doing it here
                # keeps `findings()` a complete description of the finding.
                review_only=finding.confidence <= threshold,
            )
            for finding in await self.findings(session)
        ]

    async def findings(self, session: SessionView) -> list[ClaimFinding]:
        """The findings for *session*, in deterministic order (``S3-T4``)."""
        result = await self.analyze(session)
        return [
            ClaimFinding(
                session_id=session.session_id,
                category=(
                    CATEGORY_CONTRADICTED
                    if analysis.diff.verdict is Verdict.CONFLICTED
                    else CATEGORY_UNGROUNDED
                ),
                claim=analysis.claim,
                diff=analysis.diff,
                severity=analysis.severity,
                confidence=analysis.confidence,
                event_id=analysis.event_id,
                ts=_event_ts(session, analysis.event_id),
                evidence=analysis.evidence,
                review_url=self._review_url(session.session_id),
            )
            for analysis in result.findings
        ]

    async def analyze(self, session: SessionView) -> ProvenanceResult:
        """Analyze every response in *session*; pure and deterministic."""
        result = ProvenanceResult(session_id=session.session_id)
        for response in session.llm_responses():
            result.responses_seen += 1
            context, refs = gather_evidence(session.graph, response)
            if not context.has_evidence:
                result.responses_without_evidence += 1
            for claim in await self._extractor.extract(response_text_of(response)):
                diff = diff_claim(claim, context)
                result.analyses.append(
                    ResponseAnalysis(
                        event_id=response.event_id,
                        claim=claim,
                        diff=diff,
                        severity=severity_for(claim, diff),
                        confidence=confidence_for(claim, diff, context),
                        evidence=_flag_evidence(response, refs, diff),
                    )
                )
        return result

    async def analyze_session(self, session_id: str) -> ProvenanceResult:
        """Load and analyze a session without writing flags (diagnostics/tests)."""
        view = await SessionView.load(self._store, session_id, max_events=self.config.max_events)
        return await self.analyze(view)

    def _review_url(self, session_id: str) -> str | None:
        if not self._review_url_template:
            return None
        return self._review_url_template.format(session_id=session_id, module=self.module)


# ---------------------------------------------------------------------------
# evidence gathering (``S3-T8``)
# ---------------------------------------------------------------------------


def gather_evidence(
    graph: CallGraph,
    response: Event,
) -> tuple[DiffContext, tuple[EvidenceRef, ...]]:
    """Collect the evidence available to *response*.

    Returns the :class:`DiffContext` the rules run over and the evidence refs to
    attach to any flag about this response.

    Explicit evidence is what the response cited (``RefKind.GROUNDS``). Context
    is the output of the tool calls that fed the same LLM turn, reached through
    the request's ``parent`` chain. ``implied`` maps each lexicon phrase the
    response used to the evidence that might resolve it — how "today" gets
    grounded by a date in a tool result without ever consulting a clock.
    """
    explicit: list[str] = []
    explicit_refs: list[EvidenceRef] = []
    for ref in response.refs:
        if ref.kind is not RefKind.GROUNDS:
            continue
        target = graph.nodes.get(ref.event_id)
        if target is None or target.type != TOOL_RESULT:
            continue
        text = _result_text(target)
        explicit.append(text)
        explicit_refs.append(
            EvidenceRef(
                event_id=target.event_id,
                role=EvidenceRole.EVIDENCE,
                seq=target.seq,
                note=_quote(text),
            )
        )

    context: list[str] = []
    context_refs: list[EvidenceRef] = []
    seen = {ref.event_id for ref in explicit_refs}
    for result in _turn_results(graph, response):
        if result.event_id in seen:
            continue
        seen.add(result.event_id)
        text = _result_text(result)
        context.append(text)
        context_refs.append(
            EvidenceRef(
                event_id=result.event_id,
                role=EvidenceRole.CONTEXT,
                seq=result.seq,
                note=_quote(text),
            )
        )
    if len(context) > MAX_EVIDENCE_RESULTS:
        context = context[-MAX_EVIDENCE_RESULTS:]
        context_refs = context_refs[-MAX_EVIDENCE_RESULTS:]

    return (
        DiffContext(
            explicit=tuple(explicit),
            context=tuple(context),
            implied=_implied_offerings(response, explicit + context),
        ),
        tuple(explicit_refs + context_refs),
    )


def _turn_results(graph: CallGraph, response: Event) -> list[Event]:
    """Tool results the response could have been answering from.

    Three ways in, because real logs wire a turn differently depending on the
    framework: the request the response was caused by (and that request's own
    ancestors, for a multi-step turn), the call the response hangs off as its
    parent, and the response's own parent chain. Anything the agent could have
    read before answering belongs here — being wrong about "available" is the
    expensive false positive.
    """
    results: list[Event] = []
    seen: set[str] = set()

    def add(event: Event) -> None:
        if event.type == TOOL_RESULT and event.event_id not in seen:
            seen.add(event.event_id)
            results.append(event)

    def add_calls_of(event: Event) -> None:
        for call in _calls_of_request(graph, event):
            for result in graph.tool_results_for(call.event_id):
                add(result)

    def walk(event: Event) -> None:
        add_calls_of(event)
        for ancestor in graph.parents(event.event_id):
            add_calls_of(ancestor)
            if ancestor.type == LLM_REQUEST:
                walk(ancestor)

    for ref in response.refs:
        node = graph.nodes.get(ref.event_id)
        if node is None:
            continue
        if ref.kind is RefKind.CAUSED_BY and node.type == LLM_REQUEST:
            walk(node)
        elif ref.kind is RefKind.PARENT:
            add_calls_of(node)
            for ancestor in graph.parents(node.event_id):
                walk(ancestor)
    for ancestor in graph.parents(response.event_id):
        walk(ancestor)
    return results


def _calls_of_request(graph: CallGraph, event: Event) -> list[Event]:
    """``tool.call`` events issued under *event* (its children), in order."""
    calls = [edge.src for edge in graph.refs_to(event.event_id) if edge.src.type == TOOL_CALL]
    return sorted(calls, key=lambda call: call.seq)


def _flag_evidence(
    response: Event,
    refs: Sequence[EvidenceRef],
    diff: DiffResult,
) -> tuple[EvidenceRef, ...]:
    """Assemble the evidence list for a flag: the claim, plus its sources.

    A contradicted claim also gets the ref that refutes it re-labelled as
    ``countervailance`` when the ref's snippet contains the observed value, so a
    reviewer sees the disagreement without opening the log.
    """
    claim_ref = EvidenceRef(
        event_id=response.event_id,
        role=EvidenceRole.CLAIM,
        seq=response.seq,
        note=_quote(response_text_of(response)),
    )
    evidence = [claim_ref, *refs]
    if diff.verdict is not Verdict.CONFLICTED:
        return tuple(evidence[: MAX_EVIDENCE_RESULTS + 1])
    wanted = _tokens(diff.observed or diff.matched_value)
    return tuple(
        [claim_ref]
        + [
            # The ref that carries the refuting value is promoted, so the
            # reviewer sees the disagreement in the flag rather than in the log.
            ref.model_copy(update={"role": EvidenceRole.COUNTERVAILANCE})
            if any(token in _squash(ref.note or "") for token in wanted)
            else ref
            for ref in refs
        ]
    )[: MAX_EVIDENCE_RESULTS + 1]


def _squash(text: str) -> str:
    """Letters and digits only, so ``"49 usd"`` matches ``costs $49/month``."""
    return re.sub(r"[^a-z0-9]", "", normalize_text(text))


#: Separators a canonical value uses to hold several things at once.
_TOKEN_SPLIT = re.compile(r"[|,]|\band\b|\bor\b")

#: Conjunctions carry no evidence, so a note that merely says "or" is not the
#: line that refutes anything.
_TOKEN_STOPWORDS = frozenset({"and", "or"})


def _tokens(observed: str) -> tuple[str, ...]:
    """The pieces of an observed value worth looking for in a note.

    A set is stored joined (``"card|wire"``) and a boolean as
    ``"cancellations|false"``; neither appears verbatim in prose, so each part
    is looked for on its own.
    """
    parts = (part.strip().casefold() for part in _TOKEN_SPLIT.split(observed))
    return tuple(
        token
        for token in (_squash(part) for part in parts if part)
        if token and token not in _TOKEN_STOPWORDS
    )


def _result_text(result: Event) -> str:
    """The payload a tool result offers as text, whatever its shape."""
    output = result.payload.get("output")
    if output is None:
        output = result.payload.get("result")
    return _render(output)


def _render(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " | ".join(f"{key}: {_render(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(_render(item) for item in value)
    return str(value)


def _quote(text: str, limit: int = MAX_QUOTE_CHARS) -> str | None:
    """Flatten *text* to a snippet, or ``None`` when there is nothing to show."""
    flat = " ".join((text or "").split())
    if not flat:
        return None
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _implied_offerings(response: Event, evidence: Sequence[str]) -> dict[str, str]:
    """What the evidence offers for each implicit phrase the response used."""
    if not evidence:
        return {}
    text = normalize_text(response_text_of(response))
    blob = " ".join(evidence)
    return dict.fromkeys(DEFAULT_LEXICON.matches(text), blob)


def _event_ts(session: SessionView, event_id: str) -> datetime:
    """The timestamp of the evidence event, never the wall clock (``S3-T4``)."""
    for event in session.events:
        if event.event_id == event_id:
            return event.ts
    if session.events:
        return session.events[-1].ts
    return datetime.fromtimestamp(0, tz=UTC)


__all__ = [
    "CATEGORY_CONTRADICTED",
    "CATEGORY_UNGROUNDED",
    "MAX_EVIDENCE_RESULTS",
    "MODULE",
    "MODULE_VERSION",
    "ClaimFinding",
    "ProvenanceEvaluator",
    "ProvenanceResult",
    "ResponseAnalysis",
    "gather_evidence",
]
