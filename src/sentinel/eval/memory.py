"""Memory-integrity evaluator: drift, collapse and ungrounded summaries (``S4``).

The second detection module, and the first consumer of ``provenance_core`` from
outside the provenance module itself — which is the point of the sprint. It
imports ``diff_claim`` and the lexicons rather than reimplementing "grounded",
so a sentence cannot be clean in one module and flagged in the other.

It is also the first module with a *stateful* view of a session. Provenance asks
"what did this turn claim against this turn's evidence"; memory asks "where has
this memory been going", which needs the whole ordered series of writes before it
can say anything about any of them. That is why :meth:`analyze` walks the series
before it judges anything, and why warm-up
(:attr:`DriftConfig.warmup_writes`) is a correctness requirement rather than a
nicety: the first writes of a session are what establish what the memory is for,
and drift measured against a baseline that does not exist yet is how a module
produces confident nonsense on every session's opening turn.

Routing (``S4-T8``):

``memory_drift``
    A single write moved the memory further than the threshold. The signature of
    an injected instruction, and the one finding here strong enough to be
    gate-worthy — because it is a single-event measurement with a known threshold
    behind it, not a trend.

``memory_ungrounded``
    A summary asserts something the session does not contain. Structurally
    checkable and high severity: a fabricated record is worse than a corrupted
    one, because a fabricated one gets believed later.

``memory_collapse``
    Successive writes converge. Reported, and routed review-only by default,
    because convergence is a symptom with several innocent causes — a loop, a
    one-line task, or a memory that genuinely has nothing new to say.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from sentinel.eval.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingProvider,
    HashingEmbeddingProvider,
)
from sentinel.eval.memory_core import (
    DriftConfig,
    MemoryFinding,
    MemoryState,
    MemoryVerdict,
    MemoryWrite,
    SeriesAnalysis,
    WriteIntent,
    confidence_for_finding,
    contradicted_summary,
    detect_drift,
    is_collapse,
    looks_like_an_injection,
    novelty_scores,
    numbers_seen,
    severity_for_finding,
    state_series,
    step_distances,
    ungrounded_claims,
)
from sentinel.eval.provenance_core import ClaimExtractor
from sentinel.eval.session import SessionView
from sentinel.eval.worker import CheckpointStore, EvaluatorWorker, WorkerConfig
from sentinel.models.events import Event
from sentinel.models.flags import (
    EvidenceRef,
    EvidenceRole,
    Flag,
    register_category,
)
from sentinel.store.protocol import EventStore

log = structlog.get_logger("sentinel.eval.memory")

#: Module name recorded on every memory flag, and part of the idempotency key.
MODULE = "sentinel.memory_integrity"
#: 0.1.0 is the first release of this module.
MODULE_VERSION = "0.1.0"

#: An abrupt, single-write jump in the memory's meaning.
CATEGORY_DRIFT = "memory_drift"
#: A summary or reflection asserting something the session does not contain.
CATEGORY_MEMORY_UNGROUNDED = "memory_ungrounded"
#: Successive writes converging on a stale state. A symptom, so review-only.
CATEGORY_COLLAPSE = "memory_collapse"

register_category(CATEGORY_DRIFT, CATEGORY_MEMORY_UNGROUNDED, CATEGORY_COLLAPSE)

#: Categories that never gate. Collapse is a trend rather than an event, so it
#: belongs in a queue for a human and not in a rule that blocks a deployment.
REVIEW_ONLY_CATEGORIES: frozenset[str] = frozenset({CATEGORY_COLLAPSE})

#: Cap on transcript characters handed to the summary check. The comparison is
#: O(summary x transcript), and a longer session buys no extra accuracy here.
MAX_TRANSCRIPT_CHARS = 24_000


#: Cap on the offending content quoted onto a flag. Long enough to show an
#: injected instruction in full, short enough that a flag row stays readable and
#: the store does not carry a copy of the whole memory.
MAX_QUOTE_CHARS = 400


def as_text(value: object) -> str:
    """Render a captured payload value as text.

    Deliberately a local definition rather than a shared import: ``memory_core``
    is the pure layer and must not grow a dependency on the flag model, and this
    renderer flattens nested structures in the way memory values need.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return " | ".join(f"{key}: {as_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(as_text(item) for item in value)
    return str(value)


def _quote(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    """Flatten *text* to a single-line snippet for a flag row."""
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def write_from_event(event: Event) -> MemoryWrite:
    """Reduce a ``memory.write`` event to a :class:`MemoryWrite`.

    Key and value are read leniently. An instrumentor that recorded neither is
    still a write that changed memory, and refusing to model it would let a
    misbehaving adapter blind the module entirely.
    """
    payload = event.payload
    key = payload.get("key")
    value = payload.get("value")
    summary = payload.get("summary")
    return MemoryWrite.from_fields(
        event_id=event.event_id,
        seq=event.seq,
        key=key if isinstance(key, str) else "",
        value=value if isinstance(value, str) else as_text(value),
        summary=summary if isinstance(summary, str) else "",
    )


def transcript_of(session: SessionView, limit: int = MAX_TRANSCRIPT_CHARS) -> str:
    """What the session actually said, as one blob of text.

    Built from LLM responses, tool calls and tool results — everything a summary
    could legitimately be describing. The *tail* is kept rather than the head for
    the same reason the memory state keeps its tail: a summary written at the end
    of a session describes the end of it, and a prefix would guarantee that a
    claim about recent events looked ungrounded.
    """
    parts: list[str] = []
    for event in session.events:
        if event.type == "llm.response":
            text = session.response_text(event)
            if text:
                parts.append(text)
        elif event.type == "tool.result":
            output = session.tool_output(event)
            if output:
                parts.append(output)
        elif event.type == "tool.call":
            name = session.tool_name(event)
            if name:
                parts.append(name)
    return "\n".join(parts)[-limit:]


@dataclass(frozen=True)
class MemoryIntegrityConfig:
    """Evaluator-level knobs (``S4-T7``).

    Separate from :class:`DriftConfig` because these are about *how the module is
    deployed* and that is a different conversation from *what counts as drift*.
    Thresholds live with the rule that reads them; review policy lives with the
    worker that applies it.
    """

    drift: DriftConfig = field(default_factory=DriftConfig)
    #: A summary shorter than this is not checked for grounding. Reflexive notes
    #: ("ok", "done") cannot assert a falsifiable event, so running the extractor
    #: over them costs time and produces nothing.
    min_summary_chars: int = 16
    #: Numbers the transcript must contain before a numeric claim in a summary is
    #: worth checking. Below this the transcript cannot refute anything, so the
    #: check would report silence and call it ungrounded.
    min_transcript_numbers: int = 2
    review_url_template: str | None = None


def _category(finding: MemoryFinding) -> str:
    """The flag category for *finding*."""
    if finding.verdict is MemoryVerdict.DRIFT:
        return CATEGORY_DRIFT
    if finding.verdict is MemoryVerdict.COLLAPSE:
        return CATEGORY_COLLAPSE
    return CATEGORY_MEMORY_UNGROUNDED


class MemoryIntegrityEvaluator(EvaluatorWorker):
    """Flags memory that drifts, collapses, or summarises something untrue (``S4``).

    Pass ``embeddings=`` to use a semantic model instead of the default
    deterministic hashing embedder. That is a module-version change in practice —
    different vectors give different drift numbers — and the idempotency key
    correctly treats it as a re-evaluation rather than an overwrite.
    """

    def __init__(
        self,
        store: EventStore,
        *,
        config: WorkerConfig | None = None,
        checkpoints: CheckpointStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        settings: MemoryIntegrityConfig | None = None,
        extractor: ClaimExtractor | None = None,
    ) -> None:
        """Create a memory worker over *store*."""
        self._settings = settings or MemoryIntegrityConfig()
        self._embeddings: EmbeddingProvider = CachedEmbeddingProvider(
            embeddings or HashingEmbeddingProvider()
        )
        self._extractor: ClaimExtractor | None = extractor
        super().__init__(
            store,
            config=config
            or WorkerConfig(
                module=MODULE,
                module_version=MODULE_VERSION,
                review_confidence_threshold=0.0,
            ),
            checkpoints=checkpoints,
        )

    @property
    def settings(self) -> MemoryIntegrityConfig:
        """This evaluator's configuration."""
        return self._settings

    @property
    def embeddings(self) -> EmbeddingProvider:
        """The embedding provider in use; its ``model_id`` is auditable."""
        return self._embeddings

    # -- analysis ---------------------------------------------------------

    async def analyze(self, session: SessionView) -> SeriesAnalysis:
        """Every memory finding for *session*, in write order.

        Deterministic given the same events and the same ``model_id``: the same
        session analysed twice produces byte-identical findings, which is what
        ``S3-T4`` requires of every module and what makes the corpus numbers
        reproducible across machines.
        """
        analysis = SeriesAnalysis(session_id=session.session_id)
        writes = [write_from_event(event) for event in session.memory_writes()]
        analysis.writes = writes
        if not writes:
            return analysis

        analysis.states = await state_series(writes, self._embeddings, self._settings.drift)
        analysis.steps = step_distances(analysis.states)
        analysis.novelty = novelty_scores(analysis.states)
        analysis.observations = len(analysis.states)

        transcript = transcript_of(session)
        analysis.findings.extend(self._drift_findings(writes, analysis.states, transcript))
        analysis.findings.extend(self._collapse_findings(writes))
        analysis.findings.extend(self._summary_findings(writes, transcript))
        # Sorted so the flag order does not depend on which check ran first.
        analysis.findings.sort(key=lambda finding: (finding.seq, finding.verdict))
        return analysis

    async def analyze_session(self, session_id: str) -> SeriesAnalysis:
        """Load and analyze a session without writing flags (diagnostics/tests).

        Parity with :meth:`sentinel.eval.provenance.ProvenanceEvaluator.analyze_session`,
        because "why did this fire?" is the first question anyone asks of either
        module and neither should require writing a row to answer it.
        """
        view = await SessionView.load(self._store, session_id, max_events=self.config.max_events)
        return await self.analyze(view)

    def _drift_findings(
        self,
        writes: Sequence[MemoryWrite],
        states: Sequence[MemoryState],
        transcript: str,
    ) -> list[MemoryFinding]:
        """One finding per write that moved the memory somewhere it should not (``S4-T5``).

        Warm-up is enforced here and not merely documented: a write before the
        configured number of predecessors cannot be compared to a baseline that
        exists, and comparing it to one that does not is how a module produces
        confident nonsense on the opening turn of every session.
        """
        config = self._settings.drift
        novelty = novelty_scores(states)
        findings: list[MemoryFinding] = []
        for index in range(config.warmup_writes, len(writes)):
            write = writes[index]
            if len(write.text.strip()) < config.min_drift_chars:
                # Too short to relocate a memory. A tiny vector is nearly
                # orthogonal to everything, so measuring one would report maximum
                # novelty for no content at all.
                continue
            # The baseline is the session's own prior novelties, so the threshold
            # adapts per agent instead of assuming one global noise floor.
            baseline = [score for score in novelty[1:index] if score >= 0.0]
            injected = looks_like_an_injection(write.text)
            score, z, is_drift, reason = detect_drift(
                novelty[index], baseline, config, injected=injected
            )
            if not is_drift:
                continue
            contradicted = write.intent is WriteIntent.SUMMARY and contradicted_summary(
                write, transcript
            )
            findings.append(
                MemoryFinding(
                    verdict=MemoryVerdict.DRIFT,
                    event_id=states[index].event_id,
                    seq=write.seq,
                    write_event_id=write.event_id,
                    detail=(
                        f"memory.write to {write.key!r} moved the memory away from "
                        f"what it held (novelty {score:.2f}; {reason})"
                    ),
                    severity=severity_for_finding(
                        MemoryVerdict.DRIFT, injected=injected, contradicted=contradicted
                    ),
                    confidence=confidence_for_finding(
                        MemoryVerdict.DRIFT, config=config, injected=injected, z_score=z
                    ),
                    observed=f"novelty={score:.3f} z={z:.1f}"
                    if score >= 0
                    else f"novelty=n/a z={z:.1f}",
                    claim_text=_quote(write.text),
                )
            )
        return findings

    def _collapse_findings(self, writes: Sequence[MemoryWrite]) -> list[MemoryFinding]:
        """A finding when the last writes stopped adding anything (``S4-T6``).

        Reported once, on the final write. The condition is a property of the
        series rather than of any one write, so attributing it to every step
        would put N flags on one condition and make the queue's count meaningless.
        """
        config = self._settings.drift
        collapsed, similarity, diversity = is_collapse(writes, config)
        if not collapsed:
            return []
        final = writes[-1]
        return [
            MemoryFinding(
                verdict=MemoryVerdict.COLLAPSE,
                event_id=final.event_id,
                seq=final.seq,
                write_event_id=final.event_id,
                detail=(
                    f"memory stopped adding anything over the last "
                    f"{config.collapse_window} writes: consecutive writes share "
                    f"{similarity:.0%} of their tokens (threshold "
                    f"{config.collapse_similarity:.0%}); their distinct-token "
                    f"ratio is {diversity:.2f}"
                ),
                severity=severity_for_finding(MemoryVerdict.COLLAPSE, injected=False),
                confidence=confidence_for_finding(
                    MemoryVerdict.COLLAPSE, config=config, similarity=similarity
                ),
                observed=f"similarity={similarity:.3f} diversity={diversity:.3f}",
                claim_text=_quote(
                    " | ".join(write.text for write in writes[-config.collapse_window :])
                ),
            )
        ]

    def _summary_findings(
        self, writes: Sequence[MemoryWrite], transcript: str
    ) -> list[MemoryFinding]:
        """Summaries that assert something the session does not contain (``S4-T9``).

        Only :attr:`WriteIntent.SUMMARY` writes are checked. A *fact* write is
        checked for drift, not for grounding: a memory full of true facts is
        exactly what a memory should be, so checking each one against the
        transcript would flag every novel thing the agent learned.
        """
        if not self._worth_checking(transcript):
            return []
        findings: list[MemoryFinding] = []
        for write in writes:
            if write.intent is not WriteIntent.SUMMARY:
                continue
            if len(write.text.strip()) < self._settings.min_summary_chars:
                continue
            problems = ungrounded_claims(write, transcript)
            if not problems:
                continue
            listed = ", ".join(
                f"{claim.text.strip()[:60]!r} ({verdict.value})" for claim, verdict in problems[:3]
            )
            findings.append(
                MemoryFinding(
                    verdict=MemoryVerdict.UNGROUNDED,
                    event_id=write.event_id,
                    seq=write.seq,
                    write_event_id=write.event_id,
                    detail=(
                        f"summary written to {write.key!r} asserts "
                        f"{len(problems)} claim(s) the session does not support: {listed}"
                    ),
                    severity=severity_for_finding(MemoryVerdict.UNGROUNDED, injected=False),
                    confidence=confidence_for_finding(
                        MemoryVerdict.UNGROUNDED, config=self._settings.drift
                    ),
                    observed=f"{len(problems)} ungrounded claims",
                    claim_text=_quote(problems[0][0].text),
                )
            )
        return findings

    def _worth_checking(self, transcript: str) -> bool:
        """Whether the transcript has enough substance to compare a summary to.

        A session with no numbers in it cannot refute a numeric claim, so every
        such summary would come back unsupported and be reported. That is a
        systematic false positive against a whole class of ordinary sessions, and
        it is cheaper to recognise the case than to argue about each flag.
        """
        if not transcript.strip():
            return False
        return len(numbers_seen(transcript)) >= self._settings.min_transcript_numbers

    # -- flag construction ------------------------------------------------

    async def evaluate(self, session: SessionView) -> list[Flag]:
        """The flags for *session*; the base class owns triggering and retries."""
        analysis = await self.analyze(session)
        return [
            self._to_flag(finding, session)
            for finding in analysis.findings
            if finding.is_actionable
        ]

    def _to_flag(self, finding: MemoryFinding, session: SessionView) -> Flag:
        """Build the flag row for *finding*.

        Evidence carries the write that caused it as the claim, and the state it
        departed from as countervailing evidence when there is one — a drift
        finding is only meaningful against the memory it moved away from, so a
        reviewer needs both ends of the arrow.
        """
        category = _category(finding)
        evidence = [
            EvidenceRef(
                event_id=finding.write_event_id or finding.event_id,
                role=EvidenceRole.CLAIM,
                seq=finding.seq,
                note=finding.detail,
            )
        ]
        previous = _previous_state(session, finding)
        if previous is not None:
            evidence.append(
                EvidenceRef(
                    event_id=previous.event_id,
                    role=EvidenceRole.COUNTERVAILANCE,
                    seq=previous.seq,
                    note="memory state the flagged write moved away from",
                )
            )
        return Flag.create(
            session_id=session.session_id,
            module=self.module,
            module_version=self.module_version,
            category=category,
            severity=finding.severity,
            confidence=finding.confidence,
            summary=f"Memory integrity: {finding.verdict.value} — {finding.detail}",
            evidence=[ref for ref in evidence if ref.event_id],
            details={
                "verdict": str(finding.verdict),
                "detail": finding.detail,
                "observed": finding.observed,
                "claim_text": finding.claim_text,
                "write_event_id": finding.write_event_id,
                "embedding_model": self._embeddings.model_id,
                "review_url": self._review_url(session.session_id),
            },
            event_id=finding.event_id,
            dedupe_key=f"{finding.event_id}:{finding.verdict}",
            created_at=_event_ts(session, finding.event_id),
            # INV-6 for the gate: a trend queues, an event can block.
            review_only=(
                finding.confidence <= self.config.review_confidence_threshold
                or category in REVIEW_ONLY_CATEGORIES
            ),
        )

    def _review_url(self, session_id: str) -> str | None:
        """The review link for this session, when a deployment configured one."""
        if not self._settings.review_url_template:
            return None
        return self._settings.review_url_template.format(session_id=session_id, module=self.module)


def _previous_state(session: SessionView, finding: MemoryFinding) -> Event | None:
    """The event carrying the memory state *before* the flagged write."""
    if finding.verdict is not MemoryVerdict.DRIFT or not finding.write_event_id:
        return None
    earlier = [event for event in session.memory_writes() if event.seq < finding.seq]
    return earlier[-1] if earlier else None


def _event_ts(session: SessionView, event_id: str) -> datetime:
    """The timestamp of *event_id*, never the wall clock (``S3-T4``).

    Every module derives flag time from the event rather than reading a clock, so
    two runs over the same session produce byte-identical rows. Falls back to the
    session's own bounds because a flag must always carry a time, and a finding
    whose event is missing is still worth recording.
    """
    for event in session.events:
        if event.event_id == event_id:
            return event.ts
    return session.started_at or session.ended_at or datetime(1970, 1, 1, tzinfo=UTC)


__all__ = [
    "CATEGORY_COLLAPSE",
    "CATEGORY_DRIFT",
    "CATEGORY_MEMORY_UNGROUNDED",
    "MAX_TRANSCRIPT_CHARS",
    "MODULE",
    "MODULE_VERSION",
    "REVIEW_ONLY_CATEGORIES",
    "MemoryIntegrityConfig",
    "MemoryIntegrityEvaluator",
    "as_text",
    "transcript_of",
    "write_from_event",
]
