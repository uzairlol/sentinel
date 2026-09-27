"""The universal evaluator ``Flag`` schema (sprint ``S3-T1``, docs/adr/0012).

Every detection module in Sentinel — provenance (``S3``), memory integrity
(``S4``), reasoning faithfulness (``S5``), specification gaming (``S6``),
evaluation awareness (``S8``) — reports through this one shape, so the gate
(``S7``) and the review UI can consume flags without knowing which module
raised them.

Three properties are load-bearing:

* **Structural evidence (INV-3).** A flag never states a conclusion without
  linking, by event id, the events that produced it. :class:`EvidenceRef`
  entries must name real events; ``evidence`` is non-empty by validation.
* **Deterministic identity.** :func:`flag_identity` derives the ``flag_id``
  from ``(session_id, module, module_version, category, dedupe_key)``, so
  re-running an evaluator over the same session with the same module version
  yields the *same* flag id — idempotent writes, reproducible FP/FN
  (``S3-T4``).
* **Deterministic timestamps.** ``created_at`` is derived from the evidence
  events by the module, never from the wall clock, so two runs over the same
  events produce byte-identical flags.

The category taxonomy is *open* (a validated slug) but discoverable through
:func:`known_categories`; later modules register their categories at import time
rather than requiring an enum change here.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ulid import ULID

#: The flag schema version emitted by this package (docs/adr/0007 analogue).
FLAG_SCHEMA_VERSION = "0.1"

#: Category slugs must be ``lower_snake_case``; the taxonomy is open so later
#: modules can add categories without a schema migration.
_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

#: Summary length cap. Flags are read in a review queue, not in bulk.
MAX_SUMMARY_CHARS = 2_000

#: Upper bound on evidence links per flag — enough for a claim + its sources.
MAX_EVIDENCE = 32


class Severity(StrEnum):
    """How much a flag should worry an operator (ordered low → high)."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric rank (``info`` = 0 … ``critical`` = 4) for comparisons."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class Adjudication(StrEnum):
    """Human-review state of a flag (``S7`` owns the review UI)."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class EvidenceRole(StrEnum):
    """Why one event is in a flag's evidence list."""

    #: The event carrying the claim/assertion that triggered the flag.
    CLAIM = "claim"
    #: The event the claim was (or should have been) grounded in.
    EVIDENCE = "evidence"
    #: Surrounding context a reviewer needs to judge the claim.
    CONTEXT = "context"
    #: The observed value that contradicts the claim.
    COUNTERVAILANCE = "countervailance"


#: Registered categories, populated by the modules that implement them.
_KNOWN_CATEGORIES: set[str] = set()


def register_category(*categories: str) -> None:
    """Register *categories* so tools can enumerate the live taxonomy.

    Registration is advisory (categories validate as slugs regardless); it
    exists so ``sentinel flags --categories`` and the docs cannot drift from
    the modules that actually ship.
    """
    for category in categories:
        if not _CATEGORY_RE.match(category):
            raise ValueError(f"category must be lower_snake_case, got {category!r}")
        _KNOWN_CATEGORIES.add(category)


def known_categories() -> frozenset[str]:
    """Every category registered so far, sorted."""
    return frozenset(_KNOWN_CATEGORIES)


def flag_identity(
    *,
    session_id: str,
    module: str,
    module_version: str,
    category: str,
    dedupe_key: str,
) -> str:
    """Return the deterministic ``flag_id`` for one finding (ADR-0012).

    The id is a ULID (ADR-0010) derived from the SHA-256 digest of the
    idempotency tuple, so it is a valid, sortable-looking 26-character id that
    is *purely a function of its inputs*: two evaluator runs over the same
    session produce the same id and the second write is a no-op.
    """
    payload = "\x1f".join((session_id, module, module_version, category, dedupe_key))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return str(ULID.from_bytes(digest[:16]))


class EvidenceRef(BaseModel):
    """One event a flag points at, with the role it plays in the finding."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    role: EvidenceRole
    #: The event's ``seq`` at evaluation time, so a reviewer can find it fast.
    seq: int | None = None
    note: str | None = None

    @field_validator("event_id")
    @classmethod
    def _require_ulid_event_id(cls, value: str) -> str:
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"evidence event_id must be a valid ULID, got {value!r}") from exc
        return value

    @field_validator("seq")
    @classmethod
    def _require_nonnegative_seq(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"evidence seq must be >= 0, got {value}")
        return value

    def to_row(self) -> dict[str, Any]:
        """Render the evidence entry for the store's JSONB column."""
        return {
            "event_id": self.event_id,
            "role": self.role.value,
            "seq": self.seq,
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EvidenceRef:
        """Rebuild an evidence entry from a stored JSONB row."""
        return cls.model_validate(row)


class Flag(BaseModel):
    """One evaluator finding about one session.

    Example::

        flag = Flag(
            session_id=session_id,
            module="provenance",
            module_version="0.1.0",
            category="ungrounded_claim",
            severity=Severity.HIGH,
            confidence=0.93,
            summary="Claim cites 'web_search' but no such tool call exists",
            evidence=[EvidenceRef(event_id=event.event_id, role=EvidenceRole.CLAIM, seq=7)],
            created_at=event.ts,
        )
    """

    model_config = ConfigDict(extra="forbid")

    flag_id: str
    session_id: str
    module: str
    module_version: str
    category: str
    severity: Severity = Severity.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=MAX_SUMMARY_CHARS)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=MAX_EVIDENCE)
    created_at: datetime
    #: The primary event the flag is about (the claim event, usually).
    event_id: str | None = None
    #: ``review_only`` flags never gate; they are routed to a human (``S5-T10``).
    review_only: bool = False
    #: Module-specific structured detail (claimed vs observed values, thresholds).
    details: dict[str, Any] = Field(default_factory=dict)
    adjudication: Adjudication = Adjudication.PENDING
    adjudicated_by: str | None = None
    adjudicated_at: datetime | None = None
    auto_resolved: bool | None = None
    schema_version: str = FLAG_SCHEMA_VERSION

    @field_validator("flag_id")
    @classmethod
    def _require_ulid_flag_id(cls, value: str) -> str:
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flag_id must be a valid ULID, got {value!r}") from exc
        return value

    @field_validator("event_id")
    @classmethod
    def _require_ulid_event_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            ULID.from_str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flag event_id must be a valid ULID, got {value!r}") from exc
        return value

    @field_validator("created_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if value.utcoffset() != timedelta(0):
            raise ValueError("created_at must be normalized to UTC (offset 0)")
        return value

    @field_validator("category")
    @classmethod
    def _require_slug(cls, value: str) -> str:
        if not _CATEGORY_RE.match(value):
            raise ValueError(f"category must be lower_snake_case, got {value!r}")
        return value

    @field_validator("summary")
    @classmethod
    def _require_nonempty_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be blank")
        return value

    @field_validator("evidence")
    @classmethod
    def _require_unique_evidence(cls, value: list[EvidenceRef]) -> list[EvidenceRef]:
        """Reject the same event twice, unless a second role is asserted.

        Two roles for one event is legitimate (a claim *and* its
        countervailing value live in the same ``tool.result``); two identical
        rows are not.
        """
        seen: set[tuple[str, EvidenceRole]] = set()
        for ref in value:
            key = (ref.event_id, ref.role)
            if key in seen:
                raise ValueError(f"duplicate evidence entry {key[0]}/{key[1].value}")
            seen.add(key)
        return value

    @model_validator(mode="after")
    def _require_adjudication_provenance(self) -> Flag:
        """A decision without an author, or an author without a decision, is a lie.

        The row has to be able to answer "who ruled on this, and when" without a
        second lookup, because that answer is the point of keeping the flag
        after the log it came from has been pruned.
        """
        decided = self.adjudication is not Adjudication.PENDING
        attributed = self.adjudicated_by is not None or self.adjudicated_at is not None
        if decided and not attributed:
            raise ValueError(
                f"adjudication={self.adjudication.value!r} requires adjudicated_by and "
                "adjudicated_at"
            )
        if attributed and not decided:
            raise ValueError("adjudicated_by/adjudicated_at require a non-pending adjudication")
        return self

    @property
    def evidence_event_ids(self) -> list[str]:
        """The referenced event ids, in evidence order."""
        return [ref.event_id for ref in self.evidence]

    def has_role(self, role: EvidenceRole) -> bool:
        """Whether any evidence entry carries *role*."""
        return any(ref.role is role for ref in self.evidence)

    def for_review(self) -> bool:
        """Whether this flag must be reviewed by a human before it can gate.

        True for explicitly ``review_only`` flags and for any flag whose
        confidence is below the module's review threshold (wired in ``S7``).
        """
        return self.review_only

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        module: str,
        module_version: str,
        category: str,
        confidence: float,
        summary: str,
        evidence: list[EvidenceRef],
        created_at: datetime,
        dedupe_key: str,
        severity: Severity = Severity.MEDIUM,
        event_id: str | None = None,
        review_only: bool = False,
        details: dict[str, Any] | None = None,
    ) -> Flag:
        """Build a flag with a deterministic ``flag_id`` (ADR-0012).

        ``dedupe_key`` is the finding's identity *within* the
        (session, module, version, category) tuple — typically the claiming
        event id plus the claim's index or digest. Two identical findings in
        the same category collapse to one flag; two genuinely different claims
        do not.
        """
        return cls(
            flag_id=flag_identity(
                session_id=session_id,
                module=module,
                module_version=module_version,
                category=category,
                dedupe_key=dedupe_key,
            ),
            session_id=session_id,
            module=module,
            module_version=module_version,
            category=category,
            severity=severity,
            confidence=confidence,
            summary=summary,
            evidence=evidence,
            created_at=created_at,
            event_id=event_id,
            review_only=review_only,
            details=details or {},
        )


def evidence_digest(evidence: list[EvidenceRef]) -> str:
    """Canonical digest of an evidence list, for identity/debugging."""
    payload = "|".join(
        f"{ref.role.value}:{ref.event_id}:{ref.seq if ref.seq is not None else ''}"
        for ref in evidence
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalize_utc(value: datetime) -> datetime:
    """Return *value* as a UTC datetime, assuming naive input is already UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "FLAG_SCHEMA_VERSION",
    "MAX_EVIDENCE",
    "MAX_SUMMARY_CHARS",
    "Adjudication",
    "EvidenceRef",
    "EvidenceRole",
    "Flag",
    "Severity",
    "evidence_digest",
    "flag_identity",
    "known_categories",
    "normalize_utc",
    "register_category",
]
