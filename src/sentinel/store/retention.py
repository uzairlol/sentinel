"""Retention policy for the event store (``S2-T12``).

Events are append-only (INV-2), but storage is finite: a deployment must be
able to expire old events while keeping the deletion *provable*. Deletion flows
through three safeguards:

* **Per-type TTL** — each event type may carry a different time-to-live; the
  ``default_ttl`` applies to types without a rule.
* **Legal hold** — a session on hold is never pruned no matter how old it is.
* **Tombstones** — nothing is silently deleted. Every pruned event leaves a
  row in the ``tombstones`` table recording *what* was removed, *when*, and a
  digest of the removed payload.

This module holds the pure policy logic; the backends apply it
(:meth:`~sentinel.store.protocol.EventStore.prune`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class RetentionRule:
    """A TTL applied to one specific event type (``None`` = all types)."""

    event_type: str | None
    ttl: timedelta


@dataclass(frozen=True)
class RetentionPolicy:
    """Configuration that decides when an event may be pruned."""

    #: TTL applied to event types without an explicit :attr:`rules` entry.
    default_ttl: timedelta | None = None
    rules: tuple[RetentionRule, ...] = ()
    #: Session ids that must outlive every TTL (compliance/legal hold).
    legal_hold_sessions: frozenset[str] = field(default_factory=frozenset)

    @property
    def enabled(self) -> bool:
        """Whether any pruning is configured (a constrained default TTL)."""
        return self.default_ttl is not None or bool(self.rules)

    def cutoff_for(self, event_type: str, now: datetime | None = None) -> datetime | None:
        """The cutoff before which events of *event_type* may be pruned.

        Returns ``None`` when the type has no TTL and there is no default, i.e.
        the type is retained indefinitely.
        """
        now = now if now is not None else datetime.now(UTC)
        ttl = None
        for rule in self.rules:
            if rule.event_type == event_type:
                ttl = rule.ttl
                break
            if rule.event_type is None and ttl is None:
                ttl = rule.ttl
        if ttl is None:
            ttl = self.default_ttl
        if ttl is None:
            return None
        return now - ttl


@dataclass
class PruneReport:
    """Outcome of one retention pass."""

    #: The cutoff instant the pass used (``None`` when the policy was disabled).
    executed_at: datetime
    pruned_events: int = 0
    tombstoned: int = 0
    locked_sessions: int = 0
    retained_evidence: int = 0

    @property
    def total_affected(self) -> int:
        """Total events removed by the pass (tombstoned == pruned by design)."""
        return self.pruned_events


def merge_cutoffs(
    policy: RetentionPolicy, now: datetime | None = None
) -> dict[str | None, datetime]:
    """Map event-type-or-default to its pruning cutoff, without leading Nones.

    ``None`` keys mean "the default cutoff, applies to untyped events". Keys
    resolve longest-prefix-first by explicit rule in :meth:`RetentionPolicy
    .cutoff_for`; here we only need the distinct cutoff instants to run the SQL.
    """
    now = now if now is not None else datetime.now(UTC)
    cutoffs: dict[str | None, datetime] = {}
    for rule in policy.rules:
        cutoff = policy.cutoff_for(rule.event_type or "", now)
        if cutoff is not None and rule.event_type not in cutoffs:
            cutoffs[rule.event_type] = cutoff
    default = policy.cutoff_for("", now)
    if default is not None:
        cutoffs[None] = default
    return cutoffs
