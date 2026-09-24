"""Sequence gap detection for capture sessions (``S2-T13``).

A session's ``seq`` numbers must form a contiguous ``0..N`` run — a gap means
events were lost, dropped, or mis-sequenced, and any evaluator built on that
session would be reasoning over incomplete data. ``detect_gaps`` is a pure
function over a stream of ``seq`` values so every backend (SQLite, Postgres)
reports gaps identically.
"""

from __future__ import annotations

from collections.abc import Iterable

from sentinel.store.errors import RefIntegrityError


class SeqGap:
    """A contiguous run of missing ``seq`` numbers in one session."""

    __slots__ = ("first", "last")

    def __init__(self, first: int, last: int) -> None:
        """Record the missing run ``[first, last]`` (inclusive)."""
        if first > last:
            raise ValueError(f"gap must satisfy first <= last, got {first} > {last}")
        if first < 0:
            raise ValueError(f"gap must not include negative seq, got {first}")
        self.first = first
        self.last = last

    @property
    def count(self) -> int:
        """The number of missing events in this run."""
        return self.last - self.first + 1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Stable debug rendering of the missing run."""
        return f"SeqGap({self.first}..{self.last} missing={self.count})"

    def __eq__(self, other: object) -> bool:
        """Value equality over the ``[first, last]`` inclusive bounds."""
        if not isinstance(other, SeqGap):
            return NotImplemented
        return self.first == other.first and self.last == other.last


def seq_gaps(seqs: Iterable[int]) -> list[SeqGap]:
    """Return the contiguous runs of missing ``seq`` values.

    Expects *seqs* in ascending order. Because every boundary between two
    consecutive present values with ``diff > 1`` delimits exactly one missing
    run, a single pass over the stream is sufficient.
    """
    gaps: list[SeqGap] = []
    previous: int | None = None
    for seq in seqs:
        if seq < 0:
            raise ValueError(f"seq must be >= 0, got {seq}")
        if previous is None:
            # a session's seq run starts at 0; rows starting higher mean the
            # prefix was lost or never captured
            if seq > 0:
                gaps.append(SeqGap(0, seq - 1))
        elif seq - previous > 1:
            gaps.append(SeqGap(previous + 1, seq - 1))
        previous = seq
    return gaps


def assert_no_gaps(seqs: Iterable[int]) -> None:
    """Raise :class:`RefIntegrityError` unless *seqs* is a contiguous run.

    More precisely, fails when the first seq is not ``0`` or when the stream
    skips any integer. Used by the store health / losslessness checks.
    """
    first_seen = False
    expected = 0
    for seq in seqs:
        if not first_seen:
            first_seen = True
            expected = seq
        if seq != expected:
            raise RefIntegrityError(f"session seq gap: expected {expected}, found {seq}")
        expected += 1
