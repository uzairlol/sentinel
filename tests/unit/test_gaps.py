"""Unit tests for the seq-gap detector (``S2-T13``)."""

from __future__ import annotations

import pytest

from sentinel.store.errors import RefIntegrityError
from sentinel.store.gaps import SeqGap, assert_no_gaps, seq_gaps


def test_gaps_empty_stream() -> None:
    assert seq_gaps([]) == []


def test_gaps_contiguous_stream() -> None:
    assert seq_gaps([0, 1, 2, 3]) == []


def test_gaps_identifies_one_missing_run() -> None:
    assert seq_gaps([0, 1, 2, 5, 6]) == [SeqGap(3, 4)]


def test_gaps_identifies_leading_missing_run() -> None:
    assert seq_gaps([2, 3]) == [SeqGap(0, 1)]


def test_gaps_identifies_multiple_runs_and_counts() -> None:
    gaps = seq_gaps([0, 3, 4, 9])
    assert gaps == [SeqGap(1, 2), SeqGap(5, 8)]
    assert gaps[1].count == 4


def test_seq_gap_validation() -> None:
    with pytest.raises(ValueError, match="first <= last"):
        SeqGap(3, 1)
    with pytest.raises(ValueError, match="negative seq"):
        SeqGap(-1, 2)


def test_assert_no_gaps_passes_for_contiguous_run() -> None:
    assert_no_gaps([0, 1, 2, 3])
    assert_no_gaps([])


def test_assert_no_gaps_raises_on_missing_seq() -> None:
    with pytest.raises(RefIntegrityError, match="expected 1, found 3"):
        assert_no_gaps([0, 3])
