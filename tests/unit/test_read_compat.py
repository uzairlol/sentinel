"""Unit tests for schema-0.1 -> 0.2 ref normalisation (``S2-T6``)."""

from __future__ import annotations

import pytest

from sentinel.models.events import RefKind, RefLink
from sentinel.store.read_compat import materialize_refs


def test_none_and_empty_become_empty_links() -> None:
    assert materialize_refs(None) == []
    assert materialize_refs([]) == []


def test_flat_01_strings_become_parent_links() -> None:
    links = materialize_refs(["01ARZ3NDEKTSV4RRFFQ69G5FAV", "01ARZ3NDEKTSV4RRFFQ69G5FB0"])
    assert links == [
        RefLink(event_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", kind=RefKind.PARENT),
        RefLink(event_id="01ARZ3NDEKTSV4RRFFQ69G5FB0", kind=RefKind.PARENT),
    ]


def test_typed_02_dicts_preserve_kind() -> None:
    links = materialize_refs([{"event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV", "kind": "caused_by"}])
    assert links == [RefLink(event_id="01ARZ3NDEKTSV4RRFFQ69G5FAV", kind=RefKind.CAUSED_BY)]


@pytest.mark.parametrize(
    "raw",
    [
        [{"event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV", "kind": "bogus"}],
        [{"kind": "parent"}],
        [{"event_id": 5}],
        [42],
    ],
)
def test_malformed_entries_raise_instead_of_coercing(raw: list[object]) -> None:
    with pytest.raises(ValueError, match=r"malformed ref entry|is not a valid RefKind"):
        materialize_refs(raw)
