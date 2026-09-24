"""Unit tests for the retention policy engine (``S2-T12``)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sentinel.store.retention import (
    RetentionPolicy,
    RetentionRule,
    merge_cutoffs,
)

NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
DAY = timedelta(days=1)


def test_policy_disabled_when_no_rules_and_no_default() -> None:
    policy = RetentionPolicy()
    assert not policy.enabled
    assert policy.cutoff_for("llm.request", NOW) is None


def test_default_ttl_applies_cutoff() -> None:
    policy = RetentionPolicy(default_ttl=DAY)
    assert policy.enabled
    assert policy.cutoff_for("llm.request", NOW) == NOW - DAY
    assert policy.cutoff_for("session.start", NOW) == NOW - DAY


def test_explicit_rule_overrides_default() -> None:
    policy = RetentionPolicy(
        default_ttl=DAY,
        rules=(RetentionRule(event_type="error", ttl=timedelta(days=30)),),
    )
    assert policy.cutoff_for("error", NOW) == NOW - timedelta(days=30)
    assert policy.cutoff_for("llm.request", NOW) == NOW - DAY


def test_no_ttl_means_retain_indefinitely() -> None:
    policy = RetentionPolicy(rules=(RetentionRule(event_type="error", ttl=DAY),))
    assert policy.cutoff_for("error", NOW) == NOW - DAY
    assert policy.cutoff_for("llm.request", NOW) is None


def test_merge_cutoffs_maps_types_and_default() -> None:
    policy = RetentionPolicy(
        default_ttl=DAY,
        rules=(
            RetentionRule(event_type=None, ttl=timedelta(days=7)),
            RetentionRule(event_type="error", ttl=timedelta(days=30)),
        ),
    )
    cutoffs = merge_cutoffs(policy, NOW)
    # a wildcard rule (event_type=None) becomes the "everything else" bucket
    # and overrides default_ttl for untyped events
    assert cutoffs[None] == NOW - timedelta(days=7)
    assert cutoffs["error"] == NOW - timedelta(days=30)


def test_merge_cutoffs_default_only_when_no_wildcard() -> None:
    policy = RetentionPolicy(
        default_ttl=DAY,
        rules=(RetentionRule(event_type="error", ttl=timedelta(days=30)),),
    )
    cutoffs = merge_cutoffs(policy, NOW)
    assert cutoffs == {None: NOW - DAY, "error": NOW - timedelta(days=30)}


def test_merge_cutoffs_wildcard_rule_only() -> None:
    policy = RetentionPolicy(rules=(RetentionRule(event_type=None, ttl=timedelta(days=7)),))
    cutoffs = merge_cutoffs(policy, NOW)
    assert cutoffs == {None: NOW - timedelta(days=7)}
