# 0005 — Minimal, auditable gating rules engine

Date: 2026-09-21

## Status
Accepted

## Context
A safety product's gating logic must be verifiable by the security team of any
adopting organization in an afternoon. Sophisticated detection modules are
worthless if the enforcement layer is a black box or an afterthought.

## Decision
The policy/gating layer is a thin rules engine: declarative rules over flag
predicates (severity, confidence, category) and organization-defined stakes
thresholds (dollar amount, restricted action classes), each rule mapping to
`proceed` / `hold` / `block`, with explicit priority and an explicit default
behavior. The engine core is kept intentionally small (a few hundred lines),
versioned, linted against ambiguous/conflicting rules, and documented in plain
language. INV-4.

## Consequences
- Positive: auditable by construction; trivial to reason about failure modes.
- Positive: every threshold is per-deployment configuration, never hardcoded.
- Negative: expressive power is bounded; complex policies require composing rules.
- Neutral: configuration is JSON/YAML + schema, versioned and diffable.

## Alternatives considered
- A full rules DSL / policy engine framework — rejected: auditability cost.
- Hardcoded gating in module code — rejected: not configurable, not auditable.
