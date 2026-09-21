# 0009 — Small stable public API surface; SemVer policy

Date: 2026-09-21

## Status
Accepted

## Context
This is a library that hosts embed in their agent code. Their code compiles
against our public API, and the product's credibility depends on versioning that
they can trust.

## Decision
The public API surface is deliberately small and lives under `sentinel` (package
root) and `sentinel.instrument`. Everything else is private (leading-underscore
modules, or non-exported symbols). Rules:

- Additions are minor bumps.
- Removals/renames require a deprecation shim emitting `DeprecationWarning` for
  at least one minor cycle.
- Breaking changes are major bumps per SemVer.
- `schema_version` bumps for event/flag schemas follow ADR-0007.

## Consequences
- Positive: consumers can upgrade predictably; accidental API is minimized.
- Positive: the CLI (`sentinel._cli`) is explicitly not public, so UX can evolve.
- Negative: new capabilities sometimes need an explicit API-design step.
- Neutral: enforced by keeping `__all__` and public-module enumerations explicit.

## Alternatives considered
- Large, "everything exported" API — rejected: locks in mistakes.
- Feature-flagged full re-write per module — rejected: unwieldy.
