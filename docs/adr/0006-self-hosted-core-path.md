# 0006 — Self-hosted core path; opt-in external models

Date: 2026-09-21

## Status
Accepted

## Context
Organizations that need safety instrumentation are often the least willing to
route full agent execution traces through a third-party cloud; data residency is
frequently a contractual requirement. The cleanest way to avoid an accidental
third-party dependency is to make "no external egress" the default path.

## Decision
The core capture → evaluate → gate pipeline runs entirely self-hosted with no
mandatory third-party network dependency. Embedding and judge-model calls default
to a local Ollama instance; any external model/judge integration is an explicit,
opt-in configuration that is clearly surfaced in logs and documentation (INV-5).

## Consequences
- Positive: sells to data-residency and air-gapped requirements from day one.
- Positive: no surprise egress of potentially sensitive traces.
- Negative: default deployments need a local inference runtime (Ollama) for the
  LLM-dependent evaluators.
- Neutral: later managed offerings must preserve this self-hosted as the
  open-core promise (see risk register R9-adjacent in `SENTINEL_TDD.md`).

## Alternatives considered
- SaaS-first core — rejected: contradicts the target market's constraints.
- Own hosted inference — deferred: operational cost; pluggable instead.
