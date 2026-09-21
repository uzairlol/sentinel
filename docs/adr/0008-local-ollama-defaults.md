# 0008 — Local Ollama defaults for embeddings/judge; pluggable

Date: 2026-09-21

## Status
Accepted

## Context
Two evaluator families need model calls: embedding-based (memory integrity) and
judge-based (faithfulness, spec gaming, evaluation awareness). These must be
self-hostable (ADR-0006), deterministic/reproducible, and swappable as the local
model landscape evolves.

## Decision
We define two interfaces — `EmbeddingProvider` and `JudgeProvider` — with
default implementations that call a local Ollama instance. Model versions are
pinned and recorded in flags; embedding results are cached by `(content_hash,
model_id)` for reproducibility and cost control.

## Consequences
- Positive: default path is fully local and deterministic for a pinned model.
- Positive: swapping to a different local runtime (or later, a provider) is an
  interface implementation, not a rewrite.
- Negative: evaluator quality is bounded by the locally-available model quality.
- Neutral: model upgrade = explicit change of a pinned version + backtest.

## Alternatives considered
- Hardcoded OpenAI/hosted embeddings — rejected: violates ADR-0006.
- Only rule/lexical methods (no model calls) — rejected: insufficient for
  drift/similarity judgments; retained as fast-path optimization, not replacement.
