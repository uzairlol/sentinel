# 0001 — Python 3.12+ async-first SDK

Date: 2026-09-21

## Status
Accepted

## Context
Sentinel instruments autonomous LLM agents. The agent ecosystem (LangChain,
LangGraph, Ollama clients, embedding libraries) is Python-first. The project is
developed primarily by one person and must ship a detect → flag → gate product
with measurable reliability. A statically-typed, lower-level core would offer
better raw performance but at substantial solo-development cost and FFI
complexity.

## Decision
The SDK is written in Python (>= 3.12) with an asyncio-first design:

- The capture path is async and non-blocking so it never stalls a host agent.
- Type safety and correctness are enforced with `mypy --strict`.
- Performance-sensitive paths are optimized in Python (batching, prepared
  statements, connection pooling) and re-evaluated against hard budgets
  (`SENTINEL_TDD.md` §2.7). If budgets are unmeetable in Python at GA, revisit
  this ADR.

## Consequences
- Positive: fastest possible velocity for a solo developer; frictionless
  integration with the target frameworks.
- Positive: the self-hosted, local-first design (ADR-0006) removes the network
  overhead argument for a systems language.
- Negative: raw throughput/latency ceilings are lower than Rust/Go; must be
  proven against futures and marketing claims.
- Neutral: some consumers may prefer a Perl-free, GC'd runtime; accepted.

## Alternatives considered
- Rust/Go core with Python bindings — higher ceiling, much higher cost; deferred.
- TypeScript/Node — poor fit for the Python agent tooling we instrument.
