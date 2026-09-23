## Summary

<!-- One or two sentences: what this does and why. -->

## Linked work

- Task: `S1-T1` (see [SENTINEL_TDD.md](../../docs/design/SENTINEL_TDD.md))
- Closes #<!-- issue number -->

## What changed

<!-- Bullet list of the meaningful changes. -->

## Tests

- [ ] Happy path added/updated
- [ ] Failure/adversarial path added/updated
- [ ] Property test added (if a core algorithm changed)
- [ ] `uv run pytest` green
- [ ] `uv run ruff check .` and `uv run ruff format --check .` clean
- [ ] `uv run mypy` clean (strict)

## Docs & decisions

- [ ] Public behavior changed → docs updated
- [ ] Design decision changed → ADR added/updated
- [ ] Schema changed → migration included

## Release notes

- [ ] Breaking change? If yes: semver note + deprecation plan.
- [ ] Security impact: none / <!-- describe review performed for inputs, secrets, SQL, network -->

## Rollback

<!-- How to revert this safely (e.g. version pin, revert commit, config flag). -->
