# Example — memory integrity, end to end

This renders [`examples/memory_integrity.py`](https://github.com/uzairlol/sentinel/blob/main/examples/memory_integrity.py)
from the repository. The script is included rather than linked because MkDocs
resolves relative links only within `docs_dir`, so a link from a page to a file
outside it is reported as an unresolved target — and under `--strict` one
unresolved link fails the whole build.

Run it with:

```
uv run python examples/memory_integrity.py
```

The walkthrough shows one agent session producing two flags: `memory_drift` when
an instruction that arrived inside a tool result is written into memory verbatim,
and `memory_ungrounded` when the closing summary asserts an escalation the session
never had. Both come from the same evaluator over the same session.

Related: [Memory integrity](../modules/memory-integrity.md).

## Source

```python
--8<-- "examples/memory_integrity.py"
```
