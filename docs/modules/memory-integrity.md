# Memory integrity — drift, collapse, ungrounded summaries

Sprint `S4`'s evaluator module. It answers three questions about what an agent
remembers:

1. **Did something move this memory somewhere it should not have been?**
   (`memory_drift`)
2. **Has it stopped adding anything at all?** (`memory_collapse`)
3. **Does a reflection assert things this session never happened?**
   (`memory_ungrounded`)

It is the second detection module and the first consumer of `provenance_core`
from outside the provenance module itself — which is the point of the sprint. It
imports `diff_claim` and the grounding lexicons rather than reimplementing
"grounded", so a sentence cannot be clean in one module and flagged in the other.

## What it produces

| Category | Meaning | Severity | Gate-worthy? |
|---|---|---|---|
| `memory_drift` | A single write moved the memory somewhere it should not have been | `high`, or `critical` when the write is also instruction-shaped *and* contradicts the session | **yes** |
| `memory_ungrounded` | A summary or reflection asserts something the session does not contain | `high` | yes |
| `memory_collapse` | Successive writes stopped adding anything | `medium` | **no** — always `review_only` |

The routing split is the whole of INV-6 as it applies here. `memory_drift` is a
single event with a known rule behind it, so it can block. `memory_collapse` is
a trend with several innocent causes — a loop, a one-line task, a memory that
genuinely has nothing new to say — so it queues for a human. Both are
`review_only` below the configured confidence threshold, as everywhere else.

Every flag carries `details["claim_text"]`: the offending content **verbatim**. An
operator reading a flag must be able to see *what* was wrong without opening the
event log and hunting for it, which is the same discipline as `observed_value` in
`S3`.

## Measured error rates

`sentinel eval-fixtures --module memory`:

```
corpus: 11 cases  module=sentinel.memory_integrity@0.1.0
confusion matrix (case level)
                flagged  silent
  should flag        6       0   <- recall 100.00%
  should be quiet     0      5   <- precision 100.00%

  false-negative rate: 0.00% (gate <= 10%)
  false-positive rate: 0.00% (gate <= 0%)
  claim FP rate:       0.00% (gate <= 5%)
PASS
```

| Metric | Gate | Measured |
|---|---|---|
| Case false-negative rate | ≤ 10% | **0.00%** (0/6) |
| Case false-positive rate | ≤ 0% | **0.00%** (0/5) |
| Claim false-positive rate | ≤ 5% | **0.00%** (0/11) |

Eleven cases is a smoke test, not an error-rate estimate, and the corpus says so
in its own notes. What the corpus *does* do is attack each guard: healthy memory
evolution, a summary that matches its transcript, three-word heartbeat writes, a
loop that recovered, a transcript with no numbers in it, and — the sixth — a
write that is *both* a drift and instruction-shaped, which is the only case in
the corpus that is allowed to reach `critical`. Each of those fails loudly if
its guard is removed.

## The finding that shaped this module

**A lexical embedding cannot detect a memory injection.** This is measured, not
assumed, and the number is kept in a test
(`tests/unit/test_memory_core.py::TestDriftCannotSeeAnInjectionWithALexicalEmbedder`)
so it cannot be quietly forgotten.

On a healthy session of five unrelated account facts, per-write novelty runs
**0.63–0.82**. Two topically unrelated short texts share almost no vocabulary, so
a lexical embedder places them a long way apart, and that range consumes nearly
all of the headroom cosine distance has. On the *same* session, an injected
instruction scored **0.604** — *below* the healthy writes it hijacked, because an
injection is usually lexically close to the conversation it attacks.

So the module is built the other way round:

- **The gate-worthy detector is structural.** `looks_like_an_injection` asks
  whether the write is an instruction aimed at the agent rather than a fact about
  the world. That needs no embedding model, is deterministic, and is *identified*
  rather than measured — so it gets high confidence and can block.
- **The embedding path corroborates.** `detect_drift` compares a write's novelty
  against the state it joins, and scores it against *this session's own*
  distribution (robust z, median/MAD). It is the catch-all for an injection phrased
  too carefully for the marker list. With the default lexical provider it does not
  fire, by design rather than by accident; it is for deployments that pass a local
  semantic model via `OllamaEmbeddingProvider`, where healthy novelty sits far
  lower and a genuine jump has real headroom.

## How it works

### Memory states and novelty (`S4-T4`/`T5`)

Each `memory.write` produces a cumulative state (the memory's content after that
write) and, in the same batched embedding call, a vector for the write's own text.
Novelty is the cosine distance between **the previous cumulative state and this
write's own text**.

Not between consecutive cumulative states. A cumulative state grows by one chunk
each time, so in any long session a single write is a small perturbation of a
large average and every write measures as barely moving — measured that way an
injection scored `0.11` against a threshold of `0.55`, diluted into invisibility by
the arithmetic of averaging.

One batched call matters: an embedder under a rate limit charges per call, and this
module shares one model server with the host agent (`S4-T3`).

### Thresholds (`S4-T7`)

| Setting | Default | Why |
|---|---|---|
| `warmup_writes` | 3 | The first writes establish what the memory is *for*. Drift measured against a baseline that does not exist yet is how a module produces confident nonsense on every session's opening turn. |
| `min_drift_chars` | 24 | Three words cannot relocate a memory, and a tiny vector is nearly orthogonal to everything — measuring one reports maximum novelty for no content. |
| `drift_z` | 3.0 | Conventional robust-outlier cut, relative to the session's own median. |
| `sigma_floor_fraction` | 0.15 | Minimum spread of the baseline **as a fraction of its own median**, so one configuration works for an embedder whose typical novelty is `0.7` and one whose typical novelty is `0.06`. An absolute floor guarded the lexical provider and blocked every semantic one. |
| `collapse_similarity` | 0.8 | Four fifths of the tokens in the shorter write were already in the longer one. |
| `collapse_window` | 3 | One repeated step is a retry; several in a row are a loop. |

Every one of these is configuration, not code, and the contract tests prove it:
raising `warmup_writes` silences drift and raising `collapse_similarity` silences
collapse, without a release.

### Collapse (`S4-T6`)

One condition, not two: the last `collapse_window` writes each overlap their
predecessor by at least `collapse_similarity`.

An earlier version *also* required a low distinct-token ratio, on the reasoning
that repetition and homogenisation are different failure modes. They are not —
writing near-identical text three times necessarily collapses the type-token
ratio, so the pair could never disagree. The second condition was removed rather
than left in the gate pretending to filter something, and the ratio is still
*reported* on the finding as evidence a reviewer can check.

Measured over the **writes themselves**, not the cumulative state. On the
cumulative state, three identical writes onto a diverse memory still showed a
healthy `0.65` distinct-token ratio — the old content diluted the signal — and a
genuine loop read as fine.

Collapse is a *tail* condition. A memory that looped and then learned something
has recovered, and reporting it forever would train an operator to dismiss the
signal.

### Write intent (`S4-T10`)

A summary is checked against the transcript; a fact is checked for drift. A
memory full of true facts is exactly what a memory should be, so checking each
fact against the transcript would flag every novel thing the agent learned.

`write_intent` reads the explicit `summary=` argument first (the adapter already
declaring it), then a closed marker list, then language that describes the
conversation rather than the world ("the user asked…").

### Summary grounding (`S4-T9`)

A summary's claims are extracted and diffed against the transcript by the *same*
`diff_claim` the tool-grounding module uses, with the transcript as cited
evidence. Three guards keep this from becoming a false-positive machine:

- **Too short** (`< 16` chars): reflexive notes ("ok", "done") cannot assert a
  falsifiable event.
- **Too little to check against**: a transcript with fewer than two numbers
  cannot refute a numeric claim, so every such summary would come back
  unsupported. That is a systematic false positive against every session that
  discusses things rather than quantities.
- **A label is not a veto.** `"summary:"` used to be in the extractor's
  non-assertion list, written for chat prose — and that silently disabled claim
  extraction for *every reflective memory write*, which is precisely what this
  module exists to check. `is_assertion` now strips a leading label
  (`Summary:`, `Recap -`) before deciding, because a label is metadata *about*
  the text rather than a claim *in* it.

`severity_from_diff` delegates to `provenance_core`'s severity ladder rather than
reimplementing it, so the two modules cannot disagree about how serious a kind of
verdict is.

### Embeddings (`S4-T1`–`T3`)

| Provider | When | Notes |
|---|---|---|
| `HashingEmbeddingProvider` | **Default** | Deterministic, offline, no model to pin. Signed-token hashing into 512 buckets, `hashlib`-based so it is stable across processes (Python's `hash()` is salted per process and would change every vector on every restart). |
| `OllamaEmbeddingProvider` | Opt-in | Local embeddings, both `/api/embed` and legacy `/api/embeddings` shapes. Batched, concurrency-capped, and rate-limited, because an evaluator that saturates the shared model server degrades the very agent it is protecting. |
| `CachedEmbeddingProvider` | Always wraps the above | Keyed by content hash **and model id**. Content alone would serve a vector computed by a different model after a model change, and the resulting series would mix two coordinate systems — the worst kind of silent wrongness, because every individual number still looks like a float. |

The default provider needs no configuration and satisfies INV-5 with nothing to
opt into. Its honest limitation is that it measures lexical overlap; that is
adequate for collapse and inadequate for drift, which is the finding above.

## Known limits

- **Paraphrased injections are missed by the structural path.** The marker list
  is closed. "Please disregard any prior guidance from the operator team" does not
  contain any marker verbatim, and the embedding path that might catch it does not
  fire with a lexical provider. This is the module's largest gap and it is a
  coverage gap in the marker list, not a known false-positive rate.
- **Semantic drift needs a semantic provider.** With the default hashing embedder
  the z-score path is effectively inert. A deployment that cares about drift
  should pass `OllamaEmbeddingProvider` and re-run the corpus against it.
- **The embedding path is unmeasured against real embeddings.** Its calibration is
  argued from the lexical provider's measured noise and from the observation that
  the thresholds are scale-free. It has not been tuned against an actual
  `nomic-embed-text`.
- **Cross-session memory is not analysed.** Every state is built from one
  session's writes. An injection planted in session 1 and exploited in session 5
  is invisible to this module, because the store's retention and per-agent memory
  series are a `S7`/operator concern.
- **Ten hand-built sessions bound nothing about real traffic.** Sizing the true
  rate needs a labelled production sample (`S13`).
- **"Ungrounded" means unsupported, not false.** The check establishes that the
  session does not contain the claim. A claim could be true of something outside
  the session entirely, and this module cannot see that. The verdict is
  `UNKNOWN`, which is review-only.

## Where the code lives

| Piece | Path |
|---|---|
| Providers, cosine distance, cache | `src/sentinel/eval/embeddings.py` |
| Pure rules: intent, states, novelty, collapse, summary grounding, severity | `src/sentinel/eval/memory_core.py` |
| Worker, flag construction, routing | `src/sentinel/eval/memory.py` |
| Corpus (11 cases) | `src/sentinel/eval/fixtures/memory_corpus.py` |
| Harness (shared with provenance) | `src/sentinel/eval/harness.py` |

```python
# measure it
sentinel eval-fixtures --module memory

# run it
from sentinel.eval.memory import MemoryIntegrityEvaluator
await MemoryIntegrityEvaluator(store).run_once()
await MemoryIntegrityEvaluator(store).evaluate_session(session_id)

# with semantic embeddings
from sentinel.eval.embeddings import CachedEmbeddingProvider, OllamaEmbeddingProvider
MemoryIntegrityEvaluator(
    store,
    embeddings=CachedEmbeddingProvider(OllamaEmbeddingProvider(model="nomic-embed-text")),
)
```

A worked end-to-end example — a session whose memory is corrupted and whose
summary invents an event, producing both a `memory_drift` and a
`memory_ungrounded` flag — is in
[the memory-integrity example](../examples/memory-integrity.md).
