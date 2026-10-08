"""Embedding providers for the memory-integrity module (``S4-T1`` - ``S4-T3``).

Three things live here, in the order a deployment meets them:

* :class:`EmbeddingProvider` — the interface a module depends on, so swapping a
  local hashing embedder for a local semantic model is configuration.
* :class:`HashingEmbeddingProvider` — the default. Deterministic, offline, no
  model to pin, no network. It hashes tokens into a fixed-width vector, which is
  enough to see the failure this module looks for: memory corruption is almost
  always a *large* lexical discontinuity, not a subtle semantic one.
* :class:`OllamaEmbeddingProvider` — the semantic option, local by default per
  INV-5. Batched and rate-limited so an evaluator cannot starve the host agent's
  own model calls.
* :class:`CachedEmbeddingProvider` — a cache keyed by content hash **and model
  id**, which is what makes a model change invalidate itself rather than leaving
  vectors from two models mixed in the same series.

Determinism is a hard requirement here, not a nicety. A module that reaches a
different verdict on the same events after a restart produces duplicate flags
with different bodies, and the FP/FN numbers in
:mod:`sentinel.eval.harness` stop being reproducible. Everything below is built
to be byte-identical across processes for the same input and model id:

* token hashing uses :mod:`hashlib`, never Python's ``hash()``, whose salt is
  randomised per process and would change every vector on every restart;
* the same seed and dimensionality always give the same vector;
* the cache key includes the model id, so vectors from two providers can never
  be compared against each other by accident.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import httpx
import structlog

from sentinel.eval.provenance_core import normalize_text

log = structlog.get_logger("sentinel.eval.embeddings")

#: Default width of the hashing embedder. 512 buckets is far above the number of
#: distinct tokens in a working memory and far below the point where cosine
#: distance stops discriminating: below ~128 buckets unrelated texts collide, and
#: above ~4096 almost every pair is orthogonal and every jump looks large.
DEFAULT_DIMENSIONS = 512

#: Tokens shorter than this carry no topical signal and mostly add collisions.
MIN_TOKEN_LENGTH = 2

_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into a fixed-width vector (``S4-T1``).

    ``model_id`` is part of the contract rather than a label: the embedding
    cache is keyed on it, so a provider that cannot name itself cannot be cached
    safely. It must change whenever the vectors would change.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier for the vectors this provider produces."""

    @property
    def dimensions(self) -> int:
        """Width of every vector returned. Constant for a given ``model_id``."""

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed *texts*, preserving order. One vector per input."""


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine distance in ``[0, 2]``: ``0`` is identical direction, ``1`` orthogonal.

    Lengths must match. Zero vectors — an empty document, or one made entirely of
    stopwords — return ``1.0`` rather than raising, because "no signal" and "no
    relationship" are the same answer for a threshold comparison and a crash here
    would take down a worker mid-session.
    """
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} != {len(right)}")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - dot / math.sqrt(left_norm * right_norm)


def _tokens(text: str) -> list[str]:
    """The topical tokens of *text*, lowercased and order-preserving."""
    return [
        token for token in _TOKEN_RE.findall(normalize_text(text)) if len(token) >= MIN_TOKEN_LENGTH
    ]


def _bucket(token: str, dimensions: int) -> tuple[int, float]:
    """Which bucket *token* lands in, and with what sign.

    ``hashlib`` rather than ``hash()``: Python randomises string hashing per
    process, so the builtin would put the same word in a different bucket on
    every restart and make every stored vector stale the moment the worker
    restarted. The signed bucket is the standard hashing-vectorizer trick —
    signed hashing keeps two documents that share every token at cosine
    distance ``0`` rather than letting shared features add in phase.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dimensions, 1.0 if value % 2 == 0 else -1.0


class HashingEmbeddingProvider:
    """Deterministic offline embedder (``S4-T1`` default).

    Hashes tokens into ``dimensions`` signed buckets and L2-normalises, so the
    result depends only on the text and the configuration. That makes it the
    right default for a safety module in two ways: it needs no model pinned and
    no network, satisfying INV-5 with nothing to configure; and its vectors are
    reproducible on any machine, so a flag raised today can be re-derived
    exactly on a different host years later.

    The honest limitation is that it measures *lexical* overlap. Two memories
    that mean the same thing in different words look distant. That is acceptable
    for this module because the failure it detects is an abrupt jump, and an
    injected instruction is lexically alien by construction; it would not be
    acceptable for paraphrase detection. Deployments that need semantic
    continuity pass an :class:`OllamaEmbeddingProvider` instead.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, *, seed: str = "sentinel") -> None:
        """Create a provider of *dimensions* buckets, salted with *seed*."""
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions
        self._seed = seed
        self._cache: dict[str, tuple[float, ...]] = {}

    @property
    def model_id(self) -> str:
        """``hashing/<dims>/<seed>`` — changes whenever the vectors would."""
        return f"hashing/v1/{self._dimensions}/{self._seed}"

    @property
    def dimensions(self) -> int:
        """Width of every vector this provider returns."""
        return self._dimensions

    def embed_sync(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed without an event loop; the body of :meth:`embed`."""
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            cached = self._cache.get(text)
            if cached is not None:
                vectors.append(cached)
                continue
            buckets = [0.0] * self._dimensions
            for token in _tokens(text):
                index, sign = _bucket(f"{self._seed}:{token}", self._dimensions)
                buckets[index] += sign
            vector = _normalize(buckets)
            self._cache[text] = vector
            vectors.append(vector)
        return vectors

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed *texts* in order; :meth:`embed_sync` holds the body."""
        return self.embed_sync(texts)


class OllamaEmbeddingProvider:
    """Local Ollama embeddings, batched and rate-limited (``S4-T1``/``T3``).

    INV-5 says the core path is local, and a semantic memory metric is worth
    having, so this is offered — but it is opt-in and never the default, because
    a safety module that silently depends on a model server being up is a safety
    module that silently stops working.

    Two endpoint shapes are handled: the modern ``/api/embed``, which takes a
    list and returns ``embeddings``, and the older ``/api/embeddings``, which
    takes one ``prompt`` and returns ``embedding``. Which one is used is
    configured, because guessing at it per request would turn a deployment
    mismatch into a per-call timeout.

    ``rate_limit_per_s`` spaces out batches. An evaluator and a host agent share
    one local model server, and an evaluator that saturates it degrades the very
    agent it is supposed to be protecting. Concurrency is capped for the same
    reason.
    """

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        client: httpx.AsyncClient | None = None,
        batch_size: int = 16,
        rate_limit_per_s: float = 4.0,
        max_concurrency: int = 2,
        timeout_s: float = 30.0,
        endpoint: str = "embed",
    ) -> None:
        """Create a provider for *model* at *base_url*."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._batch_size = batch_size
        self._min_interval = 1.0 / rate_limit_per_s if rate_limit_per_s > 0 else 0.0
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout = timeout_s
        self._endpoint = endpoint
        self._last_call = 0.0
        self._lock = asyncio.Lock()
        self._dimensions = 0

    @property
    def model_id(self) -> str:
        """``ollama/<model>@<endpoint>``."""
        return f"ollama/{self._model}@{self._endpoint}"

    @property
    def dimensions(self) -> int:
        """Width of the last vector seen, or ``0`` before the first call.

        Resolved from the model's own output rather than configured: a
        hard-coded width that disagreed with the model would produce cosine
        distances computed over the wrong coordinates and silently meaningless
        drift numbers. Modules read this after the first embed and assert on it.
        """
        return self._dimensions

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _throttle(self) -> None:
        """Wait until the next batch is allowed through.

        Serialised through a lock rather than by sleeping independently, so N
        concurrent callers space themselves out instead of all measuring the same
        idle gap and then firing together — which is the opposite of rate
        limiting.
        """
        if self._min_interval <= 0:
            return
        async with self._lock:
            wait = self._last_call + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed *texts* in ``batch_size`` batches, spaced by the rate limit."""
        if not texts:
            return []
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    async def _embed_batch(self, batch: Sequence[str]) -> list[tuple[float, ...]]:
        client = await self._http()
        payload = (
            {"model": self._model, "embed": list(batch)}
            if self._endpoint == "embed"
            else {"model": self._model, "prompt": batch[0]}
        )
        path = "/api/embed" if self._endpoint == "embed" else "/api/embeddings"
        async with self._semaphore:
            await self._throttle()
            try:
                response = await client.post(f"{self._base_url}{path}", json=payload)
                response.raise_for_status()
                body = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                # A module that cannot embed cannot judge. Raising here would
                # surface as a retried worker failure and eventually a stuck
                # session; returning empty vectors makes the caller skip the
                # comparison instead, which is the honest outcome.
                log.warning(
                    "embeddings.failed",
                    model=self._model,
                    endpoint=self._endpoint,
                    error=str(exc),
                )
                return [() for _ in batch]

        parsed = _parse_embeddings(body, len(batch))
        if not parsed:
            return [() for _ in batch]
        self._dimensions = len(parsed[0])
        return [_normalize(vector) for vector in parsed]

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this provider created it.

        A client injected by the caller is the caller's to close; closing it
        behind their back would break every other user of it.
        """
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_embeddings(body: object, expected: int) -> list[tuple[float, ...]]:
    """Pull the vectors out of either Ollama response shape."""
    if not isinstance(body, Mapping):
        return []
    rows = body.get("embeddings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = None
    if rows is None:
        single = body.get("embedding")
        rows = (
            [single]
            if isinstance(single, Sequence) and not isinstance(single, (str, bytes))
            else []
        )
    vectors = [tuple(float(value) for value in row) for row in rows if isinstance(row, Sequence)]
    if len(vectors) < expected:
        vectors.extend(() for _ in range(expected - len(vectors)))
    return vectors[:expected]


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """L2-normalise, so cosine distance depends on direction alone.

    Without this, a memory that simply grew longer would look like drift, and the
    metric would be measuring length.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def content_key(model_id: str, text: str) -> str:
    """The cache key for *text* under *model_id*.

    Both parts are load-bearing. Content alone would serve a vector computed by a
    different model after a model change, and the resulting drift series would
    mix two coordinate systems — the worst kind of silent wrongness, because every
    individual number still looks like a float.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_id}:{digest}"


@dataclass
class CachedEmbeddingProvider:
    """Memoising wrapper keyed by content hash and model id (``S4-T2``).

    Caching matters more here than it looks. A memory series re-embeds its whole
    state at every write to compute drift against the previous state, so an
    uncached module does O(n²) embedding work over a session and pays for it on
    every re-evaluation after a restart. Keying on the content hash means an
    unchanged memory is embedded once, ever.

    The cache is bounded and evicts least-recently-used, because a long-lived
    worker over a large corpus would otherwise accumulate every memory state it
    has ever seen. Bounding it also keeps the failure mode familiar: the worst a
    full cache can do is recompute.
    """

    provider: EmbeddingProvider
    max_entries: int = 4_096
    _entries: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)
    hits: int = field(default=0, repr=False)
    misses: int = field(default=0, repr=False)

    @property
    def model_id(self) -> str:
        """The wrapped provider's id, prefixed so two caches never collide."""
        return f"cached/{self.provider.model_id}"

    @property
    def dimensions(self) -> int:
        """Whatever the wrapped provider reports."""
        return self.provider.dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed *texts*, computing only the ones not already cached."""
        keys = [content_key(self.provider.model_id, text) for text in texts]
        missing = [text for key, text in zip(keys, texts, strict=True) if key not in self._entries]
        fresh: dict[str, tuple[float, ...]] = {}
        if missing:
            computed = await self.provider.embed(missing)
            fresh = {
                content_key(self.provider.model_id, text): vector
                for text, vector in zip(missing, computed, strict=True)
            }
            self._entries.update(fresh)
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        # Resolved *before* eviction. Evicting first can drop a key this very
        # call needs — a batch larger than ``max_entries`` evicted entries it had
        # just inserted, and the lookup then raised ``KeyError`` on the caller's
        # result rather than returning the vector it had already paid for.
        resolved = [fresh[key] if key in fresh else self._entries[key] for key in keys]
        self._evict()
        return resolved

    def _evict(self) -> None:
        """Drop the oldest half once over budget.

        Half rather than one: evicting one entry per insert would turn every
        insert into an O(n) scan and make the cache's own cost the thing being
        optimised.
        """
        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return
        for key in list(self._entries)[: max(overflow, self.max_entries // 2)]:
            del self._entries[key]


async def close_provider(provider: EmbeddingProvider) -> None:
    """Close *provider*'s HTTP client if it owns one; never raises.

    A cleanup path must not be able to fail the run: a module tearing down after
    a successful evaluation should not convert that into an error the retry loop
    then records against the session.
    """
    closer = getattr(provider, "aclose", None)
    if closer is None:
        return
    try:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    except (httpx.HTTPError, RuntimeError) as exc:  # pragma: no cover - defensive
        log.warning("embeddings.close_failed", error=str(exc))
