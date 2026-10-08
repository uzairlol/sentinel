"""Tests for the embedding providers (``S4-T1`` - ``S4-T3``).

Determinism is the property that matters most here and gets the most attention:
a module that reaches a different verdict after a restart produces duplicate
flags with different bodies and makes the corpus numbers unreproducible.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from sentinel.eval.embeddings import (
    CachedEmbeddingProvider,
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OllamaEmbeddingProvider,
    close_provider,
    content_key,
    cosine_distance,
)

pytestmark = pytest.mark.unit

OLLAMA = "http://127.0.0.1:11434"


# -- cosine -----------------------------------------------------------------


class TestCosineDistance:
    def test_identical_directions_are_zero(self) -> None:
        assert cosine_distance((1.0, 0.0), (2.0, 0.0)) == pytest.approx(0.0)

    def test_orthogonal_is_one(self) -> None:
        assert cosine_distance((1.0, 0.0), (0.0, 1.0)) == pytest.approx(1.0)

    def test_opposite_is_two(self) -> None:
        assert cosine_distance((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(2.0)

    def test_a_zero_vector_is_orthogonal_rather_than_an_error(self) -> None:
        """An empty document must not crash a worker mid-session; "no signal"
        and "no relationship" are the same answer for a threshold test."""
        assert cosine_distance((0.0, 0.0), (1.0, 0.0)) == 1.0

    def test_mismatched_widths_are_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_distance((1.0, 0.0), (1.0, 0.0, 0.0))


# -- hashing provider -------------------------------------------------------


class TestHashingEmbeddingProvider:
    def test_it_is_an_embedding_provider(self) -> None:
        assert isinstance(HashingEmbeddingProvider(), EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_the_same_text_always_gives_the_same_vector(self) -> None:
        """The property the whole module's determinism rests on."""
        first = await HashingEmbeddingProvider().embed(["the customer is on the pro plan"])
        second = await HashingEmbeddingProvider().embed(["the customer is on the pro plan"])
        assert first == second

    @pytest.mark.asyncio
    async def test_two_instances_agree(self) -> None:
        """Not just one instance being stable: hashing must not depend on
        process salt, which ``hashlib`` guarantees and ``hash()`` does not."""
        a = await HashingEmbeddingProvider().embed(["billing contact is finance"])
        b = await HashingEmbeddingProvider().embed(["billing contact is finance"])
        assert a == b

    @pytest.mark.asyncio
    async def test_unrelated_texts_are_further_apart_than_shared_vocabulary(self) -> None:
        provider = HashingEmbeddingProvider()
        near, far = await provider.embed(
            ["the account has 40 seats", "the account has 40 seats and 12 are in use"]
        )
        other, _ = await provider.embed(["retention policy is 400 days", ""])
        shared = cosine_distance(near, far)
        distant = cosine_distance(near, other)
        assert shared < distant

    @pytest.mark.asyncio
    async def test_vectors_are_unit_length(self) -> None:
        (vector,) = await HashingEmbeddingProvider().embed(["some memory content"])
        norm = sum(value * value for value in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_empty_text_yields_a_zero_vector_not_an_error(self) -> None:
        (vector,) = await HashingEmbeddingProvider().embed([""])
        assert all(value == 0.0 for value in vector)

    def test_model_id_changes_with_the_configuration(self) -> None:
        """The id is part of the cache key, so it must change when the vectors
        would."""
        assert HashingEmbeddingProvider(256).model_id != HashingEmbeddingProvider(512).model_id
        assert (
            HashingEmbeddingProvider(512, seed="other").model_id
            != HashingEmbeddingProvider(512).model_id
        )

    def test_a_tiny_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            HashingEmbeddingProvider(4)


# -- cache ------------------------------------------------------------------


class TestCachedEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_a_second_call_is_served_from_cache(self) -> None:
        inner = HashingEmbeddingProvider()
        cached = CachedEmbeddingProvider(inner, max_entries=64)
        await cached.embed(["billing contact is finance"])
        await cached.embed(["billing contact is finance"])
        assert cached.hits == 1
        assert cached.misses == 1

    @pytest.mark.asyncio
    async def test_cache_hits_do_not_reach_the_inner_provider(self) -> None:
        """The point of the cache is not calling the model again."""

        class Counting:
            model_id = "counting/v1"
            dimensions = 4
            calls = 0

            async def embed(self, texts):  # type: ignore[no-untyped-def]
                self.calls += 1
                return [(1.0, 0.0, 0.0, 0.0) for _ in texts]

        inner = Counting()
        cached = CachedEmbeddingProvider(inner)
        await cached.embed(["a", "a", "a"])
        assert inner.calls == 1

    @pytest.mark.asyncio
    async def test_a_different_model_id_is_a_different_cache_entry(self) -> None:
        """Mixing two models' vectors in one series is the worst kind of silent
        wrongness: every individual number still looks like a float."""

        class Fixed:
            dimensions = 3

            def __init__(self, model_id: str) -> None:
                self.model_id = model_id

            async def embed(self, texts):  # type: ignore[no-untyped-def]
                return [(1.0, 0.0, 0.0) for _ in texts]

        cached = CachedEmbeddingProvider(Fixed("model/a"), max_entries=8)
        cached._entries[content_key("model/b", "same text")] = (0.0, 0.0, 1.0)
        (vector,) = await cached.embed(["same text"])
        assert vector == (1.0, 0.0, 0.0), "model/b's cached vector leaked into model/a"

    @pytest.mark.asyncio
    async def test_the_cache_evicts_rather_than_growing_without_bound(self) -> None:
        cached = CachedEmbeddingProvider(HashingEmbeddingProvider(), max_entries=4)
        await cached.embed([f"memory number {index}" for index in range(20)])
        assert len(cached._entries) <= 4

    @pytest.mark.asyncio
    async def test_a_miss_and_a_hit_in_one_call_are_both_answered(self) -> None:
        cached = CachedEmbeddingProvider(HashingEmbeddingProvider(), max_entries=64)
        first = (await cached.embed(["billing contact is finance"]))[0]
        mixed = await cached.embed(["billing contact is finance", "retention is 400 days"])
        assert len(mixed) == 2
        assert mixed[0] == first, "the cached text changed its vector"
        assert mixed[1] != first, "two different texts produced the same vector"

    def test_content_key_includes_both_text_and_model(self) -> None:
        assert content_key("m1", "text") != content_key("m2", "text")
        assert content_key("m1", "a") != content_key("m1", "b")


# -- Ollama provider --------------------------------------------------------


class TestOllamaEmbeddingProvider:
    @pytest.mark.asyncio
    async def test_it_reads_the_modern_embed_shape(self) -> None:
        body = {"embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(return_value=httpx.Response(200, json=body))
            provider = OllamaEmbeddingProvider(client=httpx.AsyncClient())
            vectors = await provider.embed(["a", "b"])
        assert vectors == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        assert provider.dimensions == 3

    @pytest.mark.asyncio
    async def test_it_reads_the_legacy_single_prompt_shape(self) -> None:
        body = {"embedding": [0.0, 1.0]}
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embeddings").mock(return_value=httpx.Response(200, json=body))
            provider = OllamaEmbeddingProvider(client=httpx.AsyncClient(), endpoint="embeddings")
            (vector,) = await provider.embed(["a"])
        assert vector == (0.0, 1.0)

    @pytest.mark.asyncio
    async def test_a_model_error_degrades_to_no_vector(self) -> None:
        """Capture must never crash the host. A down embedding model has to
        degrade to "cannot judge", not to a failed worker."""
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(return_value=httpx.Response(500))
            provider = OllamaEmbeddingProvider(client=httpx.AsyncClient())
            vectors = await provider.embed(["a"])
        assert vectors == [()]

    @pytest.mark.asyncio
    async def test_a_malformed_body_degrades_rather_than_raising(self) -> None:
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(
                return_value=httpx.Response(200, json={"choices": "nope"})
            )
            provider = OllamaEmbeddingProvider(client=httpx.AsyncClient())
            assert await provider.embed(["a"]) == [()]

    @pytest.mark.asyncio
    async def test_a_short_response_is_padded_not_truncated_silently(self) -> None:
        """Two texts in, one vector back: padding makes the caller see "no
        vector", where truncating would misalign every later index."""
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(
                return_value=httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
            )
            provider = OllamaEmbeddingProvider(client=httpx.AsyncClient())
            vectors = await provider.embed(["a", "b"])
        assert vectors == [(1.0, 0.0), ()]

    @pytest.mark.asyncio
    async def test_batching_splits_a_large_request(self) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = request.read()
            calls.append(len(payload))
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * 1})

        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(side_effect=handler)
            provider = OllamaEmbeddingProvider(
                client=httpx.AsyncClient(), batch_size=2, rate_limit_per_s=0
            )
            await provider.embed(["a", "b", "c"])
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_the_rate_limit_spaces_batches_out(self) -> None:
        """An evaluator sharing a model server with the host agent must not
        saturate it."""
        with respx.mock:
            respx.post(f"{OLLAMA}/api/embed").mock(
                return_value=httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})
            )
            provider = OllamaEmbeddingProvider(
                client=httpx.AsyncClient(), batch_size=1, rate_limit_per_s=20.0
            )
            started = asyncio.get_running_loop().time()
            await provider.embed(["a", "b", "c"])
            elapsed = asyncio.get_running_loop().time() - started
        # 3 batches at 20/s is at least 2 inter-batch gaps of 50ms.
        assert elapsed >= 0.09

    @pytest.mark.asyncio
    async def test_closing_does_not_touch_an_injected_client(self) -> None:
        async with httpx.AsyncClient() as client:
            provider = OllamaEmbeddingProvider(client=client)
            await provider.aclose()
            assert not client.is_closed
        await close_provider(provider)

    @pytest.mark.asyncio
    async def test_closing_twice_is_harmless(self) -> None:
        provider = OllamaEmbeddingProvider()
        await provider.aclose()
        await close_provider(provider)

    def test_absurd_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            OllamaEmbeddingProvider(batch_size=0)
        with pytest.raises(ValueError, match="max_concurrency"):
            OllamaEmbeddingProvider(max_concurrency=0)
