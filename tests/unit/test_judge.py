"""Tests for the judge layer (``S5-T1`` - ``S5-T4``).

The judge is the only component here allowed to be wrong about meaning, so most
of these tests are about the *guards* rather than the judgements: an unparseable
response, an invented span, a missing span, a provider that is down. Every one of
them must produce no flag, because a judge that reports a finding when it has not
understood anything is worse than no judge at all.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from sentinel.eval.judge import (
    JUDGE_RUBRIC,
    JUDGE_TEMPERATURE,
    JudgeOutcome,
    JudgeProvider,
    JudgeVerdict,
    OllamaJudge,
    RuleBasedJudge,
    close_judge,
    parse_judge_output,
    validate_verdict,
)

pytestmark = pytest.mark.unit

OLLAMA = "http://127.0.0.1:11434"

REASONING = "I checked the billing API and it reported 49 USD. That is the price."
ANSWER = "The price is $49."


# -- parsing ----------------------------------------------------------------


class TestParseJudgeOutput:
    def test_a_well_formed_object_parses(self) -> None:
        verdict = parse_judge_output(
            '{"outcome": "inconsistent", "score": 0.2, "spans": ["49 USD"],'
            ' "rationale": "the figures disagree"}'
        )
        assert verdict is not None
        assert verdict.outcome is JudgeOutcome.INCONSISTENT
        assert verdict.score == pytest.approx(0.2)
        assert verdict.spans == ("49 USD",)

    def test_outcome_is_case_insensitive(self) -> None:
        verdict = parse_judge_output('{"outcome": "INCONSISTENT", "score": 0.1, "spans": []}')
        assert verdict is not None
        assert verdict.outcome is JudgeOutcome.INCONSISTENT

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "the reasoning is consistent",
            'Sure! {"outcome": "consistent", "score": 1.0, "spans": []}',
            '{"outcome": "consistent", "score": 1.0, "spans": []} hope that helps',
            '{"outcome": "maybe", "score": 0.5, "spans": []}',
            '{"score": 0.5, "spans": []}',
            '{"outcome": "consistent", "score": 1.7, "spans": []}',
            '{"outcome": "consistent", "score": -0.2, "spans": []}',
            '{"outcome": "consistent", "score": "very low", "spans": []}',
            '{"outcome": "consistent", "score": 0.5, "spans": "a string"}',
            '{"outcome": "consistent", "score": 0.5, "spans": [1, 2]}',
            "[1, 2, 3]",
            "{not json}",
        ],
    )
    def test_anything_unusable_is_none(self, raw: str) -> None:
        """Leniency here would trade a few detections for a class of silent
        misparsing nobody would notice until a flag was wrong."""
        assert parse_judge_output(raw) is None

    def test_a_missing_score_defaults_to_the_middle(self) -> None:
        verdict = parse_judge_output('{"outcome": "consistent", "spans": []}')
        assert verdict is not None
        assert verdict.score == pytest.approx(0.5)

    def test_spans_default_to_empty(self) -> None:
        verdict = parse_judge_output('{"outcome": "consistent", "score": 0.9}')
        assert verdict is not None
        assert verdict.spans == ()


# -- span verification (``S5-T3``) -----------------------------------------


class TestValidateVerdict:
    def test_a_real_span_survives(self) -> None:
        verdict = JudgeVerdict(
            outcome=JudgeOutcome.INCONSISTENT,
            score=0.1,
            spans=("49 USD",),
        )
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is JudgeOutcome.INCONSISTENT
        assert kept.spans == ("49 USD",)

    def test_an_invented_span_discards_the_whole_verdict(self) -> None:
        """A judge asked to cite will sometimes cite something from its own
        imagination. That is the whole failure this check removes."""
        verdict = JudgeVerdict(
            outcome=JudgeOutcome.INCONSISTENT, score=0.0, spans=("the quarterly ledger",)
        )
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is JudgeOutcome.UNDETERMINED
        assert kept.is_finding is False

    def test_a_finding_with_no_spans_is_discarded(self) -> None:
        verdict = JudgeVerdict(outcome=JudgeOutcome.INCONSISTENT, score=0.0)
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is JudgeOutcome.UNDETERMINED

    def test_a_discarded_verdict_is_never_downgraded_to_consistent(self) -> None:
        """Turning 'bad' into 'good' would inflate the measured false-negative
        rate with everything the guard rejected, and hide a misconfigured judge
        behind a flattering number."""
        verdict = JudgeVerdict(outcome=JudgeOutcome.INCONSISTENT, score=0.0, spans=("invented",))
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is not JudgeOutcome.CONSISTENT

    def test_matching_is_case_and_whitespace_insensitive(self) -> None:
        verdict = JudgeVerdict(
            outcome=JudgeOutcome.INCONSISTENT, score=0.0, spans=("  I CHECKED the billing ",)
        )
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is JudgeOutcome.INCONSISTENT

    def test_partly_invented_spans_are_pruned_not_discarded_whole(self) -> None:
        """One bad span does not invalidate a judgement that cited three good
        ones; it just means we know less than the judge claimed."""
        verdict = JudgeVerdict(
            outcome=JudgeOutcome.INCONSISTENT,
            score=0.1,
            spans=("49 USD", "invented entirely", "the price"),
        )
        kept = validate_verdict(verdict, reasoning=REASONING, answer=ANSWER)
        assert kept.outcome is JudgeOutcome.INCONSISTENT
        assert kept.spans == ("49 USD", "the price")

    def test_a_consistent_verdict_is_never_rewritten(self) -> None:
        verdict = JudgeVerdict(outcome=JudgeOutcome.CONSISTENT, score=0.9, spans=("nonsense",))
        assert validate_verdict(verdict, reasoning=REASONING, answer=ANSWER) is verdict


# -- the rule-based default -------------------------------------------------


class TestRuleBasedJudge:
    def test_it_is_a_judge_provider(self) -> None:
        assert isinstance(RuleBasedJudge(), JudgeProvider)

    @pytest.mark.asyncio
    async def test_it_declines_when_the_reasoning_names_no_source(self) -> None:
        """A structural rule cannot judge semantic support, and pretending
        otherwise would spend the false-positive budget to buy recall nobody
        asked for."""
        judge = RuleBasedJudge()
        verdict = await judge.judge("I think it is probably fine.", ANSWER)
        assert verdict.outcome is JudgeOutcome.UNDETERMINED
        assert verdict.is_finding is False

    @pytest.mark.asyncio
    async def test_it_never_produces_a_finding_on_its_own(self) -> None:
        """The deterministic default is a *filter*, not a detector. Every finding
        it produced would be a rule pretending to be a judge."""
        judge = RuleBasedJudge()
        for reasoning in (REASONING, "According to the ledger, no.", "the report shows 3"):
            verdict = await judge.judge(reasoning, ANSWER)
            assert verdict.is_finding is False

    @pytest.mark.asyncio
    async def test_it_keeps_count_of_what_it_examined(self) -> None:
        judge = RuleBasedJudge()
        await judge.judge("nothing to see", ANSWER)
        await judge.judge("nothing to see", ANSWER)
        assert judge.judged == 2

    def test_its_model_id_names_the_version(self) -> None:
        """Bumping the rule has to change the id, or findings computed under the
        old rule would keep the old identity."""
        assert RuleBasedJudge().model_id == "rule/v1"


# -- the Ollama judge -------------------------------------------------------


class TestOllamaJudge:
    @pytest.mark.asyncio
    async def test_a_well_formed_response_is_validated(self) -> None:
        payload = {
            "response": (
                '{"outcome": "inconsistent", "score": 0.15,'
                ' "spans": ["49 USD"], "rationale": "the reasoning quotes a different figure"}'
            )
        }
        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(
                return_value=httpx.Response(200, json=payload)
            )
            judge = OllamaJudge(client=httpx.AsyncClient())
            verdict = await judge.judge(REASONING, ANSWER)
        assert verdict.outcome is JudgeOutcome.INCONSISTENT
        assert verdict.spans == ("49 USD",)
        assert verdict.judged_sha256

    @pytest.mark.asyncio
    async def test_it_asks_for_deterministic_output(self) -> None:
        """Non-determinism here means a flag that reappears and disappears
        between evaluations, which is worse than no flag."""
        captured: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured.append(_json.loads(request.read()))
            return httpx.Response(200, json={"response": '{"outcome":"consistent","score":1}'})

        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(side_effect=handler)
            judge = OllamaJudge(client=httpx.AsyncClient())
            await judge.judge(REASONING, ANSWER)
        options = captured[0]["options"]
        assert isinstance(options, dict)
        assert options["temperature"] == JUDGE_TEMPERATURE
        assert options["seed"] == 0
        assert captured[0]["format"] == "json"

    @pytest.mark.asyncio
    async def test_it_shows_the_judge_the_evidence(self) -> None:
        """A judge that cannot see what the agent saw will call 'I checked the
        billing API' unfaithful when the agent did exactly that."""
        prompt = OllamaJudge(client=httpx.AsyncClient()).build_prompt(
            REASONING, ANSWER, context="billing API returned 49 USD"
        )
        assert "billing API returned 49 USD" in prompt
        assert JUDGE_RUBRIC in prompt

    @pytest.mark.asyncio
    async def test_a_provider_outage_produces_no_finding(self) -> None:
        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(return_value=httpx.Response(500))
            judge = OllamaJudge(client=httpx.AsyncClient())
            verdict = await judge.judge(REASONING, ANSWER)
        assert verdict.outcome is JudgeOutcome.UNDETERMINED
        assert verdict.is_finding is False

    @pytest.mark.asyncio
    async def test_a_nonsense_body_produces_no_finding(self) -> None:
        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(
                return_value=httpx.Response(200, json={"response": "I think it's fine"})
            )
            judge = OllamaJudge(client=httpx.AsyncClient())
            verdict = await judge.judge(REASONING, ANSWER)
        assert verdict.outcome is JudgeOutcome.UNDETERMINED

    @pytest.mark.asyncio
    async def test_a_missing_response_field_produces_no_finding(self) -> None:
        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(
                return_value=httpx.Response(200, json={"done": True})
            )
            judge = OllamaJudge(client=httpx.AsyncClient())
            assert (await judge.judge(REASONING, ANSWER)).outcome is JudgeOutcome.UNDETERMINED

    @pytest.mark.asyncio
    async def test_an_invented_span_is_discarded_even_from_a_real_model(self) -> None:
        payload = {
            "response": (
                '{"outcome": "inconsistent", "score": 0.0,'
                ' "spans": ["the customer cancelled their subscription"],'
                ' "rationale": "unrelated"}'
            )
        }
        with respx.mock:
            respx.post(f"{OLLAMA}/api/generate").mock(
                return_value=httpx.Response(200, json=payload)
            )
            judge = OllamaJudge(client=httpx.AsyncClient())
            verdict = await judge.judge(REASONING, ANSWER)
        assert verdict.outcome is JudgeOutcome.UNDETERMINED

    @pytest.mark.asyncio
    async def test_a_long_trace_is_clipped_at_both_ends(self) -> None:
        """The opening says what the agent set out to do and the closing says
        what it concluded; a judge shown only one judges a different agent."""
        long_reasoning = "START-OF-TRACE " + ("x " * 20_000) + " END-OF-CONCLUSION"
        prompt = OllamaJudge(client=httpx.AsyncClient(), max_chars=1_000).build_prompt(
            long_reasoning, ANSWER
        )
        assert "START-OF-TRACE" in prompt
        assert "END-OF-CONCLUSION" in prompt
        assert "…[truncated]" in prompt

    def test_the_model_id_records_the_temperature(self) -> None:
        """Two judges differing only in temperature produce different verdicts,
        so the id has to say which."""
        assert "t0" in OllamaJudge().model_id

    @pytest.mark.asyncio
    async def test_closing_does_not_touch_an_injected_client(self) -> None:
        async with httpx.AsyncClient() as client:
            judge = OllamaJudge(client=client)
            await judge.aclose()
            assert not client.is_closed
        await close_judge(judge)

    @pytest.mark.asyncio
    async def test_close_judge_on_a_provider_without_a_client_is_harmless(self) -> None:
        await close_judge(RuleBasedJudge())
