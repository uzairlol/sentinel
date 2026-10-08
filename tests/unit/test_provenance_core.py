"""Tests for the pure grounding rules (``S3-T8`` - ``S3-T11``).

The rules are the part of S3 that decides a verdict, so they are tested
directly rather than through the evaluator: each test names the rule it pins
down, and the corpus in ``test_provenance_corpus.py`` is what checks the
combination against hand-written expectations.

The through-line of these cases is that silence, support, and refutation are
three different answers. A missing value is an ``UNGROUNDED`` finding, a
different value for the same subject is a ``CONFLICTED`` one, and a value the
evidence implies without stating it is support, not a finding.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from sentinel.eval.provenance_core import (
    ClaimKind,
    DiffContext,
    GroundingLexicon,
    RuleBasedClaimExtractor,
    Source,
    SourceMatch,
    SupportKind,
    Value,
    ValueKind,
    Verdict,
    _item_statuses,
    diff_claim,
    extract_values,
    is_actionable,
    normalize_text,
    resolve_source,
    severity_for,
    split_sentences,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def extractor() -> RuleBasedClaimExtractor:
    return RuleBasedClaimExtractor()


def _only(extractor: RuleBasedClaimExtractor, text: str):  # type: ignore[no-untyped-def]
    claims = extractor.extract_sync(text)
    assert len(claims) == 1, f"expected one claim from {text!r}, got {claims!r}"
    return claims[0]


def _cited(extractor: RuleBasedClaimExtractor, text: str, evidence: str):  # type: ignore[no-untyped-def]
    """Diff a one-sentence claim against cited evidence."""
    return diff_claim(_only(extractor, text), DiffContext(explicit=(evidence,)))


# -- extraction ------------------------------------------------------------


class TestExtraction:
    def test_a_price_becomes_a_number_with_its_currency(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "The pro plan costs $29 per month.")
        assert claim.kind is ClaimKind.NUMERIC
        assert claim.value is not None
        assert claim.value.number == Decimal(29)
        assert claim.value.unit == "usd"

    def test_a_duration_is_read_as_a_duration_not_a_count(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "The trial lasts 14 days.")
        assert claim.kind is ClaimKind.DURATION
        assert claim.value is not None
        assert claim.value.number == Decimal(14)
        assert claim.value.same_magnitude(
            Value(kind=ValueKind.NUMBER, canonical="336h", number=Decimal(336), unit="h")
        )

    def test_seats_is_a_count_not_a_duration(self, extractor: RuleBasedClaimExtractor) -> None:
        """``20 seats`` must not parse the "s" of seats as seconds."""
        claim = _only(extractor, "The team uses 20 seats.")
        assert claim.value is not None
        assert claim.value.unit is None
        assert claim.value.number == Decimal(20)

    def test_a_unit_in_the_key_is_recovered(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = _only(extractor, "The export took 120 seconds.")
        assert claim.value is not None
        assert claim.value.unit == "s"
        assert claim.value.same_magnitude(
            Value(kind=ValueKind.NUMBER, canonical="120000", number=Decimal(120000), unit="ms")
        )

    def test_an_enumeration_becomes_a_set(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = _only(extractor, "Card is one of card, bank transfer, and cryptocurrency.")
        assert claim.kind is ClaimKind.SET
        assert claim.value is not None
        assert "cryptocurrency" in claim.value.members

    def test_a_set_drops_its_own_subject(self, extractor: RuleBasedClaimExtractor) -> None:
        """``Supported regions are us-east, eu-west`` has two members, not three.

        Leaving "regions are" in the member list would make the evidence set
        look like it contains an element nobody listed.
        """
        values = extract_values("Supported regions are us-east, eu-west, ap-south.")
        members = next(value.members for value in values if value.kind is ValueKind.MEMBER)
        assert members == ("us-east", "eu-west", "ap-south")

    def test_a_negation_is_scoped_to_its_verb(self, extractor: RuleBasedClaimExtractor) -> None:
        """``no fee`` must not be read as negating the acceptance around it."""
        claim = _only(extractor, "Cancellations are not accepted at any time.")
        assert claim.value is not None
        assert claim.value.kind is ValueKind.BOOLEAN
        assert claim.value.negated is True
        assert claim.value.canonical.startswith("cancellations|")

    def test_bound_phrasing_is_not_read_as_a_superlative(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "The team plan includes at most 3 seats.")
        assert claim.kind is ClaimKind.NUMERIC
        assert claim.value is not None
        assert claim.value.number == Decimal(3)

    def test_a_superlative_is_a_claim(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = _only(extractor, "Today is the biggest sale day of the year.")
        assert claim.kind is ClaimKind.COMPARATIVE

    def test_a_deictic_reference_is_recorded_as_a_cue(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "Today is the biggest sale day of the year.")
        assert "comparative:biggest" in claim.cue
        assert "today" in claim.cue

    def test_a_claim_without_assertion_is_dropped(self, extractor: RuleBasedClaimExtractor) -> None:
        assert extractor.extract_sync("What is our churn?") == []

    def test_a_key_value_row_does_not_yield_a_number_from_its_key(self) -> None:
        """``p95_latency_ms`` must not produce a stray 95.

        A phantom number is enough to manufacture a disagreement, so the key
        is masked before the prose pass runs.
        """
        values = extract_values("p95_latency_ms = 5.04")
        assert [value.number for value in values] == [Decimal("5.04")]
        assert values[0].unit == "ms"

    def test_evidence_booleans_keep_their_subject(self) -> None:
        values = extract_values("Cancellations are accepted at any time with no fee.")
        booleans = [value for value in values if value.kind is ValueKind.BOOLEAN]
        assert booleans, values
        assert booleans[0].canonical == "cancellations|False"

    def test_lexicon_matches_a_reference_resolvable_by_the_evidence(self) -> None:
        lexicon = GroundingLexicon()
        assert lexicon.matches("Currently we serve 400 accounts.")
        assert lexicon.matches("The latest report covers churn.")
        assert not lexicon.matches("The export completed.")

    def test_split_sentences_keeps_abbreviations_intact(self) -> None:
        assert len(split_sentences("Dr. Smith paid $5. It cost $6.")) == 2

    def test_normalize_text_folds_case_and_whitespace(self) -> None:
        assert normalize_text("The  PRO   plan  ") == "the pro plan"


# -- support ---------------------------------------------------------------


class TestSupport:
    def test_a_matching_value_in_cited_evidence_is_supported(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(extractor, "Churn is 4.2%.", "Monthly churn is 4.2%.")
        assert result.verdict is Verdict.SUPPORTED
        assert result.support_kind is SupportKind.EXPLICIT
        assert result.cited is True

    def test_uncited_evidence_still_supports(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = _only(extractor, "Churn is 4.2%.")
        result = diff_claim(claim, DiffContext(context=("Monthly churn is 4.2%.",)))
        assert result.verdict is Verdict.SUPPORTED
        assert result.cited is False

    def test_units_convert_before_they_conflict(self, extractor: RuleBasedClaimExtractor) -> None:
        """The verdict is support either way; the *kind* is what tells a reviewer
        the source said milliseconds when the claim said minutes."""
        result = _cited(extractor, "The export took 2 minutes.", "export_duration_ms = 120000")
        assert result.verdict in {Verdict.SUPPORTED, Verdict.IMPLIED}
        assert result.support_kind is SupportKind.UNIT_CONVERSION

    def test_a_durational_claim_matches_a_unitless_number(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(extractor, "The export took 120 seconds.", "export_duration_ms = 120000")
        assert result.verdict in {Verdict.SUPPORTED, Verdict.IMPLIED}
        assert result.support_kind is SupportKind.UNIT_CONVERSION

    def test_rounding_is_implied_support_when_the_claim_committed_to_precision(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """``5.0`` is true of ``5.04``, and the verdict says *implied*, not
        *supported*: the evidence never stated 5.0."""
        result = _cited(extractor, "The p95 latency is 5.0 ms.", "p95_latency_ms = 5.04")
        assert result.verdict is Verdict.IMPLIED
        assert result.support_kind is SupportKind.ROUNDING

    def test_a_value_beyond_the_claimed_precision_is_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(extractor, "The p95 latency is 5.0 ms.", "p95_latency_ms = 6.00")
        assert result.verdict is Verdict.CONFLICTED

    def test_an_integer_claim_is_rounded_by_the_evidence_value(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """``5`` is a fair statement about ``5.04`` — the value rounds to 5."""
        result = _cited(extractor, "The p95 latency is 5 ms.", "p95_latency_ms = 5.04")
        assert result.verdict is Verdict.IMPLIED
        assert result.support_kind is SupportKind.ROUNDING

    def test_a_subset_of_an_offered_set_is_supported(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor,
            "us-east is one of us-east, eu-west.",
            "Supported regions are us-east, eu-west, ap-south.",
        )
        assert result.verdict is Verdict.SUPPORTED

    def test_a_number_inside_a_bound_is_implied_support(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor, "The team plan includes 7 seats.", "The team plan includes at most 10 seats."
        )
        assert result.verdict is Verdict.IMPLIED
        assert result.support_kind is SupportKind.BOUND

    def test_a_weekday_matches_case_insensitively(self, extractor: RuleBasedClaimExtractor) -> None:
        result = _cited(
            extractor,
            "The maintenance window is on Sunday.",
            "maintenance_window_weekday = sunday",
        )
        assert result.verdict is Verdict.SUPPORTED

    def test_no_evidence_at_all_is_unknown(self, extractor: RuleBasedClaimExtractor) -> None:
        result = diff_claim(_only(extractor, "Churn is 4.2%."), DiffContext.empty())
        assert result.verdict is Verdict.UNKNOWN
        assert "no tool output" in result.detail

    def test_a_superlative_needs_a_ranking_statement(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "This is the biggest sale day of the year.")
        ranked = diff_claim(claim, DiffContext(explicit=("The biggest sale day was March 3.",)))
        assert ranked.verdict is Verdict.SUPPORTED
        quiet = diff_claim(claim, DiffContext(explicit=("The spring sale covers accessories.",)))
        assert quiet.verdict is Verdict.UNKNOWN


# -- contradiction ---------------------------------------------------------


class TestContradiction:
    def test_a_wrong_price_for_the_same_plan_is_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor, "The pro plan costs $29 per month.", "Plan pro costs $49 per month."
        )
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.DISAGREEMENT

    def test_a_price_about_a_different_subject_is_silence(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """The guard on the disagreement rule: another product's price is not
        a refutation of this one."""
        result = _cited(
            extractor,
            "The pro plan costs $29 per month.",
            "The enterprise plan costs $49 per month.",
        )
        assert result.verdict is Verdict.UNKNOWN

    def test_an_unrelated_number_of_a_different_unit_is_not_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """The same subject, but a unitless seat count cannot refute a price."""
        result = _cited(
            extractor, "The pro plan costs $29 per month.", "The pro plan supports 500 seats."
        )
        assert result.verdict is Verdict.UNKNOWN

    def test_a_limit_bearing_sentence_does_not_manufacture_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """``up to 100000`` bounds the number, so the 50000 beside it is not a
        refutation of 12000 — the bound rule supports the claim instead."""
        result = _cited(extractor, "We serve 12000 users.", "We serve 50000 users, up to 100000.")
        assert result.verdict is not Verdict.CONFLICTED
        assert result.support_kind is SupportKind.BOUND

    def test_a_claim_beyond_a_stated_bound_is_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(extractor, "We serve 12000 users.", "We serve up to 10000 users per month.")
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.BOUND

    def test_a_number_outside_a_stated_bound_is_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor, "The team plan includes 12 seats.", "The team plan includes at most 3 seats."
        )
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.BOUND

    def test_an_extra_set_member_is_a_conflict(self, extractor: RuleBasedClaimExtractor) -> None:
        result = _cited(
            extractor,
            "Card is one of card, bank transfer, and cryptocurrency.",
            "We accept card and bank transfer.",
        )
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.EXCLUSION

    def test_a_negated_claim_against_an_affirmative_evidence_is_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor,
            "Cancellations are not accepted at any time.",
            "Cancellations are accepted at any time with no fee.",
        )
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.NEGATION

    def test_a_different_weekday_is_a_refutation(self, extractor: RuleBasedClaimExtractor) -> None:
        result = _cited(
            extractor,
            "The maintenance window is on Sunday.",
            "maintenance_window_weekday = tuesday",
        )
        assert result.verdict is Verdict.CONFLICTED

    def test_a_different_full_date_is_a_refutation(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor, "The contract ends on 2024-12-31.", "The contract ends 2024-05-01."
        )
        assert result.verdict is Verdict.CONFLICTED

    def test_a_claim_about_a_different_entity_is_silence(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        result = _cited(
            extractor, "The pro plan costs $29 per month.", "The enterprise plan costs $49."
        )
        assert result.verdict is Verdict.UNKNOWN


# -- actionability and severity -------------------------------------------


class TestSeverity:
    def test_a_money_contradiction_is_high(self, extractor: RuleBasedClaimExtractor) -> None:
        result = _cited(
            extractor, "The pro plan costs $29 per month.", "Plan pro costs $49 per month."
        )
        claim = _only(extractor, "The pro plan costs $29 per month.")
        assert severity_for(claim, result) == "high"

    def test_a_contradiction_is_never_low(self, extractor: RuleBasedClaimExtractor) -> None:
        result = _cited(extractor, "Churn is 4.2%.", "Monthly churn is 9.9%.")
        claim = _only(extractor, "Churn is 4.2%.")
        assert severity_for(claim, result) in {"medium", "high"}

    def test_silence_is_never_critical(self, extractor: RuleBasedClaimExtractor) -> None:
        result = diff_claim(_only(extractor, "The pro plan costs $29."), DiffContext.empty())
        claim = _only(extractor, "The pro plan costs $29.")
        assert severity_for(claim, result) != "critical"

    def test_a_vague_clause_is_not_a_finding(self, extractor: RuleBasedClaimExtractor) -> None:
        """Nothing to check, nothing to flag: a clause about "it" is not a
        proposition a reviewer could act on."""
        vague = extractor.extract_sync("It is important for us to keep improving reliability.")
        assert vague == []

    def test_a_specific_claim_is_actionable(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = _only(extractor, "Churn is 4.2%.")
        assert is_actionable(claim, diff_claim(claim, DiffContext.empty())) is True

    def test_a_supported_claim_is_never_actionable(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = _only(extractor, "Churn is 4.2%.")
        grounded = diff_claim(claim, DiffContext(explicit=("Monthly churn is 4.2%.",)))
        assert is_actionable(claim, grounded) is False


class TestSourceResolution:
    """Which result a named source points at (``resolve_source``).

    Every test here is a false-positive question rather than a detection one.
    The only outcomes that may become a flag are ``ABSENT`` and ``UNCITED``;
    ``AMBIGUOUS`` and ``UNRESOLVED`` exist precisely so the rule declines to
    guess, and a regression that turns either into a flag would be the most
    damaging change this module could take.
    """

    @staticmethod
    def _sources(*rows: tuple[str, str, bool]) -> tuple[Source, ...]:
        return tuple(
            Source(event_id=f"e{i}", tool=tool, text=text, cited=cited)
            for i, (tool, text, cited) in enumerate(rows)
        )

    def test_the_named_source_resolves_to_the_result_it_was_cited_from(self) -> None:
        match, found = resolve_source(
            "the compliance report",
            self._sources(("compliance.report", "0 incidents.", True)),
        )
        assert match is SourceMatch.CITED
        assert len(found) == 1

    def test_a_source_that_ran_but_was_not_cited_is_uncited(self) -> None:
        match, found = resolve_source(
            "the compliance report",
            self._sources(
                ("weather.forecast", "light rain.", True),
                ("compliance.report", "0 incidents.", False),
            ),
        )
        assert match is SourceMatch.UNCITED
        assert found[0].tool == "compliance.report"

    def test_a_source_that_never_ran_is_absent(self) -> None:
        match, found = resolve_source(
            "the vendor security assessment", self._sources(("billing.lookup", "$49.", True))
        )
        assert match is SourceMatch.ABSENT
        assert found == ()

    def test_two_matches_do_not_identify_a_source(self) -> None:
        match, _ = resolve_source(
            "the revenue table",
            self._sources(
                ("revenue.q3", "closed late.", True),
                ("revenue.q2", "closed on time.", False),
            ),
        )
        assert match is SourceMatch.AMBIGUOUS

    @pytest.mark.parametrize(
        "generic",
        ["the report", "the document", "the api", "the file", "the data"],
    )
    def test_a_generic_name_cannot_identify_anything(self, generic: str) -> None:
        match, _ = resolve_source(generic, self._sources(("compliance.report", "0.", True)))
        assert match is SourceMatch.UNRESOLVED

    def test_no_sources_resolves_to_nothing(self) -> None:
        assert resolve_source("the compliance report", ())[0] is SourceMatch.UNRESOLVED

    def test_a_camel_case_tool_name_is_tokenised(self) -> None:
        match, _ = resolve_source(
            "the compliance report", self._sources(("fetchComplianceReport", "0.", True))
        )
        assert match is SourceMatch.CITED

    def test_the_output_text_can_name_the_source(self) -> None:
        """Not every instrumentor puts a meaningful name on the tool."""
        match, _ = resolve_source(
            "Acme", self._sources(("company.lookup", '{"name": "Acme Corp"}', True))
        )
        assert match is SourceMatch.CITED

    def test_plurals_match_without_rewriting_the_word(self) -> None:
        """A word is matched with its 's', never folded — folding mangles
        'acmes' into 'acme' and then silently fails to match."""
        match, _ = resolve_source(
            "the audit findings", self._sources(("audit.findings", "x.", True))
        )
        assert match is SourceMatch.CITED

    def test_every_distinctive_word_must_match(self) -> None:
        """Partial matches are not matches: naming 'q3 compliance' does not
        identify the result that is merely about compliance."""
        match, _ = resolve_source(
            "the q3 compliance report", self._sources(("compliance.register", "0.", True))
        )
        assert match is SourceMatch.ABSENT


class TestUniversalClaims:
    """``all checks passed`` against an enumeration with per-item statuses.

    The counted-denominator rule cannot reach these: a universal claim states no
    denominator, so there is no number to compare with the source. What refutes
    it is the statuses.
    """

    @staticmethod
    def _universal(extractor: RuleBasedClaimExtractor, text: str):  # type: ignore[no-untyped-def]
        claim = _only(extractor, text)
        assert claim.kind is ClaimKind.UNIVERSAL, claim.kind
        return claim

    def test_a_universal_claim_is_extracted(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = self._universal(extractor, "All checks passed.")
        assert claim.value is not None
        assert claim.value.canonical == "check"

    @pytest.mark.parametrize(
        "text",
        ["Every test succeeded.", "Each job is green.", "All 4 smoke tests passed."],
    )
    def test_the_phrasings_are_recognised(
        self, extractor: RuleBasedClaimExtractor, text: str
    ) -> None:
        assert self._universal(extractor, text).value is not None

    def test_a_bare_noun_without_a_quantifier_is_not_universal(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """ "checks passed" says how many passed without saying how many there
        were — a different claim, and one the rule must not answer for."""
        claims = extractor.extract_sync("Checks passed this morning.")
        assert all(claim.kind is not ClaimKind.UNIVERSAL for claim in claims)

    def test_a_skipped_item_refutes_the_claim(self, extractor: RuleBasedClaimExtractor) -> None:
        claim = self._universal(extractor, "All checks passed.")
        result = diff_claim(
            claim,
            DiffContext(explicit=("check_1 pass, check_2 pass, check_3 skipped, check_4 pass.",)),
        )
        assert result.verdict is Verdict.CONFLICTED
        assert result.support_kind is SupportKind.CHERRY_PICK
        # The finding has to name what was dropped, or a reviewer cannot act.
        assert "check_3" in result.detail

    def test_every_item_passing_supports_the_claim(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        claim = self._universal(extractor, "All checks passed.")
        result = diff_claim(
            claim,
            DiffContext(explicit=("check_1 pass, check_2 pass, check_3 pass, check_4 pass.",)),
        )
        assert result.verdict is Verdict.SUPPORTED
        assert is_actionable(claim, result) is False

    def test_too_few_items_to_check_is_not_a_conflict(
        self, extractor: RuleBasedClaimExtractor
    ) -> None:
        """Below the enumeration floor the rule cannot tell a complete set from
        a partial one, so it says UNKNOWN rather than asserting a conflict."""
        claim = self._universal(extractor, "All smoke tests passed.")
        result = diff_claim(claim, DiffContext(explicit=("smoke_1 pass, smoke_2 skipped.",)))
        assert result.verdict is Verdict.UNKNOWN
        assert result.support_kind is not SupportKind.CHERRY_PICK

    def test_status_words_in_prose_are_not_item_statuses(self) -> None:
        """ "no", "done" and "missing" are ordinary English. Without an
        identifier or a separator they must not build an enumeration."""
        statuses = _item_statuses("There is no further work. It is done and nothing is missing.")
        assert statuses == {}

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("check_1 pass, check_2 fail, check_3 skipped", {"check_1", "check_2", "check_3"}),
            ("job_1: succeeded, job_2: failed", {"job_1", "job_2"}),
            ("Test_1=PASS, test_1=pass", {"test_1"}),
        ],
    )
    def test_both_status_shapes_are_read(self, text: str, expected: set[str]) -> None:
        assert set(_item_statuses(text)) == expected
