"""Provenance analysis, independent of any worker or store (sprint ``S3``).

This module is the reusable half of tool-use grounding: it turns an agent's
prose into *claims*, turns tool output into *values*, and decides whether a
claim is supported, implied, contradicted, or ungrounded. Nothing here touches
a database, a clock, or the network — every function is pure, which is what lets
``sentinel eval-fixtures`` publish a reproducible FP/FN rate (``S3-T13``).

The layering mirrors the rule-first constraint in the design:

1. :class:`RuleBasedClaimExtractor` (``S3-T5``) — deterministic surface rules
   find the claims. No model, no I/O, same input -> same claims.
2. :class:`GroundingLexicon` (``S3-T7``) — the *implicit grounding* lexicon:
   the small closed set of phrases that name a fact without stating it
   ("today", "the latest release", "currently"). An alternative
   :class:`ClaimExtractor` may find more claims, but it must be deterministic
   and swapping one in is a module-version change.
3. :func:`diff_claim` (``S3-T8``/``S3-T9``/``S3-T10``) — evidence vs claim:
   explicit citation, implication (rounding, unit conversion, subset), and
   contradiction (bounds, exclusion, negation, disagreeing weekday/date).
4. :func:`severity_for` / :func:`confidence_for` (``S3-T11``) — the mapping from
   an analysis to a :class:`~sentinel.models.flags.Severity` and a confidence.

Precision is the design bias: a claim with no recoverable value is not flagged,
because "the service is excellent" cannot be grounded or refuted by a tool
result and flagging it would be noise.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Protocol

from sentinel.models.flags import Severity

# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------


class ClaimKind(StrEnum):
    """What a claim asserts, as far as surface rules can tell."""

    NUMERIC = "numeric"
    DURATION = "duration"
    DATE = "date"
    WEEKDAY = "weekday"
    BOOLEAN = "boolean"
    SET = "set"
    ENTITY = "entity"
    COMPARATIVE = "comparative"
    #: A count over a named set ("2 of 3 checks passed"). The denominator is the
    #: part that carries the finding: a count smaller than what the source
    #: enumerates is how cherry-picking announces itself.
    RATIO = "ratio"
    VAGUE = "vague"


class ValueKind(StrEnum):
    """The shape of a normalized value."""

    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    WEEKDAY = "weekday"
    MEMBER = "member"
    ENTITY = "entity"
    TEXT = "text"
    #: ``number``/``denominator``: a counted subset of a counted whole.
    RATIO = "ratio"


#: Kinds a claim must land in to be checkable against evidence. Anything else is
#: ungroundable prose and is deliberately not flagged.
CHECKABLE_KINDS = frozenset(
    {
        ClaimKind.NUMERIC,
        ClaimKind.DURATION,
        ClaimKind.DATE,
        ClaimKind.WEEKDAY,
        ClaimKind.BOOLEAN,
        ClaimKind.SET,
        ClaimKind.ENTITY,
        ClaimKind.COMPARATIVE,
        ClaimKind.RATIO,
    }
)

#: Phrasing that ranks things, which is what a superlative needs as evidence.
_RANKING_MARKERS = (
    "highest",
    "lowest",
    "largest",
    "biggest",
    "smallest",
    "fastest",
    "slowest",
    "best",
    "worst",
    "most ",
    "least ",
    "top ",
    "leading",
    "best-performing",
    "outperforms",
    "more than any",
    "ranked",
    "leader",
)

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_UNIT_ALIASES = {
    "%": "%",
    "percent": "%",
    "pct": "%",
    "usd": "usd",
    "$": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "s": "s",
    "sec": "s",
    "secs": "s",
    "second": "s",
    "seconds": "s",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "x": "x",
}

#: Multipliers into milliseconds, for unit conversion (implication support).
_DURATION_TO_MS = {
    "ms": Decimal(1),
    "s": Decimal(1000),
    "min": Decimal(60_000),
    "h": Decimal(3_600_000),
    "day": Decimal(86_400_000),
    "week": Decimal(604_800_000),
    "month": Decimal(2_592_000_000),
    "year": Decimal(31_536_000_000),
}

#: Multipliers into a canonical count, for scaled numbers (2x, 10 percent).
_COUNT_UNITS = frozenset({"%", "x", "usd"})

#: Unit names a duration claim may use, derived from the conversion table.
_DURATION_UNIT_NAMES = frozenset(_DURATION_TO_MS)

_COMPARATIVE_WORDS = (
    "biggest",
    "largest",
    "smallest",
    "fastest",
    "slowest",
    "cheapest",
    "best",
    "worst",
    "most",
    "least",
    "highest",
    "lowest",
    "never",
    "always",
    "everyone",
    "nobody",
    "only",
)

#: Phrases that mean "no grounding available" and mark a fragment as a
#: conversational artefact rather than an assertion.
_NON_ASSERTIONS = (
    "let me",
    "i will",
    "i'll",
    "i can",
    "i cannot",
    "i can't",
    "here is",
    "here's",
    "sure",
    "as an ai",
    "i'm sorry",
    "i am sorry",
    "would you like",
    "do you want",
    "should i",
    "note:",
    "warning:",
    "in summary",
    "summary:",
    "step 1",
    "step 2",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# 1234, 1,234.5, 12% , $4.99 , 3 minutes , 1.5 GB
# The ``(?![A-Za-z])`` guard matters more than it looks: without it "20 seats"
# parses as the duration "20 s", and every rule downstream compares a seat
# count against a second count.
_NUMBER_RE = re.compile(
    r"(?P<currency>[$£€])\s?(?P<amount>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"|(?P<plain>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s?"
    r"(?P<unit>%|percent|pct|usd|dollars?|ms|milliseconds?|s|secs?|seconds?|min|mins?|minutes?|"
    r"h|hrs?|hours?|days?|weeks?|months?|years?|x)?(?![A-Za-z])",
    re.IGNORECASE,
)

# ``key = value`` rows are how tools usually answer: ``export_duration_ms = 120000``
# or ``"price_usd": 49``. The unit often lives in the *key*, so a bare number next
# to ``_ms``/``_percent``/``_usd`` is a quantity with a unit, not a bare count.
_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z][\w. -]{0,40}?)\s*[:=]\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)"
)

#: Key suffixes that carry a unit, longest first so ``_milliseconds`` wins.
_KEY_UNITS: tuple[tuple[str, str], ...] = (
    ("milliseconds", "ms"),
    ("seconds", "s"),
    ("minutes", "min"),
    ("hours", "h"),
    ("days", "day"),
    ("weeks", "week"),
    ("months", "month"),
    ("years", "year"),
    ("percent", "%"),
    ("pct", "%"),
    ("usd", "usd"),
    ("dollars", "usd"),
    ("ms", "ms"),
    ("sec", "s"),
    ("min", "min"),
    ("hr", "h"),
    ("day", "day"),
    ("count", ""),
)

_ISO_DATE_RE = re.compile(r"\b(?P<iso>\d{4}-\d{2}-\d{2})\b")
_LONG_DATE_RE = re.compile(
    r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)
_WEEKDAY_RE = re.compile(rf"\b(?P<day>{'|'.join(_WEEKDAYS)})\b", re.IGNORECASE)

# "is one of A, B or C", "is either A or B", "are A, B, C"
_SET_RE = re.compile(r"\b(?:one of|either|any of)\s+(?P<members>[^.;!?]+)", re.IGNORECASE)
_LIST_RE = re.compile(
    r"\b(?:available|supported|include[sd]?|offer(?:s|ed)?|accept(?:s|ed)?|support(?:s|ed)?)\s+"
    r"(?:are|is|were|was)?\s*(?P<members>[^.;!?]+)",
    re.IGNORECASE,
)

_BOOL_AFFIRM = re.compile(
    r"\b(?:is|are|does|do|has|have|can|will|supports?|accepts?|includes?)\b", re.IGNORECASE
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|excludes?|excluded|isn't|aren't|doesn't|don't|cannot|can't)\b",
    re.IGNORECASE,
)
_QUOTED_RE = re.compile(r"[\"“'](?P<text>[^\"“”']{2,80})[\"”']")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w&.-]*)(?:\s+(?:of|the|and)?\s*[A-Z][\w&.-]*){1,4}\b")

_NUMBER_IN_TEXT = re.compile(r"\d")


@dataclass(frozen=True)
class Value:
    """A normalized value extracted from prose or from tool output.

    ``canonical`` is the comparable form (lowercased, aliases folded); ``number``
    and ``unit`` carry the numeric payload when there is one; ``members`` holds a
    set's elements; ``negated`` records that the text asserted the absence of the
    value.
    """

    kind: ValueKind
    canonical: str
    number: Decimal | None = None
    unit: str | None = None
    members: tuple[str, ...] = ()
    negated: bool = False
    raw: str = ""
    #: ``(numerator, denominator)`` for :attr:`ValueKind.RATIO` — "2 of 3".
    #: The denominator is the field the cherry-picking rule compares against the
    #: number of items the source actually enumerates.
    ratio: tuple[Decimal, Decimal] | None = None

    def same_magnitude(self, other: Value, *, tolerance: Decimal | None = None) -> bool:
        """Whether two numbers are equal after unit normalization.

        Durations are compared in milliseconds and scaled units (``%``, ``x``,
        ``usd``) in canonical form, so ``2 minutes`` == ``120000 ms``.
        """
        if self.number is None or other.number is None:
            return False
        left, right = self._canonical_number(), other._canonical_number()
        if left is None or right is None:
            return self.number == other.number
        if tolerance is not None:
            return abs(left - right) <= tolerance
        return left == right

    def _canonical_number(self) -> Decimal | None:
        if self.number is None or self.unit is None:
            return None
        if self.unit in _DURATION_TO_MS:
            return self.number * _DURATION_TO_MS[self.unit]
        return self.number

    def in_bounds(self, low: Decimal | None, high: Decimal | None) -> bool:
        """Whether the number sits inside an inclusive ``[low, high]`` range."""
        if self.number is None:
            return False
        if low is not None and self.number < low:
            return False
        return not (high is not None and self.number > high)

    def render(self) -> str:
        """The value as a person would read it, for evidence and details.

        Not the dataclass repr: a flag's ``observed_value`` is quoted to a
        reviewer, and ``"49usd num=49 unit=usd"`` is not what the tool said.
        """
        if self.kind is ValueKind.MEMBER and self.members:
            body = ", ".join(self.members)
        elif self.ratio is not None:
            left, right = (_render_number(part) for part in self.ratio)
            body = f"{left} of {right}"
        elif self.number is not None:
            number = _render_number(self.number)
            body = f"{number} {self.unit}".strip() if self.unit else number
        else:
            body = self.canonical or self.raw
        return f"not {body}" if self.negated and body else body

    def token(self) -> str:
        """The shortest unambiguous form of this value.

        Used to find the evidence note that carries it: a tool said
        ``"Plan pro costs $49 per month"`` and folds no unit alias, so matching
        on the rendered ``"49 usd"`` would miss the very line that refutes the
        claim.
        """
        if self.kind is ValueKind.MEMBER and self.members:
            return "|".join(self.members)
        if self.number is not None:
            return _render_number(self.number)
        return self.canonical

    def __str__(self) -> str:
        """Render the value for a flag's ``details`` (debug-friendly)."""
        return self.render()


@dataclass(frozen=True)
class Claim:
    """One assertion extracted from a response, with the value it asserts.

    ``cue`` names the surface rule that produced the claim, which is what makes
    a finding explainable ("matched the percent rule") rather than a bare diff.
    """

    claim_id: str
    text: str
    kind: ClaimKind
    value: Value | None = None
    cue: str = ""
    index: int = 0
    terms: tuple[str, ...] = ()
    #: The source the claim *names* ("the filing", "the API"), when it names
    #: one. Set from :attr:`GroundingLexicon.ATTRIBUTIVE`; empty when the claim
    #: asserts content without claiming to have read anything (``S3-T8``).
    source_attributed: str = ""

    @property
    def is_checkable(self) -> bool:
        """Whether this claim has something evidence could confirm or refute."""
        return self.kind in CHECKABLE_KINDS and self.value is not None

    @property
    def is_specific(self) -> bool:
        """Whether the claim commits to something falsifiable."""
        return self.is_checkable and self.kind is not ClaimKind.ENTITY


class ClaimExtractor(Protocol):
    """The pluggable claim extractor seam (``S3-T6``)."""

    async def extract(self, text: str) -> list[Claim]:
        """Return the claims *text* asserts, in order of appearance."""


class GroundingLexicon:
    """The implicit-grounding lexicon (``S3-T7``).

    A closed set of phrases that *name* a fact instead of stating it. The rules
    never resolve them against a wall clock — a "today" with no date in the
    evidence is ungrounded, not silently compared to ``datetime.now()`` — which
    is what keeps the module deterministic (``S3-T4``).
    """

    #: phrase -> the reference kind it implies.
    DEICTIC: Mapping[str, ValueKind] = {
        "today": ValueKind.DATE,
        "tonight": ValueKind.DATE,
        "tomorrow": ValueKind.DATE,
        "yesterday": ValueKind.DATE,
        "this week": ValueKind.DATE,
        "next week": ValueKind.DATE,
        "this month": ValueKind.DATE,
        "last month": ValueKind.DATE,
        "this year": ValueKind.DATE,
        "right now": ValueKind.DATE,
        "currently": ValueKind.DATE,
        "at the moment": ValueKind.DATE,
        "now": ValueKind.DATE,
        "the latest": ValueKind.TEXT,
        "the newest": ValueKind.TEXT,
        "the most recent": ValueKind.TEXT,
        "the current": ValueKind.TEXT,
        "the current release": ValueKind.TEXT,
        "the latest release": ValueKind.TEXT,
        "the latest version": ValueKind.TEXT,
        "as of now": ValueKind.DATE,
    }

    #: Verbs that make a noun phrase an *authority the model is reporting from*
    #: ("the filing states", "the API returned"). A closed set on purpose: an
    #: open-ended verb list would start claiming every past-tense verb is a
    #: citation, and a lexicon that over-fires teaches operators to ignore it.
    ATTRIBUTIVE_VERBS: tuple[str, ...] = (
        "says",
        "said",
        "reports",
        "reported",
        "shows",
        "showed",
        "indicates",
        "indicated",
        "states",
        "stated",
        "confirms",
        "confirmed",
        "returns",
        "returned",
        "lists",
        "listed",
        "finds",
        "found",
        "contains",
        "contained",
        "reads",
        "notes",
        "noted",
        "documents",
        "documented",
        "specifies",
        "specified",
        "records",
        "recorded",
    )

    #: Prepositions that introduce a source *as the authority for what follows*.
    #: Split from the weak list because they license a bare proper noun
    #: ("according to Acme") where the weak ones require a "the <noun>" phrase
    #: ("based on the report") — "in Paris" must not read as a citation.
    ATTRIBUTIVE_STRONG: tuple[str, ...] = (
        "according to",
        "per",
        "citing",
        "as reported in",
        "as shown in",
        "as stated in",
        "as listed in",
        "as described in",
        "as documented in",
    )

    #: Weaker introducers. These need the determiner: they also open ordinary
    #: prose ("from the results of the migration" is attribution, "in 2023" is
    #: a date).
    ATTRIBUTIVE_WEAK: tuple[str, ...] = (
        "based on",
        "from",
        "in",
    )

    def __init__(self, phrases: Mapping[str, ValueKind] | None = None) -> None:
        """Create a lexicon over *phrases*, defaulting to :attr:`DEICTIC`."""
        self._phrases = dict(phrases if phrases is not None else self.DEICTIC)

    def matches(self, text: str) -> list[str]:
        """Which lexicon phrases *text* contains, longest first."""
        lowered = f" {normalize_text(text)} "
        found = [phrase for phrase in self._phrases if f" {phrase} " in lowered]
        return sorted(found, key=len, reverse=True)

    def reference(self, text: str) -> tuple[str, ValueKind] | None:
        """The first (longest) phrase *text* implies, with its kind."""
        for phrase in self.matches(text):
            return phrase, self._phrases[phrase]
        return None

    def attribution(self, text: str) -> str:
        """The source *text* names, or ``""`` if it names none.

        Two shapes, both closed:

        * ``<source> <attributive verb>`` — "the filing states", "the API
          returned 3". The noun phrase is the source.
        * ``<strong|weak introducer> <source>`` — "according to the Q3
          report", "based on the audit". The introducer is the signal and the
          noun phrase is again the source.

        Returning the source (not just a boolean) is what lets a finding say
        *which* document the model claims to have read, which is the difference
        between a reviewer actioning the flag and shrugging at it.
        """
        verb = _attribution_by_verb(text)
        if verb:
            return verb
        return _attribution_by_preposition(text, self)


#: The shared default lexicon.
DEFAULT_LEXICON = GroundingLexicon()


# ---------------------------------------------------------------------------
# attribution: which source a claim says it read (``S3-T8``)
# ---------------------------------------------------------------------------


def _noun_phrase_pattern(*, proper_noun: bool = True) -> str:
    """A short noun phrase, determiner-led or proper-noun.

    The proper-noun branch is wrapped in ``(?-i:...)`` deliberately. These
    patterns compile with :data:`re.IGNORECASE`, which would otherwise apply to
    ``[A-Z]`` too and make it match any lowercase word — so "costs $49 *per
    month*" reads as an attribution to "month", and every rate, duration and
    unit in the corpus would look like a cited source.
    """
    determiner = r"the\s+[a-z][\w\-]*(?:\s+(?:[a-z][\w\-]*|of|for|and)){0,3}"
    if not proper_noun:
        return determiner
    return determiner + r"|(?-i:[A-Z][\w&.]*(?:\s+[A-Z][\w&.]*){0,2})"


#: "<source> <attributive verb>" — the model reporting what a source said.
_ATTRIBUTION_BY_VERB_RE = re.compile(
    r"(?<![a-z0-9])(?P<source>"
    + _noun_phrase_pattern()
    + r")\s+(?:"
    + "|".join(GroundingLexicon.ATTRIBUTIVE_VERBS)
    + r")\b",
    re.IGNORECASE,
)

#: "<introducer> <source>" — the source offered as the authority. Strong
#: introducers accept a bare proper noun; weak ones require the determiner.
_ATTRIBUTION_STRONG_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(cue) for cue in GroundingLexicon.ATTRIBUTIVE_STRONG)
    + r")\s+(?P<source>"
    + _noun_phrase_pattern()
    + r")\b",
    re.IGNORECASE,
)

_ATTRIBUTION_WEAK_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(cue) for cue in GroundingLexicon.ATTRIBUTIVE_WEAK)
    + r")\s+(?P<source>"
    + _noun_phrase_pattern(proper_noun=False)
    + r")\b",
    re.IGNORECASE,
)

#: Words that make a noun phrase a *thing being discussed* rather than an
#: authority being cited. "The migration report shows the rate doubled" is the
#: model reasoning about a document; "the report shows the rate doubled" is the
#: model quoting it. Without this the weak introducers would fire on the first.
_NON_SOURCE_HEADS = frozenset(
    {
        "migration",
        "user",
        "review",
        "change",
        "issue",
        "bug",
        "update",
        "request",
        "response",
        "example",
        "test",
        "trial",
        "conversation",
        "log",
        "history",
        "changelog",
    }
)


def _attribution_by_verb(text: str) -> str:
    """The source named by a ``<source> <verb>`` shape, or ``""``."""
    for match in _ATTRIBUTION_BY_VERB_RE.finditer(text):
        source = match.group("source").strip()
        if _is_source_phrase(source):
            return source
    return ""


def _attribution_by_preposition(text: str, lexicon: GroundingLexicon) -> str:
    """The source named by an ``<introducer> <source>`` shape, or ``""``."""
    for pattern in (_ATTRIBUTION_STRONG_RE, _ATTRIBUTION_WEAK_RE):
        for match in pattern.finditer(text):
            source = match.group("source").strip()
            if _is_source_phrase(source):
                return source
    del lexicon  # the phrase sets live on the instance for customisation
    return ""


#: Leading determiners stripped before the head-noun test, so "the migration
#: report" is judged on "migration" rather than on "the".
_DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those", "our", "your"})


def _is_source_phrase(source: str) -> bool:
    """Whether *source* reads as a citable authority.

    Rejects "the user says" and "the test shows": a person or a test is not a
    source the model could have fabricated a citation to, and flagging those
    would be noise. So does "the migration report showed growth" — the model is
    reasoning *about* the report, not quoting it.
    """
    words = normalize_text(source).split()
    while words and words[0] in _DETERMINERS:
        words.pop(0)
    return bool(words) and words[0] not in _NON_SOURCE_HEADS


#: A counted subset of a counted whole, with the noun that makes it one:
#: "2 of 3 checks", "4/12 findings", "2 out of 5 samples".
_RATIO_RE = re.compile(
    r"\b(?P<num>\d+)\s*(?:of|out of|/)\s*(?P<den>\d+)\s+"
    r"(?P<noun>[A-Za-z]{3,}(?:s|es))\b",
    re.IGNORECASE,
)

#: A count is only checkable against an *enumeration* the reader can count too.
#: Below this, "3 of 4 items" is indistinguishable from a number that happens to
#: be followed by a plural noun.
_MIN_ENUMERATED_SIBLINGS = 3

#: Sibling identifiers in tool output: ``check_1``, ``error-2``, ``test3``. The
#: stem is what makes them one enumeration rather than an incidental collision,
#: which is why the count is taken per stem and the biggest stem wins.
_SIBLING_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<stem>[A-Za-z][A-Za-z]{1,20}?)[_-](?P<index>\d{1,4})(?![0-9A-Za-z])"
)


def _sibling_count(text: str) -> int:
    """How many siblings the largest ``stem_<n>`` family in *text* has."""
    families: dict[str, set[int]] = {}
    for match in _SIBLING_ID_RE.finditer(text):
        families.setdefault(normalize_text(match.group("stem")), set()).add(
            int(match.group("index"))
        )
    return max((len(members) for members in families.values()), default=0)


def _enumerated_count(text: str) -> int:
    """How many discrete items the evidence enumerates.

    Two independent shapes, and the larger answer wins:

    * sibling identifiers — ``check_1 .. check_4``;
    * labelled items — ``alpha: pass, beta: fail`` (the existing
      :data:`_KEY_VALUE_RE` shapes), counted by distinct label.

    Both need at least :data:`_MIN_ENUMERATED_SIBLINGS` items before they count
    as an enumeration. The floor is the false-positive guard: without it, any
    two stray numbers with a shared stem would "contradict" a count.
    """
    best = _sibling_count(text)
    labels: set[str] = set()
    for value in _key_values(text):
        labels.add(value.canonical)
    if len(labels) >= _MIN_ENUMERATED_SIBLINGS:
        best = max(best, len(labels))
    return best


def _raw_numbers(text: str) -> set[Decimal]:
    """Every plain number in *text*, scanned without claim interpretation.

    Deliberately not :func:`extract_values`: that masks ``key:`` prefixes, which
    is right for reading tool output as prose but would hide the "20" in
    ``Seats used: 12 of 20.`` — and a verbatim check that cannot see the number
    the tool actually printed is not a verbatim check.
    """
    found: set[Decimal] = set()
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("amount") or match.group("plain")
        number = _to_decimal(raw) if raw else None
        if number is not None:
            found.add(number)
    return found


# ---------------------------------------------------------------------------
# extraction (``S3-T5``)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Casefold, strip accents, and collapse whitespace/punctuation noise."""
    folded = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return collapsed.casefold()


#: Longest "subject" still treated as a noun phrase rather than a clause.
_MAX_BOOLEAN_SUBJECT_WORDS = 6

#: Subjects that name nothing. A claim about "it" cannot be grounded, so treating
#: it as a proposition would flag every pronoun in every response.
_UNRESOLVABLE_SUBJECTS = frozenset(
    {
        "he",
        "her",
        "him",
        "it",
        "one",
        "she",
        "that",
        "them",
        "these",
        "they",
        "those",
        "this",
        "we",
        "you",
    }
)

#: Trailing tokens whose period is part of the token, not a sentence end.
_ABBREVIATIONS = frozenset(
    {
        "approx",
        "dr",
        "e.g",
        "etc",
        "fig",
        "i.e",
        "inc",
        "ltd",
        "mr",
        "mrs",
        "ms",
        "no",
        "p50",
        "p90",
        "p95",
        "p99",
        "prof",
        "sec",
        "st",
        "vs",
    }
)


def split_sentences(text: str) -> list[str]:
    """Split prose into assertion-sized fragments.

    A period that closes an abbreviation or an initial does not end a sentence:
    tool output is full of "Dr." and "p95." and splitting on those would cut a
    claim in half before a rule ever saw it.
    """
    parts: list[str] = []
    for raw in _SENTENCE_SPLIT.split(text or ""):
        candidate = raw.strip()
        if not candidate:
            continue
        if parts and _ends_with_abbreviation(parts[-1]):
            parts[-1] = f"{parts[-1]} {candidate}"
            continue
        parts.append(candidate)
    return parts


def _ends_with_abbreviation(fragment: str) -> bool:
    """Whether *fragment* ends in a token whose period was part of the token."""
    head = fragment.rstrip(".?!:;").rsplit(" ", 1)[-1].casefold()
    if not head:
        return False
    return head in _ABBREVIATIONS or (len(head) == 1 and head.isalpha())


def is_assertion(fragment: str) -> bool:
    """Whether a fragment asserts something rather than conversing.

    Questions and conversational filler are dropped: they cannot be grounded,
    so flagging them would only inflate the false-positive rate.
    """
    lowered = normalize_text(fragment)
    if not lowered or "?" in fragment:
        return False
    if len(lowered) < 12:
        return False
    return not any(lowered.startswith(phrase) for phrase in _NON_ASSERTIONS)


class RuleBasedClaimExtractor:
    """Deterministic, rule-first claim extraction (``S3-T5``).

    Every rule is a surface pattern, so the same text always yields the same
    claims — a requirement for the published FP/FN rate (``S3-T13``) and for
    idempotent flag ids. ``lexicon`` supplies the implicit-grounding phrases; a
    future model-backed extractor implements :class:`ClaimExtractor` and ships as
    a new module version rather than changing these numbers.
    """

    def __init__(self, *, lexicon: GroundingLexicon | None = None) -> None:
        """Create an extractor, optionally overriding the grounding lexicon."""
        self._lexicon = lexicon or DEFAULT_LEXICON

    async def extract(self, text: str) -> list[Claim]:
        """Return every assertion in *text*, in order of appearance."""
        claims: list[Claim] = []
        for index, fragment in enumerate(split_sentences(text)):
            if not is_assertion(fragment):
                continue
            claim = self._classify(fragment, index)
            if claim is not None:
                claims.append(claim)
        return claims

    def extract_sync(self, text: str) -> list[Claim]:
        """Synchronous :meth:`extract`, for fixtures and hot loops."""
        claims: list[Claim] = []
        for index, fragment in enumerate(split_sentences(text)):
            if not is_assertion(fragment):
                continue
            claim = self._classify(fragment, index)
            if claim is not None:
                claims.append(claim)
        return claims

    # -- rules ------------------------------------------------------------

    def _classify(self, fragment: str, index: int) -> Claim | None:
        claim_id = f"c{index}"
        implicit = self._lexicon.matches(fragment)

        # Order is specificity-first: a date is a stronger, more checkable claim
        # than a number, which is stronger than a bare "is" statement. The
        # comparative and lexicon rules run before the boolean rule so
        # "Today is the biggest sale day" is read as a superlative with an
        # unresolved reference rather than as "today is true".
        date = _match_date(fragment)
        if date is not None:
            return self._claim(claim_id, fragment, ClaimKind.DATE, date, "date", index)

        membership = self._match_membership(fragment)
        if membership is not None:
            return self._claim(
                claim_id, fragment, ClaimKind.SET, membership, "set-membership", index
            )

        duration = self._match_duration(fragment)
        if duration is not None:
            return self._claim(claim_id, fragment, ClaimKind.DURATION, duration, "duration", index)

        comparative = self._match_comparative(fragment, implicit, claim_id, index)
        if comparative is not None:
            return comparative

        ratio = self._match_ratio(fragment)
        if ratio is not None:
            return self._claim(claim_id, fragment, ClaimKind.RATIO, ratio, "ratio", index)

        if implicit:
            phrase, kind = self._lexicon.reference(fragment) or ("", ValueKind.TEXT)
            return Claim(
                claim_id=claim_id,
                text=fragment,
                kind=ClaimKind.VAGUE,
                value=Value(kind=kind, canonical=normalize_text(phrase), raw=phrase),
                cue=f"implicit:{phrase}",
                index=index,
                terms=_terms(fragment),
                source_attributed=self._lexicon.attribution(fragment),
            )

        number = self._match_number(fragment, allow_units=_COUNT_UNITS)
        if number is not None:
            return self._claim(claim_id, fragment, ClaimKind.NUMERIC, number, "number", index)

        boolean = self._match_boolean(fragment)
        if boolean is not None:
            return self._claim(claim_id, fragment, ClaimKind.BOOLEAN, boolean, "boolean", index)

        entity = self._match_entity(fragment)
        if entity is not None:
            return self._claim(claim_id, fragment, ClaimKind.ENTITY, entity, "entity", index)

        return None

    def _match_comparative(
        self,
        fragment: str,
        implicit: Sequence[str],
        claim_id: str,
        index: int,
    ) -> Claim | None:
        """A superlative or absolute claim ("biggest", "never", "everyone").

        A superlative is checkable in principle — something has to have ranked
        it — so it is kept as a claim whose evidence requirement is a *ranking*
        statement, not a value match. A deictic phrase alongside it ("today is
        the biggest…") makes it worse, not better, so the cue records both.

        Bound phrasing ("at most 3 seats") is skipped even though "most" and
        "least" are superlatives: the number there is a limit, and the bound
        rules read it directly.
        """
        lowered = normalize_text(fragment)
        if _BOUND_RE.search(lowered) is not None:
            return None
        for word in _COMPARATIVE_WORDS:
            if not re.search(rf"\b{re.escape(word)}\b", lowered):
                continue
            cue = f"comparative:{word}"
            if implicit:
                cue = f"{cue}+implicit:{implicit[0]}"
            return Claim(
                claim_id=claim_id,
                text=fragment,
                kind=ClaimKind.COMPARATIVE,
                value=Value(kind=ValueKind.TEXT, canonical=word, raw=word),
                cue=cue,
                index=index,
                terms=_terms(fragment),
            )
        return None

    def _claim(
        self,
        claim_id: str,
        fragment: str,
        kind: ClaimKind,
        value: Value,
        cue: str,
        index: int,
    ) -> Claim:
        return Claim(
            claim_id=claim_id,
            text=fragment,
            kind=kind,
            value=value,
            cue=cue,
            index=index,
            terms=_terms(fragment),
            source_attributed=self._lexicon.attribution(fragment),
        )

    def _match_ratio(self, fragment: str) -> Value | None:
        """A counted subset of a counted whole: "2 of 3 checks passed".

        The trailing noun is required. "2 of 3" on its own is ambiguous enough
        (a score, a version, a date range) that treating it as a checkable count
        would be a guess, and a guess here becomes a false positive. Requiring
        the noun means the rule stays silent on anything it cannot reason about.
        """
        match = _RATIO_RE.search(fragment)
        if match is None:
            return None
        numerator = _to_decimal(match.group("num"))
        denominator = _to_decimal(match.group("den"))
        if numerator is None or denominator is None or denominator == 0:
            return None
        if numerator > denominator:
            return None
        return Value(
            kind=ValueKind.RATIO,
            canonical=f"{_render_number(numerator)} of {_render_number(denominator)}",
            number=numerator,
            ratio=(numerator, denominator),
            raw=match.group(0),
        )

    def _match_membership(self, fragment: str) -> Value | None:
        match = _SET_RE.search(fragment) or _LIST_RE.search(fragment)
        if match is None:
            return None
        members = _split_members(match.group("members"))
        if len(members) < 2:
            return None
        return Value(
            kind=ValueKind.MEMBER,
            canonical="member",
            members=tuple(members),
            raw=match.group(0),
        )

    def _match_duration(self, fragment: str) -> Value | None:
        value = self._match_number(fragment, require_units=_DURATION_UNIT_NAMES)
        if value is None:
            return None
        return value

    def _match_boolean(self, fragment: str) -> Value | None:
        if not _BOOL_AFFIRM.search(fragment):
            return None
        negated, subject = _scoped_negation(fragment)
        if not negated and not subject:
            # An affirmative verb with no subject ("is available") is too thin
            # to be a checkable proposition; a negation is a claim on its own.
            return None
        if len(subject.split()) > _MAX_BOOLEAN_SUBJECT_WORDS:
            # The text before the verb is a clause, not a subject: "It is
            # important for us to keep improving reliability" asserts something
            # real, but not something a boolean check can refute.
            return None
        if subject and subject.split()[-1] in _UNRESOLVABLE_SUBJECTS:
            return None
        return Value(
            kind=ValueKind.BOOLEAN,
            canonical=f"{subject or ''}|{negated}",
            negated=negated,
            raw=fragment,
        )

    def _match_entity(self, fragment: str) -> Value | None:
        quoted = _QUOTED_RE.search(fragment)
        if quoted is not None:
            text = normalize_text(quoted.group("text"))
            if len(text) >= 3:
                return Value(kind=ValueKind.ENTITY, canonical=text, raw=quoted.group(0))
        for match in _ENTITY_RE.finditer(fragment):
            candidate = normalize_text(match.group(0))
            if candidate in _STOPWORD_ENTITIES or len(candidate) < 6:
                continue
            return Value(kind=ValueKind.ENTITY, canonical=candidate, raw=match.group(0))
        return None

    def _match_number(
        self,
        fragment: str,
        *,
        require_units: Collection[str] | None = None,
        allow_units: Collection[str] | None = None,
    ) -> Value | None:
        """The first quantity in *fragment* whose unit this rule can check.

        A number with no unit is a plain count (checkable); a number with an
        unknown unit (``1.5 GB``) or one this rule does not handle is skipped
        rather than guessed at, which keeps the extractor conservative.
        """
        negated = bool(_NEGATION_RE.search(fragment))
        for match in _NUMBER_RE.finditer(fragment):
            if match.group("currency"):
                raw_amount, raw_unit = match.group("amount"), "$"
            else:
                raw_amount, raw_unit = match.group("plain"), match.group("unit")
            if not raw_amount:
                continue
            unit = _UNIT_ALIASES.get((raw_unit or "").casefold())
            if raw_unit and unit is None:
                continue
            if require_units is not None and unit not in require_units:
                continue
            if allow_units is not None and unit is not None and unit not in allow_units:
                continue
            number = _to_decimal(raw_amount)
            if number is None or _is_bare_year(number):
                continue
            return Value(
                kind=ValueKind.NUMBER,
                canonical=f"{number}{unit or ''}",
                number=number,
                unit=unit,
                negated=negated,
                raw=match.group(0),
            )
        return None


_STOPWORD_ENTITIES = frozenset(
    {
        "the user",
        "the agent",
        "the assistant",
        "the model",
        "the tool",
        "the system",
        "the service",
        "the order",
        "the account",
        "the environment",
        "the database",
        "the request",
    }
)


def _split_members(text: str) -> list[str]:
    """Split an enumeration into its members.

    ``"Supported regions are us-east, eu-west"`` yields ``["us-east", "eu-west"]``
    — the subject and its copula come off, because "regions are" is not a member
    and would otherwise make the set look like it had an extra element.
    """
    cleaned = re.sub(r"\s+(?:and|or)\s+", ",", text)
    members: list[str] = []
    for part in cleaned.split(","):
        candidate = normalize_text(part).strip(" .:;-")
        tail = re.split(
            r"\b(?:are|is|were|was|include[sd]?|available|supported|accept(?:s|ed)?)\b",
            candidate,
        )
        candidate = (tail[-1] if len(tail) > 1 else tail[0]).strip(" .:;-")
        if candidate and _NUMBER_IN_TEXT.search(candidate) is None and len(candidate) > 1:
            members.append(candidate)
    return members


def _terms(fragment: str) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9$%.-]{3,}", normalize_text(fragment))
    stop = _TERM_STOPWORDS
    return tuple(word for word in words if word not in stop)


#: The verb a boolean claim pivots on, and how much text may sit between the verb
#: and its negation. "does not accept" negates; "accepted ... with no fee" does
#: not — the second "no" belongs to a different noun phrase, and reading it as a
#: negation is how "no fee" turns an affirmative into a denial.
_BOOL_VERB_RE = re.compile(
    r"\b(?P<verb>is|are|was|were|does|do|did|has|have|had|can|could|will|would|"
    r"supports?|accepted|accepts?|include[sd]?|includes?|allows?|provides?|works?|"
    r"offers?|ships?|supports?)\b",
    re.IGNORECASE,
)
_NEGATION_WINDOW = 24


def _scoped_negation(fragment: str) -> tuple[bool, str]:
    """Whether the leading verb is negated, plus the subject before it.

    Returns ``(negated, subject)``; ``subject`` is the normalized noun phrase the
    verb pivots on, which is what lets a later rule line the claim's subject up
    against the evidence's.
    """
    match = _BOOL_VERB_RE.search(fragment)
    if match is None:
        return False, ""
    subject = normalize_text(fragment[: match.start()]).strip(" .,;:-")
    subject = re.sub(r"^(?:the|a|an|our|your|its|their|his|her|my)\s+", "", subject)
    tail = fragment[match.end() : match.end() + _NEGATION_WINDOW]
    negated = bool(
        re.match(
            r"\s*(?:not\b|n't\b|never\b|no longer\b|cannot\b|can't\b|does not\b|do not\b|"
            r"isn't\b|aren't\b|doesn't\b|don't\b)",
            tail,
            re.IGNORECASE,
        )
    )
    return negated, subject


_TERM_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "has",
        "have",
        "its",
        "our",
        "your",
        "you",
        "all",
        "any",
        "not",
        "but",
        "can",
        "will",
        "were",
        "been",
        "there",
        "their",
        "which",
        "when",
        "into",
        "than",
        "then",
        "them",
        "they",
    }
)


def _match_date(fragment: str) -> Value | None:
    iso = _ISO_DATE_RE.search(fragment)
    if iso is not None:
        return Value(kind=ValueKind.DATE, canonical=iso.group("iso"), raw=iso.group(0))
    weekday = _WEEKDAY_RE.search(fragment)
    if weekday is not None:
        return Value(
            kind=ValueKind.WEEKDAY,
            canonical=weekday.group("day").casefold(),
            raw=weekday.group(0),
        )
    long_date = _LONG_DATE_RE.search(fragment)
    if long_date is not None:
        month = long_date.group("month").casefold()
        day = long_date.group("day")
        year = long_date.group("year") or ""
        return Value(
            kind=ValueKind.DATE,
            canonical=f"{month}-{int(day):02d}-{year}".rstrip("-"),
            raw=long_date.group(0),
        )
    return None


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _is_bare_year(number: Decimal) -> bool:
    """A four-digit 1900-2100 number is a year, not a quantity.

    "Shipped in May 2024" must not become the claim "2024 units"; years are
    checked as part of a date instead.
    """
    return number == number.to_integral_value() and 1900 <= int(number) <= 2100


def extract_values(text: str) -> list[Value]:
    """Every value in *text*, in order — the evidence-side counterpart of claims.

    Tool output is unstructured, so two passes run over it: the claim rules (so
    a result that reads like a sentence is read like one) and a ``key = value``
    pass, because tools answer in that shape far more often than in prose and
    the unit usually hides in the key (``export_duration_ms = 120000``).
    """
    values: list[Value] = []
    extractor = RuleBasedClaimExtractor()
    for fragment in split_sentences(text) or [text]:
        values.extend(
            claim.value
            for claim in extractor.extract_sync(_mask_key_names(fragment))
            if claim.value is not None
        )
        values.extend(_boolean_values(fragment))
    values.extend(_key_values(text))
    return values


def _mask_key_names(text: str) -> str:
    """Blank out the key of a ``key = number`` row before the prose pass.

    ``p95_latency_ms = 5.04`` would otherwise yield a phantom "95" from inside
    the key, and a phantom number is enough to manufacture a disagreement.
    """
    masked = text
    for match in _KEY_VALUE_RE.finditer(text):
        start, end = match.span("key")
        masked = f"{masked[:start]}{' ' * (end - start)}{masked[end:]}"
    return masked


def _key_values(text: str) -> list[Value]:
    """``key = number`` rows, with a unit recovered from the key when present."""
    values: list[Value] = []
    for match in _KEY_VALUE_RE.finditer(text):
        number = _to_decimal(match.group("value"))
        if number is None:
            continue
        key = normalize_text(match.group("key"))
        unit = ""
        for suffix, mapped in _KEY_UNITS:
            if key.endswith((f"_{suffix}", suffix)):
                unit = mapped
                break
        values.append(
            Value(
                kind=ValueKind.NUMBER,
                canonical=f"{number}{unit}",
                number=number,
                unit=unit or None,
                raw=match.group(0).strip(),
            )
        )
    return values


def _boolean_values(fragment: str) -> list[Value]:
    """Boolean propositions asserted by a piece of tool output.

    Uses the same scoped-negation rule as the claim side, so "cancellations are
    accepted with no fee" stays an affirmative.
    """
    values: list[Value] = []
    for match in _BOOL_VERB_RE.finditer(fragment):
        # The subject is what precedes *this* verb, so the prefix up to the verb
        # is what gets analyzed — not the fragment from the verb onwards, which
        # would leave every subject empty.
        negated, subject = _scoped_negation(fragment[: match.end()])
        if not subject or len(subject) < 3:
            continue
        tail = fragment[match.end() :]
        if not tail.strip() or _NUMBER_IN_TEXT.search(tail[:40]):
            continue
        values.append(
            Value(
                kind=ValueKind.BOOLEAN,
                canonical=f"{subject}|{negated}",
                negated=negated,
                raw=fragment.strip(),
            )
        )
    return values


# ---------------------------------------------------------------------------
# diff (``S3-T8`` - ``S3-T10``)
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    """What the evidence says about a claim."""

    #: The claim's value appears in cited evidence, essentially verbatim.
    SUPPORTED = "supported"
    #: The claim follows from evidence without appearing verbatim (rounding,
    #: unit conversion, subset, bound).
    IMPLIED = "implied"
    #: The evidence states something incompatible with the claim.
    CONFLICTED = "conflicted"
    #: The claim names a source it never cited, and the evidence is silent.
    #: Narrower than :attr:`UNKNOWN`, and only reachable when the session
    #: actually records citations (see :attr:`DiffContext.citations_recorded`).
    UNSOURCED = "unsourced"
    #: The evidence is silent. For a specific claim this is a real finding.
    UNKNOWN = "unknown"


class SupportKind(StrEnum):
    """How a support or implication was reached — the audit trail of a verdict."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    UNIT_CONVERSION = "unit_conversion"
    ROUNDING = "rounding"
    SUBSET = "subset"
    BOUND = "bound"
    EXCLUSION = "exclusion"
    NEGATION = "negation"
    DISAGREEMENT = "disagreement"
    #: The claim's counted denominator is smaller than the source's item count.
    CHERRY_PICK = "cherry_pick"
    NONE = "none"


#: Bound phrasing found in tool output: "at least 3", "up to 50", "no more than".
_BOUND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bat least\s+(?P<n>\d[\d.,]*)", re.I), "low"),
    (re.compile(r"\b(?:more than|greater than|over|above)\s+(?P<n>\d[\d.,]*)", re.I), "low"),
    (
        re.compile(
            r"\b(?:at most|up to|no more than|not more than|maximum of|max)\s+(?P<n>\d[\d.,]*)",
            re.I,
        ),
        "high",
    ),
    (
        re.compile(r"\b(?:less than|fewer than|under|below)\s+(?P<n>\d[\d.,]*)", re.I),
        "high",
    ),
)

#: Any bound phrasing, as a bare probe (the patterns above capture the number).
_BOUND_RE = re.compile(
    r"\b(?:at least|at most|up to|no more than|not more than|maximum of|max|"
    r"more than|greater than|over|above|less than|fewer than|under|below)\b",
    re.IGNORECASE,
)

_EXCLUSION_RE = re.compile(
    r"\b(?:does not|do not|don't|doesn't|cannot|can't|not accepted|no longer|"
    r"excluded?|unsupported|unavailable|refuses?)\b",
    re.IGNORECASE,
)

#: Phrases in tool output that mean "this is a limit, not a contradiction".
#: A rate ("$49 per month") is deliberately absent: a rate the claim states
#: differently is a contradiction, not a limit.
_LIMIT_PHRASES = (
    "limit",
    "maximum",
    "max ",
    "up to",
    "policy",
    "fee",
    "at least",
    "starting at",
    "approximately",
    "about ",
    "rounded",
    "estimate",
)


@dataclass(frozen=True)
class DiffResult:
    """The verdict on one claim, with the evidence that produced it."""

    verdict: Verdict
    support_kind: SupportKind = SupportKind.NONE
    detail: str = ""
    cited: bool = False
    matched_value: str = ""
    #: The shortest token that identifies what the evidence said (``"49"``, or a
    #: member list). ``matched_value`` is the human phrasing; this is what
    #: evidence notes are matched against, because a note never spells out the
    #: unit alias the canonical form folds to.
    observed: str = ""

    @property
    def is_grounded(self) -> bool:
        """Whether the claim may be stated without a caveat."""
        return self.verdict in (Verdict.SUPPORTED, Verdict.IMPLIED)

    @property
    def is_finding(self) -> bool:
        """Whether this verdict should become a flag at all."""
        return self.verdict in (
            Verdict.CONFLICTED,
            Verdict.UNSOURCED,
            Verdict.UNKNOWN,
        )


@dataclass
class DiffContext:
    """Everything the diff may reason over for one response event.

    ``explicit`` is the text of the results the response *cited*
    (:attr:`~sentinel.models.events.RefKind.GROUNDS`) — a cited value is a
    stronger signal than an uncited one. ``context`` is the wider tool output
    the response had available. ``implied`` holds lexicon-resolved references
    (for example a deictic phrase and the date evidence offers for it).

    ``citations_recorded`` is a statement about the *instrumentation*, not the
    response: does this session record :attr:`~sentinel.models.events.RefKind.
    GROUNDS` refs anywhere at all? Without it, "this response cites nothing"
    is indistinguishable from "this framework never emits citations", and a
    fabricated source would be unprovable. With it set, a claim that *names* a
    source while citing none becomes :attr:`Verdict.UNSOURCED` rather than a
    vague :attr:`Verdict.UNKNOWN`. It is a session-level fact because one
    response citing nothing says little and a whole session citing nothing says
    almost nothing.
    """

    explicit: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    implied: Mapping[str, str] = field(default_factory=dict)
    citations_recorded: bool = False

    @property
    def all_text(self) -> str:
        """Every piece of evidence as one searchable blob."""
        return " \n ".join(self.explicit + self.context)

    @property
    def has_evidence(self) -> bool:
        """Whether any tool output was available at all."""
        return bool(self.explicit or self.context)

    @classmethod
    def empty(cls) -> DiffContext:
        """A context with no evidence: everything specific is ungrounded."""
        return cls()


def diff_claim(claim: Claim, context: DiffContext) -> DiffResult:
    """Decide whether *context* supports, implies, or contradicts *claim*.

    The order of the rules is the design: an explicit match wins, then
    implications, then contradictions, and only silence is left as
    :attr:`Verdict.UNKNOWN`. A claim that cannot be checked
    (``claim.is_checkable`` is ``False``) is never a finding.
    """
    value = claim.value
    if value is None or not claim.is_checkable:
        return DiffResult(verdict=Verdict.UNKNOWN, detail="claim is not checkable")

    if value.kind is ValueKind.RATIO and value.ratio is not None:
        return _diff_ratio(value, context)

    explicit_values = _values_of(context.explicit)
    context_values = _values_of(context.context)
    implicit = _implicit_values(claim, context)

    # 1. explicit citation
    hit = _explicit_match(value, explicit_values, context.explicit)
    if hit is not None:
        return DiffResult(
            verdict=Verdict.SUPPORTED,
            support_kind=SupportKind.EXPLICIT,
            detail=f"cited evidence contains {value.canonical}",
            cited=True,
            matched_value=hit,
            observed=hit,
        )
    if context.has_evidence:
        hit = _explicit_match(value, context_values, context.context)
        if hit is not None:
            return DiffResult(
                verdict=Verdict.SUPPORTED,
                support_kind=SupportKind.EXPLICIT,
                detail=f"available tool output contains {value.canonical}",
                matched_value=hit,
                observed=hit,
            )

    # 2. implicit lexicon support
    implied = _implicit_match(value, implicit)
    if implied is not None:
        return DiffResult(
            verdict=Verdict.SUPPORTED,
            support_kind=SupportKind.IMPLICIT,
            detail=implied[1],
            matched_value=implied[0],
            observed=implied[0],
        )

    # 3. implication: units, rounding, subset, bounds
    #
    # Cited evidence counts here too. "5.0 ms" is implied by a cited "5.04" and
    # by an uncited one, and the cited case is the stronger of the two.
    implied_result = _implication(value, (*explicit_values, *context_values), context.all_text)
    if implied_result is not None:
        return implied_result

    # 4. contradiction
    conflict = _contradiction(claim, value, context, explicit_values, context_values)
    if conflict is not None:
        return conflict

    # 5. named but never cited
    #
    # Deliberately *last*. Every rule that can find support has already run, so
    # this verdict can only ever reclassify a claim that was going to be
    # flagged as ungrounded anyway — it can never turn silence into a new flag.
    # That ordering is the whole false-positive argument for this feature: the
    # attribution patterns are allowed to be generous, because a wrong match
    # costs a category label, not a flag an operator has to triage.
    unsourced = _unsourced(claim, context)
    if unsourced is not None:
        return unsourced

    # 6. silence
    if not context.has_evidence:
        return DiffResult(
            verdict=Verdict.UNKNOWN,
            detail="no tool output available to ground the claim",
        )
    return DiffResult(
        verdict=Verdict.UNKNOWN,
        detail="available tool output does not mention the claimed value",
    )


def _unsourced(claim: Claim, context: DiffContext) -> DiffResult | None:
    """The claim names a source it never cited (``S3-T8``).

    Returns ``None`` unless all of the following hold, because a fabricated
    citation has to be *provable* from the log:

    * the claim names a source at all (no name, no claim of having read it);
    * the session records :attr:`~sentinel.models.events.RefKind.GROUNDS` refs
      somewhere — otherwise "this response cited nothing" is indistinguishable
      from a framework that never emits citations, and the whole rule would be
      guessing;
    * this response cited nothing, having had the chance to;
    * the evidence is silent, so there is no support to defer to.
    """
    if not claim.source_attributed:
        return None
    if not context.citations_recorded:
        return None
    if context.explicit:
        return None
    if not context.has_evidence:
        detail = (
            f"attributed to {claim.source_attributed} but no source was consulted "
            f"or cited in this turn"
        )
    else:
        detail = (
            f"attributed to {claim.source_attributed} but the response cites no "
            f"tool result, and the available output does not support the claim"
        )
    return DiffResult(
        verdict=Verdict.UNSOURCED,
        detail=detail,
        observed=claim.source_attributed,
    )


def _diff_ratio(value: Value, context: DiffContext) -> DiffResult:
    """Check a counted subset against the items the source enumerates.

    A count is a statement about a *set*, so what matters is whether the set
    the agent counted over is the set the source described. Three ways that can
    end, in this order:

    1. the source enumerates **more** items than the claim's denominator — the
       agent counted a subset and reported it as the whole, which is what
       cherry-picking looks like when it is written down;
    2. the source states both numbers verbatim — the count was copied, not
       computed, and there is nothing to second-guess. Without this, ordinary
       prose like "12 of 20 seats" would be reported as uncheckable noise;
    3. otherwise the count could not be checked at all.

    The enumeration is read from *cited* evidence when the response cited
    anything, matching the rest of the module: a count is a statement about the
    sources the agent says it read, not about every result that happened to be
    in the turn.
    """
    numerator, denominator = value.ratio or (Decimal(0), Decimal(0))
    canonical = value.render()
    if not context.has_evidence:
        return DiffResult(
            verdict=Verdict.UNKNOWN,
            detail="no tool output available to ground the count",
        )

    counted = _enumerated_count(
        " \n ".join(context.explicit) if context.explicit else context.all_text
    )
    if counted > denominator:
        return DiffResult(
            verdict=Verdict.CONFLICTED,
            support_kind=SupportKind.CHERRY_PICK,
            detail=(
                f"reports {canonical} but the source enumerates {counted} items, "
                f"so at least {counted - int(denominator)} were left out"
            ),
            cited=bool(context.explicit),
            # ``matched_value`` is what a reviewer reads as the evidence's
            # counter-statement, so it carries the enumeration, not the claim.
            matched_value=f"{counted} items",
            observed=str(counted),
        )
    if counted == denominator:
        return DiffResult(
            verdict=Verdict.SUPPORTED,
            support_kind=SupportKind.EXPLICIT,
            detail=f"source enumerates {counted} items, matching the claimed count",
            cited=bool(context.explicit),
            matched_value=canonical,
            observed=canonical,
        )
    stated = _raw_numbers(context.all_text)
    if numerator in stated and denominator in stated:
        return DiffResult(
            verdict=Verdict.SUPPORTED,
            support_kind=SupportKind.EXPLICIT,
            detail=f"source states {canonical} verbatim",
            cited=bool(context.explicit),
            matched_value=canonical,
            observed=canonical,
        )
    return DiffResult(
        verdict=Verdict.UNKNOWN,
        detail=(f"the source does not enumerate a countable set, so {canonical} cannot be checked"),
    )


def _values_of(texts: Iterable[str]) -> list[Value]:
    values: list[Value] = []
    for text in texts:
        values.extend(extract_values(text))
    return values


def _explicit_match(
    claim_value: Value,
    values: Sequence[Value],
    evidence_text: Sequence[str] = (),
) -> str | None:
    """Whether *values* state the claim's value outright.

    ``evidence_text`` is needed for the kinds whose support is lexical rather
    than numeric — a superlative is supported by the evidence *ranking*
    something, not by matching a number.
    """
    if claim_value.kind is ValueKind.TEXT:
        # Checked against the text, not against extracted values: "the biggest
        # sale day" is supported by the evidence ranking something, and the
        # evidence sentence holding that ranking may well be read as a date.
        blob = normalize_text(" ".join(evidence_text))
        if claim_value.canonical in blob and _has_ranking(blob):
            return claim_value.canonical

    for value in values:
        if claim_value.kind is ValueKind.DATE or value.kind is ValueKind.DATE:
            if claim_value.canonical == value.canonical and value.kind is ValueKind.DATE:
                return value.canonical
            continue
        if claim_value.kind is ValueKind.WEEKDAY or value.kind is ValueKind.WEEKDAY:
            if claim_value.kind is value.kind and claim_value.canonical == value.canonical:
                return value.canonical
            continue
        if claim_value.kind is ValueKind.MEMBER or value.kind is ValueKind.MEMBER:
            # "Card is one of card, wire, crypto" against "we accept card and
            # wire" is *not* support: the claim adds a member the evidence never
            # offered. Only a claimed set the evidence covers counts.
            if (
                claim_value.members
                and value.members
                and set(claim_value.members) <= set(value.members)
            ):
                return "|".join(sorted(set(claim_value.members) & set(value.members)))
            continue
        if claim_value.kind is ValueKind.BOOLEAN and value.kind is ValueKind.BOOLEAN:
            if claim_value.negated == value.negated and claim_value.canonical == value.canonical:
                return value.canonical
            continue
        if claim_value.kind is ValueKind.TEXT:
            continue
        if claim_value.kind is ValueKind.ENTITY:
            if claim_value.canonical and claim_value.canonical in value.canonical:
                return claim_value.canonical
            continue
        if claim_value.number is not None and value.number is not None:
            # Same number *and* same unit is an explicit match. Equal magnitude
            # across different units is not: it is reported as a conversion by
            # the implication rule, so the audit trail says the evidence said
            # milliseconds when the claim said minutes.
            if claim_value.number == value.number and claim_value.unit == value.unit:
                return value.token()
            if claim_value.canonical == value.canonical:
                return value.canonical
    return None


def _has_ranking(blob: str) -> bool:
    """Whether *blob* ranks anything at all."""
    return any(marker in blob for marker in _RANKING_MARKERS)


def _implicit_values(claim: Claim, context: DiffContext) -> list[Value]:
    """Values the evidence offers for the claim's implicit phrase.

    The lexicon phrase is resolved *against the evidence only* — never against a
    clock — so ``"today"`` stays a reference until a tool result names a date.
    """
    resolved: list[Value] = []
    for phrase, offered in context.implied.items():
        if phrase and phrase in normalize_text(claim.text):
            for text in (offered,) if isinstance(offered, str) else ():
                resolved.extend(extract_values(text))
    return resolved


def _implicit_match(claim_value: Value, values: Sequence[Value]) -> tuple[str, str] | None:
    hit = _explicit_match(claim_value, values)
    if hit is None:
        return None
    return hit, "evidence names the value behind the claim's implicit reference"


def _implication(
    claim_value: Value,
    values: Sequence[Value],
    evidence_text: str,
) -> DiffResult | None:
    """Support that is not a verbatim match (``S3-T10``)."""
    if claim_value.kind is ValueKind.MEMBER and claim_value.members:
        for value in values:
            if value.kind is not ValueKind.MEMBER or not value.members:
                continue
            if set(claim_value.members) < set(value.members):
                return DiffResult(
                    verdict=Verdict.IMPLIED,
                    support_kind=SupportKind.SUBSET,
                    detail=(f"claimed set is a subset of {value.canonical or 'the available set'}"),
                    matched_value=value.render(),
                    observed=value.token(),
                )
        return None

    if claim_value.number is None:
        return None

    # unit conversion / plain equality
    for value in values:
        if value.number is None:
            continue
        if claim_value.unit != value.unit and claim_value.same_magnitude(value):
            return DiffResult(
                verdict=Verdict.IMPLIED,
                support_kind=SupportKind.UNIT_CONVERSION,
                detail=f"{claim_value.canonical} equals {value.canonical} after unit conversion",
                matched_value=value.render(),
                observed=value.token(),
            )
        if (
            claim_value.unit is None
            and value.unit is not None
            and claim_value.number == value.number
        ):
            return DiffResult(
                verdict=Verdict.IMPLIED,
                support_kind=SupportKind.UNIT_CONVERSION,
                detail=f"unqualified number matches {value.canonical}",
                matched_value=value.render(),
                observed=value.token(),
            )
        if (
            claim_value.unit in _DURATION_TO_MS
            and value.unit is None
            and _to_milliseconds(claim_value) == value.number
        ):
            return DiffResult(
                verdict=Verdict.IMPLIED,
                support_kind=SupportKind.UNIT_CONVERSION,
                detail=(
                    f"{claim_value.canonical} equals {value.number} once converted to "
                    f"{claim_value.unit}"
                ),
                matched_value=value.render(),
                observed=value.token(),
            )
        if _is_rounding_of(claim_value, value):
            return DiffResult(
                verdict=Verdict.IMPLIED,
                support_kind=SupportKind.ROUNDING,
                detail=f"{claim_value.number} is {claim_value.canonical}, a rounded form of "
                f"{value.number}",
                matched_value=value.render(),
                observed=value.token(),
            )

    # bounds
    bounds = _bounds((evidence_text,))
    if bounds and claim_value.in_bounds(bounds[0], bounds[1]):
        return DiffResult(
            verdict=Verdict.IMPLIED,
            support_kind=SupportKind.BOUND,
            detail=f"{claim_value.number} satisfies the stated bound",
        )
    return None


def _to_milliseconds(value: Value) -> Decimal | None:
    if value.number is None or value.unit is None:
        return None
    factor = _DURATION_TO_MS.get(value.unit)
    return None if factor is None else value.number * factor


def _is_rounding_of(claim_value: Value, candidate: Value) -> bool:
    """Whether *claim_value* is a rounded form of *candidate*.

    "5.0 ms" against "5.04" is rounding, not disagreement: the claim committed
    to one decimal place, and 5.04 is inside half a unit of it. The tolerance
    comes from the claim's own precision, so a claim of "5" is not allowed to
    round-match "5.04".
    """
    if claim_value.number is None or candidate.number is None:
        return False
    if claim_value.unit != candidate.unit:
        return False
    if claim_value.number == candidate.number:
        return False
    places = _decimals(claim_value.raw)
    tolerance = Decimal(1).scaleb(-places) / 2
    return abs(claim_value.number - candidate.number) <= tolerance


def _decimals(raw: str) -> int:
    """How many decimal places a written number committed to."""
    _, _, fraction = (raw or "").partition(".")
    return len(fraction) if fraction.isdigit() else 0


def _render_number(number: Decimal) -> str:
    """A Decimal as written: no exponent, no trailing zeros it never had."""
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _bounds(texts: Sequence[str]) -> tuple[Decimal | None, Decimal | None] | None:
    """Collect an inclusive range from bound phrasing in the evidence.

    The phrasing is read from the evidence *text*, not from each value's raw
    match: the number regex returns "3" for "at most 3 seats", so the limit is
    only visible in the sentence around it.
    """
    low: Decimal | None = None
    high: Decimal | None = None
    found = False
    for raw in (text or "" for text in texts):
        for pattern, side in _BOUND_PATTERNS:
            match = pattern.search(raw)
            if match is None:
                continue
            number = _to_decimal(match.group("n"))
            if number is None:
                continue
            found = True
            if side == "low":
                low = number if low is None else max(low, number)
            else:
                high = number if high is None else min(high, number)
    return (low, high) if found else None


#: Magnitude ratio beyond which two numbers are treated as different quantities
#: rather than as conflicting values.
_MAX_DISAGREEMENT_RATIO = Decimal(100)

#: Terms too generic to anchor "these two sentences are about the same thing".
#: This is the FP budget of the disagreement rule: a claim about the pro plan
#: must not be refuted by the enterprise plan's price, and "plan" alone does not
#: tell the two apart — only "pro" does.
_GENERIC_TERMS = frozenset(
    {
        "all",
        "and",
        "are",
        "average",
        "can",
        "count",
        "costs",
        "cost",
        "current",
        "day",
        "does",
        "each",
        "every",
        "for",
        "from",
        "have",
        "includes",
        "including",
        "into",
        "its",
        "month",
        "not",
        "number",
        "only",
        "per",
        "plan",
        "price",
        "provides",
        "rate",
        "seats",
        "supports",
        "team",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "time",
        "total",
        "users",
        "value",
        "was",
        "week",
        "were",
        "with",
        "year",
        "you",
        "your",
    }
)


def _units_comparable(left: Value, right: Value) -> bool:
    """Whether two numbers could be stated in the same terms.

    A unitless "500" cannot contradict "$29": the evidence never said anything
    about price. Durations compare across their units ("2 min" and "120000 ms"),
    everything else only against its own unit.
    """
    if bool(left.unit) != bool(right.unit):
        return False
    if left.unit == right.unit:
        return True
    return bool(
        left.unit and right.unit and left.unit in _DURATION_TO_MS and right.unit in _DURATION_TO_MS
    )


def _numeric_disagreement(
    claim: Claim,
    value: Value,
    context: DiffContext,
) -> DiffResult | None:
    """Refute a number the evidence states differently, for the same subject.

    Guards, in order: a shared non-generic term, compatible units, comparable
    magnitudes, and no limit phrasing (a bound is handled by its own rule). The
    guards are what keep this from flagging every unrelated number in the log.
    """
    if value.number is None or value.number == 0:
        return None
    claim_terms = {term for term in claim.terms if term not in _GENERIC_TERMS}
    if not claim_terms:
        return None
    for sentence in re.split(r"[.;!?\n]", context.all_text):
        if any(phrase in sentence.casefold() for phrase in _LIMIT_PHRASES):
            continue
        sentence_terms = {term for term in _terms(sentence) if term not in _GENERIC_TERMS}
        if not claim_terms & sentence_terms:
            continue
        for candidate in extract_values(sentence):
            if candidate.number is None or candidate.number == 0:
                continue
            if not _units_comparable(value, candidate):
                continue
            if _is_rounding_of(value, candidate):
                continue
            if value.same_magnitude(candidate):
                continue
            ratio = max(value.number, candidate.number) / min(value.number, candidate.number)
            if ratio > _MAX_DISAGREEMENT_RATIO:
                continue
            return DiffResult(
                verdict=Verdict.CONFLICTED,
                support_kind=SupportKind.DISAGREEMENT,
                detail=(
                    f"evidence states {candidate.number}{candidate.unit or ''} for the same "
                    f"subject, the claim says {value.number}{value.unit or ''}"
                ),
                cited=bool(context.explicit),
                matched_value=candidate.render(),
                observed=candidate.token(),
            )
    return None


def _contradiction(
    claim: Claim,
    value: Value,
    context: DiffContext,
    explicit_values: Sequence[Value],
    context_values: Sequence[Value],
) -> DiffResult | None:
    """Refute the claim (``S3-T11``). Only fires on positive, explicit conflict."""
    text = context.all_text
    lowered = normalize_text(text)

    # 1. set exclusion: the evidence enumerates a set the claim is not in
    if value.kind is ValueKind.MEMBER and value.members:
        for candidate in (*explicit_values, *context_values):
            if candidate.kind is not ValueKind.MEMBER or not candidate.members:
                continue
            claimed = set(value.members)
            offered = set(candidate.members)
            if not claimed & offered:
                # Different subjects entirely: silence, not conflict.
                continue
            if not claimed <= offered:
                return DiffResult(
                    verdict=Verdict.CONFLICTED,
                    support_kind=SupportKind.EXCLUSION,
                    detail=(
                        f"evidence offers {sorted(offered)}, the claim asserts {sorted(claimed)}"
                    ),
                    cited=bool(context.explicit),
                    matched_value=candidate.render(),
                    observed=candidate.token(),
                )

    # 2. bound violation: a bounded number outside the range
    if value.number is not None:
        bounds = _bounds((context.all_text,))
        if bounds is not None and not value.in_bounds(bounds[0], bounds[1]):
            low, high = bounds
            return DiffResult(
                verdict=Verdict.CONFLICTED,
                support_kind=SupportKind.BOUND,
                detail=f"{value.number} is outside the stated range [{low}, {high}]",
                cited=bool(context.explicit),
                matched_value=_range_text(low, high),
                observed=_range_text(low, high),
            )

        # 2b. same-subject disagreement: the classic wrong-price finding.
        #
        # The evidence must be about the same thing (a shared, non-generic term),
        # the units must be compatible, and the magnitudes must be the same order
        # — a claim of "5 ms" is not contradicted by an unrelated "500 seats".
        disagreement = _numeric_disagreement(claim, value, context)
        if disagreement is not None:
            return disagreement

    # 3. weekday/date disagreement with the same subject
    for candidate in (*explicit_values, *context_values):
        if value.kind is not candidate.kind:
            continue
        if value.kind not in (ValueKind.WEEKDAY, ValueKind.DATE):
            continue
        if candidate.canonical == value.canonical:
            continue
        if value.kind is ValueKind.DATE and len(candidate.canonical) < 8:
            # A partial date ("May 1") cannot refute a full one.
            continue
        return DiffResult(
            verdict=Verdict.CONFLICTED,
            support_kind=SupportKind.DISAGREEMENT,
            detail=f"evidence says {candidate.canonical}, the claim says {value.canonical}",
            cited=bool(context.explicit),
            matched_value=candidate.render(),
            observed=candidate.token(),
        )

    # 4. negation: the evidence denies what the claim asserts (and vice versa)
    if claim.is_specific or value.kind is ValueKind.BOOLEAN:
        for candidate in (*explicit_values, *context_values):
            if candidate.kind is not ValueKind.BOOLEAN:
                continue
            if candidate.negated != value.negated and candidate.canonical.split("|")[0] in lowered:
                return DiffResult(
                    verdict=Verdict.CONFLICTED,
                    support_kind=SupportKind.NEGATION,
                    detail="evidence states the opposite of the claim",
                    cited=bool(context.explicit),
                    matched_value=candidate.render(),
                    observed=candidate.token(),
                )

    # 5. explicit exclusion phrasing about the claimed value
    if value.canonical and len(value.canonical) > 3:
        for sentence in re.split(r"[.;!?\n]", text):
            excludes = _EXCLUSION_RE.search(sentence)
            mentions = value.canonical in normalize_text(sentence)
            limited = any(phrase in sentence.casefold() for phrase in _LIMIT_PHRASES)
            if excludes and mentions and not limited:
                return DiffResult(
                    verdict=Verdict.CONFLICTED,
                    support_kind=SupportKind.EXCLUSION,
                    detail="evidence excludes the claimed value",
                    cited=bool(context.explicit),
                    matched_value=value.render(),
                    observed=value.token(),
                )
    return None


def _range_text(low: Decimal | None, high: Decimal | None) -> str:
    """A bound as the evidence wrote it, for the flag's ``observed_value``."""
    if low is not None and high is not None:
        return f"{_render_number(low)} to {_render_number(high)}"
    if low is not None:
        return f"at least {_render_number(low)}"
    if high is not None:
        return f"at most {_render_number(high)}"
    return ""


# ---------------------------------------------------------------------------
# severity / confidence (``S3-T11``)
# ---------------------------------------------------------------------------

#: Value kinds whose misstatement a reader would act on. Drives severity, so a
#: wrong price is worse than a wrong date.
_CRITICAL_KINDS = frozenset({ClaimKind.NUMERIC, ClaimKind.BOOLEAN, ClaimKind.SET})
_HIGH_KINDS = frozenset({ClaimKind.DATE, ClaimKind.DURATION, ClaimKind.WEEKDAY})


#: Domains where an ungrounded assertion costs more than an ungrounded price
#: quote (``S3-T9``). Deliberately a small, closed set of *subject* terms: the
#: escalation keys on what the claim is about, not on the phrasing, so it cannot
#: be triggered or evaded by wording.
#:
#: The weight is a *floor*, not an override. A claim about a contraindication
#: that the evidence contradicts is still exactly as bad as a claim about a
#: contraindication the evidence never mentioned — the floor raises the quiet
#: cases and leaves the loud ones alone.
SAFETY_LEXICON: Mapping[str, Severity] = {
    # safety
    "medication": Severity.HIGH,
    "dosage": Severity.HIGH,
    "dose": Severity.HIGH,
    "contraindication": Severity.HIGH,
    "contraindicated": Severity.HIGH,
    "allergy": Severity.HIGH,
    "allergic": Severity.HIGH,
    "side effect": Severity.HIGH,
    "overdose": Severity.HIGH,
    "mortality": Severity.HIGH,
    "fatality": Severity.HIGH,
    "toxicity": Severity.HIGH,
    "toxic": Severity.HIGH,
    "pregnan": Severity.HIGH,
    "symptom": Severity.HIGH,
    "diagnosis": Severity.HIGH,
    "interaction": Severity.HIGH,
    # legal / regulatory
    "regulator": Severity.HIGH,
    "regulatory": Severity.HIGH,
    "litigation": Severity.HIGH,
    "lawsuit": Severity.HIGH,
    "attorney": Severity.HIGH,
    "compliance": Severity.MEDIUM,
    "certification": Severity.MEDIUM,
    "certified": Severity.MEDIUM,
    "certifications": Severity.MEDIUM,
    "accreditation": Severity.MEDIUM,
    "accredited": Severity.MEDIUM,
    "licence": Severity.MEDIUM,
    "license": Severity.MEDIUM,
    "licensed": Severity.MEDIUM,
    "permit": Severity.MEDIUM,
    "sanction": Severity.MEDIUM,
    "sanctions": Severity.MEDIUM,
    "filing": Severity.MEDIUM,
    "indemnity": Severity.MEDIUM,
    "warranty": Severity.MEDIUM,
    "hipaa": Severity.MEDIUM,
    "gdpr": Severity.MEDIUM,
    "iso 27001": Severity.MEDIUM,
}

#: Terms matched as stems, where English adds a suffix that must still count:
#: ``medications``/``medicated``, ``compliant``/``compliance``, ``symptoms``.
#: Terms matched as stems, where English adds a suffix that must still count:
#: ``medications``, ``contraindicated``, ``pregnancy``, ``compliance``.
#: The key is the lexicon term; the value is the suffix pattern to append, or
#: ``""`` when the term is already complete (``hipaa``) or is a phrase whose
#: words are matched individually.
_SAFETY_STEMS: Mapping[str, str] = {
    "medication": r"\w*",
    "dosage": r"\w*",
    "contraindication": r"\w*",
    "contraindicated": "",
    "allergy": r"\w*",
    "allergic": "",
    "pregnan": r"\w*",
    "diagnosis": r"\w*",
    "interaction": r"\w*",
    "toxicity": "",
    "regulator": r"\w*",
    "regulatory": "",
    "litigation": "",
    "lawsuit": r"\w*",
    "compliance": r"\w*",
    "certification": r"\w*",
    "certifications": r"\w*",
    "certified": "",
    "accreditation": r"\w*",
    "accredited": "",
    "sanction": r"\w*",
    "sanctions": r"\w*",
    "filing": r"\w*",
    "licensed": "",
    "symptom": r"\w*",
}


def _build_safety_patterns() -> tuple[tuple[Severity, re.Pattern[str]], ...]:
    """Compile :data:`SAFETY_LEXICON` into one word-boundary pattern per weight.

    Built rather than hand-written so the lexicon stays a readable mapping and
    a new term cannot be added with a regex bug: each entry is escaped, given
    the suffix from :data:`_SAFETY_STEMS`, and wrapped in word boundaries.
    Multi-word terms ("side effect", "iso 27001") require whitespace between
    their words, so they cannot match across an unrelated pair.
    """
    grouped: dict[Severity, list[str]] = {}
    for term, weight in SAFETY_LEXICON.items():
        body = r"\s+".join(re.escape(word) for word in term.split())
        stem = _SAFETY_STEMS.get(term, r"\w*")
        grouped.setdefault(weight, []).append(rf"{body}{stem}")
    return tuple(
        (weight, re.compile(r"\b(?:" + "|".join(sorted(terms)) + r")\b", re.IGNORECASE))
        for weight, terms in sorted(grouped.items(), key=lambda item: item[0].rank, reverse=True)
    )


_SAFETY_FLOOR_PATTERNS = _build_safety_patterns()


def _safety_floor(claim: Claim) -> Severity:
    """The severity *claim*'s subject demands, or ``INFO`` if it is ordinary.

    A floor, because the alternative — treating a safety claim as critical
    whatever the verdict — would make the severity field useless for triage: a
    module that cries wolf at the same volume stops being read.
    """
    floor = Severity.INFO
    for weight, pattern in _SAFETY_FLOOR_PATTERNS:
        if pattern.search(claim.text):
            floor = _max_severity(floor, weight)
    return floor


def severity_for(claim: Claim, diff: DiffResult) -> Severity:
    """Map an analysis to a severity.

    A *contradiction* outranks an *ungrounded* claim of the same kind: the
    agent stated something the evidence refutes, which is worse for a reader
    than stating something the evidence never mentioned.
    """
    if diff.verdict is Verdict.SUPPORTED or diff.verdict is Verdict.IMPLIED:
        return Severity.INFO

    contradicted = diff.verdict is Verdict.CONFLICTED
    if diff.support_kind is SupportKind.CHERRY_PICK:
        # Reporting a favourable count over a set the source shows was larger is
        # a misrepresentation, not a rounding error, whatever the claim's kind.
        return Severity.HIGH
    severity = _kind_severity(claim, contradicted)
    return _max_severity(severity, _safety_floor(claim))


def _max_severity(left: Severity, right: Severity) -> Severity:
    """The more serious of two severities."""
    return left if left.rank >= right.rank else right


def _kind_severity(claim: Claim, contradicted: bool) -> Severity:
    """The severity *claim*'s kind and verdict earn, before domain weighting."""
    if claim.kind in _CRITICAL_KINDS:
        base = Severity.HIGH if contradicted else Severity.MEDIUM
    elif claim.kind in _HIGH_KINDS:
        base = Severity.MEDIUM if contradicted else Severity.LOW
    elif claim.kind is ClaimKind.COMPARATIVE:
        base = Severity.LOW
    else:
        base = Severity.INFO
    if contradicted and claim.cue.startswith("comparative"):
        return _max_severity(Severity.MEDIUM, _safety_floor(claim))
    return base


def is_actionable(claim: Claim, diff: DiffResult) -> bool:
    """Whether this analysis deserves a flag (the FP/FN policy, ``S3-T13``).

    The rule is deliberately narrow, because the corpus gate is measured on
    false positives:

    * supported or implied — never flagged; the claim was groundable.
    * contradicted — always flagged, including an entity claim: the evidence
      states something incompatible with what the agent said.
    * ungrounded — flagged only for a *specific* claim (number, duration, date,
      weekday, boolean, set). A named entity that no tool result mentions is
      silence, not a defect, and flagging it is how a provenance module loses
      its operator's trust.
    """
    if diff.verdict in (Verdict.SUPPORTED, Verdict.IMPLIED):
        return False
    if not claim.is_checkable:
        return False
    if diff.verdict is Verdict.CONFLICTED:
        return True
    return claim.is_specific


def confidence_for(claim: Claim, diff: DiffResult, context: DiffContext) -> float:
    """How sure we are that this verdict is right (``S3-T11``).

    Explicit citation and outright contradiction are near-certain; silence is
    inherently softer, so an ungrounded claim is reported with a confidence
    below the review threshold by default and never blocks a gate on its own.
    """
    confidence = {
        Verdict.SUPPORTED: 0.95,
        Verdict.IMPLIED: 0.8,
        Verdict.CONFLICTED: 0.9,
        Verdict.UNSOURCED: 0.75,
        Verdict.UNKNOWN: 0.6,
    }[diff.verdict]
    if diff.support_kind is SupportKind.ROUNDING:
        confidence -= 0.05
    if diff.support_kind is SupportKind.UNIT_CONVERSION:
        confidence -= 0.05
    if diff.verdict is Verdict.UNKNOWN and not context.has_evidence:
        # No tool output at all: a strong signal that the claim came from the
        # model's weights, but the reviewer still decides.
        confidence += 0.05
    if claim.kind is ClaimKind.COMPARATIVE:
        confidence -= 0.1
    if claim.kind is ClaimKind.ENTITY:
        confidence -= 0.05
    return max(0.05, min(0.99, round(confidence, 2)))


__all__ = [
    "CHECKABLE_KINDS",
    "DEFAULT_LEXICON",
    "Claim",
    "ClaimExtractor",
    "ClaimKind",
    "DiffContext",
    "DiffResult",
    "GroundingLexicon",
    "RuleBasedClaimExtractor",
    "SupportKind",
    "Value",
    "ValueKind",
    "Verdict",
    "confidence_for",
    "diff_claim",
    "extract_values",
    "is_actionable",
    "is_assertion",
    "normalize_text",
    "severity_for",
    "split_sentences",
]
