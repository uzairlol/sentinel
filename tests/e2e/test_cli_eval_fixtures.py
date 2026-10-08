"""End-to-end tests for ``sentinel eval-fixtures`` (``S3-T13``, ``S3-T17``).

The command runs a module over its adversarial corpus, so these tests need no
database: the corpus is the event sequence. What they do check is the part an
operator relies on — the printed matrix is the measured one, the exit code is
the gate verdict, and an agent that fabricates a citation is caught end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from sentinel._cli import main
from sentinel.eval.fixtures.memory_corpus import MEMORY_CORPUS
from sentinel.eval.fixtures.provenance_corpus import CORPUS

pytestmark = pytest.mark.e2e


def test_the_gate_passes_and_says_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The sprint cannot be called done on a hope: the command has to print PASS."""
    code = main(["eval-fixtures", "--module", "provenance"])

    out = capsys.readouterr().out
    assert code == 0
    assert "confusion matrix" in out
    assert "false-negative rate: 0.00%" in out
    assert "false-positive rate: 0.00%" in out
    assert out.rstrip().endswith("PASS")


def test_the_matrix_counts_both_directions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["eval-fixtures", "--module", "provenance"])

    out = capsys.readouterr().out
    assert f"{len(CORPUS)} cases" in out
    # 24 cases expect a flag, 14 expect silence: a detector that only ever said
    # "clean" could not tell the two apart, so the matrix has to show both.
    assert "should flag" in out
    assert "should be quiet" in out
    assert "recall 100.00%" in out
    assert "precision 100.00%" in out


def test_json_output_carries_the_gates_for_ci(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["eval-fixtures", "--module", "provenance", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["confusion"] == {
        "true_positives": 24,
        "false_negatives": 0,
        "false_positives": 0,
        "true_negatives": 14,
    }
    assert payload["gates"]["max_false_negative_rate"] == 0.1
    assert payload["gates"]["max_false_positive_rate"] == 0.0
    assert payload["failures"] == []


def test_an_unknown_module_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["eval-fixtures", "--module", "nonsense"])

    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown module" in err
    assert "provenance" in err


def test_the_module_flag_is_required() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["eval-fixtures"])
    assert exit_info.value.code == 2


def test_the_console_script_runs_without_a_database() -> None:
    """A real subprocess, so CI proves the entry point needs no DSN."""
    done = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-m", "sentinel._cli", "eval-fixtures", "--module", "provenance"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert "PASS" in done.stdout


def test_a_fabricated_citation_is_caught_end_to_end() -> None:
    """S3-T17: an agent told to invent a plausible citation is still caught.

    The agent looks at a price of $49 and answers $29, inventing a discount it
    never saw. A weaker detector would call the citation ungrounded and leave
    it there; this one names the contradiction, points at the event that
    refutes it, and quotes the value that was actually observed.
    """
    import asyncio

    from sentinel.eval.fixtures.provenance_corpus import case_by_id
    from sentinel.eval.provenance import CATEGORY_CONTRADICTED, ProvenanceEvaluator
    from sentinel.store.sqlite import SQLiteEventStore

    case = case_by_id("contradicted_price")
    store = SQLiteEventStore(":memory:")
    try:
        for event in case.cited_events():
            asyncio.run(store.append(event))

        flags = asyncio.run(
            ProvenanceEvaluator(
                store, review_url_template="https://review.test/{session_id}"
            ).evaluate_session(case.cited_events()[0].session_id)
        )
    finally:
        asyncio.run(store.close())

    assert len(flags) == 1
    flag = flags[0]
    assert flag.category == CATEGORY_CONTRADICTED
    assert flag.evidence[0].role.value == "claim"
    assert "$29" in flag.details["claim_text"]
    assert flag.details["observed_value"] == "49 usd"
    assert flag.details["review_url"].endswith(flag.session_id)
    # A contradiction is a finding, not a hypothesis: it stays out of the
    # review-only bucket so it can gate.
    assert flag.review_only is False
    assert flag.severity.value in {"high", "critical"}


# -- the memory module (``S4-T12``)


def test_the_memory_corpus_passes_its_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``S4-T12``: the FP/FN number the sprint's exit criterion asks for."""
    code = main(["eval-fixtures", "--module", "memory"])

    out = capsys.readouterr().out
    assert code == 0
    assert "sentinel.memory_integrity@0.1.0" in out
    assert "false-negative rate: 0.00%" in out
    assert "false-positive rate: 0.00%" in out
    assert out.rstrip().endswith("PASS")


def test_the_memory_report_counts_both_directions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["eval-fixtures", "--module", "memory"])

    out = capsys.readouterr().out
    assert f"{len(MEMORY_CORPUS)} cases" in out
    assert "should flag" in out
    assert "should be quiet" in out


def test_the_memory_json_carries_the_gates_for_ci(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["eval-fixtures", "--module", "memory", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["passed"] is True
    assert payload["module"] == "sentinel.memory_integrity"
    assert payload["confusion"]["false_positives"] == 0
    assert payload["confusion"]["false_negatives"] == 0
    assert payload["failures"] == []


def test_the_two_modules_are_measured_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A shared harness must not let one module's numbers stand in for another's."""
    main(["eval-fixtures", "--module", "provenance", "--json"])
    provenance = json.loads(capsys.readouterr().out)
    main(["eval-fixtures", "--module", "memory", "--json"])
    memory = json.loads(capsys.readouterr().out)

    assert provenance["module"] != memory["module"]
    assert provenance["confusion"] != memory["confusion"]


def test_both_modules_are_listed_in_usage() -> None:
    """The help text must not advertise a module that does not exist, or omit
    one that does."""
    result = subprocess.run(
        [sys.executable, "-m", "sentinel._cli", "eval-fixtures", "--module", "nope"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "memory" in result.stderr
    assert "provenance" in result.stderr
