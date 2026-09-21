"""Basic package smoke tests for the S-1 skeleton."""

from __future__ import annotations

import pytest

from sentinel import __version__
from sentinel._cli import main


def test_version_is_reported() -> None:
    """The package advertises a version string."""
    assert isinstance(__version__, str)
    assert len(__version__.split(".")) >= 2


def test_cli_runs_without_args() -> None:
    """The `sentinel` command exits cleanly with no arguments."""
    assert main([]) == 0


def test_cli_version_flag_prints_and_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """`sentinel --version` prints the version and exits zero."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sentinel-sdk" in capsys.readouterr().out
