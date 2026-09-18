"""CLI stub tests: `earbench --help` must succeed (Phase 0 acceptance)."""

from __future__ import annotations

from typer.testing import CliRunner

from earbench.cli import app

runner = CliRunner()


def test_help_succeeds() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "earbench" in result.output.lower()
