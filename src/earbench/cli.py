"""The `earbench` command. Just a stub for now; real commands come later."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="earbench",
    help="Measure how well off-the-shelf speech recognition hears older people in noise.",
    no_args_is_help=True,
)


def _todo(phase: str) -> None:
    typer.echo(f"not implemented yet (Phase {phase}); see PLAN.md")
    raise typer.Exit(code=1)


@app.command()
def prepare(
    config: str = typer.Option(..., "--config", help="Path to prepare YAML config."),
) -> None:
    """Pick clips and write the manifest (not built yet)."""
    _todo("1")


@app.command()
def sweep(
    config: str = typer.Option(..., "--config", help="Path to sweep YAML config."),
) -> None:
    """Run the noise sweep (not built yet)."""
    _todo("3")


@app.command()
def report(run_id: str = typer.Argument(..., help="Run id under runs/.")) -> None:
    """Build charts + HTML report (not built yet)."""
    _todo("4")


@app.command()
def record(
    config: str = typer.Option(..., "--config", help="Path to room session YAML config."),
) -> None:
    """Play + record audio in a real room (not built yet)."""
    _todo("5")


@app.command(name="score-room")
def score_room(session_id: str = typer.Argument(..., help="Session id under recordings/.")) -> None:
    """Score real recordings (not built yet)."""
    _todo("5")


if __name__ == "__main__":
    app()
