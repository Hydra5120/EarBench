"""The `earbench` command."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from earbench import audio, manifest
from earbench.config import ConfigError, PrepareConfig, load_config

app = typer.Typer(
    name="earbench",
    help="Measure how well off-the-shelf speech recognition hears older people in noise.",
    no_args_is_help=True,
)

DEMAND_NOISE_FILES = (
    Path("DLIVING/ch01.wav"),
    Path("DKITCHEN/ch01.wav"),
    Path("PCAFETER/ch01.wav"),
)
TV_RAW_PATH = Path("data/raw/tv.wav")
TV_OUT_PATH = Path("data/noise/tv.wav")


@app.callback()
def _init_logging() -> None:
    """Send library warnings to the console for every command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _todo(phase: str) -> None:
    typer.echo(f"not implemented yet (Phase {phase}); see PLAN.md")
    raise typer.Exit(code=1)


@app.command()
def prepare(
    config: str = typer.Option(..., "--config", help="Path to prepare YAML config."),
) -> None:
    """Pick clips and write the manifest."""
    try:
        cfg = load_config(config, PrepareConfig)
        summary = manifest.run_prepare(cfg)
    except (ConfigError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(manifest.format_summary(summary))
    typer.echo(f"manifest: {cfg.manifest_path}")


@app.command(name="prepare-noise")
def prepare_noise() -> None:
    """Verify DEMAND noise files and convert the TV recording to 16 kHz mono."""
    failed = False
    for noise_path in DEMAND_NOISE_FILES:
        problems = audio.check_wav_16k_mono(noise_path)
        if problems:
            failed = True
            for problem in problems:
                typer.echo(f"error: {problem}", err=True)
        else:
            typer.echo(f"ok: {noise_path} is 16 kHz mono (used in place, no copy)")
    if not TV_RAW_PATH.is_file():
        typer.echo(
            f"error: {TV_RAW_PATH} not found; record 5 minutes of TV first (see PLAN.md)",
            err=True,
        )
        raise typer.Exit(code=1)
    try:
        tv_mono, tv_rate_hz = audio.load_mono(TV_RAW_PATH)
        converted = audio.resample_to_16k(tv_mono, tv_rate_hz)
        audio.save_wav_16k_mono(TV_OUT_PATH, converted)
    except (OSError, ValueError) as exc:
        typer.echo(f"error: cannot convert {TV_RAW_PATH}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ok: {TV_RAW_PATH} -> {TV_OUT_PATH} (16 kHz mono)")
    if failed:
        raise typer.Exit(code=1)


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
