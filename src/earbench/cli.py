"""The `earbench` command."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from earbench import audio, manifest, noise
from earbench import sweep as sweep_mod
from earbench.config import NoiseType, PrepareConfig, SweepConfig, load_config

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


@contextmanager
def _exit_on_error() -> Iterator[None]:
    """Turn config, data and file errors into a one-line message and exit code 1."""
    try:
        yield
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _todo(phase: str) -> None:
    typer.echo(f"not implemented yet (Phase {phase}); see PLAN.md")
    raise typer.Exit(code=1)


@app.command()
def prepare(
    config: str = typer.Option(..., "--config", help="Path to prepare YAML config."),
) -> None:
    """Pick clips and write the manifest."""
    with _exit_on_error():
        cfg = load_config(config, PrepareConfig)
        summary = manifest.run_prepare(cfg)
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
    with _exit_on_error():
        audio.save_wav_16k_mono(TV_OUT_PATH, audio.load_16k(TV_RAW_PATH))
    typer.echo(f"ok: {TV_RAW_PATH} -> {TV_OUT_PATH} (16 kHz mono)")
    if failed:
        raise typer.Exit(code=1)


def _render_listen(
    cfg: SweepConfig,
    clip_id: str,
    distance_m: float,
    noise_type: NoiseType,
    snr_db: float | None,
    seed: int,
) -> np.ndarray:
    """Build one simulated condition for a clip from the manifest."""
    rows = {row.clip_id: row for row in manifest.read_manifest(cfg.manifest_path)}
    if clip_id not in rows:
        raise ValueError(f"clip {clip_id!r} not in {cfg.manifest_path}")
    clip = audio.load_16k(rows[clip_id].wav_path)
    noise_file = None
    if noise_type != "none":
        if snr_db is None:
            raise ValueError("--snr is needed when --noise is not 'none'")
        noise_file = sweep_mod.load_noise(cfg, noise_type)
    mixed, _ = noise.make_condition(clip, distance_m, cfg.room, noise_file, snr_db, seed)
    # Scales speech and noise together, so the SNR is unchanged.
    return audio.peak_normalise(mixed)


@app.command()
def listen(
    clip: str = typer.Option(..., "--clip", help="Clip id from the manifest."),
    distance: float = typer.Option(..., "--distance", help="Speaker-to-mic distance in metres."),
    noise_type: Annotated[NoiseType, typer.Option("--noise", help="Noise type.")] = "none",
    snr: float | None = typer.Option(None, "--snr", help="Target SNR in dB at the mic."),
    config: str = typer.Option("configs/full.yaml", "--config", help="Sweep YAML config."),
    seed: int | None = typer.Option(None, "--seed", help="Noise offset seed (default: config)."),
    out: str | None = typer.Option(None, "--out", help="Output WAV path."),
) -> None:
    """Write one simulated condition to a WAV so you can hear it."""
    with _exit_on_error():
        cfg = load_config(config, SweepConfig)
        mixed = _render_listen(
            cfg, clip, distance, noise_type, snr, cfg.seed if seed is None else seed
        )
    snr_tag = "" if noise_type == "none" else f"_snr{snr:g}"
    out_path = (
        Path(out) if out else Path(f"data/listen/{clip}_d{distance:g}m_{noise_type}{snr_tag}.wav")
    )
    audio.save_wav_16k_mono(out_path, mixed)
    typer.echo(f"wrote {out_path}")


@app.command()
def sweep(
    config: str = typer.Option(..., "--config", help="Path to sweep YAML config."),
    run_id: str | None = typer.Option(None, "--run-id", help="Override the run id."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Hide the progress bar."),
) -> None:
    """Run the noise sweep: transcribe every condition and write results."""
    with _exit_on_error():
        cfg = load_config(config, SweepConfig)
        result = sweep_mod.run_sweep(cfg, progress=not no_progress, run_id=run_id)
    typer.echo(f"run: {result.run_id}")
    typer.echo(f"results: {result.run_dir / 'results.csv'}")
    typer.echo(f"summary: {result.run_dir / 'summary.csv'}")
    for model in cfg.models:
        model_rows = [row for row in result.summary if row.model == model]
        if model_rows:
            worst = max(row.wer for row in model_rows)
            typer.echo(f"{model}: {len(model_rows)} conditions, worst WER {worst:.1%}")


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
