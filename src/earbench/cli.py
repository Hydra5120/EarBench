"""The `earbench` command."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from earbench import audio, manifest, noise, site_export
from earbench import fix as fix_mod
from earbench import record as record_mod
from earbench import report as report_mod
from earbench import sweep as sweep_mod
from earbench.config import (
    NoiseType,
    PrepareConfig,
    RoomSessionConfig,
    SiteConfig,
    SweepConfig,
    load_config,
)

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
# Third-party loggers that log every HTTP request or every clip at INFO.
QUIET_LOGGERS = ("httpx", "huggingface_hub", "faster_whisper")


@app.callback()
def _init_logging() -> None:
    """Send earbench info and library warnings to the console for every command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def _exit_on_error() -> Iterator[None]:
    """Turn config, data and file errors into a one-line message and exit code 1."""
    try:
        yield
    except (ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


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
    typer.echo(f"conditions: {len(result.summary)} ({len(result.rows)} transcriptions)")


@app.command()
def report(
    run: Annotated[str, typer.Argument(help="Run dir (e.g. runs/<run_id>).")],
    out: Annotated[str | None, typer.Option(help="Output dir (default: reports/<run_id>).")] = None,
    best_model: Annotated[str | None, typer.Option(help="Override the best model.")] = None,
    distance_snr: Annotated[
        str | None, typer.Option(help="Chart C SNR levels, e.g. 'clean,10'.")
    ] = None,
    distance_noise: Annotated[NoiseType | None, typer.Option(help="Chart C noise type.")] = None,
    reference_distance: Annotated[
        float | None, typer.Option(help="Charts A/B distance in metres.")
    ] = None,
    before: Annotated[
        str | None, typer.Option(help="Before run dir for the Phase 6 fix comparison.")
    ] = None,
) -> None:
    """Build charts + HTML report for a sweep run."""
    with _exit_on_error():
        run_dir = Path(run)
        if not run_dir.is_dir():
            fallback = Path("runs") / run
            run_dir = fallback if fallback.is_dir() else run_dir
        levels: list[float | None] | None = None
        if distance_snr is not None:
            levels = [
                None if part.strip().lower() in ("clean", "none") else float(part)
                for part in distance_snr.split(",")
            ]
        out_dir = report_mod.write_report(
            run_dir,
            Path(out) if out else None,
            best_model=best_model,
            distance_snr_db=levels,
            distance_noise=distance_noise,
            reference_distance_m=reference_distance,
            before_dir=Path(before) if before else None,
        )
    typer.echo(f"report: {out_dir / 'report.html'}")


@app.command(name="make-playlist")
def make_playlist(
    config: str = typer.Option(..., "--config", help="Path to room session YAML config."),
) -> None:
    """Write one playlist WAV + timing CSV per block, plus calibration.wav."""
    with _exit_on_error():
        cfg = load_config(config, RoomSessionConfig)
        built = record_mod.make_playlist(cfg)
    for playlist in built:
        typer.echo(f"playlist: {playlist.wav_path} ({len(playlist.rows)} clips)")
    typer.echo(f"calibration: {Path(cfg.playlists_dir) / record_mod.CALIBRATION_FILENAME}")


@app.command()
def record(
    config: str = typer.Option(..., "--config", help="Path to room session YAML config."),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Reuse a session (retakes get the next take number)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Align a synthetic recording without touching audio."
    ),
) -> None:
    """Record the mic while the MacBook plays each block's playlist. You type every measurement."""
    with _exit_on_error():
        cfg = load_config(config, RoomSessionConfig)
        backend = record_mod.make_backend(cfg)
        session_dir = record_mod.run_session(
            cfg, session_id=session_id, backend=backend, dry_run=dry_run
        )
    typer.echo(f"session: {session_dir}")


@app.command(name="score-room")
def score_room(
    session: str = typer.Argument(..., help="Session dir or id under recordings/."),
    vad_filter: bool = typer.Option(
        False, "--vad-filter", help="Transcribe with the VAD fix on (Phase 6)."
    ),
) -> None:
    """Transcribe and score a real-room session into room-source result rows."""
    with _exit_on_error():
        session_dir = record_mod.resolve_session_dir(session)
        factory = None
        if vad_filter:
            factory = lambda model: record_mod.default_transcriber_factory(  # noqa: E731
                model, vad_filter=True
            )
        rows = record_mod.score_session(session_dir, transcriber_factory=factory)
    typer.echo(f"results: {session_dir / 'results.csv'} ({len(rows)} takes)")


@app.command(name="compare-room")
def compare_room(
    session: str = typer.Argument(..., help="Session dir or id under recordings/."),
    config: str = typer.Option(..., "--config", help="Sweep YAML config (noise, fallback room)."),
) -> None:
    """Simulate each real take's condition and plot real vs simulated WER."""
    with _exit_on_error():
        sweep_cfg = load_config(config, SweepConfig)
        session_dir = record_mod.resolve_session_dir(session)
        _paired, note = record_mod.compare_session(session_dir, sweep_cfg)
        cells = record_mod.read_compare_summary(session_dir / record_mod.COMPARE_SUMMARY_FILENAME)
    typer.echo(f"compare: {session_dir / 'compare.csv'}")
    typer.echo(f"summary: {session_dir / record_mod.COMPARE_SUMMARY_FILENAME}")
    typer.echo(f"chart: {session_dir / record_mod.COMPARE_GROUP_PNG}")
    typer.echo(record_mod.format_compare_table(cells))
    if note:
        typer.echo(note)


@app.command(name="compare-fix")
def compare_fix(
    sim_before: str = typer.Option(..., "--sim-before", help="Before sweep run dir."),
    sim_after: str = typer.Option(..., "--sim-after", help="After sweep run dir."),
    room_before: str | None = typer.Option(
        None, "--room-before", help="Before session dir (real room, later)."
    ),
    room_after: str | None = typer.Option(
        None, "--room-after", help="After session dir (real room, later)."
    ),
    out: str | None = typer.Option(None, "--out", help="Output dir."),
    iters: int | None = typer.Option(None, "--iters", help="Bootstrap iters."),
    seed: int | None = typer.Option(None, "--seed", help="Bootstrap seed."),
) -> None:
    """Before/after numbers with intervals for the fix, sim now and room later."""
    with _exit_on_error():
        before_rows = fix_mod.read_any_results(sim_before)
        after_rows = fix_mod.read_any_results(sim_after)
        if (room_before is None) != (room_after is None):
            raise ValueError("give both --room-before and --room-after, or neither")
        if room_before is not None and room_after is not None:
            before_rows = [*before_rows, *fix_mod.read_any_results(room_before)]
            after_rows = [*after_rows, *fix_mod.read_any_results(room_after)]
        n_iters, use_seed, usable = 1000, 0, 0.20
        after_cfg = Path(sim_after) / "config.yaml"
        if after_cfg.is_file():
            cfg = load_config(after_cfg, SweepConfig)
            n_iters, use_seed, usable = cfg.bootstrap_iters, cfg.seed, cfg.usable_wer
        cells = fix_mod.compare(
            before_rows,
            after_rows,
            iters=n_iters if iters is None else iters,
            seed=use_seed if seed is None else seed,
            usable_wer=usable,
        )
        out_dir = Path(out) if out else Path("reports") / f"fix-{Path(sim_after).name}"
        fix_mod.write_compare(cells, out_dir)
        fig = fix_mod.fig_before_after(cells)
        try:
            fig.savefig(out_dir / fix_mod.FIX_CHART, dpi=100)
        finally:
            report_mod.close_figure(fig)
    typer.echo(f"compare: {out_dir / fix_mod.FIX_CSV}")
    typer.echo(fix_mod.format_table(cells, "before", "after"))
    if room_before is None:
        typer.echo("room: pending -- rerun with --room-before/--room-after after the session")


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


@app.command(name="export-site")
def export_site(
    config: Annotated[str, typer.Option(help="Path to site YAML config.")] = "configs/site.yaml",
) -> None:
    """Export the demo site's data from existing runs (never transcribes)."""
    with _exit_on_error():
        site = load_config(config, SiteConfig)
        result = site_export.export_site(site)
    typer.echo(f"wrote {result.out_dir}")
    typer.echo(f"clips: {result.clips}   transcripts: {result.transcripts}")
    typer.echo(
        f"audio: {result.audio_files} MP3s + {result.noise_beds} noise beds, "
        f"{result.audio_bytes / 1e6:.2f} MB   "
        f"json: {result.json_bytes / 1e3:.0f} kB"
    )
    for clip_id, size in result.per_clip_bytes.items():
        typer.echo(f"  {clip_id}: {size / 1e3:.0f} kB")
    if result.cache_dir_found:
        typer.echo(
            f"audio check: {result.cache_matched}/{result.cache_checked} "
            "audio cells match the transcription cache"
        )
    else:
        typer.echo("audio check: skipped (no transcription cache on this machine)")
    for clip_id, gain in result.gain_db.items():
        typer.echo(f"  {clip_id}: turned down {-gain:.1f} dB (whole clip) so the MP3 never clips")
    hero = result.hero
    snr = "clean" if hero["snr_db"] is None else f"{hero['snr_db']:g} dB"
    typer.echo(f"hero: {hero['model']}, {hero['noise_type']} {snr}, {hero['distance_m']:g} m")
    for group in ("older", "younger"):
        point = hero[group]
        typer.echo(
            f"  {group}: WER {_pct(point['wer'])} "
            f"[{_pct(point['ci_low'])}-{_pct(point['ci_high'])}], "
            f"usable {_pct(point['usable_rate'])}, n={point['n_clips']}"
        )
    typer.echo("room vs sim (WER):")
    for row in result.room_rows:
        typer.echo(
            f"  {row['distance_m']:g} m {row['condition']:<5} room {_pct(row['room_wer'])}  "
            f"sim {_pct(row['sim_wer'])}"
        )


if __name__ == "__main__":
    app()
