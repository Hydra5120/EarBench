"""Export the demo site's data from existing runs (Phase 8). Never transcribes.

Reads results.csv/summary.csv from the full run and the VAD run, regenerates
each curated clip's mixed audio through the sweep's own chain (so it is exactly
what Whisper heard, checked against the transcription cache), and writes JSON
plus MP3s under `site/public/data/`. TV conditions never get audio.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from earbench import audio, manifest, noise
from earbench import score as scoring
from earbench import sweep as sweep_mod
from earbench.config import (
    NoiseType,
    ResultRow,
    SiteChart,
    SiteConfig,
    SummaryRow,
    SweepConfig,
    load_config,
)
from earbench.report import pool, read_results, read_summary
from earbench.transcribe import audio_key

# (clip_id, noise_type, snr_db, model) -> row, for one run at the site's distance.
CellKey = tuple[str, str, float | None, str]
VAD_STATES = ("off", "on")


@dataclass
class ExportSummary:
    """What the export wrote, for the CLI to print."""

    out_dir: Path
    clips: int = 0
    transcripts: int = 0
    audio_files: int = 0
    audio_bytes: int = 0
    json_bytes: int = 0
    cache_checked: int = 0
    cache_matched: int = 0
    cache_dir_found: bool = True
    gain_db: dict[str, float] = field(default_factory=dict)  # clips turned down for MP3
    hero: dict[str, Any] = field(default_factory=dict)
    room_rows: list[dict[str, Any]] = field(default_factory=list)
    per_clip_bytes: dict[str, int] = field(default_factory=dict)


def cond_key(noise_type: str, snr_db: float | None) -> str:
    """Stable condition id used in file names and JSON keys: `clean`, `cafeteria_0`."""
    return "clean" if snr_db is None else f"{noise_type}_{snr_db:g}"


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _vad_flag(settings: str) -> str:
    for part in settings.split(","):
        if part.startswith("vad="):
            return part.removeprefix("vad=")
    raise ValueError(f"settings {settings!r} has no vad= field")


def _index_cells(
    rows: Sequence[ResultRow], distance_m: float, run_name: str
) -> dict[CellKey, ResultRow]:
    """Rows at `distance_m` keyed by cell; a duplicate cell is an error."""
    cells: dict[CellKey, ResultRow] = {}
    for row in rows:
        if row.distance_m != distance_m:
            continue
        key = (row.clip_id, row.noise_type, row.snr_db, row.model)
        if key in cells:
            raise ValueError(f"{run_name}: duplicate result for {key}")
        cells[key] = row
    return cells


def _conditions(
    noise_types: Sequence[NoiseType], snr_levels: Sequence[float]
) -> list[tuple[NoiseType, float | None]]:
    out: list[tuple[NoiseType, float | None]] = [("none", None)]
    out.extend((nt, snr) for nt in noise_types for snr in snr_levels)
    return out


def _lookup(
    cells: dict[CellKey, ResultRow], key: CellKey, run_name: str, want_vad: str
) -> ResultRow:
    row = cells.get(key)
    if row is None:
        clip, nt, snr, model = key
        raise ValueError(
            f"{run_name}: missing cell clip={clip} noise={nt} snr={snr} model={model} "
            "(export never transcribes; run the sweep that covers it)"
        )
    if _vad_flag(row.settings) != want_vad:
        raise ValueError(
            f"{run_name}: {key} has settings {row.settings!r}, expected vad={want_vad}"
        )
    return row


def _transcript(row: ResultRow, usable_wer: float) -> dict[str, Any]:
    tokens = scoring.align_normalised(row.reference, row.hypothesis)
    n_errors = sum(1 for token in tokens if token.op != "ok")
    if n_errors != row.errors:
        raise ValueError(
            f"alignment for {row.clip_id} {row.noise_type} {row.snr_db} {row.model} has "
            f"{n_errors} errors but results.csv says {row.errors}"
        )
    return {
        "reference": row.reference,
        "hypothesis": row.hypothesis,
        "wer": row.wer,
        "errors": row.errors,
        "ref_words": row.ref_words,
        "usable": row.wer < usable_wer,
        "tokens": [asdict(token) for token in tokens],
    }


def _cache_key(mixed: np.ndarray, transcriber_name: str, row: ResultRow) -> str:
    parts = (
        audio_key(mixed, audio.SAMPLE_RATE_HZ),
        transcriber_name,
        row.model_version,
        row.settings,
    )
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def speech_span(
    clip: np.ndarray, frame_ms: float = 20.0, range_db: float = 30.0
) -> tuple[float, float]:
    """Seconds where speech starts and ends in a dry clip: first and last 20 ms frame
    within `range_db` of the loudest. Used only to pace the site's live transcript."""
    frame = int(audio.SAMPLE_RATE_HZ * frame_ms / 1000)
    n = clip.shape[0] // frame
    if n == 0:
        return (0.0, clip.shape[0] / audio.SAMPLE_RATE_HZ)
    powers = np.mean(clip[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1)
    if powers.max() == 0.0:
        return (0.0, clip.shape[0] / audio.SAMPLE_RATE_HZ)
    active = np.flatnonzero(powers >= powers.max() / 10.0 ** (range_db / 10.0))
    return (
        active[0] * frame / audio.SAMPLE_RATE_HZ,
        (active[-1] + 1) * frame / audio.SAMPLE_RATE_HZ,
    )


def _check_cache(
    site: SiteConfig,
    clip_id: str,
    noise_type: NoiseType,
    snr: float | None,
    mixed: np.ndarray,
    full_cells: dict[CellKey, ResultRow],
    vad_cells: dict[CellKey, ResultRow],
    cache_dirs: tuple[Path, Path],
) -> None:
    """Every model and VAD state must have transcribed exactly this audio."""
    for model in site.models:
        for state, cells, cache_dir in (
            ("off", full_cells, cache_dirs[0]),
            ("on", vad_cells, cache_dirs[1]),
        ):
            row = cells[(clip_id, noise_type, snr, model)]
            key = _cache_key(mixed, site.transcriber_name, row)
            if not (cache_dir / f"{key}.txt").is_file():
                raise ValueError(
                    f"regenerated audio for {clip_id} {cond_key(noise_type, snr)} "
                    f"({model}, vad={state}) is not in the transcription cache: "
                    "it is not what Whisper heard"
                )


def encode_mp3(samples: np.ndarray) -> bytes:
    """16 kHz mono MP3 via libsndfile (soundfile), no extra encoder dependency."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, audio.SAMPLE_RATE_HZ, format="MP3")
    return buffer.getvalue()


def _summary_lookup(
    summary: Sequence[SummaryRow],
) -> dict[tuple[str, str, float, str, float | None], SummaryRow]:
    return {
        (row.model, row.age_group, row.distance_m, row.noise_type, row.snr_db): row
        for row in summary
    }


def wilson_interval(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (here: share of usable clips)."""
    if n == 0:
        return (0.0, 0.0)
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summary_point(row: SummaryRow) -> dict[str, Any]:
    usable_low, usable_high = wilson_interval(row.usable_rate, row.n_clips)
    return {
        "snr_db": row.snr_db,
        "wer": row.wer,
        "ci_low": row.ci_low,
        "ci_high": row.ci_high,
        "usable_rate": row.usable_rate,
        "usable_low": usable_low,
        "usable_high": usable_high,
        "usable_count": round(row.usable_rate * row.n_clips),
        "n_clips": row.n_clips,
    }


def hero_numbers(site: SiteConfig, summary: Sequence[SummaryRow]) -> dict[str, Any]:
    """The hero condition's numbers per age group, straight from summary.csv."""
    lookup = _summary_lookup(summary)
    hero = site.hero
    out: dict[str, Any] = {
        "model": hero.model,
        "noise_type": hero.noise_type,
        "snr_db": hero.snr_db,
        "distance_m": site.distance_m,
    }
    for group in ("older", "younger"):
        row = lookup.get((hero.model, group, site.distance_m, hero.noise_type, hero.snr_db))
        if row is None:
            raise ValueError(f"summary.csv has no hero row for {hero} {group}")
        out[group] = _summary_point(row)
        quiet = lookup.get((hero.model, group, site.distance_m, "none", None))
        if quiet is None:
            raise ValueError(f"summary.csv has no clean row for {hero.model} {group}")
        out[f"{group}_quiet"] = _summary_point(quiet)
    return out


def chart_data(site: SiteConfig, chart: SiteChart, summary: Sequence[SummaryRow]) -> dict[str, Any]:
    """One findings chart: a line per series value, a point per noise level (clean first)."""
    lookup = _summary_lookup(summary)
    all_noise: list[NoiseType] = [*site.audio_noise_types, *site.text_only_noise_types]
    values: dict[str, list[str]] = {
        "age_group": ["older", "younger"],
        "model": list(site.models),
        "noise_type": list(all_noise),
    }
    fixed: dict[str, str | None] = {
        "model": chart.model,
        "age_group": chart.age_group,
        "noise_type": chart.noise_type,
    }
    for dim, value in fixed.items():
        if dim != chart.series and value is None:
            raise ValueError(f"chart {chart.id!r}: series is {chart.series}, so set {dim}")
    snr_levels = [level.snr_db for level in site.levels]
    series = []
    for value in values[chart.series]:
        pick = {**fixed, chart.series: value}
        points = []
        for snr in snr_levels:
            noise_type = "none" if snr is None else str(pick["noise_type"])
            key = (str(pick["model"]), str(pick["age_group"]), site.distance_m, noise_type, snr)
            row = lookup.get(key)
            if row is None:
                raise ValueError(f"chart {chart.id!r}: summary.csv has no row for {key}")
            points.append(_summary_point(row))
        series.append({"key": value, "points": points})
    return {
        "id": chart.id,
        "title": chart.title,
        "series_by": chart.series,
        "fixed": {k: v for k, v in fixed.items() if k != chart.series},
        "series": series,
    }


def vad_comparison(
    site: SiteConfig,
    full_cells: dict[CellKey, ResultRow],
    vad_cells: dict[CellKey, ResultRow],
    cfg: SweepConfig,
) -> dict[str, Any]:
    """VAD off vs on, pooled over the curated clips only (same clips both sides)."""
    clip_ids = [clip.id for clip in site.clips]
    rows_out = []
    for model in site.models:
        for noise_type, snr in _conditions(
            site.audio_noise_types,
            [level.snr_db for level in site.levels if level.snr_db is not None],
        ):
            pair = {}
            for state, cells in (("off", full_cells), ("on", vad_cells)):
                group = [cells[(clip, noise_type, snr, model)] for clip in clip_ids]
                pooled = pool(group, cfg.bootstrap_iters, cfg.seed, cfg.usable_wer)
                pair[state] = asdict(pooled)
            rows_out.append({"model": model, "noise_type": noise_type, "snr_db": snr, **pair})
    return {"n_clips": len(clip_ids), "distance_m": site.distance_m, "rows": rows_out}


def read_room(path: Path, label: str) -> dict[str, Any]:
    """compare-summary.csv as numbers (empty cells -> None), plus the caveat label."""
    if not path.is_file():
        raise ValueError(f"room comparison file not found: {path}")
    rows = []
    with open(path, encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in ("condition", "noise_type"):
                    row[key] = value
                elif key == "n_clips":
                    row[key] = int(value)
                else:
                    row[key] = None if value == "" else float(value)
            rows.append(row)
    if not rows:
        raise ValueError(f"room comparison file is empty: {path}")
    for row in rows:
        for side in ("room", "sim"):
            low, high = wilson_interval(row[f"{side}_usable"], row["n_clips"])
            row[f"{side}_usable_low"], row[f"{side}_usable_high"] = low, high
    session = path.parent / "session.yaml"
    model = None
    if session.is_file():
        model = (yaml.safe_load(session.read_text(encoding="utf-8")) or {}).get("model")
    return {"session_id": path.parent.name, "model": model, "label": label, "rows": rows}


def _write_json(path: Path, payload: Any) -> int:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def _run_info(role: str, run_dir: Path, rows: Sequence[ResultRow]) -> dict[str, Any]:
    return {
        "role": role,
        "run_id": run_dir.name,
        "config_hashes": sorted({row.config_hash for row in rows}),
        "model_versions": sorted({row.model_version for row in rows}),
        "settings": sorted({row.settings for row in rows}),
        "n_results": len(rows),
    }


def export_site(site: SiteConfig) -> ExportSummary:
    """Write site/public/data from existing runs. Missing cells fail the export."""
    full_cfg = load_config(site.full_run / "config.yaml", SweepConfig)
    vad_cfg = load_config(site.vad_run / "config.yaml", SweepConfig)
    full_rows = read_results(site.full_run / "results.csv")
    vad_rows = read_results(site.vad_run / "results.csv")
    full_summary = read_summary(site.full_run / "summary.csv")
    full_cells = _index_cells(full_rows, site.distance_m, site.full_run.name)
    vad_cells = _index_cells(vad_rows, site.distance_m, site.vad_run.name)
    by_clip = {row.clip_id: row for row in manifest.read_manifest(full_cfg.manifest_path)}

    snr_levels = [level.snr_db for level in site.levels if level.snr_db is not None]
    audio_conditions = _conditions(site.audio_noise_types, snr_levels)
    text_conditions = [(nt, snr) for nt in site.text_only_noise_types for snr in snr_levels]
    usable_wer = full_cfg.usable_wer

    # Check every cell exists before touching the output directory.
    for clip in site.clips:
        if clip.id not in by_clip:
            raise ValueError(f"clip {clip.id} is not in {full_cfg.manifest_path}")
        for noise_type, snr in [*audio_conditions, *text_conditions]:
            for model in site.models:
                _lookup(full_cells, (clip.id, noise_type, snr, model), site.full_run.name, "off")
                if (noise_type, snr) in audio_conditions:
                    _lookup(vad_cells, (clip.id, noise_type, snr, model), site.vad_run.name, "on")
        featured = (clip.featured_noise, clip.featured_snr_db)
        if featured not in audio_conditions and featured not in text_conditions:
            raise ValueError(f"clip {clip.id}: featured condition {featured} is not on the site")

    out_dir = site.out_dir
    for sub in ("audio", "clips"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
    summary = ExportSummary(out_dir=out_dir)
    summary.cache_dir_found = full_cfg.cache_dir.is_dir() and vad_cfg.cache_dir.is_dir()

    noise_audio = {nt: sweep_mod.load_noise(full_cfg, nt) for nt in site.audio_noise_types}
    index_clips = []
    for clip in site.clips:
        info = by_clip[clip.id]
        speech = audio.load_16k(info.wav_path)
        mixes: dict[tuple[NoiseType, float | None], np.ndarray] = {}
        for noise_type, snr in audio_conditions:
            seed = sweep_mod.condition_seed(full_cfg.seed, clip.id, noise_type)
            noise_file = None if noise_type == "none" else noise_audio[noise_type]
            mixed, _speech = noise.make_condition(
                speech, site.distance_m, full_cfg.room, noise_file, snr, seed
            )
            if summary.cache_dir_found:
                summary.cache_checked += len(site.models) * 2
                _check_cache(
                    site,
                    clip.id,
                    noise_type,
                    snr,
                    mixed,
                    full_cells,
                    vad_cells,
                    (
                        full_cfg.cache_dir,
                        vad_cfg.cache_dir,
                    ),
                )
                summary.cache_matched += len(site.models) * 2
            mixes[(noise_type, snr)] = mixed
        # One gain per clip, so moving the slider only changes the noise, never the voice.
        # A level change only: the waveform shape Whisper heard is unchanged.
        peak = max(float(np.max(np.abs(mixed))) for mixed in mixes.values())
        gain = min(1.0, 0.99 / peak) if peak > 0 else 1.0
        if gain < 1.0:
            summary.gain_db[clip.id] = 20.0 * float(np.log10(gain))
        cells_out: dict[str, Any] = {}
        clip_bytes = 0
        for noise_type, snr in [*audio_conditions, *text_conditions]:
            has_audio = (noise_type, snr) in mixes
            results: dict[str, Any] = {}
            for model in site.models:
                for state in VAD_STATES if has_audio else ("off",):
                    cells, run = (
                        (full_cells, site.full_run) if state == "off" else (vad_cells, site.vad_run)
                    )
                    row = _lookup(cells, (clip.id, noise_type, snr, model), run.name, state)
                    results[f"{model}|{state}"] = _transcript(row, usable_wer)
                    summary.transcripts += 1
            cell: dict[str, Any] = {"noise_type": noise_type, "snr_db": snr, "results": results}
            if has_audio:
                data = encode_mp3(mixes[(noise_type, snr)] * gain)
                name = f"audio/{clip.id}/{cond_key(noise_type, snr)}.{_short_hash(data)}.mp3"
                (out_dir / name).parent.mkdir(parents=True, exist_ok=True)
                (out_dir / name).write_bytes(data)
                cell["audio"] = name
                summary.audio_files += 1
                summary.audio_bytes += len(data)
                clip_bytes += len(data)
            else:
                cell["audio"] = None
                cell["note"] = site.tv_note
            cells_out[cond_key(noise_type, snr)] = cell

        payload = {
            "id": clip.id,
            "sentence": info.sentence,
            "distance_m": site.distance_m,
            "audio_gain_db": 20.0 * float(np.log10(gain)),
            # Measured from the dry clip; the site spreads words across it (approximate).
            "speech_span_s": speech_span(speech),
            "cells": cells_out,
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        clip_name = f"clips/{clip.id}.{_short_hash(body)}.json"
        summary.json_bytes += _write_json(out_dir / clip_name, payload)
        summary.per_clip_bytes[clip.id] = clip_bytes + len(body)
        index_clips.append(
            {
                "id": clip.id,
                "age_group": info.age_group,
                "age_bucket": info.age_bucket,
                "gender": info.gender,
                "duration_s": info.duration_s,
                "sentence": info.sentence,
                "featured": {"noise_type": clip.featured_noise, "snr_db": clip.featured_snr_db},
                "data": clip_name,
            }
        )
        summary.clips += 1

    charts = {
        "hero": hero_numbers(site, full_summary),
        "findings": [chart_data(site, chart, full_summary) for chart in site.charts],
        "vad": vad_comparison(site, full_cells, vad_cells, full_cfg),
    }
    room = read_room(site.room_compare, site.room_label)
    site_manifest = {
        "runs": [
            _run_info("benchmark", site.full_run, full_rows),
            _run_info("vad", site.vad_run, vad_rows),
        ],
        "room_session": room["session_id"],
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
    }
    index = {
        "site_manifest": site_manifest,
        "distance_m": site.distance_m,
        "usable_wer": usable_wer,
        "models": site.models,
        "levels": [level.model_dump() for level in site.levels],
        "noise_types": [
            {"id": nt, "label": site.noise_labels.get(nt, nt), "audio": True}
            for nt in site.audio_noise_types
        ]
        + [
            {"id": nt, "label": site.noise_labels.get(nt, nt), "audio": False}
            for nt in site.text_only_noise_types
        ],
        "vad_noise_types": ["none", *site.audio_noise_types],
        "tv_note": site.tv_note,
        "clips": index_clips,
    }
    summary.json_bytes += _write_json(out_dir / "index.json", index)
    summary.json_bytes += _write_json(out_dir / "charts.json", charts)
    summary.json_bytes += _write_json(out_dir / "room.json", room)
    summary.hero = charts["hero"]
    summary.room_rows = room["rows"]

    text_only = tuple(f"{nt}_" for nt in site.text_only_noise_types)
    leaked = [
        p for p in (out_dir / "audio").rglob("*") if p.is_file() and p.name.startswith(text_only)
    ]
    if leaked:
        raise ValueError(f"TV audio must never be exported: {leaked}")
    return summary
