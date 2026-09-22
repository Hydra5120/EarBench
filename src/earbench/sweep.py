"""Build the condition grid, run the sweep and write results (Phase 3)."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from pydantic import BaseModel
from tqdm import tqdm

from earbench import audio, manifest, noise
from earbench import score as scoring
from earbench.config import (
    AgeGroup,
    NoiseType,
    ResultRow,
    SummaryRow,
    SweepConfig,
    config_hash,
)
from earbench.transcribe import CachedTranscriber, FasterWhisperTranscriber, Transcriber

logger = logging.getLogger(__name__)

TranscriberFactory = Callable[[str], Transcriber]


@dataclass
class Condition:
    """One (clip, distance, noise type, SNR) point, before the model is chosen."""

    clip: manifest.ManifestRow
    distance_m: float
    noise_type: str
    snr_db: float | None


@dataclass
class SweepResult:
    """Where a sweep wrote its output and what it produced."""

    run_id: str
    run_dir: Path
    rows: list[ResultRow]
    summary: list[SummaryRow]


def select_clips(
    rows: Sequence[manifest.ManifestRow], clips_per_group: int | None, seed: int
) -> list[manifest.ManifestRow]:
    """Pick `clips_per_group` clips per age group with a seeded, order-independent shuffle."""
    if clips_per_group is None:
        return sorted(rows, key=lambda row: (row.age_group, row.clip_id))
    by_group: dict[str, list[manifest.ManifestRow]] = {}
    for row in rows:
        by_group.setdefault(row.age_group, []).append(row)
    groups = sorted(by_group)
    streams = np.random.SeedSequence(seed).spawn(max(1, len(groups)))
    selected: list[manifest.ManifestRow] = []
    for index, group in enumerate(groups):
        group_rows = sorted(by_group[group], key=lambda row: row.clip_id)
        order = np.random.default_rng(streams[index]).permutation(len(group_rows))
        selected.extend(group_rows[int(i)] for i in order[:clips_per_group])
    return sorted(selected, key=lambda row: (row.age_group, row.clip_id))


def build_grid(cfg: SweepConfig, clips: Sequence[manifest.ManifestRow]) -> list[Condition]:
    """Cross clips × distances × noise types × SNRs. The clean condition has no SNR."""
    noise_types: list[str] = ["none"] if cfg.include_clean else []
    noise_types.extend(cfg.noise_files)
    conditions: list[Condition] = []
    for clip in clips:
        for distance_m in cfg.distances_m:
            for noise_type in noise_types:
                snr_values: list[float | None] = (
                    [None] if noise_type == "none" else list(cfg.snr_db)
                )
                for snr_db in snr_values:
                    conditions.append(Condition(clip, distance_m, noise_type, snr_db))
    return conditions


def _condition_seed(seed: int, clip_id: str, noise_type: str) -> int:
    """Stable per-condition noise offset seed, independent of grid order."""
    digest = hashlib.sha256(f"{seed}:{clip_id}:{noise_type}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _new_run_id(cfg: SweepConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{config_hash(cfg)}"


def default_transcriber_factory(cfg: SweepConfig) -> TranscriberFactory:
    """One cached faster-whisper transcriber per model size."""

    def factory(model_size: str) -> Transcriber:
        return CachedTranscriber(FasterWhisperTranscriber(model_size), cfg.cache_dir)

    return factory


def _load_clip(cache: dict[str, np.ndarray], wav_path: str) -> np.ndarray:
    if wav_path not in cache:
        mono, sample_rate_hz = audio.load_mono(wav_path)
        cache[wav_path] = audio.resample_to_16k(mono, sample_rate_hz)
    return cache[wav_path]


def _load_noise(cache: dict[str, np.ndarray], cfg: SweepConfig, noise_type: str) -> np.ndarray:
    if noise_type not in cache:
        paths = cfg.noise_files.get(noise_type) or []
        if not paths:
            raise ValueError(f"noise type {noise_type!r} has no file in the config's noise_files")
        path = paths[0]
        if not path.is_file():
            raise ValueError(f"noise file {path} not found (run `earbench prepare-noise`)")
        mono, sample_rate_hz = audio.load_mono(path)
        cache[noise_type] = audio.resample_to_16k(mono, sample_rate_hz)
    return cache[noise_type]


def _speech_at_mic(
    speech_cache: dict[tuple[str, float], np.ndarray],
    clip_audio: dict[str, np.ndarray],
    cfg: SweepConfig,
    condition: Condition,
) -> np.ndarray:
    """Room-simulated speech for a (clip, distance), simulated once per run."""
    key = (condition.clip.wav_path, condition.distance_m)
    if key not in speech_cache:
        clip = _load_clip(clip_audio, condition.clip.wav_path)
        speech_cache[key] = noise.speech_at_mic(clip, condition.distance_m, cfg.room)
    return speech_cache[key]


def _noise_at_mic(
    noise_cache: dict[tuple[str, int, int], np.ndarray],
    noise_audio: dict[str, np.ndarray],
    cfg: SweepConfig,
    condition: Condition,
    seed: int,
    n_samples: int,
) -> np.ndarray:
    """Room-simulated noise for a condition, simulated once per run.

    The noise source does not move with distance, so distance is not in the key.
    The segment does depend on the clip length and seed, so both are.
    """
    key = (condition.noise_type, seed, n_samples)
    if key not in noise_cache:
        noise_file = _load_noise(noise_audio, cfg, condition.noise_type)
        noise_cache[key] = noise.noise_at_mic(noise_file, n_samples, seed, cfg.room)
    return noise_cache[key]


def _summary_rows(rows: Sequence[ResultRow], cfg: SweepConfig) -> list[SummaryRow]:
    groups: dict[tuple[str, AgeGroup, float, str, float | None], list[ResultRow]] = {}
    for row in rows:
        key = (row.model, row.age_group, row.distance_m, row.noise_type, row.snr_db)
        groups.setdefault(key, []).append(row)
    summary: list[SummaryRow] = []
    for (model, age_group, distance_m, noise_type, snr_db), group in groups.items():
        ci_low, ci_high = scoring.bootstrap_ci(group, cfg.bootstrap_iters, cfg.seed)
        summary.append(
            SummaryRow(
                model=model,
                age_group=age_group,
                distance_m=distance_m,
                noise_type=cast(NoiseType, noise_type),
                snr_db=snr_db,
                n_clips=len(group),
                errors=sum(row.errors for row in group),
                ref_words=sum(row.ref_words for row in group),
                wer=scoring.corpus_wer(group),
                ci_low=ci_low,
                ci_high=ci_high,
                usable_rate=scoring.usable_rate(group, cfg.usable_wer),
            )
        )
    return summary


def _write_csv(model: type[BaseModel], rows: Sequence[BaseModel], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    return out_path


def write_results(rows: Sequence[ResultRow], path: str | Path) -> Path:
    """Write results rows with the exact ResultRow column order."""
    return _write_csv(ResultRow, rows, path)


def write_summary(summary: Sequence[SummaryRow], path: str | Path) -> Path:
    """Write summary rows with the exact SummaryRow column order."""
    return _write_csv(SummaryRow, summary, path)


def write_config(cfg: SweepConfig, path: str | Path) -> Path:
    """Write the resolved config next to the results, so a run is self-describing."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    out_path.write_text(payload, encoding="utf-8")
    return out_path


def run_sweep(
    cfg: SweepConfig,
    transcriber_factory: TranscriberFactory | None = None,
    *,
    progress: bool = True,
    run_id: str | None = None,
) -> SweepResult:
    """Run the whole grid, write results.csv/config.yaml/summary.csv, return the result."""
    run_id = run_id or _new_run_id(cfg)
    run_dir = Path(cfg.runs_dir) / run_id
    clips = select_clips(manifest.read_manifest(cfg.manifest_path), cfg.clips_per_group, cfg.seed)
    conditions = build_grid(cfg, clips)
    chash = config_hash(cfg)
    factory = transcriber_factory or default_transcriber_factory(cfg)

    clip_audio: dict[str, np.ndarray] = {}
    noise_audio: dict[str, np.ndarray] = {}
    speech_cache: dict[tuple[str, float], np.ndarray] = {}
    noise_cache: dict[tuple[str, int, int], np.ndarray] = {}
    transcribers = {model: factory(model) for model in cfg.models}
    rows: list[ResultRow] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config(cfg, run_dir / "config.yaml")

    current_clip_id: str | None = None
    total = len(cfg.models) * len(conditions)
    with tqdm(total=total, disable=not progress, unit="cond", desc="sweep") as bar:
        for condition in conditions:
            if condition.clip.clip_id != current_clip_id:
                # The grid groups conditions by clip: free the previous clip's audio.
                speech_cache.clear()
                noise_cache.clear()
                current_clip_id = condition.clip.clip_id
            speech = _speech_at_mic(speech_cache, clip_audio, cfg, condition)
            if condition.noise_type == "none":
                mixed = speech.copy()
            else:
                seed = _condition_seed(cfg.seed, condition.clip.clip_id, condition.noise_type)
                noise_sim = _noise_at_mic(
                    noise_cache, noise_audio, cfg, condition, seed, speech.shape[0]
                )
                mixed = noise.mix(speech, noise_sim, cast(float, condition.snr_db))
            for model in cfg.models:
                transcriber = transcribers[model]
                hypothesis = transcriber.transcribe(mixed, audio.SAMPLE_RATE_HZ)
                scored = scoring.score_clip(condition.clip.sentence, hypothesis)
                rows.append(
                    ResultRow(
                        run_id=run_id,
                        source="sim",
                        clip_id=condition.clip.clip_id,
                        age_group=cast(AgeGroup, condition.clip.age_group),
                        distance_m=condition.distance_m,
                        noise_type=cast(NoiseType, condition.noise_type),
                        snr_db=condition.snr_db,
                        model=model,
                        model_version=transcriber.version,
                        settings=transcriber.settings,
                        reference=scored.reference,
                        hypothesis=scored.hypothesis,
                        ref_words=scored.ref_words,
                        errors=scored.errors,
                        wer=scored.wer,
                        config_hash=chash,
                    )
                )
                bar.update(1)

    summary = _summary_rows(rows, cfg)
    write_results(rows, run_dir / "results.csv")
    write_summary(summary, run_dir / "summary.csv")
    logger.info("wrote %s: %d results in %d conditions", run_dir, len(rows), len(summary))
    return SweepResult(run_id=run_id, run_dir=run_dir, rows=rows, summary=summary)
