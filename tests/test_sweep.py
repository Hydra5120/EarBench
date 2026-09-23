"""Tests for the sweep grid, runner and output files (Phase 3)."""

from __future__ import annotations

import csv
from collections import Counter

import numpy as np
import pytest

from earbench import audio, manifest, noise, sweep
from earbench.config import ResultRow, SweepConfig
from earbench.score import normalise
from earbench.transcribe import CachedTranscriber, FakeTranscriber, Transcriber, audio_key


def _fake_factory(text: str = "hello world") -> sweep.TranscriberFactory:
    return lambda model: FakeTranscriber(default_text=text)


def test_missing_noise_file_fails_before_any_model_loads(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1, noise_types=("living", "tv"))
    cfg.noise_files["tv"][0].unlink()
    created: list[str] = []

    def factory(model: str) -> Transcriber:
        created.append(model)
        return FakeTranscriber()

    with pytest.raises(ValueError, match="not found"):
        sweep.run_sweep(cfg, factory, progress=False, run_id="missing")
    assert created == []
    assert not (cfg.runs_dir / "missing").exists()


def test_select_clips_respects_limit_and_seed(sweep_env) -> None:
    cfg, rows = sweep_env(n_per_group=3)
    picks = sweep.select_clips(rows, clips_per_group=2, seed=0)
    assert len(picks) == 4  # two groups, two clips each
    assert picks == sweep.select_clips(rows, clips_per_group=2, seed=0)
    assert Counter(row.age_group for row in picks) == {"older": 2, "younger": 2}
    assert len(sweep.select_clips(rows, clips_per_group=None, seed=0)) == 6


def test_build_grid_has_clean_conditions(sweep_env) -> None:
    cfg, rows = sweep_env(n_per_group=1, distances_m=(1.0, 2.0), snr_db=(10.0, 0.0))
    conditions = sweep.build_grid(cfg, rows)
    # 2 clips × 2 distances × (clean 1 + living 2) = 12
    assert len(conditions) == 12
    clean = [c for c in conditions if c.noise_type == "none"]
    assert len(clean) == 4
    assert all(c.snr_db is None for c in clean)
    noisy = [c for c in conditions if c.noise_type == "living"]
    assert sorted({c.snr_db for c in noisy if c.snr_db is not None}) == [0.0, 10.0]


def test_run_sweep_writes_files_and_headers(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    result = sweep.run_sweep(cfg, _fake_factory(), progress=False, run_id="testrun")
    results_path = result.run_dir / "results.csv"
    assert results_path.is_file()
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "summary.csv").is_file()
    with open(results_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        data = list(reader)
    assert header == list(ResultRow.model_fields)
    # 2 clips × 1 distance × (clean 1 + living 1) × 1 model = 4
    assert len(data) == 4
    assert {row["run_id"] for row in data} == {"testrun"}
    assert {row["source"] for row in data} == {"sim"}


def test_summary_splits_age_groups_per_condition(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    result = sweep.run_sweep(cfg, _fake_factory(), progress=False, run_id="summary")
    # conditions (distance, noise, SNR): clean + living @ 10 dB, for older and younger
    pairs = {(row.age_group, row.noise_type) for row in result.summary}
    assert pairs == {
        ("older", "none"),
        ("older", "living"),
        ("younger", "none"),
        ("younger", "living"),
    }
    assert all(row.n_clips == 1 for row in result.summary)
    assert all(row.ci_low <= row.wer <= row.ci_high for row in result.summary)


def test_rerun_makes_zero_transcriber_calls(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    fake = FakeTranscriber(default_text="cached answer")

    def factory(model: str) -> CachedTranscriber:
        return CachedTranscriber(fake, cfg.cache_dir)

    sweep.run_sweep(cfg, factory, progress=False, run_id="first")
    assert fake.calls > 0
    fake.calls = 0
    sweep.run_sweep(cfg, factory, progress=False, run_id="second")
    assert fake.calls == 0


def test_same_seed_gives_identical_results(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    first = sweep.run_sweep(cfg, _fake_factory(), progress=False, run_id="a")
    second = sweep.run_sweep(cfg, _fake_factory(), progress=False, run_id="b")

    def signature(rows: list[ResultRow]) -> list[tuple]:
        return [
            (r.clip_id, r.distance_m, r.noise_type, r.snr_db, r.hypothesis, r.wer) for r in rows
        ]

    assert signature(first.rows) == signature(second.rows)


class _AudioEchoTranscriber:
    """Hypothesis is a fingerprint of the audio.

    If the (condition-outer, model-inner) reorder changed the mixed audio at all,
    the hypotheses would stop matching the model-outer reference below.
    """

    name = "echo"
    version = "1"
    settings = "echo"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return audio_key(audio, sample_rate)[:16]


def _model_outer_reference(cfg: SweepConfig, transcriber: Transcriber) -> list[tuple]:
    """Results in the old order: model outer, condition inner, chain per (model, condition)."""
    clips = sweep.select_clips(
        manifest.read_manifest(cfg.manifest_path), cfg.clips_per_group, cfg.seed
    )
    conditions = sweep.build_grid(cfg, clips)
    rows: list[tuple] = []
    for model in cfg.models:
        for condition in conditions:
            clip = audio.load_16k(condition.clip.wav_path)
            noise_file = None
            if condition.noise_type != "none":
                noise_file = audio.load_16k(cfg.noise_files[condition.noise_type][0])
            seed = sweep._condition_seed(cfg.seed, condition.clip.clip_id, condition.noise_type)
            mixed, _ = noise.make_condition(
                clip, condition.distance_m, cfg.room, noise_file, condition.snr_db, seed
            )
            hypothesis = transcriber.transcribe(mixed, audio.SAMPLE_RATE_HZ)
            rows.append(
                (
                    condition.clip.clip_id,
                    condition.distance_m,
                    condition.noise_type,
                    condition.snr_db,
                    model,
                    normalise(hypothesis),
                )
            )
    return rows


def test_condition_outer_matches_model_outer_results(sweep_env) -> None:
    cfg, _ = sweep_env(
        n_per_group=2, snr_db=(10.0, 0.0), distances_m=(1.0, 2.0), models=("tiny", "base")
    )
    transcriber = _AudioEchoTranscriber()
    result = sweep.run_sweep(cfg, lambda model: transcriber, progress=False, run_id="reordered")
    got = sorted(
        repr((r.clip_id, r.distance_m, r.noise_type, r.snr_db, r.model, r.hypothesis))
        for r in result.rows
    )
    reference = sorted(repr(row) for row in _model_outer_reference(cfg, transcriber))
    assert got == reference


def test_default_factory_passes_device_and_compute_type(monkeypatch, sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    cfg = cfg.model_copy(update={"device": "cuda", "compute_type": "float16"})
    seen: list[tuple[str, str, str]] = []

    def fake_whisper(model_size: str, *, device: str, compute_type: str) -> FakeTranscriber:
        seen.append((model_size, device, compute_type))
        return FakeTranscriber()

    monkeypatch.setattr(sweep, "FasterWhisperTranscriber", fake_whisper)
    transcriber = sweep.default_transcriber_factory(cfg)("small")
    assert isinstance(transcriber, CachedTranscriber)
    assert seen == [("small", "cuda", "float16")]
