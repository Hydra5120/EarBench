"""Test helpers: tiny fake audio clips made with numpy (no real data needed)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from earbench.audio import SAMPLE_RATE_HZ, save_wav_16k_mono
from earbench.config import AgeGroup, NoiseType, SweepConfig
from earbench.manifest import ManifestRow, write_manifest


@pytest.fixture
def tone_mono() -> np.ndarray:
    """1 s of 440 Hz sine at 0.5 peak amplitude, float32 mono."""
    t = np.arange(SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


def _write_tone(path: Path, duration_s: float, freq_hz: float) -> Path:
    """A short tone burst, enough to stand in for a speech clip."""
    t = np.arange(int(duration_s * SAMPLE_RATE_HZ), dtype=np.float64) / SAMPLE_RATE_HZ
    return save_wav_16k_mono(path, 0.4 * np.sin(2.0 * np.pi * freq_hz * t))


@pytest.fixture
def sweep_env(tmp_path: Path) -> Callable[..., tuple[SweepConfig, list[ManifestRow]]]:
    """Build a tiny, offline sweep: synthetic clips, a manifest and one noise file.

    Returns a `make(...)` that yields `(SweepConfig, rows)` with all paths under
    `tmp_path`, so runs are isolated and reproducible.
    """

    def _make(
        n_per_group: int = 2,
        duration_s: float = 0.6,
        include_clean: bool = True,
        noise_types: tuple[NoiseType, ...] = ("living",),
        snr_db: tuple[float, ...] = (10.0,),
        distances_m: tuple[float, ...] = (1.0,),
        models: tuple[str, ...] = ("tiny",),
        clips_per_group: int | None = None,
    ) -> tuple[SweepConfig, list[ManifestRow]]:
        rows: list[ManifestRow] = []
        groups: tuple[tuple[AgeGroup, str, float], ...] = (
            ("older", "sixties", 300.0),
            ("younger", "twenties", 700.0),
        )
        for group, bucket, base_freq_hz in groups:
            for index in range(n_per_group):
                clip_id = f"{group}_{index}"
                wav_path = _write_tone(
                    tmp_path / "clips" / f"{clip_id}.wav",
                    duration_s,
                    base_freq_hz + 20.0 * index,
                )
                rows.append(
                    ManifestRow(
                        clip_id=clip_id,
                        wav_path=wav_path.as_posix(),
                        sentence=f"the {group} speaker says number {index}",
                        age_bucket=bucket,
                        age_group=group,
                        gender="female",
                        duration_s=duration_s,
                        speaker_hash=f"{group}{index}",
                    )
                )
        manifest_path = tmp_path / "manifest.csv"
        write_manifest(rows, manifest_path)

        noise_files: dict[NoiseType, list[Path]] = {}
        for noise_type in noise_types:
            rng = np.random.default_rng(0)
            noise_path = save_wav_16k_mono(
                tmp_path / f"{noise_type}.wav", 0.2 * rng.standard_normal(5 * SAMPLE_RATE_HZ)
            )
            noise_files[noise_type] = [noise_path]

        cfg = SweepConfig(
            manifest_path=manifest_path,
            clips_per_group=clips_per_group,
            distances_m=list(distances_m),
            noise_files=noise_files,
            snr_db=list(snr_db),
            include_clean=include_clean,
            models=list(models),
            bootstrap_iters=100,
            cache_dir=tmp_path / "cache",
            runs_dir=tmp_path / "runs",
        )
        return cfg, rows

    return _make
