"""Test helpers: tiny fake audio clips made with numpy (no real data needed)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from earbench.config import SweepConfig
from earbench.manifest import ManifestRow, write_manifest

SAMPLE_RATE_HZ = 16_000


@pytest.fixture
def sample_rate_hz() -> int:
    """Internal sample rate: 16 kHz mono."""
    return SAMPLE_RATE_HZ


@pytest.fixture
def silence_mono() -> np.ndarray:
    """1 s of silence as float32 mono."""
    return np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)


@pytest.fixture
def tone_mono() -> np.ndarray:
    """1 s of 440 Hz sine at 0.5 peak amplitude, float32 mono."""
    t = np.arange(SAMPLE_RATE_HZ, dtype=np.float64) / SAMPLE_RATE_HZ
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def noise_mono() -> np.ndarray:
    """1 s of seeded white noise at ~0.1 RMS, float32 mono."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SAMPLE_RATE_HZ).astype(np.float64)
    x *= 0.1 / float(np.sqrt(np.mean(x**2)))
    return x.astype(np.float32)


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> Path:
    """Save mono float32 audio as a 16-bit WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    frames = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(frames.tobytes())
    return path


def _write_tone(path: Path, duration_s: float, freq_hz: float) -> Path:
    """A short tone burst, enough to stand in for a speech clip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * SAMPLE_RATE_HZ), dtype=np.float64) / SAMPLE_RATE_HZ
    return write_wav_mono(path, (0.4 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32))


@pytest.fixture
def sweep_env(tmp_path: Path):
    """Build a tiny, offline sweep: synthetic clips, a manifest and one noise file.

    Returns a `make(...)` that yields `(SweepConfig, rows, manifest_path)` with all
    paths under `tmp_path`, so runs are isolated and reproducible.
    """

    def _make(
        n_per_group: int = 2,
        duration_s: float = 0.6,
        include_clean: bool = True,
        noise_types: tuple[str, ...] = ("living",),
        snr_db: tuple[float, ...] = (10.0,),
        distances_m: tuple[float, ...] = (1.0,),
        models: tuple[str, ...] = ("tiny",),
        clips_per_group: int | None = None,
    ) -> tuple[SweepConfig, list[ManifestRow], Path]:
        rows: list[ManifestRow] = []
        for group, bucket, base_freq_hz in (
            ("older", "sixties", 300.0),
            ("younger", "twenties", 700.0),
        ):
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

        noise_files: dict[str, list[Path]] = {}
        for noise_type in noise_types:
            noise_path = tmp_path / f"{noise_type}.wav"
            rng = np.random.default_rng(0)
            write_wav_mono(
                noise_path, (0.2 * rng.standard_normal(5 * SAMPLE_RATE_HZ)).astype(np.float32)
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
        return cfg, rows, manifest_path

    return _make
