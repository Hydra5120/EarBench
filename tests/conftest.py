"""Test helpers: tiny fake audio clips made with numpy (no real data needed)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

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
