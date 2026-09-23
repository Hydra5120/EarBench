"""Tests for noise segment picking and SNR mixing (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from earbench import noise
from earbench.audio import SAMPLE_RATE_HZ, active_speech_power, power
from earbench.config import RoomConfig

TARGETS_DB = [-5.0, 0.0, 5.0, 10.0, 20.0, 30.0]


def measured_snr_db(speech: np.ndarray, mixed: np.ndarray) -> float:
    """SNR of a mix, measured from the active speech level and what was added."""
    added = mixed.astype(np.float64) - speech.astype(np.float64)
    return 10.0 * np.log10(active_speech_power(speech) / power(added))


def _speech_like(n: int, seed: int) -> np.ndarray:
    """Tone bursts with gaps, a rough stand-in for speech with pauses."""
    t = np.arange(n) / SAMPLE_RATE_HZ
    env = (np.sin(2 * np.pi * 3.0 * t) > 0).astype(np.float64)
    rng = np.random.default_rng(seed)
    x = env * np.sin(2 * np.pi * 220.0 * t) + 0.05 * rng.standard_normal(n)
    return (0.3 * x).astype(np.float32)


def _noises(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    t = np.arange(n) / SAMPLE_RATE_HZ
    bursts = rng.standard_normal(n) * (np.sin(2 * np.pi * 0.7 * t) > 0.5)
    return {
        "white": 0.1 * rng.standard_normal(n),
        "tone": 0.2 * np.sin(2 * np.pi * 1000.0 * t),
        "bursty": 0.3 * bursts,
        "very_loud": 50.0 * rng.standard_normal(n),
        "very_quiet": 1e-4 * rng.standard_normal(n),
    }


@pytest.mark.parametrize("snr_db", TARGETS_DB)
@pytest.mark.parametrize("noise_name", ["white", "tone", "bursty", "very_loud", "very_quiet"])
def test_mix_hits_target_snr(noise_name: str, snr_db: float) -> None:
    n = 3 * SAMPLE_RATE_HZ
    speech = _speech_like(n, seed=0)
    noise_sig = _noises(n)[noise_name].astype(np.float32)
    mixed = noise.mix(speech, noise_sig, snr_db)
    assert mixed.dtype == np.float32
    assert abs(measured_snr_db(speech, mixed) - snr_db) < 0.1


@pytest.mark.parametrize("snr_db", TARGETS_DB)
def test_full_chain_snr_measured_at_mic(snr_db: float) -> None:
    """SNR is set on speech and noise after room simulation."""
    room = RoomConfig()
    clip = _speech_like(2 * SAMPLE_RATE_HZ, seed=2)
    noise_file = np.random.default_rng(3).standard_normal(10 * SAMPLE_RATE_HZ).astype(np.float32)
    for distance_m in [1.0, 3.0]:
        mixed, speech_at_mic = noise.make_condition(
            clip, distance_m, room, noise_file, snr_db, seed=0
        )
        assert mixed.shape == speech_at_mic.shape
        assert abs(measured_snr_db(speech_at_mic, mixed) - snr_db) < 0.1


def test_no_noise_equals_room_speech() -> None:
    from earbench.room import simulate

    room = RoomConfig()
    clip = _speech_like(SAMPLE_RATE_HZ, seed=4)
    mixed, speech_at_mic = noise.make_condition(clip, 2.0, room, None, None, seed=0)
    assert np.array_equal(mixed, speech_at_mic)
    assert np.array_equal(mixed, simulate(clip, 2.0, room))


def test_same_seed_same_audio() -> None:
    room = RoomConfig()
    clip = _speech_like(SAMPLE_RATE_HZ, seed=5)
    noise_file = np.random.default_rng(6).standard_normal(10 * SAMPLE_RATE_HZ).astype(np.float32)
    a, _ = noise.make_condition(clip, 2.0, room, noise_file, 5.0, seed=7)
    b, _ = noise.make_condition(clip, 2.0, room, noise_file, 5.0, seed=7)
    c, _ = noise.make_condition(clip, 2.0, room, noise_file, 5.0, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_pick_segment_is_seeded_and_right_length() -> None:
    noise_file = np.arange(1000, dtype=np.float32)
    a = noise.pick_segment(noise_file, 100, seed=1)
    b = noise.pick_segment(noise_file, 100, seed=1)
    assert a.shape == (100,)
    assert np.array_equal(a, b)
    # A real slice of the file: consecutive sample values.
    assert np.all(np.diff(a) == 1.0)


def test_pick_segment_too_short_raises() -> None:
    with pytest.raises(ValueError, match="too short"):
        noise.pick_segment(np.zeros(50, dtype=np.float32), 100, seed=0)


def test_mix_rejects_silent_noise() -> None:
    speech = _speech_like(1000, seed=0)
    with pytest.raises(ValueError, match="silent"):
        noise.mix(speech, np.zeros(1000, dtype=np.float32), 5.0)


def test_mix_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        noise.mix(np.ones(10, dtype=np.float32), np.ones(11, dtype=np.float32), 5.0)


@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0])
def test_silence_padding_does_not_change_snr(snr_db: float) -> None:
    n = 3 * SAMPLE_RATE_HZ
    speech = _speech_like(n, seed=0)
    padded = np.concatenate([speech, np.zeros(SAMPLE_RATE_HZ, dtype=np.float32)])
    noise_sig = _noises(len(padded))["white"].astype(np.float32)

    base = noise.mix(speech, noise_sig[:n], snr_db)
    padded_mix = noise.mix(padded, noise_sig, snr_db)

    assert abs(measured_snr_db(speech, base) - snr_db) < 0.1
    assert abs(measured_snr_db(padded, padded_mix) - snr_db) < 0.1
    difference_db = measured_snr_db(padded, padded_mix) - measured_snr_db(speech, base)
    assert abs(difference_db) < 1e-6
