"""Noise segments and mixing at an exact SNR."""

from __future__ import annotations

import numpy as np

from earbench import room as room_sim
from earbench.audio import active_speech_power, power
from earbench.config import RoomConfig


def pick_segment(noise: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Cut `n_samples` from `noise` at a random (seeded) start."""
    if noise.shape[0] < n_samples:
        raise ValueError(f"noise file too short: {noise.shape[0]} samples, need {n_samples}")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, noise.shape[0] - n_samples + 1))
    return noise[start : start + n_samples].copy()


def mix(speech_at_mic: np.ndarray, noise_at_mic: np.ndarray, snr_db: float) -> np.ndarray:
    """Add noise scaled so that active-speech power / noise power equals `snr_db`.

    The speech level ignores pauses and silence padding (see
    :func:`earbench.audio.active_speech_power`), so the target SNR reflects how
    loud the speech actually is, not how much of the clip is quiet.
    """
    if speech_at_mic.shape != noise_at_mic.shape:
        raise ValueError(
            f"speech and noise length differ: {speech_at_mic.shape} vs {noise_at_mic.shape}"
        )
    speech_power = active_speech_power(speech_at_mic)
    noise_power = power(noise_at_mic)
    if noise_power == 0.0:
        raise ValueError("noise is silent, cannot set an SNR")
    if speech_power == 0.0:
        raise ValueError("speech is silent, cannot set an SNR")
    gain = np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0)))
    # Scale the noise to float32 first, so the added part has exactly the target power.
    scaled_noise = (noise_at_mic.astype(np.float64) * gain).astype(np.float32)
    return speech_at_mic + scaled_noise


def noise_at_mic(noise_file: np.ndarray, n_samples: int, seed: int, room: RoomConfig) -> np.ndarray:
    """A seeded noise segment as heard by the mic, trimmed to `n_samples`."""
    segment = pick_segment(noise_file, n_samples, seed)
    return room_sim.simulate_noise(segment, room)[:n_samples]


def make_condition(
    clip: np.ndarray,
    distance_m: float,
    room: RoomConfig,
    noise_file: np.ndarray | None,
    snr_db: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full chain: speech + noise through the room, mixed at the mic.

    Returns (mixed, speech_at_mic). With no noise, mixed is just speech_at_mic.
    """
    speech = room_sim.simulate(clip, distance_m, room)
    if noise_file is None or snr_db is None:
        return speech, speech
    return mix(speech, noise_at_mic(noise_file, speech.shape[0], seed, room), snr_db), speech
