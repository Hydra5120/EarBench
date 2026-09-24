"""Tests for the room simulation (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from earbench import room
from earbench.audio import SAMPLE_RATE_HZ
from earbench.config import RoomConfig

SPEED_OF_SOUND_M_S = 343.0


def _impulse() -> np.ndarray:
    x = np.zeros(SAMPLE_RATE_HZ // 2, dtype=np.float32)
    x[0] = 1.0
    return x


def _first_arrival(x: np.ndarray) -> int:
    """First sample louder than 20% of the peak. (At 3 m, piled-up echoes beat the
    direct sound, so the loudest sample is not the first arrival.)"""
    y = np.abs(x)
    return int(np.argmax(y > 0.2 * y.max()))


def test_far_impulse_arrives_later() -> None:
    cfg = RoomConfig()
    near = room.simulate(_impulse(), 1.0, cfg)
    far = room.simulate(_impulse(), 3.0, cfg)
    delay_samples = _first_arrival(far) - _first_arrival(near)
    expected = 2.0 / SPEED_OF_SOUND_M_S * SAMPLE_RATE_HZ  # about 93 samples
    assert abs(delay_samples - expected) < 3


def test_far_is_quieter() -> None:
    cfg = RoomConfig()
    near = room.simulate(_impulse(), 1.0, cfg)
    far = room.simulate(_impulse(), 3.0, cfg)
    assert np.max(np.abs(far)) < np.max(np.abs(near))


def test_output_is_float32_and_longer_than_input() -> None:
    clip = _impulse()
    out = room.simulate(clip, 2.0, RoomConfig())
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] >= clip.shape[0]


def test_same_input_same_output() -> None:
    clip = np.random.default_rng(0).standard_normal(4000).astype(np.float32)
    cfg = RoomConfig()
    assert np.array_equal(room.simulate(clip, 2.0, cfg), room.simulate(clip, 2.0, cfg))


def test_source_outside_room_raises() -> None:
    with pytest.raises(ValueError, match="outside the room"):
        room.simulate(_impulse(), 5.0, RoomConfig())


def test_noise_outside_room_raises() -> None:
    cfg = RoomConfig(noise_pos_m=(9.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="outside the room"):
        room.simulate_noise(_impulse(), cfg)


def test_simulate_noise_runs() -> None:
    noise_sig = np.random.default_rng(1).standard_normal(4000).astype(np.float32)
    out = room.simulate_noise(noise_sig, RoomConfig())
    assert out.dtype == np.float32
    assert out.shape[0] >= noise_sig.shape[0]


def test_simulate_at_matches_plus_x_position() -> None:
    """An explicit position on the +x axis sounds exactly like `simulate` there."""
    clip = np.random.default_rng(2).standard_normal(4000).astype(np.float32)
    cfg = RoomConfig()
    expected = room.simulate(clip, 2.0, cfg)
    np.testing.assert_array_equal(
        room.simulate_at(clip, room.speaker_position(2.0, cfg), cfg), expected
    )


def test_simulate_at_outside_room_raises() -> None:
    with pytest.raises(ValueError, match="outside the room"):
        room.simulate_at(_impulse(), (9.0, 1.0, 1.0), RoomConfig())
