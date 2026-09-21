"""Room simulation with pyroomacoustics: distance and echo."""

from __future__ import annotations

import numpy as np
import pyroomacoustics as pra

from earbench.audio import SAMPLE_RATE_HZ
from earbench.config import RoomConfig

Position = tuple[float, float, float]


def _check_inside(pos_m: Position, room: RoomConfig, what: str) -> None:
    for coord, size in zip(pos_m, room.dims_m, strict=True):
        if not 0.0 < coord < size:
            raise ValueError(
                f"{what} at {pos_m} m is outside the room {room.dims_m} m; "
                "use a shorter distance or move the mic"
            )


def _run(signal: np.ndarray, source_m: Position, room: RoomConfig) -> np.ndarray:
    """Play `signal` from `source_m` in the room and return what the mic hears."""
    _check_inside(room.mic_pos_m, room, "mic")
    absorption, max_order = pra.inverse_sabine(room.rt60_s, list(room.dims_m))
    shoebox = pra.ShoeBox(
        list(room.dims_m),
        fs=SAMPLE_RATE_HZ,
        materials=pra.Material(absorption),
        max_order=max_order,
    )
    shoebox.add_source(list(source_m), signal=np.asarray(signal, dtype=np.float64))
    shoebox.add_microphone(list(room.mic_pos_m))
    shoebox.simulate()
    return shoebox.mic_array.signals[0].astype(np.float32)  # type: ignore[union-attr]


def speaker_position(distance_m: float, room: RoomConfig) -> Position:
    """Speaker sits `distance_m` from the mic along +x, at mouth height."""
    mic_x, mic_y, _ = room.mic_pos_m
    return (mic_x + distance_m, mic_y, room.source_height_m)


def simulate(clip: np.ndarray, distance_m: float, room: RoomConfig) -> np.ndarray:
    """Speech as heard by the mic from `distance_m` away. Output includes the echo tail."""
    source_m = speaker_position(distance_m, room)
    _check_inside(source_m, room, f"speaker ({distance_m} m from mic)")
    return _run(clip, source_m, room)


def simulate_noise(noise: np.ndarray, room: RoomConfig) -> np.ndarray:
    """Noise as heard by the mic, played from `room.noise_pos_m`."""
    _check_inside(room.noise_pos_m, room, "noise source")
    return _run(noise, room.noise_pos_m, room)
