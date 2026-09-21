"""Audio helpers: 16 kHz mono WAV I/O, resampling, clip decoding."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE_HZ = 16_000


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any soundfile-readable audio as mono float32.

    Multi-channel audio is averaged to mono. Returns (audio, sample_rate_hz)
    without resampling; use :func:`resample_to_16k` to convert.
    """
    data, sample_rate_hz = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    return mono, int(sample_rate_hz)


def resample_to_16k(audio: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Resample mono audio to 16 kHz with linear interpolation."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate_hz == SAMPLE_RATE_HZ:
        return mono.copy()
    if sample_rate_hz <= 0:
        raise ValueError(f"invalid sample rate: {sample_rate_hz}")
    duration_s = mono.shape[0] / sample_rate_hz
    out_len = max(1, int(round(duration_s * SAMPLE_RATE_HZ)))
    positions = np.linspace(0.0, float(mono.shape[0] - 1), out_len)
    return np.interp(positions, np.arange(mono.shape[0]), mono.astype(np.float64)).astype(
        np.float32
    )


def measure_duration_s(n_samples: int, sample_rate_hz: int) -> float:
    """Return the duration in seconds of a sample count at a sample rate."""
    return n_samples / sample_rate_hz


def save_wav_16k_mono(path: str | Path, audio: np.ndarray) -> Path:
    """Save 16 kHz mono float32 audio as 16-bit PCM WAV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    frames = np.round(clipped * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE_HZ)
        wf.writeframes(frames.tobytes())
    return out_path


def decode_clip_to_wav(src: str | Path, dst: str | Path) -> float:
    """Decode a clip (MP3/WAV) to 16 kHz mono WAV. Returns duration_s."""
    src_path, dst_path = Path(src), Path(dst)
    mono, sample_rate_hz = load_mono(src_path)
    converted = resample_to_16k(mono, sample_rate_hz)
    save_wav_16k_mono(dst_path, converted)
    return measure_duration_s(converted.shape[0], SAMPLE_RATE_HZ)


def check_wav_16k_mono(path: str | Path) -> list[str]:
    """Check a WAV file is 16 kHz mono 16-bit. Returns a list of problems."""
    wav_path = Path(path)
    if not wav_path.is_file():
        return [f"file not found: {wav_path}"]
    try:
        with wave.open(str(wav_path), "rb") as wf:
            problems = []
            if wf.getnchannels() != 1:
                problems.append(f"{wav_path}: expected 1 channel, got {wf.getnchannels()}")
            if wf.getsampwidth() != 2:
                problems.append(
                    f"{wav_path}: expected 16-bit audio, got {8 * wf.getsampwidth()}-bit"
                )
            if wf.getframerate() != SAMPLE_RATE_HZ:
                problems.append(
                    f"{wav_path}: expected {SAMPLE_RATE_HZ} Hz, got {wf.getframerate()} Hz"
                )
            return problems
    except wave.Error as exc:
        return [f"{wav_path}: unreadable WAV: {exc}"]


def power(audio: np.ndarray) -> float:
    """Mean square of a signal (its average power)."""
    x = np.asarray(audio, dtype=np.float64)
    return float(np.mean(x**2))


def peak_normalise(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale audio so its largest sample is `peak`. Silence is returned unchanged."""
    biggest = float(np.max(np.abs(audio)))
    if biggest == 0.0:
        return audio.copy()
    return (audio * (peak / biggest)).astype(np.float32)
