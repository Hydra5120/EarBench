"""Audio helpers: 16 kHz mono WAV I/O, resampling, clip decoding."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE_HZ = 16_000
FRAME_MS = 20.0
ACTIVE_RANGE_DB = 40.0


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any soundfile-readable audio as mono float32.

    Multi-channel audio is averaged to mono. Returns (audio, sample_rate_hz)
    without resampling; use :func:`resample_to_16k` to convert.
    """
    data, sample_rate_hz = sf.read(str(path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    return mono, int(sample_rate_hz)


def resample_to_16k(audio: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Resample mono audio to 16 kHz with a polyphase filter (anti-aliased)."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate_hz == SAMPLE_RATE_HZ:
        return mono.copy()
    divisor = math.gcd(sample_rate_hz, SAMPLE_RATE_HZ)
    return resample_poly(mono, SAMPLE_RATE_HZ // divisor, sample_rate_hz // divisor).astype(
        np.float32
    )


def load_16k(path: str | Path) -> np.ndarray:
    """Load any soundfile-readable audio as 16 kHz mono float32."""
    return resample_to_16k(*load_mono(path))


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
    converted = load_16k(src)
    save_wav_16k_mono(dst, converted)
    return converted.shape[0] / SAMPLE_RATE_HZ


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


def active_speech_power(
    audio: np.ndarray,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    frame_ms: float = FRAME_MS,
    range_db: float = ACTIVE_RANGE_DB,
) -> float:
    """Mean square of the active (speech) frames of a signal.

    Splits the signal into 20 ms frames, keeps the frames within `range_db` of
    the loudest frame, and returns the mean square of those frames. Pauses between
    words and silence padding are ignored, so the level reflects speech, not gaps.
    """
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    frame_length = max(1, int(round(frame_ms / 1000.0 * sample_rate_hz)))
    n_frames = x.shape[0] // frame_length
    if n_frames == 0:
        return power(x)
    frames = x[: n_frames * frame_length].reshape(n_frames, frame_length)
    frame_powers = np.mean(frames**2, axis=1)
    loudest = float(frame_powers.max())
    if loudest == 0.0:
        return 0.0
    threshold = loudest / (10.0 ** (range_db / 10.0))
    active = frame_powers[frame_powers >= threshold]
    return float(np.mean(active))


def peak_normalise(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale audio so its largest sample is `peak`. Silence is returned unchanged."""
    biggest = float(np.max(np.abs(audio)))
    if biggest == 0.0:
        return audio.copy()
    return (audio * (peak / biggest)).astype(np.float32)
