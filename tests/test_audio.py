"""Tests for audio helpers: 16 kHz mono WAV I/O, resampling, clip decoding (Phase 1)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from earbench import audio

SAMPLE_RATE_HZ = 16_000


def _sine(freq_hz: float, duration_s: float, sample_rate_hz: int) -> np.ndarray:
    """Generate a mono sine wave as float32."""
    t = np.arange(int(duration_s * sample_rate_hz), dtype=np.float64) / sample_rate_hz
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _fft_peak_hz(signal: np.ndarray, sample_rate_hz: int) -> float:
    """Return the dominant frequency of a mono signal."""
    spectrum = np.abs(np.fft.rfft(signal.astype(np.float64)))
    freqs = np.fft.rfftfreq(signal.shape[0], 1.0 / sample_rate_hz)
    return float(freqs[int(np.argmax(spectrum))])


def test_save_load_roundtrip_preserves_samples(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    tone = _sine(440.0, 1.0, SAMPLE_RATE_HZ)
    audio.save_wav_16k_mono(path, tone)
    loaded, sample_rate_hz = audio.load_mono(path)
    assert sample_rate_hz == SAMPLE_RATE_HZ
    assert loaded.dtype == np.float32
    assert loaded.shape == tone.shape
    # 16-bit quantisation (round) + libsndfile's /32768 normalisation: < 1.5 LSB.
    assert np.max(np.abs(loaded - tone)) < 1e-4


def test_load_mono_averages_stereo_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = _sine(440.0, 0.5, SAMPLE_RATE_HZ)
    right = _sine(880.0, 0.5, SAMPLE_RATE_HZ)
    sf.write(str(path), np.stack([left, right], axis=1), SAMPLE_RATE_HZ)
    loaded, _ = audio.load_mono(path)
    assert loaded.ndim == 1
    np.testing.assert_allclose(loaded, (left + right) / 2.0, atol=2e-4)


def test_resample_48k_to_16k_keeps_pitch_and_length() -> None:
    source = _sine(440.0, 1.0, 48_000)
    out = audio.resample_to_16k(source, 48_000)
    assert out.shape == (SAMPLE_RATE_HZ,)
    assert abs(_fft_peak_hz(out, SAMPLE_RATE_HZ) - 440.0) < 5.0


def test_resample_same_rate_returns_equal_copy() -> None:
    tone = _sine(440.0, 0.5, SAMPLE_RATE_HZ)
    out = audio.resample_to_16k(tone, SAMPLE_RATE_HZ)
    np.testing.assert_array_equal(out, tone.astype(np.float32))


def test_decode_clip_to_wav_writes_16k_mono(tmp_path: Path) -> None:
    src = tmp_path / "clip_src.wav"
    sf.write(str(src), _sine(440.0, 2.0, 48_000), 48_000)
    dst = tmp_path / "clip.wav"
    duration_s = audio.decode_clip_to_wav(src, dst)
    assert duration_s == 2.0
    with wave.open(str(dst), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == SAMPLE_RATE_HZ
    loaded, sample_rate_hz = audio.load_mono(dst)
    assert sample_rate_hz == SAMPLE_RATE_HZ
    assert abs(_fft_peak_hz(loaded, SAMPLE_RATE_HZ) - 440.0) < 5.0


def test_check_wav_16k_mono_accepts_good_file(tmp_path: Path) -> None:
    path = tmp_path / "good.wav"
    audio.save_wav_16k_mono(path, _sine(440.0, 0.5, SAMPLE_RATE_HZ))
    assert audio.check_wav_16k_mono(path) == []


def test_check_wav_16k_mono_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "stereo8k.wav"
    stereo = np.stack([_sine(440.0, 0.5, 8_000), _sine(440.0, 0.5, 8_000)], axis=1).astype(
        np.float32
    )
    sf.write(str(path), stereo, 8_000)
    problems = audio.check_wav_16k_mono(path)
    assert any("channel" in problem for problem in problems)
    assert any("16" in problem and "000" in problem for problem in problems)
