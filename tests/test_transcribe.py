"""Tests for transcribers and the transcription cache (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from earbench.audio import SAMPLE_RATE_HZ
from earbench.transcribe import CachedTranscriber, FakeTranscriber, audio_key


def test_audio_key_stable_and_sensitive(tone_mono: np.ndarray) -> None:
    key = audio_key(tone_mono, SAMPLE_RATE_HZ)
    assert key == audio_key(tone_mono.copy(), SAMPLE_RATE_HZ)
    changed = tone_mono.copy()
    changed[0] += 0.01
    assert audio_key(changed, SAMPLE_RATE_HZ) != key
    assert audio_key(tone_mono, 8000) != key


def test_fake_transcriber_default_responses_and_counter(tone_mono: np.ndarray) -> None:
    fake = FakeTranscriber(default_text="hello world")
    fake.responses[audio_key(tone_mono, SAMPLE_RATE_HZ)] = "the quick brown fox"
    assert fake.transcribe(tone_mono, SAMPLE_RATE_HZ) == "the quick brown fox"
    assert fake.transcribe(tone_mono.copy(), SAMPLE_RATE_HZ) == "the quick brown fox"
    assert fake.transcribe(tone_mono * 0.5, SAMPLE_RATE_HZ) == "hello world"
    assert fake.calls == 3


def test_cached_transcriber_calls_inner_once(tmp_path: Path, tone_mono: np.ndarray) -> None:
    inner = FakeTranscriber(default_text="cached")
    cached = CachedTranscriber(inner, tmp_path / "cache")
    assert cached.transcribe(tone_mono, SAMPLE_RATE_HZ) == "cached"
    assert cached.transcribe(tone_mono.copy(), SAMPLE_RATE_HZ) == "cached"
    assert inner.calls == 1


def test_cache_key_depends_on_settings(tmp_path: Path, tone_mono: np.ndarray) -> None:
    cache = tmp_path / "cache"
    first = FakeTranscriber(default_text="one", settings="beam=1")
    second = FakeTranscriber(default_text="two", settings="beam=2")
    assert CachedTranscriber(first, cache).transcribe(tone_mono, SAMPLE_RATE_HZ) == "one"
    assert CachedTranscriber(second, cache).transcribe(tone_mono, SAMPLE_RATE_HZ) == "two"
    assert second.calls == 1


def test_faster_whisper_settings_separate_cache_entries(
    monkeypatch, tmp_path: Path, tone_mono: np.ndarray
) -> None:
    import faster_whisper

    from earbench.transcribe import FasterWhisperTranscriber

    seen_kwargs: list[dict[str, object]] = []

    class DummyModel:
        def __init__(
            self, model_size: str, device: str = "cpu", compute_type: str = "int8"
        ) -> None:
            self.tag = f"{device}/{compute_type}"

        def transcribe(self, audio: np.ndarray, **kwargs: object):
            seen_kwargs.append(dict(kwargs))
            return [SimpleNamespace(text=self.tag)], None

    monkeypatch.setattr(faster_whisper, "WhisperModel", DummyModel)
    int8 = FasterWhisperTranscriber("tiny", device="cpu", compute_type="int8")
    float32 = FasterWhisperTranscriber("tiny", device="cpu", compute_type="float32")
    assert "device=cpu" in int8.settings
    assert "compute=int8" in int8.settings
    assert "temp=0.0" in int8.settings
    assert int8.temperature == 0.0
    assert int8.version == float32.version  # differ only in compute_type
    assert int8.settings != float32.settings

    cache = tmp_path / "cache"
    first = CachedTranscriber(int8, cache).transcribe(tone_mono, SAMPLE_RATE_HZ)
    second = CachedTranscriber(float32, cache).transcribe(tone_mono, SAMPLE_RATE_HZ)
    assert (first, second) == ("cpu/int8", "cpu/float32")
    assert len(list(cache.glob("*.txt"))) == 2
    assert all(kwargs["temperature"] == 0.0 for kwargs in seen_kwargs)


def test_leftover_temp_file_is_not_a_cache_hit(tmp_path: Path, tone_mono: np.ndarray) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "abc123.txt.4242.tmp").write_text("stale partial write", encoding="utf-8")
    inner = FakeTranscriber(default_text="fresh result")
    cached = CachedTranscriber(inner, cache)
    assert cached.transcribe(tone_mono, SAMPLE_RATE_HZ) == "fresh result"
    assert inner.calls == 1
