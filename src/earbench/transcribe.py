"""Transcribers and the transcription cache (Phase 3)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from earbench.audio import SAMPLE_RATE_HZ


class Transcriber(Protocol):
    """Turns 16 kHz mono audio into text.

    Implementations declare `name`, `version` and `settings` so every result row
    and every cache entry records exactly which model and options ran.
    """

    name: str
    version: str
    settings: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


def audio_key(audio: np.ndarray, sample_rate: int) -> str:
    """Stable fingerprint of a chunk of audio: sha256 of its float32 samples + rate."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    digest.update(str(int(sample_rate)).encode("ascii"))
    return digest.hexdigest()


@dataclass
class FakeTranscriber:
    """Deterministic, offline transcriber for tests.

    Returns `responses[audio_key(audio, rate)]` when the audio is known, else
    `default_text`, and counts how many times it was actually asked to transcribe.
    """

    default_text: str = ""
    responses: dict[str, str] = field(default_factory=dict)
    name: str = "fake"
    version: str = "1"
    settings: str = "fake"
    calls: int = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        return self.responses.get(audio_key(audio, sample_rate), self.default_text)


class CachedTranscriber:
    """Wrap a transcriber and store its output on disk.

    The cache key is the audio fingerprint plus the wrapped transcriber's name,
    version and settings, so a rerun of the same audio with the same model and
    options never calls the wrapped transcriber again.
    """

    def __init__(self, inner: Transcriber, cache_dir: str | Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.name = inner.name
        self.version = inner.version
        self.settings = inner.settings

    def _key(self, audio: np.ndarray, sample_rate: int) -> str:
        parts = (
            audio_key(audio, sample_rate),
            self.inner.name,
            self.inner.version,
            self.inner.settings,
        )
        return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        path = self.cache_dir / f"{self._key(audio, sample_rate)}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        text = self.inner.transcribe(audio, sample_rate)
        self._store(path, text)
        return text

    def _store(self, path: Path, text: str) -> None:
        """Write atomically (temp file + os.replace), so a crash never leaves a fake hit."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        handle_fd, tmp_name = tempfile.mkstemp(
            dir=self.cache_dir, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise


class FasterWhisperTranscriber:
    """faster-whisper. The model is loaded lazily on first use."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 5,
        vad_filter: bool = False,
        temperature: float = 0.0,
    ) -> None:
        from faster_whisper import WhisperModel  # heavy import, only when a real model runs

        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.temperature = temperature
        self.version = f"{package_version('faster-whisper')}+{model_size}"
        self.settings = (
            f"device={device},compute={compute_type},"
            f"beam={beam_size},vad={'on' if vad_filter else 'off'},temp={temperature}"
        )
        self._model: Any = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE_HZ:
            raise ValueError(f"expected {SAMPLE_RATE_HZ} Hz audio, got {sample_rate} Hz")
        segments, _info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            temperature=self.temperature,
        )
        return " ".join(str(segment.text).strip() for segment in segments).strip()
