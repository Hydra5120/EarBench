"""Text normalisation, WER, usable rate and bootstrap intervals (Phase 3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from jiwer import process_words
from whisper_normalizer.english import EnglishTextNormalizer

_normaliser = EnglishTextNormalizer()


class Scored(Protocol):
    """A scored clip: everything needed to pool WER across a condition."""

    @property
    def errors(self) -> int: ...

    @property
    def ref_words(self) -> int: ...

    @property
    def wer(self) -> float: ...


@dataclass(frozen=True)
class ClipScore:
    """One clip scored: normalised texts, error/word counts and WER."""

    reference: str
    hypothesis: str
    errors: int
    ref_words: int
    wer: float


def normalise(text: str) -> str:
    """Whisper English normalisation: case, contractions, symbols and numbers."""
    return str(_normaliser(text)).strip()


def score_clip(reference: str, hypothesis: str) -> ClipScore:
    """WER for one clip: errors and reference words via jiwer, on normalised text."""
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    result = process_words(ref, hyp)
    errors = result.substitutions + result.deletions + result.insertions
    ref_words = result.hits + result.substitutions + result.deletions
    wer = errors / ref_words if ref_words else 0.0
    return ClipScore(reference=ref, hypothesis=hyp, errors=errors, ref_words=ref_words, wer=wer)


def corpus_wer(scored: Sequence[Scored]) -> float:
    """Corpus-level WER: total errors divided by total reference words."""
    total_errors = sum(row.errors for row in scored)
    total_words = sum(row.ref_words for row in scored)
    return total_errors / total_words if total_words else 0.0


def usable_rate(scored: Sequence[Scored], usable_wer: float) -> float:
    """Fraction of clips with WER strictly under `usable_wer` ("under 20%")."""
    if not scored:
        return 0.0
    return sum(1 for row in scored if row.wer < usable_wer) / len(scored)


def bootstrap_ci(
    scored: Sequence[Scored], iters: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Plain percentile bootstrap interval for corpus WER, resampling clips.

    Clips are drawn with replacement, the errors and reference words are pooled
    each time, and the interval is the alpha/2 and 1 - alpha/2 percentiles of the
    resampled corpus WERs. Reproducible for a given seed.
    """
    if not scored:
        return (0.0, 0.0)
    errors = np.array([row.errors for row in scored], dtype=np.float64)
    words = np.array([row.ref_words for row in scored], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(scored), size=(iters, len(scored)))
    resampled_errors = errors[idx].sum(axis=1)
    resampled_words = words[idx].sum(axis=1)
    stats = np.divide(
        resampled_errors,
        resampled_words,
        out=np.zeros_like(resampled_errors),
        where=resampled_words > 0,
    )
    lo = float(np.percentile(stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)
