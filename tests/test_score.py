"""Tests for normalisation, WER, usable rate and bootstrap intervals (Phase 3)."""

from __future__ import annotations

from earbench.score import bootstrap_ci, corpus_wer, normalise, score_clip, usable_rate


def test_normalise_applies_whisper_rules() -> None:
    assert normalise("Hello, world.") == "hello world"
    assert normalise("I won't go") == "i will not go"
    assert normalise("  Spaced    out  ") == "spaced out"


def test_wer_identical_strings_is_zero() -> None:
    scored = score_clip("hello world", "hello world")
    assert (scored.errors, scored.ref_words) == (0, 2)
    assert scored.wer == 0.0


def test_wer_hand_computed_cases() -> None:
    substitution = score_clip("hello world", "hello there")
    assert (substitution.errors, substitution.ref_words, substitution.wer) == (1, 2, 0.5)

    deletion = score_clip("the cat sat down", "the cat sat")
    assert (deletion.errors, deletion.ref_words, deletion.wer) == (1, 4, 0.25)

    insertion = score_clip("the cat", "the cat sat")
    assert (insertion.errors, insertion.ref_words, insertion.wer) == (1, 2, 0.5)


def test_wer_normalises_before_scoring() -> None:
    # Case, punctuation and contraction differences must not count as errors.
    assert score_clip("Hello, world!", "hello world").wer == 0.0


def test_corpus_wer_pools_counts() -> None:
    rows = [score_clip("a b c d", "a b c d"), score_clip("a b c d", "a b x d")]
    assert corpus_wer(rows) == 1 / 8


def test_usable_rate_counts_only_strictly_under_threshold() -> None:
    rows = [
        score_clip("a b", "a b"),  # 0.0
        score_clip("a b", "a x"),  # 0.5
        score_clip("a b", "x y"),  # 1.0
    ]
    assert usable_rate(rows, 0.2) == 1 / 3
    assert usable_rate(rows, 0.5) == 1 / 3  # exactly 0.5 is not "under" 0.5
    assert usable_rate(rows, 1.0) == 2 / 3  # exactly 1.0 is not "under" 1.0
    assert usable_rate([], 0.2) == 0.0


def test_bootstrap_interval_contains_point_and_is_reproducible() -> None:
    rows = [
        score_clip("a b c d", "a b c d"),
        score_clip("a b c d", "a b x d"),
        score_clip("a b c d", "a b c x"),
    ]
    point = corpus_wer(rows)
    low, high = bootstrap_ci(rows, iters=200, seed=0)
    assert low <= point <= high
    assert (low, high) == bootstrap_ci(rows, iters=200, seed=0)


def test_bootstrap_empty_rows_is_zero() -> None:
    assert bootstrap_ci([], iters=10, seed=0) == (0.0, 0.0)
