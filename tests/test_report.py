"""Tests for the HTML report and charts (Phase 4). All offline, no models."""

from __future__ import annotations

from pathlib import Path

import pytest

from earbench import report, sweep
from earbench.config import ResultRow, SummaryRow, SweepConfig, load_config
from earbench.transcribe import FakeTranscriber


def _row(clip_id: str, model: str, wer: float, ref_words: int = 10, **over: object) -> ResultRow:
    errors = round(wer * ref_words)
    base: dict[str, object] = {
        "run_id": "r1",
        "source": "sim",
        "clip_id": clip_id,
        "age_group": "older",
        "distance_m": 2.0,
        "noise_type": "tv",
        "snr_db": 10.0,
        "model": model,
        "model_version": "fake+1",
        "settings": "fake",
        "reference": "hello world",
        "hypothesis": "hello world",
        "ref_words": ref_words,
        "errors": errors,
        "wer": errors / ref_words,
        "config_hash": "abc",
    }
    base.update(over)
    return ResultRow.model_validate(base)


def _summary(model: str, noise: str, wer: float, age: str = "older", **over: object) -> SummaryRow:
    base: dict[str, object] = {
        "model": model,
        "age_group": age,
        "distance_m": 2.0,
        "noise_type": noise,
        "snr_db": 10.0,
        "n_clips": 2,
        "errors": 4,
        "ref_words": 20,
        "wer": wer,
        "ci_low": wer - 0.05,
        "ci_high": wer + 0.05,
        "usable_rate": 0.5,
    }
    base.update(over)
    return SummaryRow.model_validate(base)


def test_pick_best_model_prefers_lowest_overall_wer() -> None:
    rows = [
        _row("a", "tiny", 0.4),
        _row("b", "tiny", 0.4),
        _row("a", "small", 0.1),
        _row("b", "small", 0.1),
    ]
    assert report.pick_best_model(rows, None) == "small"
    assert report.pick_best_model(rows, "tiny") == "tiny"
    with pytest.raises(ValueError, match="small"):
        report.pick_best_model(rows, "missing-model")


def test_pool_recomputes_from_results_not_summaries() -> None:
    # Two clips: 0/10 + 10/10 errors. Pooled WER is 0.5, not the mean of per-clip WERs.
    rows = [_row("a", "tiny", 0.0), _row("b", "tiny", 1.0)]
    pooled = report.pool(rows, iters=50, seed=0, usable_wer=0.2)
    assert pooled.wer == pytest.approx(0.5)
    assert pooled.n_clips == 2
    assert pooled.ci_low <= pooled.wer <= pooled.ci_high


def test_worst_per_noise_type_reports_other_age_group() -> None:
    summary = [
        _summary("small", "tv", 0.6, "older", distance_m=2.0, snr_db=0.0),
        _summary("small", "tv", 0.2, "younger", distance_m=2.0, snr_db=0.0),
        _summary("small", "tv", 0.1, "older", distance_m=1.0, snr_db=20.0),
        _summary("small", "living", 0.3, "older"),
    ]
    worst = report.worst_per_noise_type(summary, "small")
    assert set(worst) == {"tv", "living"}
    assert worst["tv"].row.wer == pytest.approx(0.6)
    assert worst["tv"].other_wer == pytest.approx(0.2)


def test_example_clips_picks_median_and_worst() -> None:
    rows = [_row(f"c{i}", "small", wer, age_group="older") for i, wer in enumerate([0.0, 0.5, 1.0])]
    median, worst = report.example_clips(rows)
    assert median.wer == pytest.approx(0.5)
    assert worst.wer == pytest.approx(1.0)


def test_write_report_builds_html_and_pngs(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=2, distances_m=(2.0,), noise_types=("tv",), snr_db=(10.0, 0.0))
    run = sweep.run_sweep(cfg, lambda model: FakeTranscriber(default_text="hello"), progress=False)
    out = report.write_report(run.run_dir, run.run_dir.parent / "report-out")
    assert (out / "report.html").is_file()
    for name in ("wer_vs_snr.png", "age_gap.png", "wer_vs_distance.png"):
        assert (out / name).is_file()
    html = (out / "report.html").read_text(encoding="utf-8")
    assert run.run_id in html
    assert "tiny" in html  # best (only) model is named
    assert "SNR (dB)" in html  # headline table header
    assert "WER vs distance" in html


def test_charts_label_axes_with_units(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=2, distances_m=(1.0, 2.0), noise_types=("tv",), snr_db=(10.0,))
    run = sweep.run_sweep(cfg, lambda model: FakeTranscriber(default_text="hi"), progress=False)
    figs = report.build_figures(run.run_dir)
    try:
        assert len(figs) == 3
        for fig in figs:
            labels = [ax.get_xlabel() + ax.get_ylabel() for ax in fig.axes]
            assert any("dB" in text or "(m)" in text for text in labels)
            assert all("WER" in text for text in labels)
    finally:
        for fig in figs:
            report.close_figure(fig)


def test_write_report_missing_results_fails_clearly(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="results.csv"):
        report.write_report(run_dir, tmp_path / "out")


def test_distance_chart_shows_clean_and_noisy_lines(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1, distances_m=(1.0, 2.0), noise_types=("tv",), snr_db=(10.0,))
    run = sweep.run_sweep(cfg, lambda model: FakeTranscriber(default_text="hi"), progress=False)
    stored = load_config(run.run_dir / "config.yaml", SweepConfig)
    fig, notes = report.fig_wer_vs_distance(
        report.read_results(run.run_dir / "results.csv"), stored, "tiny", "tv", [None, 10.0]
    )
    try:
        assert len(fig.axes[0].lines) == 4  # older/younger x clean/10 dB
        assert notes == []
    finally:
        report.close_figure(fig)


def test_unknown_distance_snr_is_noted_not_fatal(sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1, distances_m=(1.0,), noise_types=("tv",), snr_db=(0.0,))
    run = sweep.run_sweep(cfg, lambda model: FakeTranscriber(default_text="hi"), progress=False)
    out = report.write_report(
        run.run_dir, run.run_dir.parent / "out2", distance_snr_db=[None, 10.0]
    )
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "not in this run" in html
