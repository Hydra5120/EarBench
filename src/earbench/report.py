"""Charts and HTML report for a sweep run (Phase 4)."""

from __future__ import annotations

import csv
import html
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from earbench import score as scoring
from earbench.config import ResultRow, SummaryRow, SweepConfig, load_config

logger = logging.getLogger(__name__)

CHART_A = "wer_vs_snr.png"
CHART_B = "age_gap.png"
CHART_C = "wer_vs_distance.png"


@dataclass(frozen=True)
class Pooled:
    """WER recomputed from result rows: total errors / total words, plus interval."""

    wer: float
    ci_low: float
    ci_high: float
    usable_rate: float
    n_clips: int


@dataclass(frozen=True)
class WorstCase:
    """The worst age-group row per noise type, plus the other age group's WER."""

    row: SummaryRow
    other_wer: float | None


def _read_csv(model: type[ResultRow] | type[SummaryRow], path: str | Path) -> list:
    """Read a results/summary CSV; empty cells mean None (e.g. clean-condition SNR)."""
    with open(path, encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            fixed = {
                key: (None if value == "" and key == "snr_db" else value)
                for key, value in raw.items()
            }
            rows.append(model.model_validate(fixed))
        return rows


def read_results(path: str | Path) -> list[ResultRow]:
    """Read results.csv back into validated rows."""
    return _read_csv(ResultRow, path)


def read_summary(path: str | Path) -> list[SummaryRow]:
    """Read summary.csv back into validated rows."""
    return _read_csv(SummaryRow, path)


def pool(rows: list[ResultRow], iters: int, seed: int, usable_wer: float) -> Pooled:
    """Pool result rows the honest way: total errors over total words, never averaged."""
    ci_low, ci_high = scoring.bootstrap_ci(rows, iters, seed)
    return Pooled(
        wer=scoring.corpus_wer(rows),
        ci_low=ci_low,
        ci_high=ci_high,
        usable_rate=scoring.usable_rate(rows, usable_wer),
        n_clips=len(rows),
    )


def overall_wer_by_model(results: list[ResultRow]) -> dict[str, float]:
    """Corpus WER per model over the whole run."""
    by_model: dict[str, list[ResultRow]] = {}
    for row in results:
        by_model.setdefault(row.model, []).append(row)
    return {model: scoring.corpus_wer(rows) for model, rows in by_model.items()}


def pick_best_model(results: list[ResultRow], override: str | None) -> str:
    """Lowest overall corpus WER wins, unless `override` names a model."""
    overall = overall_wer_by_model(results)
    if override is not None:
        if override not in overall:
            raise ValueError(f"best model {override!r} is not in this run (have {sorted(overall)})")
        return override
    return min(sorted(overall), key=lambda model: overall[model])


def worst_per_noise_type(summary: list[SummaryRow], model: str) -> dict[str, WorstCase]:
    """Worst age-group row per noise type for `model`, with the other age group's WER."""
    rows = [row for row in summary if row.model == model]
    by_noise: dict[str, list[SummaryRow]] = {}
    for row in rows:
        by_noise.setdefault(row.noise_type, []).append(row)
    worst: dict[str, WorstCase] = {}
    for noise_type, group in by_noise.items():
        top = max(group, key=lambda row: (row.wer, row.errors))
        other = next(
            (
                row.wer
                for row in group
                if row.age_group != top.age_group
                and row.distance_m == top.distance_m
                and row.snr_db == top.snr_db
            ),
            None,
        )
        worst[noise_type] = WorstCase(row=top, other_wer=other)
    return worst


def example_clips(rows: list[ResultRow]) -> tuple[ResultRow, ResultRow]:
    """The clip closest to the median WER and the single worst clip."""
    ordered = sorted(rows, key=lambda row: (row.wer, row.clip_id))
    median = ordered[(len(ordered) - 1) // 2]
    return median, ordered[-1]


def _snr_axis(results: list[ResultRow], noise_type: str) -> tuple[list[float], bool]:
    """Sorted noisy SNRs for a noise type, plus whether clean (None) is present."""
    own = [row for row in results if row.noise_type == noise_type]
    snrs = sorted({row.snr_db for row in own if row.snr_db is not None})
    return snrs, any(row.snr_db is None for row in own)


def _cell(results: list[ResultRow], cfg: SweepConfig, **match: object) -> Pooled | None:
    """Pool the rows matching every field, or None when the cell is empty."""
    rows = [
        row for row in results if all(getattr(row, key) == value for key, value in match.items())
    ]
    if not rows:
        return None
    return pool(rows, cfg.bootstrap_iters, cfg.seed, cfg.usable_wer)


def _plot_line(ax: object, xs: Sequence[float], cells: list[Pooled | None], label: str) -> None:
    """One line with its bootstrap band; missing cells break the line."""
    ys = [100 * cell.wer if cell else float("nan") for cell in cells]
    lows = [100 * cell.ci_low if cell else float("nan") for cell in cells]
    highs = [100 * cell.ci_high if cell else float("nan") for cell in cells]
    ax.plot(xs, ys, marker="o", label=label)  # type: ignore[attr-defined]
    ax.fill_between(xs, lows, highs, alpha=0.2)  # type: ignore[attr-defined]


def _legend_or_note(ax: object, note: str) -> None:
    """A legend when lines exist, otherwise a plain note so empty panels never warn."""
    _, labels = ax.get_legend_handles_labels()  # type: ignore[attr-defined]
    if labels:
        ax.legend()  # type: ignore[attr-defined]
    else:
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center")  # type: ignore[attr-defined]


def fig_wer_vs_snr(results: list[ResultRow], cfg: SweepConfig, ref_distance_m: float) -> Figure:
    """Chart A: WER vs SNR at the reference distance, one line per model per noise panel."""
    at_ref = [row for row in results if row.distance_m == ref_distance_m]
    noises = sorted({row.noise_type for row in at_ref if row.noise_type != "none"})
    models = sorted({row.model for row in at_ref})
    fig, axes = plt.subplots(
        1, max(len(noises), 1), figsize=(5 * max(len(noises), 1), 4), squeeze=False
    )
    for ax, noise_type in zip(axes[0], noises or ["tv"], strict=True):
        snrs, has_clean = _snr_axis(at_ref, noise_type)
        labels = [f"{snr:g}" for snr in snrs] + (["clean"] if has_clean else [])
        xs = list(range(len(labels)))
        for model in models:
            cells = [
                _cell(at_ref, cfg, model=model, noise_type=noise_type, snr_db=snr) for snr in snrs
            ]
            if has_clean:
                cells.append(_cell(at_ref, cfg, model=model, noise_type=noise_type, snr_db=None))
            _plot_line(ax, xs, cells, model)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("WER (%)")
        ax.set_title(f"{noise_type}, {ref_distance_m:g} m")
        _legend_or_note(ax, f"no rows at {ref_distance_m:g} m in this run")
    fig.suptitle(f"WER vs SNR at {ref_distance_m:g} m (both age groups pooled)")
    fig.tight_layout()
    return fig


def fig_age_gap(
    results: list[ResultRow], cfg: SweepConfig, model: str, ref_distance_m: float
) -> Figure:
    """Chart B: older vs younger at each SNR, best model, same reference distance."""
    at_ref = [row for row in results if row.distance_m == ref_distance_m and row.model == model]
    noises = sorted({row.noise_type for row in at_ref if row.noise_type != "none"})
    fig, axes = plt.subplots(
        1, max(len(noises), 1), figsize=(5 * max(len(noises), 1), 4), squeeze=False
    )
    for ax, noise_type in zip(axes[0], noises or ["tv"], strict=True):
        snrs, has_clean = _snr_axis(at_ref, noise_type)
        labels = [f"{snr:g}" for snr in snrs] + (["clean"] if has_clean else [])
        xs = list(range(len(labels)))
        for age_group in ("older", "younger"):
            cells = [
                _cell(at_ref, cfg, noise_type=noise_type, age_group=age_group, snr_db=snr)
                for snr in snrs
            ]
            if has_clean:
                cells.append(
                    _cell(at_ref, cfg, noise_type=noise_type, age_group=age_group, snr_db=None)
                )
            _plot_line(ax, xs, cells, age_group)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("WER (%)")
        ax.set_title(f"{noise_type}, {model}, {ref_distance_m:g} m")
        _legend_or_note(ax, f"no rows at {ref_distance_m:g} m in this run")
    fig.suptitle(f"Older vs younger ({model}, {ref_distance_m:g} m)")
    fig.tight_layout()
    return fig


def fig_wer_vs_distance(
    results: list[ResultRow],
    cfg: SweepConfig,
    model: str,
    noise_type: str,
    snr_levels: list[float | None],
) -> tuple[Figure, list[str]]:
    """Chart C: WER vs distance for the given noise and SNR levels. Returns notes for gaps."""
    rows = [row for row in results if row.model == model and row.noise_type in (noise_type, "none")]
    distances = sorted({row.distance_m for row in rows})
    notes: list[str] = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for age_group in ("older", "younger"):
        for snr_db in snr_levels:
            tag = "clean" if snr_db is None else f"{snr_db:g} dB"
            # Clean rows live under noise "none"; noisy rows under the chart's noise.
            want_noise = "none" if snr_db is None else noise_type
            cells = [
                _cell(
                    rows,
                    cfg,
                    age_group=age_group,
                    distance_m=distance_m,
                    snr_db=snr_db,
                    noise_type=want_noise,
                )
                for distance_m in distances
            ]
            if all(cell is None for cell in cells):
                note = f"{tag} ({want_noise}) is not in this run for {model}."
                if note not in notes:
                    notes.append(note)
                continue
            _plot_line(ax, distances, cells, f"{age_group}, {tag}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("WER (%)")
    ax.set_title(f"WER vs distance ({model}, {noise_type})")
    _legend_or_note(ax, f"no {noise_type} rows for {model} in this run")
    fig.tight_layout()
    return fig, notes


def build_figures(run_dir: str | Path) -> list[Figure]:
    """Build the three report charts from a run dir (used by tests and write_report)."""
    run_path = Path(run_dir)
    results_path = run_path / "results.csv"
    if not results_path.is_file():
        raise ValueError(f"run results not found: {results_path} (no such sweep run)")
    results = read_results(results_path)
    if not results:
        raise ValueError(f"run results are empty: {results_path}")
    cfg = load_config(run_path / "config.yaml", SweepConfig)
    model = pick_best_model(results, cfg.report.best_model)
    figs: list[Figure] = [
        fig_wer_vs_snr(results, cfg, cfg.report.reference_distance_m),
        fig_age_gap(results, cfg, model, cfg.report.reference_distance_m),
    ]
    fig_c, _ = fig_wer_vs_distance(
        results, cfg, model, cfg.report.distance_noise, list(cfg.report.distance_snr_db)
    )
    figs.append(fig_c)
    return figs


def close_figure(fig: Figure) -> None:
    """Free a figure built by this module."""
    plt.close(fig)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _headline_rows(
    results: list[ResultRow], cfg: SweepConfig, model: str, ref_distance_m: float
) -> str:
    """One row per (noise, SNR) at the reference distance: older vs younger with CIs."""
    cells: dict[tuple[str, float | None], dict[str, list[ResultRow]]] = {}
    for row in results:
        if row.model == model and row.distance_m == ref_distance_m:
            cells.setdefault((row.noise_type, row.snr_db), {}).setdefault(row.age_group, []).append(
                row
            )
    lines = [
        "<table><tr><th>Noise</th><th>SNR (dB)</th>"
        "<th>Older WER [95% CI]</th><th>Younger WER [95% CI]</th>"
        "<th>Older usable</th><th>Younger usable</th></tr>"
    ]
    for noise_type, snr_db in sorted(cells, key=lambda key: (key[0], key[1] is None, key[1])):
        parts = [f"<tr><td>{html.escape(noise_type)}</td>"]
        parts.append(f"<td>{'clean' if snr_db is None else f'{snr_db:g}'}</td>")
        for age_group in ("older", "younger"):
            pooled = pool(
                cells[(noise_type, snr_db)].get(age_group, []),
                cfg.bootstrap_iters,
                cfg.seed,
                cfg.usable_wer,
            )
            if pooled.n_clips == 0:
                parts.append("<td>—</td>")
            else:
                parts.append(
                    f"<td>{_pct(pooled.wer)} [{_pct(pooled.ci_low)}–{_pct(pooled.ci_high)}]</td>"
                )
        for age_group in ("older", "younger"):
            pooled = pool(
                cells[(noise_type, snr_db)].get(age_group, []),
                cfg.bootstrap_iters,
                cfg.seed,
                cfg.usable_wer,
            )
            parts.append(f"<td>{_pct(pooled.usable_rate)}</td>" if pooled.n_clips else "<td>—</td>")
        parts.append("</tr>")
        lines.append("".join(parts))
    lines.append("</table>")
    return "\n".join(lines)


def _worst_table(
    results: list[ResultRow], summary: list[SummaryRow], model: str, usable_wer: float
) -> str:
    """Worst condition per noise type: both age groups' WER plus median/worst transcripts."""
    worst = worst_per_noise_type(summary, model)
    lines = [
        "<table><tr><th>Noise</th><th>Condition</th><th>Worst-group WER</th>"
        "<th>Other-group WER</th><th>Usable rate</th>"
        "<th>Median clip: reference → hypothesis</th>"
        "<th>Worst clip: reference → hypothesis</th></tr>"
    ]
    for noise_type in sorted(worst):
        case = worst[noise_type]
        top = case.row
        snr = "clean" if top.snr_db is None else f"{top.snr_db:g} dB"
        clips = [
            row
            for row in results
            if row.model == model
            and row.noise_type == top.noise_type
            and row.distance_m == top.distance_m
            and row.snr_db == top.snr_db
            and row.age_group == top.age_group
        ]
        if clips:
            median, bad = example_clips(clips)
            median_text = f"{median.reference} → {median.hypothesis}"
            bad_text = f"{bad.reference} → {bad.hypothesis}"
        else:
            median_text = bad_text = "no clips in this run"
        other = _pct(case.other_wer) if case.other_wer is not None else "—"
        lines.append(
            f"<tr><td>{html.escape(noise_type)}</td>"
            f"<td>{top.distance_m:g} m, {snr}, {html.escape(top.age_group)}</td>"
            f"<td>{_pct(top.wer)}</td><td>{other}</td>"
            f"<td>{_pct(top.usable_rate)}</td>"
            f"<td>{html.escape(median_text[:320])}</td>"
            f"<td>{html.escape(bad_text[:320])}</td></tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def write_report(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    best_model: str | None = None,
    distance_snr_db: list[float | None] | None = None,
    distance_noise: str | None = None,
    reference_distance_m: float | None = None,
) -> Path:
    """Build charts + HTML report for a sweep run. Returns the output directory."""
    run_path = Path(run_dir)
    results_path = run_path / "results.csv"
    if not results_path.is_file():
        raise ValueError(f"run results not found: {results_path} (no such sweep run)")
    results = read_results(results_path)
    if not results:
        raise ValueError(f"run results are empty: {results_path}")
    cfg = load_config(run_path / "config.yaml", SweepConfig)
    if best_model is not None:
        cfg.report.best_model = best_model
    if distance_snr_db is not None:
        cfg.report.distance_snr_db = list(distance_snr_db)
    if distance_noise is not None:
        cfg.report.distance_noise = distance_noise  # type: ignore[assignment]
    if reference_distance_m is not None:
        cfg.report.reference_distance_m = reference_distance_m
    summary_path = run_path / "summary.csv"
    summary = read_summary(summary_path) if summary_path.is_file() else []

    model = pick_best_model(results, cfg.report.best_model)
    overall = overall_wer_by_model(results)
    out = Path(out_dir) if out_dir else Path("reports") / run_path.name
    out.mkdir(parents=True, exist_ok=True)

    figs = [
        fig_wer_vs_snr(results, cfg, cfg.report.reference_distance_m),
        fig_age_gap(results, cfg, model, cfg.report.reference_distance_m),
    ]
    fig_c, notes = fig_wer_vs_distance(
        results, cfg, model, cfg.report.distance_noise, list(cfg.report.distance_snr_db)
    )
    figs.append(fig_c)
    try:
        for fig, name in zip(figs, (CHART_A, CHART_B, CHART_C), strict=True):
            fig.savefig(out / name, dpi=100)
    finally:
        for fig in figs:
            close_figure(fig)

    notes_html = "".join(f"<p><i>Note: {html.escape(note)}</i></p>" for note in notes)
    config_text = (
        (run_path / "config.yaml").read_text(encoding="utf-8")
        if (run_path / "config.yaml").is_file()
        else "(no config.yaml in run dir)"
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EarBench report {html.escape(run_path.name)}</title>
<style>body {{ font-family: sans-serif; max-width: 60em; margin: 2em auto; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #999; padding: 4px 8px; }}</style>
</head><body>
<h1>EarBench report: {html.escape(run_path.name)}</h1>
<p>Run id <b>{html.escape(results[0].run_id)}</b>,
config hash {html.escape(results[0].config_hash)}.
Models: {html.escape(", ".join(sorted(overall)))}. Usable threshold:
WER under {_pct(cfg.usable_wer)}; bootstrap {cfg.bootstrap_iters} iters, seed {cfg.seed}.</p>
<p>Best model: <b>{html.escape(model)}</b> (overall WER {_pct(overall[model])}).</p>
<h2>Headline: best model at {cfg.report.reference_distance_m:g} m</h2>
{_headline_rows(results, cfg, model, cfg.report.reference_distance_m)}
<h2>WER vs SNR (all models, both age groups pooled)</h2>
<img src="{CHART_A}" alt="WER vs SNR chart">
<h2>Older vs younger (best model)</h2>
<img src="{CHART_B}" alt="Older vs younger chart">
<h2>WER vs distance (best model, {html.escape(cfg.report.distance_noise)})</h2>
<img src="{CHART_C}" alt="WER vs distance chart">
<p><i>In the sim, SNR is fixed at the mic, so distance only changes echo.</i></p>
{notes_html}
<h2>Worst conditions (best model, one row per noise type)</h2>
{_worst_table(results, summary, model, cfg.usable_wer)}
<h2>Config</h2>
<details><summary>config.yaml</summary><pre>{html.escape(config_text)}</pre></details>
</body></html>
"""
    (out / "report.html").write_text(page, encoding="utf-8")
    logger.info("wrote %s", out / "report.html")
    return out
