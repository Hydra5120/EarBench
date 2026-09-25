"""Before/after fix comparison (Phase 6).

Compares two result sets that share the same grid — e.g. a sweep run with
`vad_filter: false` (before) against the same grid with `vad_filter: true`
(after). Sim pairs work now; room-session pairs drop into the same command
later via `--room-before/--room-after`. Cell keys include the source, so sim
rows only pair with sim rows and room rows with room rows — sim output can
never be mistaken for real-room output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel

from earbench import report as report_mod
from earbench import score as scoring
from earbench.config import AgeGroup, NoiseType, ResultRow, write_csv

logger = logging.getLogger(__name__)

FIX_CSV = "fix-compare.csv"
FIX_CHART = "fix-before-after.png"


class FixCell(BaseModel):
    """One pooled condition: before vs after WER, intervals and usable rates."""

    source: str
    model: str
    age_group: AgeGroup
    distance_m: float
    noise_type: NoiseType
    snr_db: float | None = None
    n_before: int
    before_wer: float
    before_low: float
    before_high: float
    before_usable: float
    n_after: int
    after_wer: float
    after_low: float
    after_high: float
    after_usable: float
    delta_wer: float  # after minus before; negative means the fix helped


def _key(row: ResultRow) -> tuple:
    return (
        row.source,
        row.model,
        row.age_group,
        row.distance_m,
        row.noise_type,
        row.snr_db,
    )


def compare(
    before: list[ResultRow],
    after: list[ResultRow],
    *,
    iters: int = 1000,
    seed: int = 0,
    usable_wer: float = 0.2,
) -> list[FixCell]:
    """Pool matching conditions from both sides.

    Cell keys include the source, so sim rows only ever pair with sim rows
    and room rows with room rows. Unmatched cells are skipped; when nothing
    overlaps at all, an error names both sides sources to catch mix-ups.
    """
    by_key: dict[tuple, dict[str, list[ResultRow]]] = {}
    for row in before:
        by_key.setdefault(_key(row), {}).setdefault("before", []).append(row)
    for row in after:
        by_key.setdefault(_key(row), {}).setdefault("after", []).append(row)
    cells: list[FixCell] = []
    skipped = 0
    for key, sides in sorted(by_key.items()):
        if "before" not in sides or "after" not in sides:
            skipped += 1
            continue
        source, model, age_group, distance_m, noise_type, snr_db = key
        pooled_before = report_mod.pool(sides["before"], iters, seed, usable_wer)
        pooled_after = report_mod.pool(sides["after"], iters, seed, usable_wer)
        cells.append(
            FixCell(
                source=source,
                model=model,
                age_group=age_group,
                distance_m=distance_m,
                noise_type=noise_type,
                snr_db=snr_db,
                n_before=pooled_before.n_clips,
                before_wer=pooled_before.wer,
                before_low=pooled_before.ci_low,
                before_high=pooled_before.ci_high,
                before_usable=pooled_before.usable_rate,
                n_after=pooled_after.n_clips,
                after_wer=pooled_after.wer,
                after_low=pooled_after.ci_low,
                after_high=pooled_after.ci_high,
                after_usable=pooled_after.usable_rate,
                delta_wer=pooled_after.wer - pooled_before.wer,
            )
        )
    if skipped:
        logger.warning("skipped %d condition(s) present on only one side", skipped)
    if not cells:
        raise ValueError(
            "no overlapping (source, model, age group, distance, noise, SNR) cells: "
            f"before sources {sorted({row.source for row in before})}, "
            f"after sources {sorted({row.source for row in after})}"
        )
    return cells


def read_any_results(path: str | Path) -> list[ResultRow]:
    """Read results.csv from a run/session dir or from the CSV file itself."""
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "results.csv"
    if not candidate.is_file():
        raise ValueError(f"run results not found: {candidate} (no such sweep run)")
    rows = report_mod.read_results(candidate)
    if not rows:
        raise ValueError(f"run results are empty: {candidate}")
    return rows


def _label(cell: FixCell) -> str:
    snr = "clean" if cell.snr_db is None else f"{cell.snr_db:g} dB"
    return (
        f"{cell.source} {cell.model} {cell.age_group} {cell.noise_type} {snr} {cell.distance_m:g} m"
    )


def fig_before_after(
    cells: list[FixCell], before_tag: str = "before", after_tag: str = "after"
) -> Figure:
    """Grouped bars with bootstrap whiskers, one pair per condition."""
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cells)), 4))
    xs = list(range(len(cells)))
    width = 0.35
    before_xs = [x - width / 2 for x in xs]
    after_xs = [x + width / 2 for x in xs]
    before_yerr = [
        [cell.before_wer - cell.before_low, cell.before_high - cell.before_wer] for cell in cells
    ]
    after_yerr = [
        [cell.after_wer - cell.after_low, cell.after_high - cell.after_wer] for cell in cells
    ]
    ax.bar(
        before_xs,
        [100 * cell.before_wer for cell in cells],
        width,
        yerr=list(zip(*before_yerr, strict=True)) if cells else None,
        capsize=3,
        label=before_tag,
    )
    ax.bar(
        after_xs,
        [100 * cell.after_wer for cell in cells],
        width,
        yerr=list(zip(*after_yerr, strict=True)) if cells else None,
        capsize=3,
        label=after_tag,
    )
    ax.set_xticks(xs, [_label(cell) for cell in cells], rotation=30, ha="right")
    ax.set_xlabel("Condition")
    ax.set_ylabel("WER (%)")
    ax.set_title(f"Before vs after ({before_tag} → {after_tag})")
    ax.legend()
    fig.tight_layout()
    return fig


def format_table(cells: list[FixCell], before_tag: str, after_tag: str) -> str:
    """One line per condition: before/after WER with intervals, usable rates, delta."""
    lines = [
        f"{'Condition':<32} {before_tag + ' WER [95% CI]':<28} "
        f"{after_tag + ' WER [95% CI]':<28} {'Delta':>8}"
    ]
    for cell in cells:
        lines.append(
            f"{_label(cell):<32} "
            f"{100 * cell.before_wer:5.1f}% "
            f"[{100 * cell.before_low:5.1f}-{100 * cell.before_high:5.1f}] "
            f"{100 * cell.after_wer:5.1f}% "
            f"[{100 * cell.after_low:5.1f}-{100 * cell.after_high:5.1f}] "
            f"{100 * cell.delta_wer:+7.1f}pp"
        )
    return "\n".join(lines)


def format_html_table(cells: list[FixCell], before_tag: str, after_tag: str) -> str:
    """Before/after table for the HTML report, with intervals and deltas."""
    import html as html_mod

    lines = [
        "<table><tr><th>Condition</th>"
        f"<th>{html_mod.escape(before_tag)} WER [95% CI]</th>"
        f"<th>{html_mod.escape(after_tag)} WER [95% CI]</th>"
        "<th>Delta (pp)</th><th>Usable before → after</th></tr>"
    ]
    for cell in cells:
        lines.append(
            f"<tr><td>{html_mod.escape(_label(cell))}</td>"
            f"<td>{100 * cell.before_wer:.1f}% "
            f"[{100 * cell.before_low:.1f}–{100 * cell.before_high:.1f}]</td>"
            f"<td>{100 * cell.after_wer:.1f}% "
            f"[{100 * cell.after_low:.1f}–{100 * cell.after_high:.1f}]</td>"
            f"<td>{100 * cell.delta_wer:+.1f}</td>"
            f"<td>{100 * cell.before_usable:.0f}% → {100 * cell.after_usable:.0f}%</td></tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def pooled_wer(rows: list[ResultRow]) -> float:
    """Corpus WER over rows (re-exported for the CLI summary line)."""
    return scoring.corpus_wer(rows)


def write_compare(cells: list[FixCell], out_dir: str | Path) -> Path:
    """Write fix-compare.csv; returns the output directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(FixCell, cells, out / FIX_CSV)
    return out
