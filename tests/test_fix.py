"""Tests for the Phase 6 before/after fix comparison. All offline, no models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from earbench import fix, record, report, sweep
from earbench.audio import SAMPLE_RATE_HZ
from earbench.cli import app
from earbench.config import ResultRow, RoomSessionConfig, SweepConfig, write_csv
from earbench.manifest import ManifestRow, write_manifest
from earbench.transcribe import FakeTranscriber

runner = CliRunner()


def _row(clip_id: str, wer: float, source: str = "sim", **over: object) -> ResultRow:
    ref_words = 10
    errors = round(wer * ref_words)
    base: dict[str, object] = {
        "run_id": "r1",
        "source": source,
        "clip_id": clip_id,
        "age_group": "older",
        "distance_m": 2.0,
        "noise_type": "tv",
        "snr_db": 0.0,
        "model": "tiny",
        "model_version": "fake+1",
        "settings": "fake,vad=off",
        "reference": "hello world",
        "hypothesis": "hello world",
        "ref_words": ref_words,
        "errors": errors,
        "wer": errors / ref_words,
        "config_hash": "abc",
    }
    base.update(over)
    return ResultRow.model_validate(base)


def test_compare_pools_matched_cells_with_intervals() -> None:
    before = [_row("a", 1.0), _row("b", 0.0)]  # pooled WER 0.5
    after = [_row("a", 0.0), _row("b", 0.0)]  # pooled WER 0.0
    cells = fix.compare(before, after, iters=50, seed=0, usable_wer=0.2)
    assert len(cells) == 1
    cell = cells[0]
    assert cell.before_wer == pytest.approx(0.5)
    assert cell.after_wer == pytest.approx(0.0)
    assert cell.delta_wer == pytest.approx(-0.5)
    assert cell.before_low <= cell.before_wer <= cell.before_high
    assert cell.after_low <= cell.after_wer <= cell.after_high
    assert cell.n_before == 2 and cell.n_after == 2


def test_compare_never_mixes_sim_and_room() -> None:
    with pytest.raises(ValueError, match="source"):
        fix.compare([_row("a", 0.5, source="sim")], [_row("a", 0.5, source="room")])


def test_compare_skips_unmatched_cells() -> None:
    before = [_row("a", 0.5, snr_db=0.0)]
    after = [_row("a", 0.5, snr_db=0.0), _row("b", 0.5, snr_db=10.0)]
    cells = fix.compare(before, after, iters=10, seed=0, usable_wer=0.2)
    assert [(c.snr_db) for c in cells] == [0.0]


def test_chart_labels_axes_with_units() -> None:
    cells = fix.compare([_row("a", 0.5)], [_row("a", 0.1)], iters=10, seed=0, usable_wer=0.2)
    fig = fix.fig_before_after(cells)
    try:
        labels = [ax.get_xlabel() + ax.get_ylabel() for ax in fig.axes]
        assert any("WER (%)" in text for text in labels)
    finally:
        report.close_figure(fig)


def _write_run(tmp_path: Path, name: str, rows: list[ResultRow], cfg: SweepConfig) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    write_csv(ResultRow, rows, run_dir / "results.csv")
    sweep.write_config(cfg, run_dir / "config.yaml")
    return run_dir


def test_compare_fix_command_writes_csv_png_and_table(tmp_path: Path, sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    before = _write_run(tmp_path, "before", [_row("a", 0.5)], cfg)
    after = _write_run(tmp_path, "after", [_row("a", 0.1)], cfg)
    out = tmp_path / "fix-out"
    result = runner.invoke(
        app,
        [
            "compare-fix",
            "--sim-before",
            str(before),
            "--sim-after",
            str(after),
            "--out",
            str(out),
            "--iters",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "fix-compare.csv").is_file()
    assert (out / "fix-before-after.png").is_file()
    assert "room: pending" in result.output  # no fake room numbers, just a pointer
    assert "-0.4" in result.output or "-40.0" in result.output  # the delta shows up


def test_compare_fix_with_room_pair_covers_both_sources(tmp_path: Path, sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    sim_before = _write_run(tmp_path, "sim-before", [_row("a", 0.5)], cfg)
    sim_after = _write_run(tmp_path, "sim-after", [_row("a", 0.1)], cfg)
    room_before = _write_run(
        tmp_path, "room-before", [_row("a", 0.5, source="room", run_id="s")], cfg
    )
    room_after = _write_run(
        tmp_path, "room-after", [_row("a", 0.1, source="room", run_id="s")], cfg
    )
    out = tmp_path / "fix-out"
    result = runner.invoke(
        app,
        [
            "compare-fix",
            "--sim-before",
            str(sim_before),
            "--sim-after",
            str(sim_after),
            "--room-before",
            str(room_before),
            "--room-after",
            str(room_after),
            "--out",
            str(out),
            "--iters",
            "10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "room: pending" not in result.output
    content = (out / "fix-compare.csv").read_text(encoding="utf-8")
    assert "sim" in content and "room" in content


def test_compare_fix_rejects_missing_results(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(
        app, ["compare-fix", "--sim-before", str(empty), "--sim-after", str(empty)]
    )
    assert result.exit_code == 1
    assert "results.csv" in result.output


def test_report_before_section_shows_fix_table(tmp_path: Path, sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1, distances_m=(2.0,), noise_types=("tv",))
    before = _write_run(tmp_path, "before", [_row("a", 0.5)], cfg)
    after = _write_run(tmp_path, "after", [_row("a", 0.1)], cfg)
    out = report.write_report(after, tmp_path / "rep", before_dir=before)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Fix comparison" in html
    assert (out / "fix-before-after.png").is_file()


def test_sweep_config_defaults_vad_off() -> None:
    assert SweepConfig(manifest_path=Path("m.csv")).vad_filter is False


def test_example_fix_config_loads() -> None:
    base = Path(__file__).resolve().parents[1] / "configs"
    before = SweepConfig.model_validate(
        yaml.safe_load((base / "fix-baseline.yaml").read_text(encoding="utf-8"))
    )
    after = SweepConfig.model_validate(
        yaml.safe_load((base / "fix-vad.yaml").read_text(encoding="utf-8"))
    )
    assert before.vad_filter is False
    assert after.vad_filter is True
    assert after.noise_files.keys() == {"tv"}
    # Same grid, so the before/after delta is the VAD fix alone.
    assert before.model_dump(exclude={"vad_filter"}) == after.model_dump(exclude={"vad_filter"})


def test_sweep_factory_passes_vad_filter(monkeypatch, sweep_env) -> None:
    cfg, _ = sweep_env(n_per_group=1)
    cfg = cfg.model_copy(update={"vad_filter": True})
    seen: list[tuple[str, bool]] = []

    def fake_whisper(model_size: str, *, device: str, compute_type: str, vad_filter: bool = False):
        seen.append((model_size, vad_filter))
        return FakeTranscriber(settings=f"vad={'on' if vad_filter else 'off'}")

    monkeypatch.setattr(sweep, "FasterWhisperTranscriber", fake_whisper)
    transcriber = sweep.default_transcriber_factory(cfg)("tiny")
    assert seen == [("tiny", True)]
    assert "vad=on" in transcriber.settings


def _quiet_session(tmp_path: Path, monkeypatch, session_id: str) -> RoomSessionConfig:
    """A one-block quiet session recorded from a synthetic buffer (offline)."""
    import numpy as np

    from earbench import audio

    rows = []
    for group, bucket, freq in (("older", "sixties", 300.0), ("younger", "twenties", 700.0)):
        wav = tmp_path / "clips" / f"{group}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(0.6 * SAMPLE_RATE_HZ)) / SAMPLE_RATE_HZ
        audio.save_wav_16k_mono(wav, 0.5 * np.sin(2.0 * np.pi * freq * t))
        rows.append(
            ManifestRow(
                clip_id=f"{group}_0",
                wav_path=wav.as_posix(),
                sentence=f"the {group} speaker says hello",
                age_bucket=bucket,
                age_group=group,  # type: ignore[arg-type]
                gender="female",
                duration_s=0.6,
                speaker_hash=f"{group}0",
            )
        )
    write_manifest(rows, tmp_path / "manifest.csv")
    cfg = RoomSessionConfig.model_validate(
        {
            "manifest_path": tmp_path / "manifest.csv",
            "clips_per_group": 1,
            "distances_m": [1.0],
            "conditions": ["quiet"],
            "lead_in_s": 0.5,
            "sample_rate": SAMPLE_RATE_HZ,
            "model": "tiny",
            "seed": 0,
            "recordings_dir": tmp_path / "recordings",
            "playlists_dir": tmp_path / "playlists",
        }
    )
    record.make_playlist(cfg)
    play_16k, _ = record._load_block_playlist(cfg, 1.0, "quiet")
    offset = int(round(0.37 * SAMPLE_RATE_HZ))
    total = play_16k.shape[0] + offset + int(round(record.RECORD_SLACK_S * SAMPLE_RATE_HZ))
    rng = np.random.default_rng(0)
    rec = (0.05 * rng.standard_normal(total)).astype(np.float32)
    rec[offset : offset + play_16k.shape[0]] += play_16k

    class FakeBackend:
        input_name = "fake-mic"

        @property
        def native_sample_rate(self) -> int:
            return SAMPLE_RATE_HZ

        def record_block(self, n_samples: int) -> np.ndarray:
            out = np.zeros(n_samples, dtype=np.float32)
            out[: min(n_samples, rec.shape[0])] = rec[: min(n_samples, rec.shape[0])]
            return out

    monkeypatch.setattr(record, "make_backend", lambda _cfg: FakeBackend())
    monkeypatch.setattr(record, "default_transcriber_factory", lambda _model: FakeTranscriber())
    cfg_path = tmp_path / "room.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    session_input = "5\n4\n2.7\n1.0\n0.5\n1.2\n4.0\n3.0\n1.0\nMacBook speakers\nPC webcam mic\n"
    session_input += "1.0\n60\n\nsofa one metre\n\n"
    result = runner.invoke(
        app, ["record", "--config", str(cfg_path), "--session-id", session_id], input=session_input
    )
    assert result.exit_code == 0, result.output
    return cfg


def test_score_room_vad_filter_flag(tmp_path: Path, monkeypatch) -> None:
    cfg = _quiet_session(tmp_path, monkeypatch, "vad")
    seen: dict[str, object] = {}

    def spy(model: str, vad_filter: bool = False) -> FakeTranscriber:
        seen["vad"] = vad_filter
        return FakeTranscriber(settings=f"vad={'on' if vad_filter else 'off'}")

    monkeypatch.setattr(record, "default_transcriber_factory", spy)
    session = str(Path(cfg.recordings_dir) / "vad")
    assert runner.invoke(app, ["score-room", session]).exit_code == 0
    assert seen["vad"] is False
    result = runner.invoke(app, ["score-room", session, "--vad-filter"])
    assert result.exit_code == 0, result.output
    assert seen["vad"] is True
    rows = record.read_results(Path(session) / "results.csv")
    assert all(row.settings == "vad=on" for row in rows)
