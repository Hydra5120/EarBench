"""CLI stub tests: `earbench --help` must succeed (Phase 0 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from earbench.audio import check_wav_16k_mono, save_wav_16k_mono
from earbench.cli import app
from earbench.manifest import ManifestRow, write_manifest
from earbench.transcribe import FakeTranscriber

runner = CliRunner()


def test_help_succeeds() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "earbench" in result.output.lower()


def test_listen_writes_wav(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    clip_path = save_wav_16k_mono(tmp_path / "c1.wav", 0.3 * np.sin(np.arange(16_000) / 5.0))
    noise_path = save_wav_16k_mono(tmp_path / "n.wav", 0.1 * rng.standard_normal(5 * 16_000))
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(
        [
            ManifestRow(
                clip_id="c1",
                wav_path=str(clip_path),
                sentence="hello",
                age_bucket="sixties",
                age_group="older",
                gender="female",
                duration_s=1.0,
                speaker_hash="abc",
            )
        ],
        manifest_path,
    )
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        f"manifest_path: {manifest_path.as_posix()}\n"
        f"noise_files:\n  living: [{noise_path.as_posix()}]\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.wav"
    args = ["listen", "--config", str(cfg), "--clip", "c1", "--distance", "2"]
    result = runner.invoke(app, [*args, "--noise", "living", "--snr", "5", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert check_wav_16k_mono(out) == []

    # Asking for a noise type the config doesn't list fails clearly.
    result = runner.invoke(app, [*args, "--noise", "tv", "--snr", "5", "--out", str(out)])
    assert result.exit_code == 1
    assert "tv" in result.output

    # A listed noise file that doesn't exist fails clearly, not with a traceback.
    noise_path.unlink()
    result = runner.invoke(app, [*args, "--noise", "living", "--snr", "5", "--out", str(out)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_sweep_command_runs_offline(monkeypatch, tmp_path: Path, sweep_env) -> None:
    cfg, _, _ = sweep_env(n_per_group=1)
    cfg_path = tmp_path / "sweep.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")

    def fake_factory(_cfg) -> object:
        return lambda model: FakeTranscriber(default_text="a canned answer")

    monkeypatch.setattr("earbench.sweep.default_transcriber_factory", fake_factory)
    args = ["sweep", "--config", str(cfg_path), "--no-progress", "--run-id", "cli-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (Path(cfg.runs_dir) / "cli-run" / "results.csv").is_file()
