"""CLI stub tests: `earbench --help` must succeed (Phase 0 acceptance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from earbench.audio import check_wav_16k_mono, save_wav_16k_mono
from earbench.cli import app
from earbench.manifest import ManifestRow, write_manifest

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
