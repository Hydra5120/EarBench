"""Tests for config models and YAML loading (Phase 0 acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest

from earbench.config import (
    ConfigError,
    PrepareConfig,
    RoomSessionConfig,
    SweepConfig,
    config_hash,
    load_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_prepare_defaults_match_plan() -> None:
    cfg = PrepareConfig()
    assert cfg.clips_per_group == 50
    assert cfg.max_clips_per_speaker == 1
    assert cfg.age_buckets["younger"] == ["twenties", "thirties", "fourties"]
    assert cfg.age_buckets["older"] == ["sixties", "seventies", "eighties", "nineties"]


def test_wrong_field_type_error_names_file_and_field(tmp_path: Path) -> None:
    bad = tmp_path / "prepare.yaml"
    bad.write_text("clips_per_group: fifty\n", encoding="utf-8")
    with pytest.raises(ConfigError) as exc_info:
        load_config(bad, PrepareConfig)
    message = str(exc_info.value)
    assert str(bad) in message
    assert "clips_per_group" in message


def test_missing_file_error_names_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError) as exc_info:
        load_config(missing, PrepareConfig)
    assert str(missing) in str(exc_info.value)


def test_example_configs_load() -> None:
    assert isinstance(load_config(CONFIGS_DIR / "prepare.yaml", PrepareConfig), PrepareConfig)
    assert isinstance(load_config(CONFIGS_DIR / "quick.yaml", SweepConfig), SweepConfig)
    assert isinstance(load_config(CONFIGS_DIR / "full.yaml", SweepConfig), SweepConfig)
    assert isinstance(load_config(CONFIGS_DIR / "room.yaml", RoomSessionConfig), RoomSessionConfig)


def test_config_hash_stable() -> None:
    assert config_hash(PrepareConfig()) == config_hash(PrepareConfig())
