"""Tests for clip selection and manifest writing (Phase 1 acceptance)."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pytest import LogCaptureFixture

from earbench import audio, manifest
from earbench.config import PrepareConfig

CV_HEADER = [
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "sentence_domain",
    "up_votes",
    "down_votes",
    "age",
    "gender",
    "accents",
    "variant",
    "locale",
    "segment",
]

# (client_id, fname, sentence, up_votes, down_votes, age, gender)
FakeRow = tuple[str, str, str, int, int, str, str]

DecodeFn = Callable[[Path, Path], float]


def _write_fake_cv(
    root: Path, files: dict[str, list[FakeRow]], existing: set[str] | None = None
) -> Path:
    """Write tiny TSVs plus clip files. Files in `existing` (default: all) are created."""
    cv_dir = root / "cv"
    clips_dir = cv_dir / "clips"
    clips_dir.mkdir(parents=True)
    for fname, rows in files.items():
        with open(cv_dir / fname, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            writer.writerow(CV_HEADER)
            for client_id, clip, sentence, up, down, age, gender in rows:
                writer.writerow(
                    [
                        client_id,
                        clip,
                        "sent01",
                        sentence,
                        "",
                        up,
                        down,
                        age,
                        gender,
                        "Australian English",
                        "",
                        "en",
                        "",
                    ]
                )
                if existing is None or clip in existing:
                    (clips_dir / clip).write_bytes(b"\x00")
    return cv_dir


def _make_config(cv_dir: Path, root: Path, **overrides: Any) -> PrepareConfig:
    return PrepareConfig(
        cv_dir=cv_dir,
        tsv_files=["train.tsv"],
        out_dir=root / "clips",
        manifest_path=root / "manifest.csv",
        **overrides,
    )


def _decoder(duration_s: float | dict[str, float], written: list[str] | None = None) -> DecodeFn:
    """Fake clip decoder: writes 1 s of silence, returns a canned duration."""

    def _decode(src: Path, dst: Path) -> float:
        audio.save_wav_16k_mono(dst, np.zeros(audio.SAMPLE_RATE_HZ, dtype=np.float32))
        if written is not None:
            written.append(src.name)
        if isinstance(duration_s, dict):
            return duration_s[src.name]
        return duration_s

    return _decode


def test_fourty_spelling_younger_includes_fourties(tmp_path: Path) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("spk1", "a.mp3", "hello world one", 3, 0, "fourties", "male_masculine"),
                ("spk2", "b.mp3", "hello world two", 3, 0, "twenties", "male_masculine"),
            ]
        },
    )
    cfg = _make_config(cv_dir, tmp_path, clips_per_group=2, match_gender=False, min_duration_s=0.5)
    summary = manifest.run_prepare(cfg, decode_clip=_decoder(5.0))
    assert summary.selected_per_group["younger"] == 2
    rows = manifest.read_manifest(cfg.manifest_path)
    assert any(row.age_bucket == "fourties" for row in rows)


def test_vote_filter_drops_low_up_and_downvoted(tmp_path: Path) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("spk1", "a.mp3", "good clip here", 3, 0, "sixties", "male_masculine"),
                ("spk2", "b.mp3", "low votes clip", 1, 0, "sixties", "male_masculine"),
                ("spk3", "c.mp3", "downvoted clip", 5, 2, "sixties", "male_masculine"),
            ]
        },
    )
    cfg = _make_config(cv_dir, tmp_path, match_gender=False, min_duration_s=0.5)
    candidates, _ = manifest.load_candidates(cv_dir, cfg)
    assert [clip.mp3_path.name for clip in candidates] == ["a.mp3"]


def test_missing_mp3_skipped_with_warning_and_counted(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("spk1", "a.mp3", "present clip here", 3, 0, "sixties", "male_masculine"),
                ("spk2", "gone.mp3", "missing clip here", 3, 0, "sixties", "male_masculine"),
            ]
        },
        existing={"a.mp3"},
    )
    cfg = _make_config(cv_dir, tmp_path, match_gender=False, min_duration_s=0.5)
    with caplog.at_level(logging.WARNING, logger="earbench.manifest"):
        candidates, missing_mp3 = manifest.load_candidates(cv_dir, cfg)
    assert missing_mp3 == 1
    assert [clip.mp3_path.name for clip in candidates] == ["a.mp3"]
    assert "gone.mp3" in caplog.text


def test_same_seed_gives_same_selection(tmp_path: Path) -> None:
    rows = [
        (f"spk{i}", f"c{i}.mp3", f"sentence number {i}", 3, 0, "sixties", "male_masculine")
        for i in range(10)
    ]
    cv_dir = _write_fake_cv(tmp_path, {"train.tsv": rows})
    first = manifest.run_prepare(
        _make_config(cv_dir, tmp_path / "r1", clips_per_group=4, match_gender=False),
        decode_clip=_decoder(5.0),
    )
    second = manifest.run_prepare(
        _make_config(cv_dir, tmp_path / "r2", clips_per_group=4, match_gender=False),
        decode_clip=_decoder(5.0),
    )
    assert first.selected_ids == second.selected_ids


def test_max_clips_per_speaker_cap_and_shortfall_reported(tmp_path: Path) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("only", "a.mp3", "first sentence here", 3, 0, "sixties", "male_masculine"),
                ("only", "b.mp3", "second sentence here", 3, 0, "sixties", "male_masculine"),
                ("only", "c.mp3", "third sentence here", 3, 0, "sixties", "male_masculine"),
            ]
        },
    )
    cfg = _make_config(
        cv_dir,
        tmp_path,
        clips_per_group=2,
        max_clips_per_speaker=1,
        match_gender=False,
        min_duration_s=0.5,
    )
    summary = manifest.run_prepare(cfg, decode_clip=_decoder(5.0))
    assert summary.selected_per_group["older"] == 1
    assert summary.speakers_per_group["older"] == 1
    assert summary.shortfalls != []


def test_duration_out_of_range_replaced_by_same_speaker(tmp_path: Path) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("spkA", "long.mp3", "very long clip here", 3, 0, "sixties", "male_masculine"),
                ("spkA", "fine.mp3", "fine length clip here", 3, 0, "sixties", "male_masculine"),
            ]
        },
    )
    cfg = _make_config(
        cv_dir,
        tmp_path,
        clips_per_group=2,
        max_clips_per_speaker=2,
        match_gender=False,
        min_duration_s=3.0,
        max_duration_s=10.0,
    )
    summary = manifest.run_prepare(cfg, decode_clip=_decoder({"long.mp3": 30.0, "fine.mp3": 5.0}))
    assert summary.selected_ids == ["fine"]
    assert summary.duration_rejected == 1
    assert summary.shortfalls != []
    assert not (cfg.out_dir / "long.wav").exists()
    assert (cfg.out_dir / "fine.wav").exists()


def test_match_gender_mirrors_counts_and_excludes_unknown(tmp_path: Path) -> None:
    rows: list[FakeRow] = []
    for i in range(2):
        rows.append(
            (f"of{i}", f"of{i}.mp3", f"older female {i}", 3, 0, "sixties", "female_feminine")
        )
    for i in range(3):
        rows.append((f"om{i}", f"om{i}.mp3", f"older male {i}", 3, 0, "sixties", "male_masculine"))
    rows.append(("ox", "ox.mp3", "older unknown gender", 3, 0, "sixties", ""))
    for i in range(3):
        rows.append(
            (f"yf{i}", f"yf{i}.mp3", f"younger female {i}", 3, 0, "twenties", "female_feminine")
        )
    for i in range(4):
        rows.append(
            (f"ym{i}", f"ym{i}.mp3", f"younger male {i}", 3, 0, "twenties", "male_masculine")
        )
    rows.append(("yx", "yx.mp3", "younger unknown gender", 3, 0, "twenties", ""))
    cv_dir = _write_fake_cv(tmp_path, {"train.tsv": rows})
    cfg = _make_config(cv_dir, tmp_path, clips_per_group=3, min_duration_s=0.5)
    summary = manifest.run_prepare(cfg, decode_clip=_decoder(5.0))
    assert summary.female_per_group == {"older": 2, "younger": 2}
    assert summary.male_per_group == {"older": 1, "younger": 1}
    genders = {row.gender for row in manifest.read_manifest(cfg.manifest_path)}
    assert genders <= {"female", "male"}


def test_all_unknown_genders_selected_only_when_not_matching(tmp_path: Path) -> None:
    rows: list[FakeRow] = [
        (f"spk{i}", f"c{i}.mp3", f"sentence number {i}", 3, 0, "sixties", "") for i in range(2)
    ]
    cv_dir = _write_fake_cv(tmp_path, {"train.tsv": rows})
    matched = manifest.run_prepare(
        _make_config(cv_dir, tmp_path / "m", clips_per_group=2, min_duration_s=0.5),
        decode_clip=_decoder(5.0),
    )
    assert matched.selected_per_group["older"] == 0
    assert matched.shortfalls != []
    unmatched = manifest.run_prepare(
        _make_config(
            cv_dir,
            tmp_path / "u",
            clips_per_group=2,
            match_gender=False,
            min_duration_s=0.5,
        ),
        decode_clip=_decoder(5.0),
    )
    assert unmatched.selected_per_group["older"] == 2


def test_manifest_csv_exact_columns_and_hashes(tmp_path: Path) -> None:
    cv_dir = _write_fake_cv(
        tmp_path,
        {
            "train.tsv": [
                ("abc123", "a.mp3", "hello world today", 3, 0, "seventies", "female_feminine"),
            ]
        },
    )
    cfg = _make_config(cv_dir, tmp_path, clips_per_group=1, min_duration_s=0.5)
    manifest.run_prepare(cfg, decode_clip=_decoder(5.0))
    with open(cfg.manifest_path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "clip_id",
            "wav_path",
            "sentence",
            "age_bucket",
            "age_group",
            "gender",
            "duration_s",
            "speaker_hash",
        ]
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["clip_id"] == "a"
    assert row["age_bucket"] == "seventies"
    assert row["age_group"] == "older"
    assert row["gender"] == "female"
    assert float(row["duration_s"]) == 5.0
    assert row["speaker_hash"] == hashlib.sha256(b"abc123").hexdigest()[:16]
    assert (tmp_path / row["wav_path"]).exists()


def test_prepare_cli_writes_manifest_with_real_mp3(tmp_path: Path) -> None:
    """End-to-end `earbench prepare` on one real MP3 written by soundfile."""
    import soundfile as sf
    from typer.testing import CliRunner

    from earbench.cli import app

    cv_dir = _write_fake_cv(
        tmp_path,
        {"train.tsv": [("spk1", "a.mp3", "hello world today", 3, 0, "sixties", "male_masculine")]},
    )
    t = np.arange(5 * audio.SAMPLE_RATE_HZ, dtype=np.float64) / audio.SAMPLE_RATE_HZ
    tone = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(cv_dir / "clips" / "a.mp3"), tone, audio.SAMPLE_RATE_HZ, format="MP3")
    cfg_path = tmp_path / "prepare.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                f"cv_dir: {cv_dir}",
                "tsv_files: [train.tsv]",
                f"out_dir: {tmp_path / 'clips'}",
                f"manifest_path: {tmp_path / 'manifest.csv'}",
                "clips_per_group: 1",
                "match_gender: false",
                "min_duration_s: 3.0",
                "max_duration_s: 10.0",
                "seed: 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["prepare", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "manifest.csv").is_file()
    assert (tmp_path / "clips" / "a.wav").is_file()
