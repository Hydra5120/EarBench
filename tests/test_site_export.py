"""Demo site export (Phase 8): alignment, hero numbers, missing cells, no TV audio."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from earbench import site_export, sweep
from earbench.cli import app
from earbench.config import SiteConfig, SweepConfig
from earbench.report import read_summary
from earbench.score import align, align_normalised, score_clip
from earbench.transcribe import CachedTranscriber, FakeTranscriber

ROOM_CSV = (
    "distance_m,condition,noise_type,n_clips,mean_snr_db,phone_snr_db,room_wer,room_low,"
    "room_high,room_usable,sim_wer,sim_low,sim_high,sim_usable\n"
    "2.0,quiet,none,30,,,0.15,0.1,0.2,0.6,0.2,0.1,0.3,0.6\n"
    "2.0,tv,tv,30,12.3,4.0,1.3,0.5,2.5,0.1,1.2,0.8,1.6,0.2\n"
)


def _factory(cfg: SweepConfig, vad: str):
    texts = {"tiny": "the older speaker says", "base": "the older speaker says number zero"}

    def make(model: str) -> CachedTranscriber:
        fake = FakeTranscriber(default_text=texts[model], settings=f"fake,vad={vad}")
        return CachedTranscriber(fake, cfg.cache_dir)

    return make


@pytest.fixture
def site_env(sweep_env, tmp_path: Path):
    """A full run (living + tv, vad off) and a VAD run (living only, vad on) at 2 m."""
    cfg, _rows = sweep_env(
        n_per_group=2,
        noise_types=("living", "tv"),
        snr_db=(10.0,),
        distances_m=(2.0,),
        models=("tiny", "base"),
    )
    sweep.run_sweep(cfg, _factory(cfg, "off"), progress=False, run_id="full")
    vad_cfg = cfg.model_copy(
        update={
            "noise_files": {"living": cfg.noise_files["living"]},
            "vad_filter": True,
            "clip_ids": ["older_0", "younger_0"],
        }
    )
    sweep.run_sweep(vad_cfg, _factory(vad_cfg, "on"), progress=False, run_id="vad")
    room_csv = tmp_path / "recordings" / "S1" / "compare-summary.csv"
    room_csv.parent.mkdir(parents=True)
    room_csv.write_text(ROOM_CSV, encoding="utf-8")
    site = SiteConfig.model_validate(
        {
            "full_run": cfg.runs_dir / "full",
            "vad_run": cfg.runs_dir / "vad",
            "room_compare": room_csv,
            "room_label": "room layout approximated",
            "models": ["tiny", "base"],
            "audio_noise_types": ["living"],
            "text_only_noise_types": ["tv"],
            "levels": [{"snr_db": None, "label": "No noise"}, {"snr_db": 10.0, "label": "10"}],
            "noise_labels": {"living": "Living room", "tv": "TV"},
            "clips": [
                {"id": "older_0", "featured_noise": "living", "featured_snr_db": 10.0},
                {"id": "younger_0", "featured_noise": "tv", "featured_snr_db": 10.0},
            ],
            "hero": {"model": "base", "noise_type": "tv", "snr_db": 10.0},
            "charts": [
                {
                    "id": "a",
                    "title": "A",
                    "series": "age_group",
                    "model": "tiny",
                    "noise_type": "living",
                },
                {
                    "id": "m",
                    "title": "M",
                    "series": "model",
                    "age_group": "older",
                    "noise_type": "tv",
                },
                {
                    "id": "n",
                    "title": "N",
                    "series": "noise_type",
                    "model": "base",
                    "age_group": "younger",
                },
            ],
            "transcriber_name": "fake",
            "out_dir": tmp_path / "site" / "data",
        }
    )
    return site, cfg


def test_align_marks_sub_del_ins_and_matches_wer() -> None:
    tokens = align("The cat sat on the mat.", "the bat sat on mat mat now")
    ops = [(t.op, t.ref, t.hyp) for t in tokens]
    assert ("sub", "cat", "bat") in ops
    assert [t.op for t in tokens].count("ins") + [t.op for t in tokens].count("del") >= 1
    for ref, hyp in [
        ("The cat sat on the mat.", "the bat sat on mat mat now"),
        ("hello world", ""),
        ("a b c", "x a b c d"),
        ("I'm very honoured to play it.", "I am very honored to play it"),
    ]:
        errors = sum(1 for t in align(ref, hyp) if t.op != "ok")
        assert errors == score_clip(ref, hyp).errors
    assert [(t.op, t.ref, t.hyp) for t in align("a b c", "x a b c d")] == [
        ("ins", None, "x"),
        ("ok", "a", "a"),
        ("ok", "b", "b"),
        ("ok", "c", "c"),
        ("ins", None, "d"),
    ]
    assert [t.op for t in align("hello world", "")] == ["del", "del"]


def test_align_normalised_keeps_stored_tokens() -> None:
    # Real full-run row (stored, already normalised): the scorer counted the stray "."
    # as a word (24 errors). Normalising again drops it (23), so align as stored.
    ref = "through most of the following centuries the cathedral stood only half finished"
    hyp = (
        "to the most of my father in the century is the place that stood out in the heart "
        "of the nation hi babe hi do i have to ."
    )
    tokens = align_normalised(ref, hyp)
    assert sum(1 for t in tokens if t.op != "ok") == 24
    assert sum(1 for t in align(ref, hyp) if t.op != "ok") == 23


def test_export_writes_data_and_no_tv_audio(site_env) -> None:
    site, _cfg = site_env
    result = site_export.export_site(site)
    out = site.out_dir
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert [clip["id"] for clip in index["clips"]] == ["older_0", "younger_0"]
    assert index["site_manifest"]["runs"][0]["run_id"] == "full"

    clip = json.loads((out / index["clips"][0]["data"]).read_text(encoding="utf-8"))
    assert set(clip["cells"]) == {"clean", "living_10", "tv_10"}
    assert set(clip["cells"]["living_10"]["results"]) == {
        "tiny|off",
        "tiny|on",
        "base|off",
        "base|on",
    }
    assert set(clip["cells"]["tv_10"]["results"]) == {"tiny|off", "base|off"}
    tv = clip["cells"]["tv_10"]
    assert tv["audio"] is None and "copyrighted" in tv["note"]
    for cell in ("clean", "living_10"):
        head = (out / clip["cells"][cell]["audio"]).read_bytes()[:2]
        assert head[0] == 0xFF and head[1] & 0xE0 == 0xE0  # MPEG audio frame sync

    mp3s = sorted(p.name for p in (out / "audio").rglob("*.mp3"))
    assert len(mp3s) == result.audio_files == 2 * 2
    assert not [name for name in mp3s if name.startswith("tv")]
    assert not [p for p in out.rglob("*") if p.is_file() and "tv" in p.name and p.suffix != ".json"]
    # Regenerated audio is byte-for-byte what the sweep transcribed (cache keys hit).
    assert result.cache_checked == result.cache_matched > 0


def test_hero_and_charts_equal_summary_csv(site_env) -> None:
    site, cfg = site_env
    site_export.export_site(site)
    charts = json.loads((site.out_dir / "charts.json").read_text(encoding="utf-8"))
    summary = {
        (r.model, r.age_group, r.noise_type, r.snr_db): r
        for r in read_summary(cfg.runs_dir / "full" / "summary.csv")
    }
    for group in ("older", "younger"):
        row = summary[("base", group, "tv", 10.0)]
        hero = charts["hero"][group]
        assert (hero["wer"], hero["ci_low"], hero["ci_high"]) == (row.wer, row.ci_low, row.ci_high)
        assert hero["usable_rate"] == row.usable_rate
        low, high = site_export.wilson_interval(row.usable_rate, row.n_clips)
        assert (hero["usable_low"], hero["usable_high"]) == (low, high)
        assert hero["usable_low"] <= hero["usable_rate"] <= hero["usable_high"]
        quiet = summary[("base", group, "none", None)]
        assert charts["hero"][f"{group}_quiet"]["usable_rate"] == quiet.usable_rate
    by_model = {s["key"]: s["points"] for s in charts["findings"][1]["series"]}
    assert by_model["tiny"][0]["wer"] == summary[("tiny", "older", "none", None)].wer
    assert by_model["tiny"][1]["wer"] == summary[("tiny", "older", "tv", 10.0)].wer
    assert charts["vad"]["n_clips"] == 2


def test_wilson_interval_known_values() -> None:
    low, high = site_export.wilson_interval(0.08, 50)  # 4 of 50
    assert round(low, 3) == 0.032 and round(high, 3) == 0.188
    assert site_export.wilson_interval(0.0, 50)[0] == 0.0
    assert site_export.wilson_interval(1.0, 30)[1] == 1.0


def test_missing_cell_fails_export(site_env) -> None:
    site, cfg = site_env
    results = cfg.runs_dir / "full" / "results.csv"
    with open(results, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if not (r["clip_id"] == "younger_0" and r["noise_type"] == "tv")]
    with open(results, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)
    with pytest.raises(ValueError, match="missing cell clip=younger_0 noise=tv"):
        site_export.export_site(site)
    assert not site.out_dir.exists()


def test_audio_not_in_cache_fails_export(site_env) -> None:
    site, cfg = site_env
    for path in cfg.cache_dir.glob("*.txt"):
        path.unlink()
    cfg.cache_dir.joinpath("keep").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="not what Whisper heard"):
        site_export.export_site(site)


def test_site_config_rejects_tv_audio(site_env) -> None:
    site, _cfg = site_env
    data = site.model_dump()
    data["audio_noise_types"] = ["living", "tv"]
    with pytest.raises(ValueError, match="must not include tv"):
        SiteConfig.model_validate(data)


def test_export_site_cli_prints_summary(site_env, tmp_path: Path) -> None:
    site, _cfg = site_env
    config = tmp_path / "site.yaml"
    config.write_text(json.dumps(site.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(app, ["export-site", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "audio cells match the transcription cache" in result.output
    assert "hero: base, tv 10 dB" in result.output


def test_speech_span_finds_the_spoken_part() -> None:
    import numpy as np

    rate = 16_000
    clip = np.zeros(3 * rate, dtype=np.float32)
    t = np.arange(rate) / rate
    clip[rate : 2 * rate] = 0.3 * np.sin(2 * np.pi * 300 * t)
    start, end = site_export.speech_span(clip)
    assert abs(start - 1.0) <= 0.02 and abs(end - 2.0) <= 0.02
