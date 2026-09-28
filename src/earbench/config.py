"""Config models for EarBench YAML files."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

AgeGroup = Literal["older", "younger"]
NoiseType = Literal["none", "tv", "living", "kitchen", "cafeteria"]


class PrepareConfig(BaseModel):
    """Which clips to pick and where the manifest goes."""

    cv_dir: Path = Path("cv26-australian-english")
    tsv_files: list[str] = ["train.tsv", "dev.tsv", "test.tsv"]
    out_dir: Path = Path("data/clips")
    manifest_path: Path = Path("data/manifest.csv")
    age_buckets: dict[AgeGroup, list[str]] = {
        "older": ["sixties", "seventies", "eighties", "nineties"],
        "younger": ["twenties", "thirties", "fourties"],
    }
    clips_per_group: int = 50
    max_clips_per_speaker: int = 1
    min_duration_s: float = 3.0
    max_duration_s: float = 10.0
    min_up_votes: int = 2
    max_down_votes: int = 0
    match_gender: bool = True
    seed: int = 0


class RoomConfig(BaseModel):
    """Size and echo of the fake room."""

    dims_m: tuple[float, float, float] = (5.0, 4.0, 2.7)
    rt60_s: float = 0.5
    mic_pos_m: tuple[float, float, float] = (1.0, 2.0, 1.2)
    source_height_m: float = 1.2
    noise_pos_m: tuple[float, float, float] = (4.5, 3.5, 1.0)


class ReportOptions(BaseModel):
    """How `earbench report` picks rows for its charts and tables."""

    reference_distance_m: float = 2.0  # charts A/B fix distance here
    distance_snr_db: list[float | None] = [None, 10.0]  # None = clean; chart C lines
    distance_noise: NoiseType = "tv"  # chart C noise type
    best_model: str | None = None  # None = lowest overall WER


class SweepConfig(BaseModel):
    """Which clips, distances, noises, and models to test."""

    manifest_path: Path
    clips_per_group: int | None = None
    clip_ids: list[str] | None = None  # restrict to exactly these clips (e.g. the site's 8)
    distances_m: list[float] = [1.0, 2.0, 3.0]
    noise_files: dict[NoiseType, list[Path]] = Field(default_factory=dict)
    snr_db: list[float] = [20.0, 10.0, 5.0, 0.0]
    include_clean: bool = True
    models: list[str] = ["tiny", "base", "small"]
    vad_filter: bool = False  # Phase 6 fix: VAD on shows up as vad=on in settings
    device: str = "cpu"  # "cuda" needs the gpu extra: uv sync --extra gpu
    compute_type: str = "int8"  # "float16" on cuda
    room: RoomConfig = RoomConfig()
    usable_wer: float = 0.20
    bootstrap_iters: int = 1000
    cache_dir: Path = Path("data/cache/transcribe")
    runs_dir: Path = Path("runs")
    report: ReportOptions = ReportOptions()
    seed: int = 0


class SiteClip(BaseModel):
    """One curated demo clip and the condition the site opens it on."""

    id: str
    featured_noise: NoiseType
    featured_snr_db: float | None = None


class SiteHero(BaseModel):
    """The condition whose summary.csv numbers back the hero sentence."""

    model: str
    noise_type: NoiseType
    snr_db: float | None = None


class SiteChart(BaseModel):
    """One findings chart: WER vs noise level, one line per `series` value.

    The two dimensions that are not the series are fixed by the other fields.
    """

    id: str
    title: str
    series: Literal["age_group", "model", "noise_type"]
    model: str | None = None
    age_group: AgeGroup | None = None
    noise_type: NoiseType | None = None


class SiteLevel(BaseModel):
    """A loudness-slider stop: an SNR (None = no noise) and its plain label."""

    snr_db: float | None
    label: str


class SiteConfig(BaseModel):
    """What `earbench export-site` reads and where the demo site's data goes."""

    full_run: Path
    vad_run: Path
    room_compare: Path
    room_label: str
    distance_m: float = 2.0
    models: list[str] = ["tiny", "base", "small"]
    audio_noise_types: list[NoiseType] = ["living", "kitchen", "cafeteria"]
    text_only_noise_types: list[NoiseType] = ["tv"]  # transcripts only, never audio
    levels: list[SiteLevel]
    noise_labels: dict[NoiseType, str]
    clips: list[SiteClip]
    hero: SiteHero
    charts: list[SiteChart] = []
    transcriber_name: str = "faster-whisper"  # part of the transcription cache key
    tv_note: str = "audio not shown: copyrighted TV recording"
    out_dir: Path = Path("site/public/data")

    @model_validator(mode="after")
    def _check(self) -> SiteConfig:
        if "tv" in self.audio_noise_types:
            raise ValueError("audio_noise_types must not include tv (YouTube audio)")
        if "none" in self.audio_noise_types or "none" in self.text_only_noise_types:
            raise ValueError(
                "the clean condition comes from levels (snr_db: null), not noise types"
            )
        if not any(level.snr_db is None for level in self.levels):
            raise ValueError("levels needs a no-noise stop (snr_db: null)")
        ids = [clip.id for clip in self.clips]
        if len(set(ids)) != len(ids):
            raise ValueError("clips has duplicate ids")
        return self


class RoomSessionConfig(BaseModel):
    """Settings for a real-room recording session.

    Two devices: the MacBook plays the playlists, the PC records them. The PC
    records at the input device's native sample rate (`sample_rate` is only
    the fallback when the device won't report one).
    """

    manifest_path: Path
    clips_per_group: int = 15
    distances_m: list[float] = [1.0, 2.0, 3.0]
    conditions: list[Literal["quiet", "tv"]] = ["quiet", "tv"]
    lead_in_s: float = 1.5
    sample_rate: int = 48_000
    input_device: str | int | None = None
    model: str = "small"
    recordings_dir: Path = Path("recordings")
    playlists_dir: Path = Path("playlists")
    seed: int = 0


class ResultRow(BaseModel):
    """One scored (clip, condition, model) result, a row of runs/<run_id>/results.csv."""

    run_id: str
    source: Literal["sim", "room"]
    clip_id: str
    age_group: AgeGroup
    distance_m: float
    noise_type: NoiseType
    snr_db: float | None = None
    model: str
    model_version: str
    settings: str
    reference: str
    hypothesis: str
    ref_words: int
    errors: int
    wer: float
    config_hash: str


class SummaryRow(BaseModel):
    """One age group's pooled WER, interval and usable rate for one condition."""

    model: str
    age_group: AgeGroup
    distance_m: float
    noise_type: NoiseType
    snr_db: float | None = None
    n_clips: int
    errors: int
    ref_words: int
    wer: float
    ci_low: float
    ci_high: float
    usable_rate: float


class ConfigError(ValueError):
    """Bad config file. The message names the file and the field."""


def load_config[T: BaseModel](path: str | Path, model: type[T]) -> T:
    """Read a YAML file into a config model. Errors name the file and field."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path} (expected {model.__name__})")
    try:
        raw: Any = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file {cfg_path}: invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file {cfg_path}: top-level mapping required for {model.__name__}, "
            f"got {type(raw).__name__}"
        )
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"config file {cfg_path}: invalid {model.__name__}: {problems}") from exc


def config_hash(config: BaseModel) -> str:
    """Short ID string for a config, stamped on every result row."""
    payload = config.model_dump_json()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def write_csv(model: type[BaseModel], rows: Sequence[BaseModel], path: str | Path) -> Path:
    """Write rows to CSV with columns in `model`'s field order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model.model_fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump())
    return out_path
