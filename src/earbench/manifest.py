"""Clip selection from Common Voice and manifest read/write (Phase 1)."""

from __future__ import annotations

import csv
import hashlib
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from earbench import audio
from earbench.config import AgeGroup, PrepareConfig, write_csv

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "client_id",
    "path",
    "sentence",
    "up_votes",
    "down_votes",
    "age",
    "gender",
}

DecodeClipFn = Callable[[Path, Path], float]


class ManifestRow(BaseModel):
    """One prepared clip: 16 kHz mono WAV plus its test conditions."""

    clip_id: str
    wav_path: str
    sentence: str
    age_bucket: str
    age_group: AgeGroup
    gender: str
    duration_s: float
    speaker_hash: str


class PrepareSummary(BaseModel):
    """Counts and shortfalls from a `prepare` run, also printed to the console."""

    available_per_bucket: dict[str, int] = {}
    selected_per_group: dict[str, int] = {}
    speakers_per_group: dict[str, int] = {}
    female_per_group: dict[str, int] = {}
    male_per_group: dict[str, int] = {}
    missing_mp3: int = 0
    duration_rejected: int = 0
    undecodable: int = 0
    shortfalls: list[str] = []
    selected_ids: list[str] = []


@dataclass
class CandidateClip:
    """One vote/age-filtered TSV row whose MP3 exists on disk."""

    client_id: str
    mp3_path: Path
    sentence: str
    age_bucket: str
    age_group: AgeGroup
    gender: str  # "female", "male" or "unknown"


@dataclass
class SpeakerPlan:
    """Deterministic walk order plus per-gender clip quotas for each group."""

    try_lists: dict[AgeGroup, list[str]] = field(default_factory=dict)
    quotas: dict[AgeGroup, dict[str, int]] = field(default_factory=dict)
    # The group the others mirror; None means every group wants clips_per_group.
    reference_group: AgeGroup | None = None


def speaker_hash(client_id: str) -> str:
    """Anonymised speaker id: first 16 hex chars of sha256(client_id)."""
    return hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]


def normalise_gender(value: str) -> str:
    """Map Common Voice gender labels to female/male/unknown."""
    text = value.strip().lower()
    if text.startswith("female"):
        return "female"
    if text.startswith("male"):
        return "male"
    return "unknown"


def _parse_int(value: str) -> int:
    """Vote count from a TSV cell; blank or non-numeric counts as 0."""
    try:
        return int(value)
    except ValueError:
        return 0


def load_candidates(cv_dir: str | Path, cfg: PrepareConfig) -> tuple[list[CandidateClip], int]:
    """Merge the TSVs and filter by age bucket and votes.

    Returns (candidates, missing_mp3): rows whose MP3 is missing are skipped
    with a logged warning and counted, never an error.
    """
    root = Path(cv_dir)
    bucket_to_group: dict[str, AgeGroup] = {
        bucket: group for group, buckets in cfg.age_buckets.items() for bucket in buckets
    }
    candidates: list[CandidateClip] = []
    seen_paths: set[str] = set()
    missing_mp3 = 0
    for tsv_name in cfg.tsv_files:
        tsv_path = root / tsv_name
        if not tsv_path.is_file():
            raise ValueError(f"TSV file not found: {tsv_path} (check cv_dir/tsv_files)")
        with open(tsv_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
                missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
                raise ValueError(f"{tsv_path}: missing column(s): {', '.join(missing)}")
            for row in reader:
                age_bucket = row["age"].strip()
                if age_bucket not in bucket_to_group:
                    continue
                if _parse_int(row["up_votes"]) < cfg.min_up_votes:
                    continue
                if _parse_int(row["down_votes"]) > cfg.max_down_votes:
                    continue
                clip_name = row["path"].strip()
                if not clip_name or clip_name in seen_paths:
                    continue
                seen_paths.add(clip_name)
                mp3_path = root / "clips" / clip_name
                if not mp3_path.is_file():
                    missing_mp3 += 1
                    logger.warning("missing clip file, skipping: %s", mp3_path)
                    continue
                candidates.append(
                    CandidateClip(
                        client_id=row["client_id"].strip(),
                        mp3_path=mp3_path,
                        sentence=row["sentence"].strip(),
                        age_bucket=age_bucket,
                        age_group=bucket_to_group[age_bucket],
                        gender=normalise_gender(row["gender"]),
                    )
                )
    return candidates, missing_mp3


def _spawn(seed: int, index: int) -> np.random.Generator:
    """Independent deterministic RNG stream per stratum (order-independent)."""
    children = np.random.SeedSequence(seed).spawn(5)
    return np.random.default_rng(children[index])


def _speaker_rng(seed: int, client_id: str) -> np.random.Generator:
    """Deterministic RNG stream per speaker, independent of walk order."""
    digest = hashlib.sha256(client_id.encode("utf-8")).digest()[:8]
    return np.random.default_rng(np.random.SeedSequence([seed, int.from_bytes(digest, "little")]))


def plan_speaker_order(candidates: list[CandidateClip], cfg: PrepareConfig) -> SpeakerPlan:
    """Order speakers per group with seeded shuffles; apply gender quotas.

    With match_gender=True the older group takes every eligible female speaker
    (capped at clips_per_group) plus male speakers to fill up, and the younger
    group mirrors those female/male counts. Speakers with no recorded gender
    are excluded. With match_gender=False all speakers are pooled in one
    seeded shuffle.
    """
    plan = SpeakerPlan()

    def _shuffled(group: AgeGroup, gender: str, stream: int) -> list[str]:
        ids = sorted(
            {
                clip.client_id
                for clip in candidates
                if clip.age_group == group and clip.gender == gender
            }
        )
        _spawn(cfg.seed, stream).shuffle(ids)
        return ids

    if cfg.match_gender:
        ref: AgeGroup = "older"
        plan.reference_group = ref
        ref_female = _shuffled(ref, "female", 0)
        ref_male = _shuffled(ref, "male", 1)
        plan.try_lists[ref] = ref_female + ref_male
        n_female_ref = min(len(ref_female), cfg.clips_per_group)
        plan.quotas[ref] = {
            "female": n_female_ref,
            "male": min(len(ref_male), cfg.clips_per_group - n_female_ref),
        }
        for group in cfg.age_buckets:
            if group == ref:
                continue
            female = _shuffled(group, "female", 2)
            male = _shuffled(group, "male", 3)
            plan.try_lists[group] = female + male
            plan.quotas[group] = {
                "female": min(len(female), plan.quotas[ref]["female"]),
                "male": min(len(male), plan.quotas[ref]["male"]),
            }
    else:
        for group in cfg.age_buckets:
            pooled = sorted({clip.client_id for clip in candidates if clip.age_group == group})
            _spawn(cfg.seed, 4).shuffle(pooled)
            plan.try_lists[group] = pooled
            plan.quotas[group] = {"all": cfg.clips_per_group}
    return plan


def run_prepare(
    cfg: PrepareConfig, decode_clip: DecodeClipFn = audio.decode_clip_to_wav
) -> PrepareSummary:
    """Select clips, decode them to 16 kHz mono WAV, write the manifest."""
    candidates, missing_mp3 = load_candidates(cfg.cv_dir, cfg)
    available_per_bucket = dict(Counter(clip.age_bucket for clip in candidates))
    by_speaker: dict[str, list[CandidateClip]] = {}
    for clip in candidates:
        by_speaker.setdefault(clip.client_id, []).append(clip)
    plan = plan_speaker_order(candidates, cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    summary = PrepareSummary(missing_mp3=missing_mp3, available_per_bucket=available_per_bucket)
    rows: list[ManifestRow] = []
    decoded = 0
    for group in cfg.age_buckets:
        quota = plan.quotas.get(group, {})
        taken: Counter[str] = Counter()
        genders: Counter[str] = Counter()
        speakers_used: set[str] = set()
        for client_id in plan.try_lists.get(group, []):
            if all(taken[key] >= quota[key] for key in quota):
                break
            clips = list(by_speaker.get(client_id, []))
            _speaker_rng(cfg.seed, client_id).shuffle(clips)
            accepted = 0
            for clip in clips:
                if accepted >= cfg.max_clips_per_speaker:
                    break
                slot = "all" if "all" in quota else clip.gender
                if taken[slot] >= quota.get(slot, 0):
                    continue
                dst = cfg.out_dir / f"{clip.mp3_path.stem}.wav"
                try:
                    duration_s = decode_clip(clip.mp3_path, dst)
                except Exception:  # noqa: BLE001 - one bad clip must not kill prepare
                    summary.undecodable += 1
                    logger.warning("could not decode clip, skipping: %s", clip.mp3_path)
                    continue
                decoded += 1
                if decoded % 50 == 0:
                    logger.info("decoded %d clips", decoded)
                if not cfg.min_duration_s <= duration_s <= cfg.max_duration_s:
                    summary.duration_rejected += 1
                    dst.unlink(missing_ok=True)
                    continue
                rows.append(
                    ManifestRow(
                        clip_id=clip.mp3_path.stem,
                        wav_path=dst.as_posix(),
                        sentence=clip.sentence,
                        age_bucket=clip.age_bucket,
                        age_group=clip.age_group,
                        gender=clip.gender,
                        duration_s=round(duration_s, 2),
                        speaker_hash=speaker_hash(client_id),
                    )
                )
                taken[slot] += 1
                genders[clip.gender] += 1
                accepted += 1
                speakers_used.add(client_id)
        summary.selected_per_group[group] = sum(taken.values())
        summary.speakers_per_group[group] = len(speakers_used)
        summary.female_per_group[group] = genders["female"]
        summary.male_per_group[group] = genders["male"]
        mirrors_reference = plan.reference_group not in (None, group)
        wanted = sum(quota.values()) if mirrors_reference else cfg.clips_per_group
        if summary.selected_per_group[group] < wanted:
            parts = " + ".join(f"{quota[key]} {key}" for key in quota)
            summary.shortfalls.append(
                f"{group}: wanted {wanted} clips ({parts}), "
                f"selected {summary.selected_per_group[group]}"
            )
    rows.sort(key=lambda row: (row.age_group, row.clip_id))
    write_manifest(rows, cfg.manifest_path)
    summary.selected_ids = [row.clip_id for row in rows]
    return summary


def write_manifest(rows: list[ManifestRow], path: str | Path) -> Path:
    """Write manifest rows to CSV in ManifestRow field order (the Phase 1 columns)."""
    return write_csv(ManifestRow, rows, path)


def read_manifest(path: str | Path) -> list[ManifestRow]:
    """Read a manifest CSV back into validated rows."""
    with open(path, encoding="utf-8", newline="") as handle:
        return [ManifestRow.model_validate(row) for row in csv.DictReader(handle)]


def format_summary(summary: PrepareSummary) -> str:
    """Render a PrepareSummary as console text."""
    lines = ["clips available per age bucket:"]
    for bucket in sorted(summary.available_per_bucket):
        lines.append(f"  {bucket}: {summary.available_per_bucket[bucket]}")
    for group in summary.selected_per_group:
        lines.append(
            f"{group}: {summary.selected_per_group[group]} clips from "
            f"{summary.speakers_per_group[group]} speakers "
            f"({summary.female_per_group[group]} female, {summary.male_per_group[group]} male)"
        )
    lines.append(f"missing MP3 files skipped: {summary.missing_mp3}")
    lines.append(f"clips rejected by duration: {summary.duration_rejected}")
    lines.append(f"clips failed to decode: {summary.undecodable}")
    for shortfall in summary.shortfalls:
        lines.append(f"shortfall: {shortfall}")
    return "\n".join(lines)
