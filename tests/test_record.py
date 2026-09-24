"""Tests for the two-device real-room session: playlists, alignment, record/score/compare.

All offline: the mic and the transcriber are faked. The human owns all real
measurements; tests only check that typed values are saved as typed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from earbench import audio, record
from earbench.audio import SAMPLE_RATE_HZ
from earbench.cli import app
from earbench.config import RoomSessionConfig
from earbench.manifest import ManifestRow, write_manifest
from earbench.transcribe import FakeTranscriber

runner = CliRunner()


def _tone(duration_s: float, freq_hz: float = 440.0) -> np.ndarray:
    """A steady tone standing in for speech (constant level, no pauses)."""
    t = np.arange(int(duration_s * SAMPLE_RATE_HZ), dtype=np.float64) / SAMPLE_RATE_HZ
    return (0.5 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _speech_with_pause() -> np.ndarray:
    """Tone, silence, tone: the estimator must ignore the pause like `noise.mix` does."""
    return np.concatenate([_tone(0.4), np.zeros(SAMPLE_RATE_HZ // 2, dtype=np.float32), _tone(0.4)])


def _recording_at_snr(snr_db: float, seed: int = 0) -> tuple[np.ndarray, float]:
    """A synthetic mic recording with a 1.5 s noise-only lead-in at a known SNR."""
    rng = np.random.default_rng(seed)
    raw_noise = (0.2 * rng.standard_normal(5 * SAMPLE_RATE_HZ)).astype(np.float32)
    speech = _speech_with_pause()
    speech_power = audio.active_speech_power(speech)
    noise_power = audio.power(raw_noise[: speech.shape[0]])
    gain = float(np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10.0))))
    scaled = (raw_noise * gain).astype(np.float32)
    lead_in = scaled[: int(1.5 * SAMPLE_RATE_HZ)]
    mixed_speech = (
        speech + scaled[int(1.5 * SAMPLE_RATE_HZ) : int(1.5 * SAMPLE_RATE_HZ) + speech.shape[0]]
    )
    return np.concatenate([lead_in, mixed_speech]).astype(np.float32), 1.5


class FakeBackend:
    """Pretend mic: returns a prepared buffer (padded/trimmed to the request)."""

    def __init__(self, rate_hz: int = SAMPLE_RATE_HZ, buffer: np.ndarray | None = None) -> None:
        self._rate_hz = rate_hz
        self.buffer = np.zeros(0, dtype=np.float32) if buffer is None else buffer
        self.calls: list[int] = []
        self.input_name = "fake-mic"

    @property
    def native_sample_rate(self) -> int:
        return self._rate_hz

    def record_block(self, n_samples: int) -> np.ndarray:
        self.calls.append(n_samples)
        out = np.zeros(n_samples, dtype=np.float32)
        n = min(n_samples, self.buffer.shape[0])
        out[:n] = self.buffer[:n]
        return out


def _room_cfg(tmp_path: Path, **over: object) -> RoomSessionConfig:
    """A tiny session config: 1 clip per group, one distance, quiet only by default."""
    base: dict[str, object] = {
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
    base.update(over)
    return RoomSessionConfig.model_validate(base)


def _manifest(tmp_path: Path) -> list[ManifestRow]:
    """Two clips (one older, one younger) with real WAVs on disk."""
    rows = []
    for group, bucket, freq in (("older", "sixties", 300.0), ("younger", "twenties", 700.0)):
        wav = tmp_path / "clips" / f"{group}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        audio.save_wav_16k_mono(wav, _tone(0.6, freq))
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
    return rows


def _cfg_path(tmp_path: Path, cfg: RoomSessionConfig) -> Path:
    path = tmp_path / "room.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return path


def _make_playlists(cfg: RoomSessionConfig) -> list[record.BlockPlaylist]:
    return record.make_playlist(cfg)


# Session answers: room L/W/H, webcam x/y/h, TV x/y/h, devices, then per block
# measured distance / speech level / TV level / notes / ready-Enter.
SESSION_QUIET = "5\n4\n2.7\n1.0\n0.5\n1.2\n4.0\n3.0\n1.0\nMacBook speakers\nPC webcam mic\n"
SESSION_TV_EXTRA = "https://www.youtube.com/watch?v=abc123\n03:12\n6/16\n"
BLOCK_QUIET = "1.0\n60\n\nsofa one metre\n\n"
BLOCK_TV = "1.0\n60\n55\ntv on\n\n"
QUIET_INPUT = SESSION_QUIET + BLOCK_QUIET
# The TV-noise answers come before the device labels.
TV_INPUT = (
    "5\n4\n2.7\n1.0\n0.5\n1.2\n4.0\n3.0\n1.0\n"
    + SESSION_TV_EXTRA
    + "MacBook speakers\nPC webcam mic\n"
    + BLOCK_TV
)


def _playlist_buffer(
    cfg: RoomSessionConfig, rate_hz: int, offset_s: float = 0.37, noise_level: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic room recording: the block playlist, delayed, plus noise."""
    play_16k, _ = record._load_block_playlist(cfg, cfg.distances_m[0], cfg.conditions[0])
    play = audio.resample_between(play_16k, SAMPLE_RATE_HZ, rate_hz)
    rng = np.random.default_rng(0)
    offset = int(round(offset_s * rate_hz))
    total = play.shape[0] + offset + int(round(record.RECORD_SLACK_S * rate_hz))
    rec = (noise_level * rng.standard_normal(total)).astype(np.float32)
    rec[offset : offset + play.shape[0]] += play
    return rec, play


def test_estimate_snr_recovers_known_snr() -> None:
    for target_db in (10.0, 0.0):
        recording, lead_in_s = _recording_at_snr(target_db)
        assert record.estimate_snr_db(recording, lead_in_s) == pytest.approx(target_db, abs=1.0)


def test_estimate_snr_returns_none_without_speech() -> None:
    silence = np.zeros(2 * SAMPLE_RATE_HZ, dtype=np.float32)
    assert record.estimate_snr_db(silence, 1.5) is None


def test_make_playlist_writes_block_files_and_calibration(tmp_path: Path) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path, conditions=["quiet", "tv"])
    result = runner.invoke(app, ["make-playlist", "--config", str(_cfg_path(tmp_path, cfg))])
    assert result.exit_code == 0, result.output

    playlists = list((tmp_path / "playlists").glob("playlist_*.wav"))
    assert len(playlists) == 2
    clip_sets = set()
    for wav_path in playlists:
        timing = record.read_timing(wav_path.with_suffix(".csv"))
        assert len(timing) == 2  # one older + one younger clip
        assert {row.order for row in timing} == {0, 1}
        clip_sets.add(tuple(sorted(row.clip_id for row in timing)))
        first = timing[0]
        assert first.start_sample == int((0.5 + 2.0 + cfg.lead_in_s) * SAMPLE_RATE_HZ)
    assert len(clip_sets) == 1  # every block shares the session's clip set

    calibration = audio.load_16k(tmp_path / "playlists" / "calibration.wav")
    assert calibration.shape[0] == int(60 * SAMPLE_RATE_HZ)


def test_aligner_recovers_clip_starts_within_1ms(tmp_path: Path) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    play_16k, timing = record._load_block_playlist(cfg, 1.0, "quiet")

    for rate_hz in (SAMPLE_RATE_HZ, 44100):
        rec, _ = _playlist_buffer(cfg, rate_hz)
        aligned = record.align_block(rec, rate_hz, timing, cfg.lead_in_s, play_16k.shape[0])
        assert aligned is not None
        assert aligned.warnings == []
        offset = int(round(0.37 * rate_hz))
        lead_play = int(round(cfg.lead_in_s * SAMPLE_RATE_HZ))
        for row, cut in zip(timing, aligned.take_starts, strict=True):
            # Cuts start one lead-in before the clip onset.
            expected = offset + int(
                round((row.start_sample - lead_play) * rate_hz / SAMPLE_RATE_HZ)
            )
            assert abs(cut - expected) <= 0.001 * rate_hz
        assert len(aligned.segments) == len(timing)


def test_record_aligns_and_cuts_takes(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    result = runner.invoke(
        app,
        ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "s1"],
        input=QUIET_INPUT,
    )
    assert result.exit_code == 0, result.output
    assert len(backend.calls) == 1  # one continuous recording for the single block

    session_dir = Path(cfg.recordings_dir) / "s1"
    session = record.read_session(session_dir / "session.yaml")
    assert session.room_dims_m == (5.0, 4.0, 2.7)
    assert session.webcam_pos_m == (1.0, 0.5, 1.2)
    assert session.tv_pos_m == (4.0, 3.0, 1.0)
    assert session.playback_device == "MacBook speakers"
    assert session.mic == "PC webcam mic"
    assert session.input_name == "fake-mic"
    assert session.youtube_url is None  # no TV blocks, so never asked
    block = session.blocks[0]
    assert block.measured_distance_m == pytest.approx(1.0)
    assert block.speech_level_dba == pytest.approx(60.0)
    assert block.tv_level_dba is None
    assert block.notes == "sofa one metre"
    assert len(block.takes) == 2

    for take in block.takes:
        wav_path = session_dir / take.wav_file
        assert wav_path.is_file()
        assert "_t1.wav" in wav_path.name
        loaded, rate_hz = audio.load_mono(str(wav_path))
        assert rate_hz == SAMPLE_RATE_HZ
        # Lead-in plus the 0.6 s clip, cut from the aligned recording.
        assert loaded.shape[0] == int((cfg.lead_in_s + 0.6) * SAMPLE_RATE_HZ)


def test_record_chirp_missing_saves_nothing(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rng = np.random.default_rng(1)
    backend = FakeBackend(
        buffer=(0.05 * rng.standard_normal(3 * SAMPLE_RATE_HZ)).astype(np.float32)
    )
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    result = runner.invoke(
        app,
        ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "nope"],
        input=QUIET_INPUT,
    )
    assert result.exit_code == 0, result.output
    assert "chirp" in result.output.lower()
    session_dir = Path(cfg.recordings_dir) / "nope"
    assert list((session_dir / "clips").glob("*.wav")) == []
    assert record.read_session(session_dir / "session.yaml").blocks == []


def test_record_dry_run_touches_no_hardware(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    backend = FakeBackend()
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    result = runner.invoke(
        app,
        ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "dry", "--dry-run"],
        input=QUIET_INPUT,
    )
    assert result.exit_code == 0, result.output
    assert backend.calls == []
    session_dir = Path(cfg.recordings_dir) / "dry"
    assert len(list((session_dir / "clips").glob("*.wav"))) == 2
    assert len(record.read_session(session_dir / "session.yaml").blocks[0].takes) == 2


def test_record_never_overwrites_retakes_get_next_take(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)
    args = ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "s2"]

    assert runner.invoke(app, args, input=QUIET_INPUT).exit_code == 0
    first = {
        p.name: p.read_bytes() for p in (Path(cfg.recordings_dir) / "s2" / "clips").glob("*.wav")
    }
    assert all(name.endswith("_t1.wav") for name in first)

    # A resumed run reuses the stored setup (one ready-Enter) and adds take 2.
    assert runner.invoke(app, args, input="\n").exit_code == 0
    session = record.read_session(Path(cfg.recordings_dir) / "s2" / "session.yaml")
    assert len(session.blocks[0].takes) == 4  # originals kept alongside the retakes
    second = {
        p.name: p.read_bytes() for p in (Path(cfg.recordings_dir) / "s2" / "clips").glob("*.wav")
    }
    for name, blob in first.items():
        assert second[name] == blob  # take 1 bytes untouched
    assert sum(name.endswith("_t2.wav") for name in second) == 2


def test_record_uses_native_sample_rate(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, 44100)
    backend = FakeBackend(rate_hz=44100, buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    assert (
        runner.invoke(
            app,
            ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "native"],
            input=QUIET_INPUT,
        ).exit_code
        == 0
    )
    session_dir = Path(cfg.recordings_dir) / "native"
    session = record.read_session(session_dir / "session.yaml")
    assert session.sample_rate == 44100  # queried, not the config's fallback
    for take in session.blocks[0].takes:
        loaded, rate_hz = audio.load_mono(str(session_dir / take.wav_file))
        assert rate_hz == 44100
        assert loaded.shape[0] == int(round((cfg.lead_in_s + 0.6) * 44100))


def test_record_warns_on_end_chirp_drift_but_saves(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    play_16k, _ = record._load_block_playlist(cfg, 1.0, "quiet")
    chirp_len = int(0.5 * SAMPLE_RATE_HZ)
    body, end_chirp = play_16k[:-chirp_len], play_16k[-chirp_len:]
    shifted = np.concatenate(
        [body, np.zeros(int(0.05 * SAMPLE_RATE_HZ), dtype=np.float32), end_chirp]
    )
    offset = int(0.2 * SAMPLE_RATE_HZ)
    rec = np.zeros(offset + shifted.shape[0] + SAMPLE_RATE_HZ, dtype=np.float32)
    rec[offset : offset + shifted.shape[0]] = shifted
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    result = runner.invoke(
        app,
        ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "drift"],
        input=QUIET_INPUT,
    )
    assert result.exit_code == 0, result.output
    assert "drift" in result.output.lower()
    session = record.read_session(Path(cfg.recordings_dir) / "drift" / "session.yaml")
    assert len(session.blocks[0].takes) == 2  # drift warns, takes still saved


def test_session_records_tv_noise_setup(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path, conditions=["tv"])
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)

    assert (
        runner.invoke(
            app,
            ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "tv"],
            input=TV_INPUT,
        ).exit_code
        == 0
    )
    session = record.read_session(Path(cfg.recordings_dir) / "tv" / "session.yaml")
    assert session.youtube_url == "https://www.youtube.com/watch?v=abc123"
    assert session.youtube_start == "03:12"
    assert session.macbook_volume == "6/16"
    assert session.blocks[0].tv_level_dba == pytest.approx(55.0)


def test_score_room_writes_room_results(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)
    monkeypatch.setattr(record, "default_transcriber_factory", lambda _model: FakeTranscriber())
    assert (
        runner.invoke(
            app,
            ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "s3"],
            input=QUIET_INPUT,
        ).exit_code
        == 0
    )

    result = runner.invoke(app, ["score-room", str(Path(cfg.recordings_dir) / "s3")])
    assert result.exit_code == 0, result.output
    rows = record.read_results(Path(cfg.recordings_dir) / "s3" / "results.csv")
    assert len(rows) == 2
    assert {row.source for row in rows} == {"room"}
    assert {row.model for row in rows} == {"tiny"}
    # Quiet has no added noise, so there is nothing to estimate.
    assert all(row.snr_db is None for row in rows)
    assert all(row.noise_type == "none" for row in rows)


def _sweep_cfg(tmp_path: Path, **over: object) -> Path:
    rng = np.random.default_rng(0)
    tv_path = tmp_path / "tv.wav"
    audio.save_wav_16k_mono(
        tv_path, (0.2 * rng.standard_normal(5 * SAMPLE_RATE_HZ)).astype(np.float32)
    )
    base: dict[str, object] = {
        "manifest_path": str(tmp_path / "manifest.csv"),
        "distances_m": [1.0],
        "noise_files": {"tv": [str(tv_path)]},
        "snr_db": [10.0],
        "include_clean": True,
        "models": ["tiny"],
        "bootstrap_iters": 10,
        "runs_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
        "seed": 0,
    }
    base.update(over)
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_compare_room_needs_matching_model(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path, conditions=["tv"])
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)
    monkeypatch.setattr(record, "default_transcriber_factory", lambda _model: FakeTranscriber())
    assert (
        runner.invoke(
            app,
            ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "s4"],
            input=TV_INPUT,
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["score-room", str(Path(cfg.recordings_dir) / "s4")]).exit_code == 0

    bad_sweep = _sweep_cfg(tmp_path, models=["base"])
    result = runner.invoke(
        app, ["compare-room", str(Path(cfg.recordings_dir) / "s4"), "--config", str(bad_sweep)]
    )
    assert result.exit_code == 1
    assert "tiny" in result.output and "base" in result.output


def test_compare_room_uses_typed_geometry(tmp_path: Path, monkeypatch) -> None:
    _manifest(tmp_path)
    cfg = _room_cfg(tmp_path)
    _make_playlists(cfg)
    rec, _ = _playlist_buffer(cfg, SAMPLE_RATE_HZ)
    backend = FakeBackend(buffer=rec)
    monkeypatch.setattr(record, "make_backend", lambda _cfg: backend)
    monkeypatch.setattr(record, "default_transcriber_factory", lambda _model: FakeTranscriber())
    assert (
        runner.invoke(
            app,
            ["record", "--config", str(_cfg_path(tmp_path, cfg)), "--session-id", "s5"],
            input=QUIET_INPUT,
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["score-room", str(Path(cfg.recordings_dir) / "s5")]).exit_code == 0

    seen: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []

    def spy(signal: np.ndarray, source_m: tuple[float, float, float], room: object) -> np.ndarray:
        from earbench.config import RoomConfig

        assert isinstance(room, RoomConfig)
        seen.append((tuple(source_m), tuple(room.mic_pos_m), tuple(room.dims_m)))
        return np.zeros(100, dtype=np.float32)

    monkeypatch.setattr(record.room_sim, "simulate_at", spy)
    sweep_path = _sweep_cfg(tmp_path)
    ok = runner.invoke(
        app, ["compare-room", str(Path(cfg.recordings_dir) / "s5"), "--config", str(sweep_path)]
    )
    assert ok.exit_code == 0, ok.output
    assert (Path(cfg.recordings_dir) / "s5" / "compare.csv").is_file()
    assert (Path(cfg.recordings_dir) / "s5" / "compare.png").is_file()
    # Voice straight out from the webcam along +y; mic and room from the session.
    assert seen
    for source_m, mic_pos_m, dims_m in seen:
        assert source_m == pytest.approx((1.0, 0.5 + 1.0, 1.2))
        assert mic_pos_m == pytest.approx((1.0, 0.5, 1.2))
        assert dims_m == pytest.approx((5.0, 4.0, 2.7))

    # An old session without positions falls back to the sweep config's room and says so.
    session_path = Path(cfg.recordings_dir) / "s5" / "session.yaml"
    payload = yaml.safe_load(session_path.read_text(encoding="utf-8"))
    payload["room_dims_m"] = None
    payload["webcam_pos_m"] = None
    payload["tv_pos_m"] = None
    session_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    fallback = runner.invoke(
        app, ["compare-room", str(Path(cfg.recordings_dir) / "s5"), "--config", str(sweep_path)]
    )
    assert fallback.exit_code == 0, fallback.output
    assert "fallback" in fallback.output.lower()


def test_old_room_config_still_loads(tmp_path: Path) -> None:
    """Guard: RoomSessionConfig gains playlists_dir but old YAML files still validate."""
    _manifest(tmp_path)
    path = tmp_path / "old-room.yaml"
    path.write_text(
        "manifest_path: {}\nclips_per_group: 5\ntail_s: 1.0\noutput_device: mic\n".format(
            (tmp_path / "manifest.csv").as_posix()
        ),
        encoding="utf-8",
    )
    from earbench.config import load_config

    loaded = load_config(path, RoomSessionConfig)
    assert loaded.clips_per_group == 5
    assert loaded.playlists_dir == Path("playlists")
