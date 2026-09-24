"""Real-room recording sessions: make-playlist, record, score-room, compare-room (Phase 5).

Two devices: the MacBook plays one playlist per block, the PC records its
webcam mic continuously. Each playlist is a sync chirp, 2 s of silence, then
every session clip as [lead_in_s silence][clip][1 s gap] in the block's seeded
order, then an end chirp, plus a timing CSV with each clip's start sample.
`record` finds the start chirp by cross-correlation, checks the end chirp
lands where expected, and cuts each clip's segment (lead-in included) into
the agreed `_t<take>` filenames. No clear chirp means nothing is saved.

The human owns every measurement: room dimensions, webcam and TV positions,
the YouTube link/start time, the MacBook volume, and per block the measured
distance, speech level, TV level and notes. Nothing is ever defaulted or
filled in. Retakes never overwrite: retakes are appended next to the original
takes in session.yaml.

Recordings are cut at the input device's native sample rate (webcam mics
often can't do 48 kHz) and resampled to 16 kHz only on load.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from scipy import signal as scipy_signal  # noqa: E402
from scipy.signal.windows import tukey  # noqa: E402

from earbench import audio, manifest, noise  # noqa: E402
from earbench import room as room_sim  # noqa: E402
from earbench import score as scoring  # noqa: E402
from earbench import sweep as sweep_mod  # noqa: E402
from earbench.audio import SAMPLE_RATE_HZ, active_speech_power  # noqa: E402
from earbench.config import (  # noqa: E402
    AgeGroup,
    NoiseType,
    ResultRow,
    RoomConfig,
    RoomSessionConfig,
    SummaryRow,
    SweepConfig,
    config_hash,
    load_config,
    write_csv,
)
from earbench.report import read_results  # noqa: E402
from earbench.transcribe import Transcriber  # noqa: E402

logger = logging.getLogger(__name__)

RoomCondition = Literal["quiet", "tv"]

SESSION_FILENAME = "session.yaml"
MANIFEST_COPY = "manifest.csv"
CLIPS_DIRNAME = "clips"
RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.csv"
COMPARE_FILENAME = "compare.csv"
COMPARE_PNG = "compare.png"
CALIBRATION_FILENAME = "calibration.wav"

PLAYLIST_RATE_HZ = SAMPLE_RATE_HZ  # playlists are built from 16 kHz clips
CHIRP_DURATION_S = 0.5
CHIRP_F0_HZ = 400.0
CHIRP_F1_HZ = 4000.0
CHIRP_GAIN = 0.5
START_SILENCE_S = 2.0  # silence between the start chirp and the first clip
CLIP_GAP_S = 1.0  # silence between clips and before the end chirp
CALIBRATION_S = 60.0  # one clip looped this long for setting the MacBook volume
RECORD_SLACK_S = 10.0  # extra recording past the playlist length
CHIRP_CLARITY_RATIO = 4.0  # start-chirp peak must beat the background by this much
CHIRP_FIRST_PEAK_DB = 3.0  # an earlier peak within this of the max is the start chirp
END_SEARCH_MS = 150.0  # window around the expected end chirp
END_DRIFT_WARN_MS = 20.0  # warn when the end chirp lands further off than this

LEAD_IN_SKIP_S = 0.3  # ignore the lead-in onset (clicks, late playback start)
ENERGY_GATE_DB = 3.0  # speech-region frames must clear the noise floor by this much
USABLE_WER = 0.20  # same threshold as the sweep until the protocol says otherwise
BOOTSTRAP_ITERS = 1000


class TakeRecord(BaseModel):
    """One recorded take: which clip, which planned block, which file on disk."""

    clip_id: str
    age_group: AgeGroup
    wav_file: str  # relative to the session dir, e.g. clips/<clip>_1m_tv_t1.wav
    take: int
    distance_m: float
    condition: RoomCondition


class BlockRecord(BaseModel):
    """One distance x condition block: the human's typed measurements plus its takes."""

    distance_m: float
    condition: RoomCondition
    measured_distance_m: float | None = None
    speech_level_dba: float | None = None
    tv_level_dba: float | None = None
    notes: str = ""
    takes: list[TakeRecord] = Field(default_factory=list)


class SessionFile(BaseModel):
    """Everything about a recording session: setup, typed entries, device names, takes."""

    session_id: str
    room_dims_m: tuple[float, float, float] | None = None
    webcam_pos_m: tuple[float, float, float] | None = None
    tv_pos_m: tuple[float, float, float] | None = None
    playback_device: str | None = None  # e.g. "MacBook speakers", typed by the human
    mic: str | None = None  # e.g. "PC webcam mic", typed by the human
    youtube_url: str | None = None
    youtube_start: str | None = None
    macbook_volume: str | None = None
    input_device: str | int | None = None
    input_name: str | None = None
    sample_rate: int = 48_000  # the input device's native rate, found at record time
    lead_in_s: float = 1.5
    model: str = "small"
    seed: int = 0
    config_hash: str = ""
    manifest_path: str = ""
    playlists_dir: str = ""
    blocks: list[BlockRecord] = Field(default_factory=list)


class TimingRow(BaseModel):
    """One playlist entry: where a clip starts, in playlist samples."""

    playlist: str
    distance_m: float
    condition: RoomCondition
    order: int
    clip_id: str
    start_sample: int  # onset of the clip audio (after its lead-in silence)
    clip_samples: int
    sample_rate_hz: int = PLAYLIST_RATE_HZ


class CompareRow(BaseModel):
    """One real take paired with its simulated twin: WER on both sides."""

    clip_id: str
    age_group: AgeGroup
    distance_m: float
    noise_type: NoiseType
    snr_db: float | None = None
    room_wer: float
    sim_wer: float
    room_hypothesis: str
    sim_hypothesis: str


class AudioBackend(Protocol):
    """Records the PC mic. The fake used in tests returns a prepared buffer."""

    @property
    def input_name(self) -> str: ...

    @property
    def native_sample_rate(self) -> int:
        """The input device's own rate: queried, never assumed (webcam mics vary)."""
        ...

    def record_block(self, n_samples: int) -> np.ndarray:
        """Record the mic for `n_samples` at the native rate."""
        ...


class SoundDeviceBackend:
    """Real PC-mic recording. sounddevice is imported lazily, only when used."""

    def __init__(self, input_device: str | int | None) -> None:
        self._input_device = input_device

    def _describe(self) -> dict:
        import sounddevice as sd

        try:
            if self._input_device is None:
                return dict(sd.query_devices(kind="input"))
            return dict(sd.query_devices(self._input_device))
        except Exception as exc:
            raise OSError(
                f"cannot query input device {self._input_device!r} ({exc}); "
                "use `earbench record --dry-run` to rehearse without hardware"
            ) from exc

    @property
    def input_name(self) -> str:
        return str(self._describe()["name"])

    @property
    def native_sample_rate(self) -> int:
        return int(self._describe()["default_samplerate"])

    def record_block(self, n_samples: int) -> np.ndarray:
        import sounddevice as sd

        try:
            recorded = sd.rec(
                int(n_samples),
                samplerate=self.native_sample_rate,
                channels=1,
                dtype="float32",
                device=self._input_device,
            )
            sd.wait()
        except Exception as exc:
            raise OSError(f"recording failed ({exc})") from exc
        return np.asarray(recorded).reshape(-1).astype(np.float32)


def make_backend(cfg: RoomSessionConfig) -> AudioBackend:
    """The real backend. Construction touches no device; errors surface at record time."""
    return SoundDeviceBackend(cfg.input_device)


def default_transcriber_factory(model_size: str) -> Transcriber:
    """One faster-whisper transcriber (tests monkeypatch this for a FakeTranscriber)."""
    from earbench.transcribe import FasterWhisperTranscriber

    return FasterWhisperTranscriber(model_size)


def sync_chirp(sample_rate_hz: int = PLAYLIST_RATE_HZ) -> np.ndarray:
    """The playlist sync chirp: a 0.5 s linear 400 Hz → 4 kHz sweep, edges faded."""
    n = int(round(CHIRP_DURATION_S * sample_rate_hz))
    t = np.arange(n, dtype=np.float64) / sample_rate_hz
    sweep = scipy_signal.chirp(t, CHIRP_F0_HZ, CHIRP_DURATION_S, CHIRP_F1_HZ, method="linear")
    return (CHIRP_GAIN * sweep * tukey(n, 0.05)).astype(np.float32)


def _block_seed(seed: int, distance_m: float, condition: RoomCondition) -> int:
    """Stable per-block playlist shuffle seed, independent of processing order."""
    digest = hashlib.sha256(f"{seed}:{distance_m:g}:{condition}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _condition_seed(seed: int, clip_id: str, noise_type: str) -> int:
    """Stable per-condition noise offset seed, independent of processing order."""
    digest = hashlib.sha256(f"{seed}:{clip_id}:{noise_type}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def playlist_filename(distance_m: float, condition: RoomCondition) -> str:
    """One playlist per block: `playlist_<dist>m_<cond>.wav` (+ `.csv` timing)."""
    return f"playlist_{distance_m:g}m_{condition}.wav"


@dataclass
class BlockPlaylist:
    """A built playlist: the WAV, its timing rows, and what produced them."""

    distance_m: float
    condition: RoomCondition
    wav_path: Path
    timing_path: Path
    rows: list[TimingRow]


def make_playlist(cfg: RoomSessionConfig) -> list[BlockPlaylist]:
    """Write one playlist per block plus `calibration.wav` (one clip looped 60 s).

    Every block shares the session's clip set but shuffles it with its own
    seeded order. Playlists are deterministic from the config, so reruns
    overwrite identical bytes.
    """
    out_dir = Path(cfg.playlists_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = sweep_mod.select_clips(
        manifest.read_manifest(cfg.manifest_path), cfg.clips_per_group, cfg.seed
    )
    if not clips:
        raise ValueError(f"no clips selected from {cfg.manifest_path}")
    chirp = sync_chirp()
    start_gap = np.zeros(int(round(START_SILENCE_S * PLAYLIST_RATE_HZ)), dtype=np.float32)
    gap = np.zeros(int(round(CLIP_GAP_S * PLAYLIST_RATE_HZ)), dtype=np.float32)
    lead = np.zeros(int(round(cfg.lead_in_s * PLAYLIST_RATE_HZ)), dtype=np.float32)

    built: list[BlockPlaylist] = []
    for distance_m in cfg.distances_m:
        for condition in cfg.conditions:
            order = np.random.default_rng(_block_seed(cfg.seed, distance_m, condition)).permutation(
                len(clips)
            )
            parts: list[np.ndarray] = [chirp, start_gap]
            rows: list[TimingRow] = []
            pos = chirp.shape[0] + start_gap.shape[0]
            fname = playlist_filename(distance_m, condition)
            for rank, index in enumerate(order):
                clip = clips[int(index)]
                clip_16k = audio.load_16k(clip.wav_path)
                rows.append(
                    TimingRow(
                        playlist=fname,
                        distance_m=distance_m,
                        condition=condition,
                        order=rank,
                        clip_id=clip.clip_id,
                        start_sample=pos + lead.shape[0],
                        clip_samples=clip_16k.shape[0],
                        sample_rate_hz=PLAYLIST_RATE_HZ,
                    )
                )
                parts += [lead, clip_16k, gap]
                pos += lead.shape[0] + clip_16k.shape[0] + gap.shape[0]
            parts.append(chirp)  # end chirp, straight after the last gap
            wav_path = out_dir / fname
            audio.save_wav_16k_mono(wav_path, np.concatenate(parts))
            timing_path = out_dir / f"{Path(fname).stem}.csv"
            write_csv(TimingRow, rows, timing_path)
            built.append(
                BlockPlaylist(
                    distance_m=distance_m,
                    condition=condition,
                    wav_path=wav_path,
                    timing_path=timing_path,
                    rows=rows,
                )
            )
    first_clip = audio.load_16k(clips[0].wav_path)
    n_cal = int(round(CALIBRATION_S * PLAYLIST_RATE_HZ))
    looped = np.tile(first_clip, n_cal // first_clip.shape[0] + 1)[:n_cal]
    audio.save_wav_16k_mono(out_dir / CALIBRATION_FILENAME, looped)
    logger.info("wrote %d playlists + %s in %s", len(built), CALIBRATION_FILENAME, out_dir)
    return built


def read_timing(path: str | Path) -> list[TimingRow]:
    """Read a playlist timing CSV back into validated rows."""
    import csv

    with open(path, encoding="utf-8", newline="") as handle:
        return [TimingRow.model_validate(row) for row in csv.DictReader(handle)]


def _chirp_template(native_rate_hz: int) -> np.ndarray:
    if native_rate_hz == PLAYLIST_RATE_HZ:
        return sync_chirp()
    return audio.resample_between(sync_chirp(), PLAYLIST_RATE_HZ, native_rate_hz)


def find_start_chirp(
    recording: np.ndarray, native_rate_hz: int, playlist_len_play: int
) -> int | None:
    """Sample offset of the start chirp in the recording, or None when unclear.

    The earliest near-maximum is the start chirp (the end chirp matches just
    as well but comes later). Clarity means the peak beats everything outside
    both chirp regions by `CHIRP_CLARITY_RATIO`.
    """
    rec = np.asarray(recording, dtype=np.float64).reshape(-1)
    tmpl = _chirp_template(native_rate_hz).astype(np.float64)
    if rec.shape[0] < tmpl.shape[0]:
        return None
    corr = scipy_signal.correlate(rec, tmpl, mode="valid", method="fft")
    peak = float(np.max(corr))
    if peak <= 0.0:
        return None
    first = int(np.argmax(corr >= peak * 10.0 ** (-CHIRP_FIRST_PEAK_DB / 10.0)))
    half = tmpl.shape[0]
    lo = max(0, first - half)
    start = lo + int(np.argmax(corr[lo : first + half]))
    ratio = native_rate_hz / PLAYLIST_RATE_HZ
    end_offset = int(round(playlist_len_play * ratio)) - half
    mask = np.ones(corr.shape[0], dtype=bool)
    mask[max(0, start - half) : start + half] = False
    mask[max(0, start + end_offset - half) : start + end_offset + half] = False
    background = float(np.max(np.abs(corr[mask]))) if mask.any() else 0.0
    if background <= 0.0 or float(corr[start]) < CHIRP_CLARITY_RATIO * background:
        return None
    return start


def end_chirp_drift_ms(
    recording: np.ndarray,
    start_sample: int,
    end_start_play: int,
    native_rate_hz: int,
) -> float:
    """How far the end chirp lands from where the playlist says it should (ms, signed)."""
    rec = np.asarray(recording, dtype=np.float64).reshape(-1)
    tmpl = _chirp_template(native_rate_hz).astype(np.float64)
    expected = start_sample + int(round(end_start_play * native_rate_hz / PLAYLIST_RATE_HZ))
    window = int(round(END_SEARCH_MS / 1000.0 * native_rate_hz))
    lo = max(0, expected - window)
    seg = rec[lo : expected + window + tmpl.shape[0]]
    if seg.shape[0] < tmpl.shape[0]:
        return float("inf")
    corr = scipy_signal.correlate(seg, tmpl, mode="valid", method="fft")
    found = lo + int(np.argmax(corr))
    return (found - expected) / native_rate_hz * 1000.0


@dataclass
class AlignedBlock:
    """A block recording cut into takes: one native-rate segment per timing row."""

    start_sample: int  # start chirp offset, in recording samples
    drift_ms: float
    take_starts: list[int]  # cut offset per timing row, in recording samples
    segments: list[np.ndarray]  # lead-in included, at the recorded rate
    warnings: list[str] = field(default_factory=list)


def align_block(
    recording: np.ndarray,
    native_rate_hz: int,
    rows: list[TimingRow],
    lead_in_s: float,
    playlist_len_play: int,
) -> AlignedBlock | None:
    """Align a block recording to its playlist and cut each clip's segment.

    Returns None when the start chirp isn't found clearly: the caller must
    save nothing for the block and say so.
    """
    rec = np.asarray(recording, dtype=np.float32).reshape(-1)
    start = find_start_chirp(rec, native_rate_hz, playlist_len_play)
    if start is None:
        return None
    end_start_play = (
        rows[-1].start_sample + rows[-1].clip_samples + int(round(CLIP_GAP_S * PLAYLIST_RATE_HZ))
    )
    drift_ms = end_chirp_drift_ms(rec, start, end_start_play, native_rate_hz)
    warnings: list[str] = []
    if abs(drift_ms) > END_DRIFT_WARN_MS:
        warnings.append(
            f"end chirp drift {drift_ms:.1f} ms exceeds {END_DRIFT_WARN_MS:.0f} ms "
            "(clock drift or a pause); takes still saved"
        )
    ratio = native_rate_hz / PLAYLIST_RATE_HZ
    lead_play = int(round(lead_in_s * PLAYLIST_RATE_HZ))
    take_starts: list[int] = []
    segments: list[np.ndarray] = []
    for row in rows:
        cut = start + int(round((row.start_sample - lead_play) * ratio))
        n = int(round((lead_play + row.clip_samples) * ratio))
        seg = rec[cut : cut + n]
        if seg.shape[0] < n:  # recording ended early: pad, never shift
            seg = np.concatenate([seg, np.zeros(n - seg.shape[0], dtype=np.float32)])
        take_starts.append(cut)
        segments.append(seg.astype(np.float32))
    return AlignedBlock(
        start_sample=start,
        drift_ms=drift_ms,
        take_starts=take_starts,
        segments=segments,
        warnings=warnings,
    )


def _above_noise_frames(
    speech_part: np.ndarray, noise_power: float, sample_rate_hz: int
) -> np.ndarray:
    """The speech region, found by energy: 20 ms frames clearing the noise floor.

    Pauses filled with room/TV noise would otherwise fool the 40 dB gate inside
    `active_speech_power` (they sit far less than 40 dB below speech), so they
    are removed first. Dropping a speech frame or two does not bias the mean;
    keeping noise-only frames would.
    """
    x = np.asarray(speech_part, dtype=np.float64).reshape(-1)
    frame_length = max(1, int(round(audio.FRAME_MS / 1000.0 * sample_rate_hz)))
    n_frames = x.shape[0] // frame_length
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)
    frames = x[: n_frames * frame_length].reshape(n_frames, frame_length)
    threshold = noise_power * 10.0 ** (ENERGY_GATE_DB / 10.0)
    kept = frames[np.mean(frames**2, axis=1) >= threshold]
    return kept.reshape(-1)


def estimate_snr_db(
    recording: np.ndarray, lead_in_s: float, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> float | None:
    """SNR of a recording from its noise-only lead-in.

    Noise power comes from the lead-in after skipping its first 0.3 s; the
    speech level uses :func:`earbench.audio.active_speech_power` on the
    energy-found speech region — the same function `noise.mix` uses, so
    simulated and real-room SNRs are measured the same way. Returns None when
    the SNR is not estimable (silent lead-in or no speech above the noise).
    """
    x = np.asarray(recording, dtype=np.float64).reshape(-1)
    n_lead = int(round(lead_in_s * sample_rate_hz))
    n_skip = int(round(LEAD_IN_SKIP_S * sample_rate_hz))
    if x.shape[0] <= n_lead or n_lead - n_skip <= 0:
        return None
    noise_power = float(np.mean(x[n_skip:n_lead] ** 2))
    if noise_power <= 0.0:
        return None
    gated = _above_noise_frames(x[n_lead:], noise_power, sample_rate_hz)
    if gated.shape[0] == 0:
        return None
    region_power = active_speech_power(gated, sample_rate_hz)
    if region_power <= noise_power:
        return None
    return float(10.0 * np.log10((region_power - noise_power) / noise_power))


def take_filename(clip_id: str, distance_m: float, condition: RoomCondition, take: int) -> str:
    """Never-overwrite take name: `<clip>_<dist>m_<cond>_t<take>.wav`."""
    return f"{clip_id}_{distance_m:g}m_{condition}_t{take}.wav"


def read_session(path: str | Path) -> SessionFile:
    """Read session.yaml back into a validated SessionFile (errors name file + field)."""
    return load_config(path, SessionFile)


def write_session(session: SessionFile, path: str | Path) -> Path:
    """Write session.yaml next to the recordings."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(session.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    return out_path


def resolve_session_dir(
    session: str | Path, recordings_dir: str | Path = Path("recordings")
) -> Path:
    """A session dir directly, or an id under the recordings root."""
    direct = Path(session)
    if direct.is_dir():
        return direct
    under_root = Path(recordings_dir) / direct.name
    if under_root.is_dir():
        return under_root
    raise ValueError(
        f"session not found: {session} (looked in {direct} and {under_root}; "
        "run `earbench record` first)"
    )


def _ask_float(prompt: str, allow_blank: bool, input_fn: Callable[[str], str]) -> float | None:
    """A typed number. Blank is only accepted where the protocol allows it (TV level)."""
    while True:
        raw = input_fn(prompt).strip()
        if not raw and allow_blank:
            return None
        try:
            return float(raw)
        except ValueError:
            print(f"enter a number{', or blank when not applicable' if allow_blank else ''}")


def _ask_text(prompt: str, input_fn: Callable[[str], str]) -> str:
    """A typed line. Blank is never accepted: the human types every entry."""
    while True:
        raw = input_fn(prompt).strip()
        if raw:
            return raw
        print("type a value; nothing is filled in for you")


def _ask_position(
    label: str, fields: tuple[str, str, str], input_fn: Callable[[str], str]
) -> tuple[float, float, float]:
    """Three typed metre values, e.g. room length/width/height or x/y/height."""
    while True:
        try:
            values = tuple(float(input_fn(f"{label} {name} (m): ").strip()) for name in fields)
            return (values[0], values[1], values[2])
        except ValueError:
            print("enter a number for each")


def prompt_new_session(
    cfg: RoomSessionConfig, session_id: str, input_fn: Callable[[str], str] = input
) -> SessionFile:
    """Ask once per session: room, webcam/TV positions, TV-noise details, device labels."""
    print("Measure the room (metres) before starting.")
    print("Positions: x from the wall to the PC's left, y from the wall behind the PC.")
    room_dims = _ask_position("Room", ("length", "width", "height"), input_fn)
    webcam_pos = _ask_position("Webcam (mic)", ("x", "y", "height"), input_fn)
    tv_pos = _ask_position("TV", ("x", "y", "height"), input_fn)
    youtube_url = youtube_start = macbook_volume = None
    if "tv" in cfg.conditions:
        youtube_url = _ask_text("YouTube link used as TV noise: ", input_fn)
        youtube_start = _ask_text("YouTube start time (mm:ss): ", input_fn)
        macbook_volume = _ask_text("MacBook volume setting (e.g. 6/16): ", input_fn)
    playback_device = _ask_text("Playback device (e.g. MacBook speakers): ", input_fn)
    mic = _ask_text("Mic (e.g. PC webcam mic): ", input_fn)
    return SessionFile(
        session_id=session_id,
        room_dims_m=room_dims,
        webcam_pos_m=webcam_pos,
        tv_pos_m=tv_pos,
        youtube_url=youtube_url,
        youtube_start=youtube_start,
        macbook_volume=macbook_volume,
        playback_device=playback_device,
        mic=mic,
        input_device=cfg.input_device,
        lead_in_s=cfg.lead_in_s,
        model=cfg.model,
        seed=cfg.seed,
        config_hash=config_hash(cfg),
        manifest_path=cfg.manifest_path.as_posix(),
        playlists_dir=cfg.playlists_dir.as_posix(),
    )


def prompt_block(
    distance_m: float, condition: RoomCondition, input_fn: Callable[[str], str] = input
) -> BlockRecord:
    """Prompt for one block's setup values. Every measurement is typed, never defaulted."""
    print(f"--- set up: {distance_m:g} m, {condition} ---")
    while True:
        try:
            measured = float(input_fn("Measured distance (m): ").strip())
            break
        except ValueError:
            print("enter a number")
    speech_level = _ask_float("Speech level (dBA): ", allow_blank=False, input_fn=input_fn)
    tv_level = _ask_float("TV level (dBA, blank when the TV is off): ", True, input_fn)
    notes = input_fn("Notes: ")
    return BlockRecord(
        distance_m=distance_m,
        condition=condition,
        measured_distance_m=measured,
        speech_level_dba=speech_level,
        tv_level_dba=tv_level,
        notes=notes.strip(),
    )


def _new_session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_block_playlist(
    cfg: RoomSessionConfig, distance_m: float, condition: RoomCondition
) -> tuple[np.ndarray, list[TimingRow]]:
    """A block's playlist audio (16 kHz) and timing rows, built by `make-playlist`."""
    out_dir = Path(cfg.playlists_dir)
    wav_path = out_dir / playlist_filename(distance_m, condition)
    timing_path = wav_path.with_suffix(".csv")
    if not wav_path.is_file() or not timing_path.is_file():
        raise ValueError(
            f"playlist not found: {wav_path} (run `earbench make-playlist --config <room.yaml>`)"
        )
    return audio.load_16k(wav_path), read_timing(timing_path)


def run_session(
    cfg: RoomSessionConfig,
    *,
    session_id: str | None = None,
    backend: AudioBackend,
    input_fn: Callable[[str], str] = input,
    dry_run: bool = False,
) -> Path:
    """Run (or resume) a recording session. Resumed blocks keep stored measurements.

    The PC records the mic continuously while the MacBook plays the block's
    playlist; takes are cut from the aligned recording. A block whose start
    chirp isn't found clearly saves nothing. `--dry-run` aligns a synthetic
    recording instead of touching audio devices.
    """
    session_dir = Path(cfg.recordings_dir) / (session_id or _new_session_id())
    clips_dir = session_dir / CLIPS_DIRNAME
    clips_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / SESSION_FILENAME

    if session_path.is_file():
        session = read_session(session_path)
    else:
        session = prompt_new_session(cfg, session_dir.name, input_fn)
        shutil.copy2(cfg.manifest_path, session_dir / MANIFEST_COPY)
        write_session(session, session_path)

    # The native rate is queried, never assumed; dry runs stay off the hardware.
    native_rate = cfg.sample_rate if dry_run else backend.native_sample_rate
    session.sample_rate = native_rate
    session.input_name = None if dry_run else backend.input_name

    clips = {row.clip_id: row for row in manifest.read_manifest(cfg.manifest_path)}
    for distance_m in cfg.distances_m:
        for condition in cfg.conditions:
            play_16k, timing = _load_block_playlist(cfg, distance_m, condition)
            block = next(
                (
                    b
                    for b in session.blocks
                    if b.distance_m == distance_m and b.condition == condition
                ),
                None,
            )
            if block is None:
                block = prompt_block(distance_m, condition, input_fn)
            fname = playlist_filename(distance_m, condition)
            input_fn(f"Press Enter, then start {fname} on the MacBook: ")
            n_record = int(round(play_16k.shape[0] * native_rate / PLAYLIST_RATE_HZ))
            n_record += int(round(RECORD_SLACK_S * native_rate))
            if dry_run:
                play_native = audio.resample_between(play_16k, PLAYLIST_RATE_HZ, native_rate)
                recorded = np.zeros(n_record, dtype=np.float32)
                recorded[: play_native.shape[0]] = play_native
            else:
                recorded = np.asarray(backend.record_block(n_record), dtype=np.float32).reshape(-1)
            aligned = align_block(recorded, native_rate, timing, cfg.lead_in_s, play_16k.shape[0])
            if aligned is None:
                print(
                    f"start chirp not found clearly in {distance_m:g} m {condition}: "
                    "saving nothing for this block"
                )
                continue
            for warning in aligned.warnings:
                print(f"warning: {warning}")
            if not any(
                b.distance_m == distance_m and b.condition == condition for b in session.blocks
            ):
                session.blocks.append(block)
            for row, segment in zip(timing, aligned.segments, strict=True):
                clip = clips.get(row.clip_id)
                if clip is None:
                    raise ValueError(f"clip {row.clip_id!r} is not in the manifest")
                take_no = (
                    max([t.take for t in block.takes if t.clip_id == row.clip_id], default=0) + 1
                )
                filename = take_filename(row.clip_id, distance_m, condition, take_no)
                while (clips_dir / filename).is_file():  # never overwrite, even after a crash
                    take_no += 1
                    filename = take_filename(row.clip_id, distance_m, condition, take_no)
                audio.save_wav_mono(clips_dir / filename, segment, native_rate)
                block.takes.append(
                    TakeRecord(
                        clip_id=row.clip_id,
                        age_group=clip.age_group,
                        wav_file=f"{CLIPS_DIRNAME}/{filename}",
                        take=take_no,
                        distance_m=distance_m,
                        condition=condition,
                    )
                )
            write_session(session, session_path)
    logger.info("wrote %s: %d blocks", session_dir, len(session.blocks))
    return session_dir


def _session_manifest(session_dir: Path, session: SessionFile) -> dict[str, manifest.ManifestRow]:
    """Clip rows from the session's manifest copy, falling back to the original path."""
    copy = session_dir / MANIFEST_COPY
    source = copy if copy.is_file() else Path(session.manifest_path)
    if not source.is_file():
        raise ValueError(
            f"manifest not found: {source} (session copy and {session.manifest_path} both missing)"
        )
    return {row.clip_id: row for row in manifest.read_manifest(source)}


def _room_noise_type(condition: RoomCondition) -> NoiseType:
    return "none" if condition == "quiet" else "tv"


def score_session(
    session_dir: str | Path,
    transcriber_factory: Callable[[str], Transcriber] | None = None,
) -> list[ResultRow]:
    """Transcribe and score every take into room-source ResultRows.

    Quiet takes get `snr_db=None` (no added noise); TV takes get the lead-in
    estimate, or None when it is not estimable.
    """
    session_path = Path(session_dir)
    session = read_session(session_path / SESSION_FILENAME)
    clips = _session_manifest(session_path, session)
    factory = transcriber_factory or default_transcriber_factory
    transcriber = factory(session.model)
    rows: list[ResultRow] = []
    for block in session.blocks:
        distance_m = block.measured_distance_m or block.distance_m
        for take in block.takes:
            clip = clips.get(take.clip_id)
            if clip is None:
                raise ValueError(f"clip {take.clip_id!r} is not in the session manifest")
            recording_16k = audio.load_16k(session_path / take.wav_file)
            noise_type = _room_noise_type(take.condition)
            snr_db = (
                None if noise_type == "none" else estimate_snr_db(recording_16k, session.lead_in_s)
            )
            hypothesis = transcriber.transcribe(recording_16k, SAMPLE_RATE_HZ)
            scored = scoring.score_clip(clip.sentence, hypothesis)
            rows.append(
                ResultRow(
                    run_id=session.session_id,
                    source="room",
                    clip_id=take.clip_id,
                    age_group=clip.age_group,
                    distance_m=distance_m,
                    noise_type=noise_type,
                    snr_db=snr_db,
                    model=session.model,
                    model_version=transcriber.version,
                    settings=transcriber.settings,
                    reference=scored.reference,
                    hypothesis=scored.hypothesis,
                    ref_words=scored.ref_words,
                    errors=scored.errors,
                    wer=scored.wer,
                    config_hash=session.config_hash,
                )
            )
    write_csv(ResultRow, rows, session_path / RESULTS_FILENAME)
    write_csv(
        SummaryRow,
        _summarise(rows, USABLE_WER, BOOTSTRAP_ITERS, session.seed),
        session_path / SUMMARY_FILENAME,
    )
    logger.info("scored %s: %d takes", session_path, len(rows))
    return rows


def _summarise(
    rows: list[ResultRow], usable_wer: float, bootstrap_iters: int, seed: int
) -> list[SummaryRow]:
    """One pooled row per (model, age group, distance, noise, SNR), like the sweep."""
    groups: dict[tuple[str, AgeGroup, float, NoiseType, float | None], list[ResultRow]] = {}
    for row in rows:
        key = (row.model, row.age_group, row.distance_m, row.noise_type, row.snr_db)
        groups.setdefault(key, []).append(row)
    summary: list[SummaryRow] = []
    for (model, age_group, distance_m, noise_type, snr_db), group in groups.items():
        ci_low, ci_high = scoring.bootstrap_ci(group, bootstrap_iters, seed)
        summary.append(
            SummaryRow(
                model=model,
                age_group=age_group,
                distance_m=distance_m,
                noise_type=noise_type,
                snr_db=snr_db,
                n_clips=len(group),
                errors=sum(row.errors for row in group),
                ref_words=sum(row.ref_words for row in group),
                wer=scoring.corpus_wer(group),
                ci_low=ci_low,
                ci_high=ci_high,
                usable_rate=scoring.usable_rate(group, usable_wer),
            )
        )
    return summary


def sim_room_config(session: SessionFile, fallback: RoomConfig) -> tuple[RoomConfig, str]:
    """The simulation room from the typed webcam/TV positions, or the fallback.

    The voice source sits straight out from the webcam (along +y) at the
    block's distance; see :func:`voice_position`. Sessions without positions
    (older ones) fall back to the sweep config's room, reported in the note.
    """
    if session.room_dims_m is not None and session.webcam_pos_m is not None:
        if session.tv_pos_m is not None:
            return (
                RoomConfig(
                    dims_m=session.room_dims_m,
                    mic_pos_m=session.webcam_pos_m,
                    noise_pos_m=session.tv_pos_m,
                    rt60_s=fallback.rt60_s,
                    source_height_m=fallback.source_height_m,
                ),
                "",
            )
    missing = [
        name
        for name, value in (
            ("room_dims_m", session.room_dims_m),
            ("webcam_pos_m", session.webcam_pos_m),
            ("tv_pos_m", session.tv_pos_m),
        )
        if value is None
    ]
    note = (
        f"fallback: session {session.session_id} has no {', '.join(missing)}, "
        f"using the sweep config room {tuple(fallback.dims_m)} m"
    )
    logger.warning("%s", note)
    return fallback, note


def voice_position(session: SessionFile, distance_m: float) -> tuple[float, float, float]:
    """Where the voice plays in the simulation: out from the webcam along +y."""
    if session.webcam_pos_m is None:
        raise ValueError(f"session {session.session_id} has no webcam position")
    wx, wy, wh = session.webcam_pos_m
    return (wx, wy + distance_m, wh)


def compare_session(
    session_dir: str | Path,
    sweep_cfg: SweepConfig,
    transcriber_factory: Callable[[str], Transcriber] | None = None,
) -> tuple[list[CompareRow], str]:
    """Simulate each real take's condition (same clip, distance, TV noise at the
    estimated SNR) and pair real vs simulated WER.

    Reads `results.csv`, so `score-room` must run first. The sweep config must
    name exactly the model the recordings were scored with. The simulation uses
    the typed room size, webcam and TV positions; sessions without them fall
    back to the sweep config's room (reported in the returned note).
    """
    session_path = Path(session_dir)
    session = read_session(session_path / SESSION_FILENAME)
    results_path = session_path / RESULTS_FILENAME
    if not results_path.is_file():
        raise ValueError(
            f"no {RESULTS_FILENAME} in {session_path}; run `earbench score-room` first"
        )
    room_rows = read_results(results_path)
    if not room_rows:
        raise ValueError(f"no scored takes in {results_path}")
    room_models = sorted({row.model for row in room_rows})
    if len(sweep_cfg.models) != 1 or len(room_models) != 1 or sweep_cfg.models[0] != room_models[0]:
        raise ValueError(
            f"cannot compare: sweep config models {sweep_cfg.models} differ from "
            f"the recorded model {room_models} (score-room used {room_models})"
        )
    model = room_models[0]

    room, note = sim_room_config(session, sweep_cfg.room)

    clips = _session_manifest(session_path, session)
    tv_noise: np.ndarray | None = None
    if any(row.noise_type == "tv" and row.snr_db is not None for row in room_rows):
        tv_noise = sweep_mod.load_noise(sweep_cfg, "tv")
    factory = transcriber_factory or default_transcriber_factory
    transcriber = factory(model)

    sim_cache: dict[tuple[str, float, NoiseType, float | None], CompareRow] = {}
    paired: list[CompareRow] = []
    for row in room_rows:
        key = (row.clip_id, row.distance_m, row.noise_type, row.snr_db)
        cached = sim_cache.get(key)
        if cached is None:
            clip = clips.get(row.clip_id)
            if clip is None:
                raise ValueError(f"clip {row.clip_id!r} is not in the session manifest")
            if session.webcam_pos_m is not None:
                speech = room_sim.simulate_at(
                    audio.load_16k(clip.wav_path), voice_position(session, row.distance_m), room
                )
            else:  # older sessions without positions: distance along +x from the sweep mic
                speech = room_sim.simulate(audio.load_16k(clip.wav_path), row.distance_m, room)
            if row.noise_type == "none" or row.snr_db is None or tv_noise is None:
                mixed = speech
            else:
                seed = _condition_seed(session.seed, row.clip_id, row.noise_type)
                mixed = noise.mix(
                    speech, noise.noise_at_mic(tv_noise, speech.shape[0], seed, room), row.snr_db
                )
            sim_hypothesis = transcriber.transcribe(mixed, SAMPLE_RATE_HZ)
            sim_scored = scoring.score_clip(clip.sentence, sim_hypothesis)
            cached = CompareRow(
                clip_id=row.clip_id,
                age_group=row.age_group,
                distance_m=row.distance_m,
                noise_type=row.noise_type,
                snr_db=row.snr_db,
                room_wer=0.0,  # filled per take below; the sim side is cached per condition
                sim_wer=sim_scored.wer,
                room_hypothesis="",
                sim_hypothesis=sim_hypothesis,
            )
            sim_cache[key] = cached
        paired.append(
            CompareRow(
                clip_id=row.clip_id,
                age_group=row.age_group,
                distance_m=row.distance_m,
                noise_type=row.noise_type,
                snr_db=row.snr_db,
                room_wer=row.wer,
                sim_wer=cached.sim_wer,
                room_hypothesis=row.hypothesis,
                sim_hypothesis=cached.sim_hypothesis,
            )
        )
    write_csv(CompareRow, paired, session_path / COMPARE_FILENAME)
    _write_compare_plot(paired, session.session_id, session_path / COMPARE_PNG)
    logger.info("compared %s: %d takes", session_path, len(paired))
    return paired, note


def _write_compare_plot(paired: list[CompareRow], session_id: str, out_path: Path) -> Path:
    """Scatter of simulated vs real-room WER, with the x=y line for reference."""
    xs = [100.0 * row.sim_wer for row in paired]
    ys = [100.0 * row.room_wer for row in paired]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys)
    top = max([*xs, *ys, 5.0])
    ax.plot([0.0, top], [0.0, top], linestyle="--", color="grey")
    ax.set_xlabel("Simulated WER (%)")
    ax.set_ylabel("Real-room WER (%)")
    ax.set_title(f"Real vs simulated ({session_id})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
