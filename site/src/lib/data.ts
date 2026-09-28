// Shapes of the JSON written by `earbench export-site` (site/public/data/).

export type Op = "ok" | "sub" | "del" | "ins";

export interface Token {
  op: Op;
  ref: string | null;
  hyp: string | null;
}

export interface Transcript {
  reference: string;
  hypothesis: string;
  wer: number;
  errors: number;
  ref_words: number;
  usable: boolean;
  tokens: Token[];
}

export interface Cell {
  noise_type: string;
  snr_db: number | null;
  audio: string | null;
  note?: string;
  results: Record<string, Transcript>; // "<model>|<off|on>"
}

export interface ClipData {
  id: string;
  sentence: string;
  distance_m: number;
  audio_gain_db: number;
  speech_span_s: [number, number]; // measured speech start/end in the clip
  cells: Record<string, Cell>; // cond key: "clean" | "<noise>_<snr>"
}

export interface ClipInfo {
  id: string;
  age_group: "older" | "younger";
  age_bucket: string;
  gender: string;
  duration_s: number;
  sentence: string;
  featured: { noise_type: string; snr_db: number | null };
  data: string;
}

export interface Level {
  snr_db: number | null;
  label: string;
}

export interface NoiseOption {
  id: string;
  label: string;
  audio: boolean;
}

export interface RunInfo {
  role: string;
  run_id: string;
  config_hashes: string[];
  model_versions: string[];
  settings: string[];
}

export interface SiteIndex {
  site_manifest: {
    runs: RunInfo[];
    room_session: string;
    exported_at: string;
    git_commit: string | null;
  };
  distance_m: number;
  usable_wer: number;
  models: string[];
  levels: Level[];
  noise_types: NoiseOption[];
  vad_noise_types: string[];
  tv_note: string;
  clips: ClipInfo[];
}

export interface Point {
  snr_db: number | null;
  wer: number;
  ci_low: number;
  ci_high: number;
  usable_rate: number;
  usable_low: number; // 95% Wilson interval on the usable share
  usable_high: number;
  usable_count: number;
  n_clips: number;
}

export interface Finding {
  id: string;
  title: string;
  series_by: "age_group" | "model" | "noise_type";
  fixed: Record<string, string>;
  series: { key: string; points: Point[] }[];
}

export interface VadPoint {
  wer: number;
  ci_low: number;
  ci_high: number;
  usable_rate: number;
  n_clips: number;
}

export interface Charts {
  hero: {
    model: string;
    noise_type: string;
    snr_db: number | null;
    distance_m: number;
    older: Point;
    younger: Point;
    older_quiet: Point;
    younger_quiet: Point;
  };
  findings: Finding[];
  vad: {
    n_clips: number;
    distance_m: number;
    rows: {
      model: string;
      noise_type: string;
      snr_db: number | null;
      off: VadPoint;
      on: VadPoint;
    }[];
  };
}

export interface RoomRow {
  distance_m: number;
  condition: string;
  noise_type: string;
  n_clips: number;
  mean_snr_db: number | null;
  phone_snr_db: number | null;
  room_wer: number;
  room_low: number;
  room_high: number;
  room_usable: number;
  sim_wer: number;
  sim_low: number;
  sim_high: number;
  sim_usable: number;
  room_usable_low: number;
  room_usable_high: number;
  sim_usable_low: number;
  sim_usable_high: number;
}

export interface Room {
  session_id: string;
  model: string | null;
  label: string;
  rows: RoomRow[];
}

export const condKey = (noise: string, snr: number | null): string =>
  snr === null ? "clean" : `${noise}_${snr}`;

export const pct = (x: number, digits = 0): string => `${(x * 100).toFixed(digits)}%`;

export const snrLabel = (snr: number | null): string => (snr === null ? "No noise" : `${snr} dB`);

// Noise level compared with the voice: the SNR flipped, so louder noise reads as a
// bigger number (-20 dB, -10 dB, ... 0 dB = as loud as the voice).
export const noiseVsVoice = (snr: number | null): string =>
  snr === null ? "Off" : snr === 0 ? "0 dB" : `−${snr} dB`;

const GENDER: Record<string, string> = { female: "woman", male: "man" };
const DECADE: Record<string, string> = {
  twenties: "20s",
  thirties: "30s",
  fourties: "40s",
  sixties: "60s",
  seventies: "70s",
  eighties: "80s",
  nineties: "90s",
};

export const speakerLabel = (clip: ClipInfo): string =>
  `${GENDER[clip.gender] ?? "speaker"}, ${DECADE[clip.age_bucket] ?? clip.age_bucket}`;

export const speakerParts = (clip: ClipInfo): { who: string; age: string } => {
  const who = GENDER[clip.gender] ?? "speaker";
  return { who: who[0].toUpperCase() + who.slice(1), age: DECADE[clip.age_bucket] ?? clip.age_bucket };
};

export const MODEL_LABEL: Record<string, string> = {
  tiny: "tiny",
  base: "base",
  small: "small",
};
