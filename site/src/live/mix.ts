// Audio helpers for the "record your voice" panel. The SNR is measured the way the
// benchmark measures it (earbench.noise.mix): active-speech power over noise power.

export const RATE = 16_000;
const FRAME = Math.round(0.02 * RATE); // 20 ms frames
const ACTIVE_RANGE_DB = 40; // frames within 40 dB of the loudest count as speech

function power(x: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  return x.length ? sum / x.length : 0;
}

/** Mean square of the frames within 40 dB of the loudest, so pauses don't count. */
export function activeSpeechPower(x: Float32Array): number {
  const n = Math.floor(x.length / FRAME);
  if (n === 0) return power(x);
  const frames = new Float64Array(n);
  for (let f = 0; f < n; f++) frames[f] = power(x.subarray(f * FRAME, (f + 1) * FRAME));
  const loudest = Math.max(...frames);
  if (loudest === 0) return 0;
  const threshold = loudest / 10 ** (ACTIVE_RANGE_DB / 10);
  let sum = 0;
  let count = 0;
  for (const p of frames) {
    if (p >= threshold) {
      sum += p;
      count++;
    }
  }
  return sum / count;
}

/** Scale to a peak of 0.9: a level change only, so playback is comfortable. */
export function normalise(x: Float32Array): Float32Array {
  let peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  const out = new Float32Array(x.length);
  if (peak === 0) return out;
  const gain = 0.9 / peak;
  for (let i = 0; i < x.length; i++) out[i] = x[i] * gain;
  return out;
}

/** Add the noise (looped to length) at `snrDb` below the voice, then normalise. */
export function mixAtSnr(speech: Float32Array, noise: Float32Array, snrDb: number): Float32Array {
  const looped = new Float32Array(speech.length);
  for (let i = 0; i < looped.length; i++) looped[i] = noise[i % noise.length];
  const speechPower = activeSpeechPower(speech);
  const noisePower = power(looped);
  const gain = noisePower > 0 ? Math.sqrt(speechPower / (noisePower * 10 ** (snrDb / 10))) : 0;
  const mixed = new Float32Array(speech.length);
  for (let i = 0; i < mixed.length; i++) mixed[i] = speech[i] + looped[i] * gain;
  return normalise(mixed);
}

/** Decode any browser-supported audio to 16 kHz mono, the rate Whisper expects. */
export async function decode16k(data: ArrayBuffer): Promise<Float32Array> {
  const ctx = new OfflineAudioContext(1, 1, RATE);
  const buffer = await ctx.decodeAudioData(data);
  return buffer.getChannelData(0).slice();
}

/** A playable 16-bit WAV object URL. Revoke it when done. */
export function wavUrl(x: Float32Array): string {
  const out = new DataView(new ArrayBuffer(44 + x.length * 2));
  const text = (at: number, s: string) => [...s].forEach((c, i) => out.setUint8(at + i, c.charCodeAt(0)));
  text(0, "RIFF");
  out.setUint32(4, 36 + x.length * 2, true);
  text(8, "WAVEfmt ");
  out.setUint32(16, 16, true);
  out.setUint16(20, 1, true); // PCM
  out.setUint16(22, 1, true); // mono
  out.setUint32(24, RATE, true);
  out.setUint32(28, RATE * 2, true);
  out.setUint16(32, 2, true);
  out.setUint16(34, 16, true);
  text(36, "data");
  out.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) {
    out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 0x7fff, true);
  }
  return URL.createObjectURL(new Blob([out], { type: "audio/wav" }));
}
