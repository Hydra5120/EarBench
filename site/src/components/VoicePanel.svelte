<script lang="ts">
  import { onDestroy } from "svelte";
  import { NOISE_ICON, noiseVsVoice, type SiteIndex } from "../lib/data";
  import { RATE, decode16k, mixAtSnr, normalise, wavUrl } from "../live/mix";
  import {
    pickDevice,
    plainError,
    preloadModel,
    preloadWhenIdle,
    transcribe,
    type Device,
    type Stage,
  } from "../live/live";

  let { index }: { index: SiteIndex } = $props();

  const LIMIT = index.voice_seconds;
  const noises = index.noise_types.filter((n) => n.bed);
  const levels = index.levels.filter((l) => l.snr_db !== null) as { snr_db: number; label: string }[];
  const sentences = index.clips.map((c) => c.sentence);

  let noise = $state(noises[0].id);
  let levelIdx = $state(Math.max(0, levels.findIndex((l) => l.snr_db === 5)));
  const level = $derived(levels[levelIdx]);
  const noiseOpt = $derived(noises.find((n) => n.id === noise) ?? noises[0]);
  let sentenceIdx = $state(0);

  // Recording
  let phase = $state<"idle" | "asking" | "recording" | "decoding" | "ready">("idle");
  let elapsed = $state(0);
  let meter = $state(0); // input level, 0..1
  let micError = $state("");
  let recId = $state(0);
  let voice: Float32Array | null = null; // normalised recording, 16 kHz mono
  let voiceUrl = $state<string | null>(null);
  let stopRecording: () => void = () => {};

  // Mix
  let mix = $state<{ key: string; samples: Float32Array; url: string } | null>(null);
  const beds = new Map<string, Promise<Float32Array>>();
  const mixKey = $derived(`${recId}|${noise}|${level.snr_db}`);
  const cleanKey = $derived(`${recId}|clean`);

  // Whisper
  let texts = $state<Record<string, string>>({});
  let busy = $state(false);
  let stage = $state<Stage | null>(null);
  let loaded = $state(0);
  let total = $state(0);
  let device = $state<Device | null>(null);
  let whisperError = $state("");
  let noiseError = $state("");
  let noiseRetry = $state(0);
  let again = false;
  let abort: AbortController | null = null;

  const mb = (n: number) => (n / 1e6).toFixed(n < 1e7 ? 1 : 0);

  async function startRecording() {
    micError = "";
    void preloadModel(); // the model downloads while you talk
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      micError = "This browser can't record audio here.";
      return;
    }
    phase = "asking";
    let stream: MediaStream;
    try {
      // Raw microphone: no noise suppression, so Whisper hears your room as it is.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : "";
      micError =
        name === "NotAllowedError"
          ? "The microphone is blocked. Allow it from the address bar, then try again."
          : name === "NotFoundError"
            ? "I couldn't find a microphone."
            : "The microphone didn't start. Try again.";
      phase = voice ? "ready" : "idle";
      return;
    }

    const recorder = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => chunks.push(e.data);
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const frame = new Float32Array(analyser.fftSize);
    const started = performance.now();
    let raf = 0;
    const hardStop = setTimeout(() => stopRecording(), LIMIT * 1000);

    stopRecording = () => {
      clearTimeout(hardStop);
      cancelAnimationFrame(raf);
      if (recorder.state !== "inactive") recorder.stop();
    };
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
      meter = 0;
      phase = "decoding";
      try {
        const raw = await decode16k(await new Blob(chunks, { type: recorder.mimeType }).arrayBuffer());
        const samples = normalise(raw.subarray(0, Math.round(LIMIT * RATE)));
        if (samples.length < RATE * 0.3) throw new Error("too short");
        voice = samples;
        if (voiceUrl) URL.revokeObjectURL(voiceUrl);
        voiceUrl = wavUrl(samples);
        recId += 1;
        phase = "ready";
        void runWhisper(); // the clean version starts now; the noisy one once it's mixed
      } catch {
        micError = "That recording was too short or couldn't be read. Try again.";
        phase = voice ? "ready" : "idle";
      }
    };

    const tick = () => {
      elapsed = Math.min(LIMIT, (performance.now() - started) / 1000);
      analyser.getFloatTimeDomainData(frame);
      let sum = 0;
      for (const v of frame) sum += v * v;
      meter = Math.min(1, Math.sqrt(sum / frame.length) * 4);
      raf = requestAnimationFrame(tick);
    };
    elapsed = 0;
    recorder.start();
    phase = "recording";
    raf = requestAnimationFrame(tick);
  }

  function bed(id: string): Promise<Float32Array> {
    let pending = beds.get(id);
    if (!pending) {
      const path = noises.find((n) => n.id === id)?.bed;
      // An MP3 inside JSON: download-manager extensions grab fetched .mp3 files.
      pending = fetch(`/data/${path}`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json() as Promise<{ mp3_base64: string }>;
        })
        .then(({ mp3_base64 }) => decode16k(Uint8Array.from(atob(mp3_base64), (c) => c.charCodeAt(0)).buffer));
      pending.catch(() => beds.delete(id));
      beds.set(id, pending);
    }
    return pending;
  }

  // Remix whenever the recording, the noise or the level changes, then transcribe what's new.
  $effect(() => {
    const key = mixKey;
    const nt = noise;
    const snr = level.snr_db;
    void noiseRetry; // "Try again" re-runs this
    if (recId === 0 || !voice) return;
    const speech = voice;
    let stale = false;
    noiseError = "";
    bed(nt).then(
      (noiseSamples) => {
        if (stale) return;
        if (mix) URL.revokeObjectURL(mix.url);
        const samples = mixAtSnr(speech, noiseSamples, snr);
        mix = { key, samples, url: wavUrl(samples) };
        void runWhisper();
      },
      () => {
        if (!stale) noiseError = "The background noise couldn't be loaded, so only your own recording was transcribed.";
      },
    );
    return () => {
      stale = true;
    };
  });

  async function runWhisper() {
    if (busy) {
      again = true;
      return;
    }
    const jobs: [string, Float32Array][] = [];
    if (voice && !(cleanKey in texts)) jobs.push([cleanKey, voice]);
    if (mix && mix.key === mixKey && !(mix.key in texts)) jobs.push([mix.key, mix.samples]);
    if (!jobs.length) return;
    busy = true;
    whisperError = "";
    loaded = total = 0;
    abort = new AbortController();
    try {
      device = await pickDevice();
      const out = await transcribe(
        jobs.map(([, samples]) => samples.slice()), // the worker takes ownership of copies
        device,
        (u) => {
          stage = u.stage;
          if (u.total) {
            loaded = u.loaded ?? 0;
            total = u.total;
          }
        },
        abort.signal,
      );
      jobs.forEach(([key], i) => (texts[key] = out.texts[i]));
    } catch (err) {
      const cancelled = err instanceof DOMException && err.name === "AbortError";
      whisperError = cancelled ? "Cancelled." : plainError(err instanceof Error ? err.message : String(err));
      again = false;
    } finally {
      busy = false;
      stage = null;
      abort = null;
    }
    if (again) {
      again = false;
      void runWhisper();
    }
  }

  // Wide screens start the model download once the panel is on screen and the page is idle.
  $effect(() => preloadWhenIdle());

  onDestroy(() => {
    stopRecording();
    abort?.abort();
    if (voiceUrl) URL.revokeObjectURL(voiceUrl);
    if (mix) URL.revokeObjectURL(mix.url);
  });

  const recording = $derived(phase === "recording");
  const cleanText = $derived(texts[cleanKey]);
  const mixText = $derived(mix && mix.key === mixKey ? texts[mix.key] : undefined);
</script>

<div class="panel card" data-testid="voice-panel">
  <div class="controls">
    <fieldset class="group">
      <legend class="group-title">Background noise</legend>
      <div class="tabs tabs-noise" style={`--n:${noises.length};--i:${noises.findIndex((n) => n.id === noise)}`}>
        {#each noises as n (n.id)}
          <label class:active={n.id === noise}>
            <input type="radio" name="voice-noise" value={n.id} bind:group={noise} />
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" class="tab-icon">
              <path d={NOISE_ICON[n.id]} />
            </svg>
            <span>{n.label}</span>
          </label>
        {/each}
        <span class="tab-indicator" aria-hidden="true"></span>
      </div>
    </fieldset>

    <fieldset class="group">
      <legend class="group-title">Noise level <span class="sub-label">compared with your voice</span></legend>
      <div class="tabs" style={`--n:${levels.length};--i:${levelIdx}`}>
        {#each levels as l, i (l.snr_db)}
          <label class:active={i === levelIdx}>
            <input type="radio" name="voice-level" value={i} bind:group={levelIdx} />
            <span>{noiseVsVoice(l.snr_db)}</span>
          </label>
        {/each}
        <span class="tab-indicator" aria-hidden="true"></span>
      </div>
      <p class="sub-label level-label" aria-live="polite">{level.label.replace("the speaker", "you")}</p>
    </fieldset>

    <div class="group">
      <span class="group-title">Something to say</span>
      <p class="sentence">"{sentences[sentenceIdx]}"</p>
      <button
        type="button"
        class="text-button"
        onclick={() => (sentenceIdx = (sentenceIdx + 1) % sentences.length)}
      >
        Another sentence
      </button>
    </div>
  </div>

  <div class="output">
    <div class="record-row">
      <button
        type="button"
        class="record"
        class:on={recording}
        style={`--p:${recording ? elapsed / LIMIT : 0};--m:${meter}`}
        onclick={recording ? stopRecording : startRecording}
        onpointerenter={() => void preloadModel()}
        disabled={phase === "asking" || phase === "decoding"}
        aria-label={recording ? "Stop recording" : "Record"}
        data-testid="record"
      >
        {#if recording}
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" />
          </svg>
        {:else}
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
            <path d="M6 11a6 6 0 0 0 12 0M12 17v3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" />
          </svg>
        {/if}
      </button>
      <div class="record-text" aria-live="polite">
        {#if recording}
          <strong>Recording</strong>
          <span data-testid="countdown">{`${Math.max(0, LIMIT - elapsed).toFixed(1)} s left`}</span>
        {:else if phase === "asking"}
          <strong>Waiting for the microphone</strong>
          <span>Your browser will ask first.</span>
        {:else if phase === "decoding"}
          <strong>Got it</strong>
          <span>&nbsp;</span>
        {:else}
          <strong>{voice ? "Record again" : "Record"}</strong>
          <span>{`Up to ${LIMIT} seconds. It stays on your device.`}</span>
        {/if}
      </div>
    </div>

    {#if micError}
      <p class="error" role="alert" data-testid="mic-error">{micError}</p>
    {/if}

    {#if phase === "ready" || (voice && phase !== "recording")}
      <div class="results">
        <div class="result">
          <div class="result-head">
            <span class="line-label">As you recorded it</span>
            {#if voiceUrl}<audio controls src={voiceUrl} preload="auto" data-testid="voice-audio"></audio>{/if}
          </div>
          <p class="words" data-testid="voice-clean">
            {#if cleanText !== undefined}{cleanText || "(nothing)"}{:else if !whisperError}<span class="pending">…</span>{/if}
          </p>
        </div>
        <div class="result">
          <div class="result-head">
            <span class="line-label">{`With ${noiseOpt.label.toLowerCase()} noise at ${noiseVsVoice(level.snr_db)}`}</span>
            {#if mix && mix.key === mixKey}<audio controls src={mix.url} preload="auto" data-testid="voice-mix"></audio>{/if}
          </div>
          <p class="words" data-testid="voice-noisy">
            {#if mixText !== undefined}{mixText || "(nothing)"}{:else if !whisperError && !noiseError}<span class="pending">…</span>{/if}
          </p>
        </div>
      </div>

      {#if busy}
        <div class="loading" role="status" aria-live="polite">
          <p class="stage" data-testid="voice-stage">
            {#if stage === "download"}
              Downloading Whisper tiny{total ? ` (${mb(loaded)} of ${mb(total)} MB, one-time)` : ""}
            {:else if stage === "starting"}
              Starting up
            {:else}
              Transcribing
            {/if}
          </p>
          <div
            class="bar"
            class:indeterminate={stage !== "download" || !total}
            role="progressbar"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={stage === "download" && total ? Math.round((loaded / total) * 100) : undefined}
          >
            <span style={`width:${stage === "download" && total ? (loaded / total) * 100 : 100}%`}></span>
          </div>
          <div class="loading-foot">
            <span>{device === "webgpu" ? "Running on your GPU" : device ? "Running on your CPU" : ""}</span>
            <button type="button" class="text-button" onclick={() => abort?.abort()}>Cancel</button>
          </div>
        </div>
      {:else if whisperError}
        <p class="error" role="alert" data-testid="voice-error">
          {`Whisper didn't run. ${whisperError}`}
          <button type="button" class="text-button" onclick={() => void runWhisper()}>Try again</button>
        </p>
      {/if}
      {#if noiseError}
        <p class="error" role="alert" data-testid="noise-error">
          {noiseError}
          <button type="button" class="text-button" onclick={() => noiseRetry++}>Try again</button>
        </p>
      {/if}
    {:else if phase === "idle" && !micError}
      <p class="prompt">Press record and say something. Whisper tiny transcribes it twice, with and without the noise.</p>
    {/if}

    <p class="fine">
      Runs Whisper tiny in your browser. Your microphone and room are part of the test: I add
      the noise only.
    </p>
  </div>
</div>

<style>
  .panel {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    overflow: hidden;
    margin-inline: calc((100% - min(100% + 120px, 100vw - 32px)) / 2);
  }
  @media (min-width: 960px) {
    .panel {
      grid-template-columns: minmax(0, 0.8fr) minmax(0, 1.2fr);
      min-height: 420px;
    }
  }
  .controls {
    container-type: inline-size;
    padding: clamp(20px, 3.4vw, 40px);
    display: grid;
    gap: 28px;
    align-content: start;
    border-bottom: 1px solid var(--hairline);
  }
  @media (min-width: 960px) {
    .controls {
      border-bottom: 0;
      border-right: 1px solid var(--hairline);
    }
  }
  fieldset {
    border: 0;
    margin: 0;
    padding: 0;
    min-width: 0;
  }
  .group {
    display: grid;
    gap: 10px;
  }
  .group > .text-button {
    justify-self: start;
  }
  fieldset.group {
    display: block;
  }
  .group-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    padding: 0;
  }
  legend.group-title {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    width: 100%;
    margin-bottom: 12px;
  }
  .sub-label {
    font-size: 0.82rem;
    font-weight: 400;
    color: var(--muted);
  }
  .level-label {
    margin: 10px 0 0;
  }
  input[type="radio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  /* Underline tabs with a sliding indicator, as in the panel above. */
  .tabs {
    position: relative;
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(0, 1fr);
    border-bottom: 1px solid var(--hairline);
  }
  .tabs label {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 7px;
    min-height: 44px;
    padding: 8px 6px 10px;
    cursor: pointer;
    font-size: 0.9rem;
    color: var(--muted);
    border-radius: 8px 8px 0 0;
    font-variant-numeric: tabular-nums;
    transition:
      color 0.2s var(--ease),
      background-color 0.2s var(--ease);
  }
  .tabs label:hover {
    color: var(--ink);
    background: color-mix(in oklab, var(--surface-2) 60%, transparent);
  }
  .tabs label.active {
    color: var(--ink);
    font-weight: 600;
  }
  .tabs label:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
  .tab-indicator {
    position: absolute;
    left: 0;
    bottom: -1px;
    width: calc(100% / var(--n));
    height: 2px;
    background: var(--ink);
    border-radius: 2px;
    transform: translateX(calc(var(--i) * 100%));
    transition: transform 0.35s var(--ease);
  }
  .tab-icon {
    flex: none;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .tabs-noise label {
    flex-direction: column;
    gap: 4px;
    white-space: nowrap;
  }
  .sentence {
    margin: 0;
    font-family: var(--serif);
    font-size: 1.1rem;
    line-height: 1.45;
    color: var(--ink);
  }

  .output {
    padding: clamp(20px, 3.4vw, 40px);
    display: grid;
    gap: 20px;
    align-content: start;
    min-width: 0;
  }
  .record-row {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .record {
    --p: 0;
    --m: 0;
    flex: none;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 0;
    cursor: pointer;
    display: grid;
    place-items: center;
    color: var(--on-accent);
    background:
      radial-gradient(closest-side, var(--accent) 86%, transparent 88% 100%),
      conic-gradient(var(--accent-ink) calc(var(--p) * 1turn), var(--surface-3) 0);
    box-shadow: 0 6px 16px -6px color-mix(in oklab, var(--accent) 60%, transparent);
    transition:
      transform 0.2s var(--ease),
      box-shadow 0.1s linear;
  }
  .record:hover {
    transform: scale(1.04);
  }
  .record:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 3px;
  }
  .record:disabled {
    cursor: progress;
    opacity: 0.7;
  }
  /* While recording, a soft ring grows with your voice. */
  .record.on {
    box-shadow:
      0 0 0 calc(var(--m) * 14px) color-mix(in oklab, var(--accent) 22%, transparent),
      0 6px 16px -6px color-mix(in oklab, var(--accent) 60%, transparent);
  }
  .record-text {
    display: grid;
    line-height: 1.35;
  }
  .record-text strong {
    font-weight: 600;
    color: var(--ink);
  }
  .record-text span {
    font-size: 0.85rem;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }
  .results {
    display: grid;
    gap: 14px;
  }
  .result {
    display: grid;
    gap: 8px;
    padding: 14px 16px;
    border: 1px solid var(--hairline);
    border-radius: var(--radius-md);
    background: var(--surface);
  }
  .result-head {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px 12px;
  }
  .result-head audio {
    height: 32px;
    max-width: 100%;
  }
  .line-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--muted);
  }
  .words {
    margin: 0;
    min-height: 1.45em;
    font-family: var(--serif);
    font-size: 1.15rem;
    line-height: 1.45;
    color: var(--ink);
    overflow-wrap: anywhere;
  }
  .pending {
    color: var(--muted);
  }
  .loading {
    display: grid;
    gap: 8px;
  }
  .stage {
    margin: 0;
    font-size: 0.92rem;
    color: var(--ink);
  }
  .bar {
    height: 4px;
    border-radius: 999px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .bar span {
    display: block;
    height: 100%;
    background: var(--accent);
    border-radius: inherit;
    transition: width 0.3s var(--ease);
  }
  .bar.indeterminate span {
    width: 35% !important;
    animation: slide 1.1s ease-in-out infinite;
  }
  @keyframes slide {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(300%);
    }
  }
  .loading-foot {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
    font-size: 0.8rem;
    color: var(--muted);
  }
  .text-button {
    border: 0;
    background: none;
    padding: 0;
    font: inherit;
    font-size: 0.82rem;
    color: var(--accent-ink);
    cursor: pointer;
    text-decoration: underline;
    text-underline-offset: 0.2em;
  }
  .error {
    margin: 0;
    font-size: 0.9rem;
    color: var(--bad);
  }
  .prompt {
    margin: 0;
    color: var(--muted);
    font-family: var(--serif);
    font-size: 1.15rem;
  }
  .fine {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0;
  }
  @media (prefers-reduced-motion: reduce) {
    .record,
    .tab-indicator,
    .bar span {
      transition: none;
    }
    .bar.indeterminate span {
      animation: none;
    }
  }
</style>
