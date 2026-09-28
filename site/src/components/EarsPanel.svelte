<script lang="ts">
  import { flip } from "svelte/animate";
  import { fade, fly } from "svelte/transition";
  import {
    condKey,
    pct,
    noiseVsVoice,
    speakerParts,
    type ClipData,
    type SiteIndex,
    type Token,
  } from "../lib/data";

  let { index }: { index: SiteIndex } = $props();

  const reduceMotion =
    typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dur = (ms: number) => (reduceMotion ? 0 : ms);

  const first = index.clips[0];
  let clipId = $state(first.id);
  let noise = $state(first.featured.noise_type);
  let levelIdx = $state(levelIndex(first.featured.snr_db));
  let model = $state(index.models[index.models.length - 1]);
  let vad = $state(false);
  let clips = $state<Record<string, ClipData>>({});
  let loadError = $state<string | null>(null);

  let audioEl: HTMLAudioElement | undefined = $state();
  let playing = $state(false);
  let progress = $state(0);
  let now = $state(0); // playback position, seconds
  let live = $state(false); // true from play until the clip ends or is reset
  let listened = $state(false); // this audio has been played: show the transcript
  let finished = $state(false); // played to the end: show the score
  // Clips already heard (and heard to the end) stay revealed when you come back to them.
  let heardSrcs = $state<Record<string, true>>({});
  let finishedSrcs = $state<Record<string, true>>({});

  function levelIndex(snr: number | null): number {
    const i = index.levels.findIndex((l) => l.snr_db === snr);
    return i < 0 ? 0 : i;
  }

  const clipInfo = $derived(index.clips.find((c) => c.id === clipId) ?? first);
  const level = $derived(index.levels[levelIdx]);
  const noiseOpt = $derived(index.noise_types.find((n) => n.id === noise) ?? index.noise_types[0]);
  const key = $derived(condKey(noise, level.snr_db));
  const clip = $derived(clips[clipId]);
  const cell = $derived(clip?.cells[key]);
  const vadAvailable = $derived(
    index.vad_noise_types.includes(level.snr_db === null ? "none" : noise),
  );
  const result = $derived(cell?.results[`${model}|${vad && vadAvailable ? "on" : "off"}`]);
  const audioSrc = $derived(cell?.audio ? `/data/${cell.audio}` : null);
  const counts = $derived({
    ins: result?.tokens.filter((t) => t.op === "ins").length ?? 0,
    sub: result?.tokens.filter((t) => t.op === "sub").length ?? 0,
    del: result?.tokens.filter((t) => t.op === "del").length ?? 0,
  });
  const timed = $derived(
    result && clip ? timeline(result.tokens, clip.speech_span_s) : ([] as TimedToken[]),
  );
  // "10 wrong words and 6 invented", in plain words; zero counts are left out.
  const breakdown = $derived.by(() => {
    const parts = [
      counts.sub && `${counts.sub} wrong`,
      counts.del && `${counts.del} dropped`,
      counts.ins && `${counts.ins} invented`,
    ].filter(Boolean) as string[];
    if (parts.length === 0) return "Every word right";
    const joined =
      parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts.at(-1)}`;
    return joined.charAt(0).toUpperCase() + joined.slice(1);
  });
  const saidTokens = $derived(withIds(timed.filter((t) => t.ref !== null), "ref"));
  const heardTokens = $derived(withIds(timed.filter((t) => t.hyp !== null), "hyp"));
  const shown = (t: TimedToken) => !live || t.time <= now;
  // TV has no audio, so its transcript shows straight away.
  const revealText = $derived(!audioSrc || listened || !!heardSrcs[audioSrc]);
  const revealScore = $derived(!audioSrc || finished || !!finishedSrcs[audioSrc]);
  const currentSaid = $derived(
    live ? saidTokens.findLast((t) => t.time <= now && t.time > now - 0.9)?.id : undefined,
  );

  type TimedToken = Token & { time: number };

  // Estimated timing: the benchmark has no word timestamps. Spoken words are spread
  // across the measured speech span by length; Whisper's words share the time of
  // the spoken word they align to, and invented words fill the gaps between them.
  function timeline(tokens: Token[], [start, end]: [number, number]): TimedToken[] {
    const weights = tokens.map((t) => (t.ref ? t.ref.length + 1 : 0));
    const total = weights.reduce((a, b) => a + b, 0) || 1;
    const times: (number | null)[] = [];
    let acc = 0;
    for (let i = 0; i < tokens.length; i++) {
      times.push(tokens[i].ref ? start + (acc / total) * (end - start) : null);
      acc += weights[i];
    }
    for (let i = 0; i < tokens.length; i++) {
      if (times[i] !== null) continue;
      let j = i;
      while (j < tokens.length && times[j] === null) j++;
      const before = i > 0 ? (times[i - 1] as number) : start;
      const after = j < tokens.length ? (times[j] as number) : end + 0.8;
      for (let k = i; k < j; k++) times[k] = before + ((k - i + 1) / (j - i + 1)) * (after - before);
      i = j - 1;
    }
    return tokens.map((t, i) => ({ ...t, time: times[i] as number }));
  }

  // Stable ids so unchanged words stay put and only changed words animate.
  function withIds(tokens: TimedToken[], side: "ref" | "hyp") {
    const seen = new Map<string, number>();
    return tokens.map((t) => {
      const word = (side === "ref" ? t.ref : t.hyp) ?? "";
      const base = `${word}:${t.op}`;
      const n = (seen.get(base) ?? 0) + 1;
      seen.set(base, n);
      return { ...t, word, id: `${base}:${n}` };
    });
  }

  async function loadClip(id: string) {
    if (clips[id]) return;
    loadError = null;
    const info = index.clips.find((c) => c.id === id);
    if (!info) return;
    try {
      const res = await fetch(`/data/${info.data}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      clips[id] = (await res.json()) as ClipData;
    } catch (err) {
      loadError = `Couldn't load this speaker's results (${(err as Error).message}).`;
    }
  }

  function pickSpeaker(id: string) {
    const info = index.clips.find((c) => c.id === id);
    if (!info) return;
    clipId = id;
    noise = info.featured.noise_type;
    levelIdx = levelIndex(info.featured.snr_db);
    stop();
    void loadClip(id);
  }

  $effect(() => {
    void loadClip(clipId);
  });

  // Swap audio when the condition changes; keep playing from the same moment.
  let lastSrc: string | null = null;
  $effect(() => {
    const src = audioSrc;
    const el = audioEl;
    if (!el || src === lastSrc) return;
    const wasPlaying = playing;
    const t = el.currentTime;
    lastSrc = src;
    finished = false;
    if (!wasPlaying) listened = false;
    if (!src) {
      el.pause();
      el.removeAttribute("src");
      playing = false;
      live = false;
      progress = 0;
      return;
    }
    el.src = src;
    if (wasPlaying) {
      el.addEventListener(
        "loadedmetadata",
        () => {
          el.currentTime = Math.min(t, el.duration || t);
          void el.play();
        },
        { once: true },
      );
    }
  });

  function toggle() {
    if (!audioEl || !audioSrc) return;
    if (audioEl.paused) void audioEl.play();
    else audioEl.pause();
  }
  function stop() {
    audioEl?.pause();
    if (audioEl) audioEl.currentTime = 0;
    progress = 0;
    now = 0;
    live = false;
  }

  let raf = 0;
  function tick() {
    if (audioEl && audioEl.duration) {
      now = audioEl.currentTime;
      progress = now / audioEl.duration;
    }
    if (playing) raf = requestAnimationFrame(tick);
  }
  function onPlay() {
    playing = true;
    live = true;
    listened = true;
    if (audioSrc) heardSrcs[audioSrc] = true;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(tick);
  }
  function onPause() {
    playing = false;
    cancelAnimationFrame(raf);
  }
  function onEnded() {
    playing = false;
    live = false;
    finished = true;
    if (audioSrc) {
      heardSrcs[audioSrc] = true;
      finishedSrcs[audioSrc] = true;
    }
    progress = 0;
    now = 0;
  }

  // Whisper parameter counts (OpenAI model card).
  const MODEL_PARAMS: Record<string, string> = { tiny: "39M", base: "74M", small: "244M" };
  const NOISE_ICON: Record<string, string> = {
    living: "M5 11V8.5A2.5 2.5 0 0 1 7.5 6h9A2.5 2.5 0 0 1 19 8.5V11M3 12.5a1.5 1.5 0 0 1 3 0V14h12v-1.5a1.5 1.5 0 0 1 3 0V17H3zM5 17v2M19 17v2",
    kitchen: "M4 10h16v5a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4zM2 10h2M20 10h2M9 3.5c0 1.2-1 1.3-1 2.5M13 3.5c0 1.2-1 1.3-1 2.5",
    cafeteria: "M4 9h12v4.5A5.5 5.5 0 0 1 10.5 19h-1A5.5 5.5 0 0 1 4 13.5zM16 10.5h1.5a2.5 2.5 0 0 1 0 5H16M8 3.5v2.5M12 3.5v2.5",
    tv: "M4 7h16a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1zM8.5 3.5 12 7l3.5-3.5M8 21h8",
  };

  const OP_TEXT: Record<string, string> = {
    sub: "wrong word",
    del: "dropped",
    ins: "invented",
  };
  const older = index.clips.filter((c) => c.age_group === "older");
  const younger = index.clips.filter((c) => c.age_group === "younger");
  const groups = [
    { label: "Older", clips: older, offset: 0 },
    { label: "Younger", clips: younger, offset: older.length },
  ];
</script>

<div class="panel card" data-testid="ears-panel">
  <div class="controls">
    <fieldset class="group">
      <legend class="group-title">Speaker</legend>
      {#each groups as g (g.label)}
        <div class="speaker-group">
          <span class="sub-label">{g.label}</span>
          <div class="speaker-grid">
            {#each g.clips as c, i (c.id)}
              <label class="speaker" class:active={c.id === clipId}>
                <input
                  type="radio"
                  name="speaker"
                  value={c.id}
                  checked={c.id === clipId}
                  onchange={() => pickSpeaker(c.id)}
                />
                <span class="speaker-num">{g.offset + i + 1}</span>
                <span class="speaker-text">
                  <span class="speaker-who">{speakerParts(c).who}</span>
                  <span class="speaker-age">{speakerParts(c).age}</span>
                </span>
              </label>
            {/each}
          </div>
        </div>
      {/each}
    </fieldset>

    <fieldset class="group">
      <legend class="group-title">Background noise</legend>
      <div
        class="tabs tabs-noise"
        style={`--n:${index.noise_types.length};--i:${Math.max(0, index.noise_types.findIndex((n) => n.id === noise))}`}
      >
        {#each index.noise_types as n (n.id)}
          <label class:active={n.id === noise}>
            <input type="radio" name="noise" value={n.id} bind:group={noise} />
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" class="tab-icon">
              <path d={NOISE_ICON[n.id] ?? NOISE_ICON.tv} />
            </svg>
            <span>{n.label}</span>
          </label>
        {/each}
        <span class="tab-indicator" aria-hidden="true"></span>
      </div>
    </fieldset>

    <div class="group">
      <div class="group-head">
        <label for="loudness" class="group-title">Noise level</label>
        <span class="sub-label"
          >{level.snr_db === null ? "" : "compared with the voice"}</span
        >
      </div>
      <output for="loudness" class="level-label" aria-live="polite">{level.label}</output>
      <div class="slider" style={`--fill:${(levelIdx / (index.levels.length - 1)) * 100}`}>
        <div class="slider-track" aria-hidden="true">
          <span class="slider-fill"></span>
          {#each index.levels as _l, i (i)}
            <span
              class="stop"
              class:passed={i <= levelIdx}
              style={`--at:${(i / (index.levels.length - 1)) * 100}`}
            ></span>
          {/each}
        </div>
        <input
          id="loudness"
          type="range"
          min="0"
          max={index.levels.length - 1}
          step="1"
          bind:value={levelIdx}
          aria-valuetext={level.label}
        />
      </div>
      <div class="ticks" aria-hidden="true">
        {#each index.levels as l, i (i)}
          <button
            type="button"
            tabindex="-1"
            class:on={i === levelIdx}
            style={`--at:${(i / (index.levels.length - 1)) * 100}`}
            onclick={() => (levelIdx = i)}
          >
            {noiseVsVoice(l.snr_db)}
          </button>
        {/each}
      </div>
    </div>

    <fieldset class="group">
      <legend class="group-title">Whisper model</legend>
      <div
        class="tabs tabs-model"
        style={`--n:${index.models.length};--i:${Math.max(0, index.models.indexOf(model))}`}
      >
        {#each index.models as m (m)}
          <label class:active={m === model}>
            <input type="radio" name="model" value={m} bind:group={model} />
            <span class="model-name">{m}</span>
            {#if MODEL_PARAMS[m]}<span class="model-size">{MODEL_PARAMS[m]}</span>{/if}
          </label>
        {/each}
        <span class="tab-indicator" aria-hidden="true"></span>
      </div>
    </fieldset>

    <div class="group vad" class:disabled={!vadAvailable}>
      <label class="switch">
        <span class="vad-text">
          <span class="group-title">Skip non-speech</span>
          <span class="sub-label">
            {vadAvailable
              ? "Cuts the silent bits before Whisper listens (VAD)."
              : "Not tested with TV noise."}
          </span>
        </span>
        <input type="checkbox" role="switch" bind:checked={vad} disabled={!vadAvailable} />
        <span class="track" aria-hidden="true"><span class="thumb"></span></span>
      </label>
    </div>
  </div>

  <div class="output" aria-busy={!result}>
    <div class="listen">
      {#if audioSrc}
        <button
          class="play"
          type="button"
          onclick={toggle}
          aria-label={playing ? "Pause" : "Play"}
          style={`--p:${progress}`}
        >
          <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
            {#if playing}
              <rect x="6" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
              <rect x="14" y="5" width="4" height="14" rx="1.2" fill="currentColor" />
            {:else}
              <path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z" fill="currentColor" />
            {/if}
          </svg>
        </button>
        <div class="listen-text">
          <strong>{playing ? "Playing" : "Listen"}</strong>
          <span
            >{level.snr_db === null
              ? "No background noise"
              : `${noiseOpt.label} noise ${level.label.toLowerCase().replace(/^noise /, "")}`},
            heard from {index.distance_m} m away</span
          >
        </div>
      {:else if cell}
        <div class="tv-note" role="note">
          <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><rect x="3" y="5" width="18" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.8" /><path d="M8 21h8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /><path d="M4 4l16 16" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" /></svg>
          <span>No audio for TV: it's a copyrighted recording. You can still read what Whisper wrote.</span>
        </div>
      {/if}
      <audio
        bind:this={audioEl}
        preload="none"
        onplay={onPlay}
        onpause={onPause}
        onended={onEnded}
        data-testid="panel-audio"
      ></audio>
    </div>

    {#if loadError}
      <p class="error" role="alert">{loadError}</p>
    {:else if !result}
      <div class="skeleton" aria-label="Loading results">
        <span></span><span></span><span></span>
      </div>
    {:else if !revealText}
      <p class="prompt">Press play. What Whisper wrote appears as the clip plays.</p>
    {:else}
      {#if revealScore}
      <div class="score" in:fade={{ duration: dur(200) }}>
        <div class="wer">
          <span class="wer-num">{result.errors}</span>
          <span class="wer-label">
            {result.errors === 1 ? "mistake" : "mistakes"} in {result.ref_words} words<br />
            <small>
              {breakdown} <span data-testid="wer">(WER {pct(result.wer)})</span>
            </small>
          </span>
        </div>
        <span class="badge" class:good={result.usable} class:bad={!result.usable} data-testid="usable">
          {#if result.usable}
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
            Usable
          {:else}
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>
            Not usable
          {/if}
        </span>
      </div>
      {/if}

      <div class="transcripts">
        <div class="line">
          <span class="line-label">Said</span>
          <p class="words" data-testid="said">
            {#each saidTokens as t (t.id)}
              <span
                class="w {t.op === 'del' ? 'del' : 'ok'}"
                class:pending={!shown(t)}
                class:now={t.id === currentSaid}
                animate:flip={{ duration: dur(260) }} in:fly={{ y: 6, duration: dur(220) }}
                >{t.word}{#if t.op === "del"}<span class="sr-only"> (dropped)</span>{/if}</span
              >
            {/each}
          </p>
        </div>
        <div class="line">
          <span class="line-label">Whisper wrote</span>
          <p class="words" data-testid="heard">
            {#each heardTokens as t (t.id)}
              <span class="w {t.op === 'ok' ? 'ok' : 'bad'}" class:pending={!shown(t)} animate:flip={{ duration: dur(260) }} in:fly={{ y: 6, duration: dur(220) }}
                >{t.word}{#if t.op !== "ok"}<span class="sr-only"> ({OP_TEXT[t.op]})</span>{/if}</span
              >
            {:else}
              <span class="nothing">(nothing)</span>
            {/each}
          </p>
        </div>
      </div>

      <p class="fine">
        From the benchmark run (faster-whisper {model}{vad && vadAvailable ? ", VAD on" : ""}).
        {#if live}Word timing is approximate.{/if}
      </p>
    {/if}
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
      min-height: 640px;
    }
  }

  /* ---------- Controls ---------- */
  .controls {
    container-type: inline-size;
    padding: clamp(20px, 3.4vw, 40px);
    display: grid;
    gap: 30px;
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
    gap: 12px;
  }
  .group-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    padding: 0;
    margin: 0;
    float: none;
  }
  legend.group-title {
    margin-bottom: 12px;
  }
  fieldset.group {
    display: block;
  }
  fieldset.group > :not(legend) + :not(legend) {
    margin-top: 12px;
  }
  .group-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
  }
  .sub-label {
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.4;
  }
  input[type="radio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }

  /* Speaker cards: a deliberate two-line layout. */
  .speaker-group {
    display: grid;
    gap: 8px;
  }
  .speaker-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
  }
  @container (min-width: 440px) {
    .speaker-grid {
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
  }
  .speaker {
    position: relative;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--hairline);
    background: var(--surface);
    cursor: pointer;
    transition:
      border-color 0.2s var(--ease),
      background-color 0.2s var(--ease),
      box-shadow 0.2s var(--ease),
      transform 0.15s var(--ease);
  }
  .speaker:hover {
    border-color: var(--axis);
    background: color-mix(in oklab, var(--surface-2) 45%, var(--surface));
  }
  .speaker:active {
    transform: scale(0.98);
  }
  .speaker.active {
    border-color: var(--accent);
    background: color-mix(in oklab, var(--accent) 7%, var(--surface));
    box-shadow: 0 0 0 1px var(--accent);
  }
  .speaker:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .speaker-num {
    flex: none;
    display: grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--surface-2);
    color: var(--ink-2);
    font-weight: 600;
    font-size: 0.75rem;
    font-variant-numeric: tabular-nums;
    transition:
      background-color 0.2s var(--ease),
      color 0.2s var(--ease);
  }
  .speaker.active .speaker-num {
    background: var(--accent);
    color: var(--on-accent);
  }
  .speaker-text {
    display: grid;
    line-height: 1.2;
    min-width: 0;
  }
  .speaker-who {
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--ink);
  }
  .speaker-age {
    font-size: 0.78rem;
    color: var(--muted);
  }

  /* Underline tabs with a sliding indicator. */
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
    text-align: center;
    border-radius: 8px 8px 0 0;
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
  @container (max-width: 400px) {
    .tabs-noise label {
      font-size: 0.8rem;
    }
  }
  .tabs-model label {
    flex-direction: column;
    gap: 1px;
  }
  .model-name {
    line-height: 1.2;
  }
  .model-size {
    font-size: 0.72rem;
    font-weight: 400;
    color: var(--muted);
    line-height: 1.2;
  }

  /* Noise level slider: native range input over a custom track with stops. */
  .level-label {
    font-family: var(--serif);
    font-size: 1.3rem;
    line-height: 1.25;
    color: var(--ink);
  }
  .slider {
    --thumb: 22px;
    position: relative;
    height: 28px;
  }
  .slider-track {
    position: absolute;
    left: calc(var(--thumb) / 2);
    right: calc(var(--thumb) / 2);
    top: 50%;
    height: 4px;
    margin-top: -2px;
    border-radius: 999px;
    background: var(--surface-3);
  }
  .slider-fill {
    position: absolute;
    inset: 0 auto 0 0;
    width: calc(var(--fill) * 1%);
    border-radius: inherit;
    background: var(--accent);
    transition: width 0.25s var(--ease);
  }
  .stop {
    position: absolute;
    top: 50%;
    left: calc(var(--at) * 1%);
    width: 8px;
    height: 8px;
    margin: -4px 0 0 -4px;
    border-radius: 50%;
    background: var(--surface);
    border: 2px solid var(--surface-3);
    transition: border-color 0.25s var(--ease);
  }
  .stop.passed {
    border-color: var(--accent);
  }
  .slider input[type="range"] {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    margin: 0;
    background: transparent;
    -webkit-appearance: none;
    appearance: none;
    cursor: pointer;
  }
  .slider input[type="range"]::-webkit-slider-runnable-track {
    background: transparent;
    height: 100%;
  }
  .slider input[type="range"]::-moz-range-track {
    background: transparent;
  }
  .slider input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: var(--thumb);
    height: var(--thumb);
    margin-top: 3px;
    border-radius: 50%;
    background: var(--surface);
    border: 1px solid var(--ring);
    box-shadow:
      0 0 0 5px color-mix(in oklab, var(--accent) 0%, transparent),
      0 1px 3px rgba(0, 0, 0, 0.18),
      inset 0 0 0 6px var(--accent);
    transition:
      transform 0.15s var(--ease),
      box-shadow 0.2s var(--ease);
  }
  .slider input[type="range"]::-moz-range-thumb {
    width: var(--thumb);
    height: var(--thumb);
    border-radius: 50%;
    background: var(--surface);
    border: 1px solid var(--ring);
    box-shadow:
      0 1px 3px rgba(0, 0, 0, 0.18),
      inset 0 0 0 6px var(--accent);
  }
  .slider input[type="range"]:hover::-webkit-slider-thumb {
    box-shadow:
      0 0 0 6px color-mix(in oklab, var(--accent) 16%, transparent),
      0 1px 3px rgba(0, 0, 0, 0.18),
      inset 0 0 0 6px var(--accent);
  }
  .slider input[type="range"]:active::-webkit-slider-thumb {
    transform: scale(1.08);
  }
  .slider input[type="range"]:focus-visible {
    outline: none;
  }
  .slider input[type="range"]:focus-visible::-webkit-slider-thumb {
    box-shadow:
      0 0 0 3px var(--surface),
      0 0 0 5px var(--accent),
      inset 0 0 0 6px var(--accent);
  }
  .ticks {
    position: relative;
    height: 20px;
    margin: -4px calc(22px / 2) 0;
  }
  .ticks button {
    position: absolute;
    left: calc(var(--at) * 1%);
    transform: translateX(-50%);
    border: 0;
    background: none;
    padding: 2px 4px;
    font: inherit;
    font-size: 0.75rem;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    cursor: pointer;
    white-space: nowrap;
    transition: color 0.2s var(--ease);
  }
  .ticks button:hover {
    color: var(--ink-2);
  }
  .ticks button.on {
    color: var(--ink);
    font-weight: 600;
  }

  /* VAD switch row. */
  .vad .switch {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 14px 16px;
    border: 1px solid var(--hairline);
    border-radius: var(--radius-md);
    background: var(--surface);
    cursor: pointer;
    position: relative;
  }
  .vad.disabled .switch {
    cursor: not-allowed;
    opacity: 0.6;
  }
  .vad-text {
    display: grid;
    gap: 2px;
  }
  .switch input {
    position: absolute;
    opacity: 0;
    inset: 0;
    margin: 0;
    cursor: inherit;
  }
  .track {
    width: 40px;
    height: 24px;
    border-radius: 999px;
    background: var(--surface-3);
    position: relative;
    transition: background-color 0.2s var(--ease);
    flex: none;
  }
  .thumb {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #fff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    transition: transform 0.25s var(--ease);
  }
  .switch input:checked ~ .track {
    background: var(--accent);
  }
  .switch input:checked ~ .track .thumb {
    transform: translateX(16px);
  }
  .switch:has(input:focus-visible) {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }

  /* ---------- Output ---------- */
  .output {
    padding: clamp(20px, 3.4vw, 40px);
    display: grid;
    gap: 24px;
    align-content: start;
    background: color-mix(in oklab, var(--surface-2) 35%, var(--surface));
  }
  .listen {
    display: flex;
    align-items: center;
    gap: 16px;
    min-height: 64px;
  }
  .play {
    --p: 0;
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
    transition: transform 0.2s var(--ease);
  }
  .play:hover {
    transform: scale(1.04);
  }
  .play:active {
    transform: scale(0.96);
  }
  .listen-text {
    display: grid;
    gap: 2px;
    line-height: 1.3;
  }
  .listen-text strong {
    font-weight: 600;
  }
  .listen-text span {
    color: var(--muted);
    font-size: 0.88rem;
  }
  .tv-note {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    font-size: 0.9rem;
    color: var(--ink-2);
    background: var(--surface-2);
    border: 1px solid var(--hairline);
    padding: 12px 14px;
    border-radius: var(--radius-sm);
  }
  .tv-note svg {
    flex: none;
    margin-top: 2px;
  }

  .score {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px 16px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--hairline);
  }
  .wer {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .wer-num {
    font-family: var(--serif);
    font-size: clamp(2.8rem, 2rem + 2.6vw, 4rem);
    font-weight: 500;
    letter-spacing: -0.03em;
    line-height: 1;
  }
  .wer-label {
    font-size: 0.95rem;
    color: var(--ink);
    line-height: 1.35;
  }
  .wer-label small {
    font-size: 0.8rem;
    color: var(--muted);
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 5px 10px;
    border-radius: 6px;
    flex-wrap: wrap;
  }
  .badge.good {
    color: var(--good);
    background: var(--good-bg);
  }
  .badge.bad {
    color: var(--bad);
    background: var(--bad-bg);
  }
  .dot {
    margin: 0 4px;
  }
  .badge-rule {
    font-weight: 400;
    font-size: 0.75rem;
    opacity: 0.85;
  }

  .transcripts {
    display: grid;
    gap: 20px;
  }
  .line {
    display: grid;
    gap: 6px;
  }
  .line-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--muted);
  }
  .words {
    margin: 0;
    font-family: var(--serif);
    font-size: clamp(1.2rem, 1rem + 0.8vw, 1.6rem);
    line-height: 1.5;
    min-height: 2em;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3em 0.28em;
  }
  .w {
    display: inline-block;
    padding: 0 4px;
    border-radius: 5px;
    transition:
      opacity 0.25s var(--ease),
      transform 0.25s var(--ease),
      box-shadow 0.2s var(--ease);
  }
  .w.pending {
    opacity: 0.14;
    transform: translateY(4px);
  }
  .w.now {
    box-shadow: 0 0 0 2px var(--accent);
  }
  /* Whisper's mistakes (wrong or invented): red and struck through. */
  .w.bad {
    color: var(--bad);
    background: var(--bad-bg);
    text-decoration: line-through var(--bad);
    text-decoration-thickness: 2px;
  }
  /* Words Whisper dropped: greyed out in what was said. */
  .w.del {
    color: var(--muted);
    opacity: 0.55;
  }
  .w.del.pending {
    opacity: 0.1;
  }
  .prompt {
    margin: 0;
    padding: 28px 0;
    color: var(--muted);
    font-family: var(--serif);
    font-size: 1.2rem;
  }
  .nothing {
    color: var(--muted);
    font-style: italic;
  }
  .fine {
    font-size: 0.78rem;
    color: var(--muted);
    margin: 0;
  }
  .error {
    color: var(--bad);
  }
  .skeleton {
    display: grid;
    gap: 10px;
  }
  .skeleton span {
    height: 18px;
    border-radius: 6px;
    background: linear-gradient(90deg, var(--surface-2), var(--surface-3), var(--surface-2));
    background-size: 200% 100%;
    animation: shimmer 1.2s linear infinite;
  }
  .skeleton span:nth-child(2) {
    width: 80%;
  }
  .skeleton span:nth-child(3) {
    width: 60%;
  }
  @keyframes shimmer {
    to {
      background-position: -200% 0;
    }
  }
</style>
