/// <reference lib="webworker" />
// Runs Whisper tiny with transformers.js off the main thread, so the page never freezes.
// Messages in:  { type: "run", id, audios: Float32Array[] (16 kHz mono), device: "webgpu" | "wasm" }
//               { type: "preload", device } downloads the model into the browser cache only
// Messages out: progress / stage (shared by every run), result { id, texts, ms } / error { id }.
// transformers.js loads from jsDelivr at runtime, pinned to the installed version, so its
// 26 MB ONNX Runtime WebAssembly file comes from the CDN too and never sits in the site.
type TransformersModule = typeof import("@huggingface/transformers");
declare const __TRANSFORMERS_VERSION__: string;
const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${__TRANSFORMERS_VERSION__}/dist/transformers.min.js`;

const MODEL = "onnx-community/whisper-tiny"; // multilingual tiny, like the benchmark's "tiny"

let lib: TransformersModule | null = null;
async function transformers(): Promise<TransformersModule> {
  if (!lib) {
    lib = (await import(/* @vite-ignore */ TRANSFORMERS_URL)) as TransformersModule;
    lib.env.allowLocalModels = false; // models come from the Hugging Face Hub; the browser caches them
  }
  return lib;
}

type Transcriber = (
  audio: Float32Array,
  options: Record<string, unknown>,
) => Promise<{ text: string } | { text: string }[]>;

// One load per device, shared by a preload and a later run, so nothing downloads twice.
const loads = new Map<string, Promise<Transcriber>>();
const ready = new Set<string>();

const post = (message: Record<string, unknown>) => self.postMessage(message);

function load(device: "webgpu" | "wasm"): Promise<Transcriber> {
  let pending = loads.get(device);
  if (!pending) {
    pending = createPipeline(device);
    pending.then(
      () => ready.add(device),
      () => loads.delete(device), // a failed load can be retried
    );
    loads.set(device, pending);
  }
  return pending;
}

async function createPipeline(device: "webgpu" | "wasm"): Promise<Transcriber> {
  const { pipeline } = await transformers();
  const run = (await pipeline("automatic-speech-recognition", MODEL, {
    device,
    // The 8-bit build is the smallest Whisper tiny (about 41 MB), on the GPU and the CPU alike.
    dtype: "q8",
    progress_callback: (info: { status: string; loaded?: number; total?: number }) => {
      if (info.status === "progress_total") {
        post({ type: "progress", loaded: info.loaded ?? 0, total: info.total ?? 0 });
      } else if (info.status === "ready") {
        post({ type: "stage", stage: "starting" });
      }
    },
  })) as unknown as Transcriber;
  return run;
}

self.addEventListener("message", async (event: MessageEvent) => {
  const { type, id, audios, device } = event.data ?? {};
  if (type === "preload") {
    load(device).then(
      () => post({ type: "preloaded" }),
      () => {}, // a failed preload is retried, with a visible error, on the real run
    );
    return;
  }
  if (type !== "run") return;
  try {
    if (!ready.has(device)) post({ type: "stage", stage: "download" });
    const run = await load(device);
    post({ type: "stage", stage: "transcribing" });
    const started = performance.now();
    const texts: string[] = [];
    for (const audio of audios as Float32Array[]) {
      const out = await run(audio, { language: "english", task: "transcribe" });
      texts.push((Array.isArray(out) ? out.map((o) => o.text).join(" ") : out.text).trim());
    }
    post({ type: "result", id, texts, ms: Math.round(performance.now() - started) });
  } catch (err) {
    post({ type: "error", id, message: err instanceof Error ? err.message : String(err) });
  }
});

post({ type: "booted" });
