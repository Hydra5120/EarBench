// Main-thread side of in-browser Whisper: pick a device, talk to the worker.

export type Stage = "download" | "starting" | "transcribing";
export type Update = { stage: Stage; loaded?: number; total?: number };
export type Device = "webgpu" | "wasm";

export const TIMEOUT_MS = 90_000;

let worker: Worker | null = null;

function getWorker(): Worker {
  worker ??= new Worker(new URL("./whisper.worker.ts", import.meta.url), { type: "module" });
  return worker;
}

/** Stop everything: the next run starts a fresh worker (the model stays in the browser cache). */
export function killWorker() {
  worker?.terminate();
  worker = null;
  preloading = false;
}

/** WebGPU if the browser has a usable adapter, otherwise WebAssembly. `?live=wasm` forces WASM. */
export async function pickDevice(): Promise<Device> {
  if (new URLSearchParams(location.search).get("live") === "wasm") return "wasm";
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  if (!gpu) return "wasm";
  try {
    return (await gpu.requestAdapter()) ? "webgpu" : "wasm";
  } catch {
    return "wasm";
  }
}

let nextId = 0;

/**
 * Transcribe 16 kHz mono clips in order. Downloads the model on first use.
 * A failure, timeout or abort stops the worker, so the next call starts clean.
 */
export function transcribe(
  audios: Float32Array[],
  device: Device,
  onUpdate: (update: Update) => void,
  signal?: AbortSignal,
): Promise<{ texts: string[]; ms: number }> {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const id = ++nextId;
    const timer = setTimeout(() => finish(new Error("It took longer than 90 seconds.")), TIMEOUT_MS);
    const onMessage = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.id !== undefined && msg.id !== id) return;
      if (msg.type === "stage") onUpdate({ stage: msg.stage });
      else if (msg.type === "progress") onUpdate({ stage: "download", loaded: msg.loaded, total: msg.total });
      else if (msg.type === "result") finish(null, { texts: msg.texts, ms: msg.ms });
      else if (msg.type === "error") finish(new Error(msg.message));
    };
    const onError = (event: ErrorEvent) => finish(new Error(event.message || "The worker crashed."));
    const onAbort = () => finish(new DOMException("Cancelled.", "AbortError"));

    function finish(err: Error | null, out?: { texts: string[]; ms: number }) {
      clearTimeout(timer);
      w.removeEventListener("message", onMessage);
      w.removeEventListener("error", onError);
      signal?.removeEventListener("abort", onAbort);
      if (err) {
        killWorker();
        reject(err);
      } else resolve(out!);
    }

    w.addEventListener("message", onMessage);
    w.addEventListener("error", onError);
    signal?.addEventListener("abort", onAbort);
    w.postMessage(
      { type: "run", id, audios, device },
      audios.map((a) => a.buffer as ArrayBuffer),
    );
  });
}

/** Turn library and network errors into one plain sentence. */
export function plainError(message: string): string {
  if (/90 seconds/.test(message)) return message;
  if (/fetch|network|load|download|404|403/i.test(message)) {
    return "The model couldn't be downloaded. Check your connection and try again.";
  }
  if (/memory|allocation|out of/i.test(message)) return "Your device ran out of memory running the model.";
  return `Something went wrong (${message.replace(/\.$/, "")}).`;
}

const saveData = () =>
  !!(navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData;

let preloading = false;

/** Start downloading the model into the browser cache. Never when the browser asks to save data. */
export async function preloadModel() {
  if (preloading || saveData()) return;
  preloading = true;
  getWorker().postMessage({ type: "preload", device: await pickDevice() });
}

/** On wide screens, preload the model once the page is idle. */
export function preloadWhenIdle() {
  if (!matchMedia("(min-width: 900px)").matches) return;
  const idle = (cb: () => void) =>
    "requestIdleCallback" in window ? requestIdleCallback(cb, { timeout: 8000 }) : setTimeout(cb, 4000);
  idle(() => void preloadModel());
}
