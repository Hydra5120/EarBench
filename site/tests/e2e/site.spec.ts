import { expect, test, type Page } from "@playwright/test";

const panel = (page: Page) => page.getByTestId("ears-panel");
const MODEL_HOSTS = /huggingface\.co|hf\.co/;

/** Tests that don't use the live panel stop the idle preload from downloading the model. */
async function noModelDownloads(page: Page) {
  await page.route(MODEL_HOSTS, (route) => route.abort());
}

const voice = (page: Page) => page.getByTestId("voice-panel");

/** Scroll to the your-voice panel and wait for it to hydrate (it loads when visible). */
async function openVoice(page: Page) {
  await voice(page).scrollIntoViewIfNeeded();
  await expect(page.locator("astro-island:has([data-testid=voice-panel])")).not.toHaveAttribute("ssr", /.*/);
}

/** Record from the fake microphone; stop early after `ms`, or let the time limit stop it. */
async function record(page: Page, ms?: number) {
  await voice(page).getByTestId("record").click();
  await expect(voice(page).getByTestId("countdown")).toBeVisible();
  if (ms) {
    await page.waitForTimeout(ms);
    await voice(page).getByRole("button", { name: "Stop recording" }).click();
  }
  await expect(voice(page).getByTestId("voice-audio")).toBeVisible({ timeout: 10_000 });
}

/** Press play, then pause: the transcript is revealed once a clip has been played. */
async function reveal(page: Page) {
  await panel(page).getByRole("button", { name: "Play", exact: true }).click();
  await page.waitForTimeout(400);
  await panel(page).getByRole("button", { name: "Pause", exact: true }).click();
  await expect(page.getByTestId("heard")).toBeVisible();
}

test("page loads with the hero and the panel", async ({ page }) => {
  await noModelDownloads(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("older person over the TV");
  await expect(panel(page)).toBeVisible();
  await expect(panel(page).getByText("Press play.")).toBeVisible();
  await expect(page.getByTestId("heard")).toHaveCount(0);
});

test("moving the slider changes the transcript and the audio", async ({ page }) => {
  await noModelDownloads(page);
  await page.goto("/");
  const audio = page.getByTestId("panel-audio");
  await reveal(page);
  const noisySrc = await audio.getAttribute("src");
  const noisyText = await page.getByTestId("heard").textContent();

  await page.locator("#loudness").focus();
  await page.keyboard.press("Home"); // all the way down: no noise
  await expect(audio).not.toHaveAttribute("src", noisySrc ?? "");
  await expect(panel(page).getByText("Press play.")).toBeVisible(); // new clip, not heard yet

  await reveal(page);
  expect(await audio.getAttribute("src")).toContain("/clean.");
  expect(await page.getByTestId("heard").textContent()).not.toBe(noisyText);
});

test("recording stops at the time limit and mixes in the noise", async ({ page, context }) => {
  await context.route(MODEL_HOSTS, (route) => route.abort());
  await page.goto("/?live=wasm");
  await openVoice(page);
  const started = Date.now();
  await record(page); // no stop: the 5 second limit ends it
  expect(Date.now() - started).toBeLessThan(9_000);
  const duration = await voice(page)
    .getByTestId("voice-audio")
    .evaluate(async (el: HTMLAudioElement) => {
      if (!el.duration) await new Promise((r) => el.addEventListener("loadedmetadata", r, { once: true }));
      return el.duration;
    });
  expect(duration).toBeGreaterThan(3);
  expect(duration).toBeLessThanOrEqual(5.01);
  await expect(voice(page).getByTestId("voice-mix")).toBeVisible();

  // The model is blocked: a plain failure message, never a library or worker error.
  const error = voice(page).getByTestId("voice-error");
  await expect(error).toBeVisible({ timeout: 60_000 });
  await expect(error).toContainText("couldn't be downloaded");
  await expect(error).not.toContainText(/module specifier|import|worker crashed/i);

  // Another noise remixes the recording without recording again.
  const before = await voice(page).getByTestId("voice-mix").getAttribute("src");
  await voice(page).getByText("Kitchen").click();
  await expect(voice(page).getByTestId("voice-mix")).not.toHaveAttribute("src", before ?? "");
});

test("if the noise can't load, the recording still plays and says why", async ({ page }) => {
  await noModelDownloads(page);
  await page.route(/\/data\/noise\//, (route) => route.abort());
  await page.goto("/?live=wasm");
  await openVoice(page);
  await record(page, 1_500);
  await expect(voice(page).getByTestId("noise-error")).toContainText("only your own recording");
  await expect(voice(page).getByTestId("voice-mix")).toHaveCount(0);
  // The clean recording still went to Whisper (the model is blocked here, so that fails visibly).
  await expect(voice(page).getByTestId("voice-error")).toBeVisible({ timeout: 60_000 });
});

test("a blocked microphone shows a plain message", async ({ page }) => {
  await noModelDownloads(page);
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () =>
      Promise.reject(new DOMException("Permission denied", "NotAllowedError"));
  });
  await page.goto("/");
  await openVoice(page);
  await voice(page).getByTestId("record").click();
  await expect(voice(page).getByTestId("mic-error")).toContainText("microphone is blocked");
  await expect(voice(page).getByTestId("voice-audio")).toHaveCount(0);
});

test("@slow your voice is transcribed with and without noise on the WebAssembly path", async ({
  page,
}) => {
  test.setTimeout(240_000);
  await page.goto("/?live=wasm");
  await openVoice(page);
  await record(page);
  await expect(voice(page).getByTestId("voice-stage")).toBeVisible();
  const clean = voice(page).getByTestId("voice-clean");
  const noisy = voice(page).getByTestId("voice-noisy");
  await expect(clean).not.toContainText("…", { timeout: 200_000 });
  await expect(noisy).not.toContainText("…");
  await expect(clean).toContainText(/cambridge/i); // the fake microphone's sentence
  await expect(voice(page).getByTestId("voice-error")).toHaveCount(0);

  // A new level re-transcribes only the noisy version; the clean text stays.
  const cleanText = await clean.textContent();
  await voice(page).getByText("0 dB", { exact: true }).click();
  await expect(voice(page).getByTestId("voice-noisy")).not.toContainText("…", { timeout: 60_000 });
  expect(await clean.textContent()).toBe(cleanText);
});

test("@slow a second visit uses the browser cache instead of downloading again", async ({ page }) => {
  test.setTimeout(240_000);
  const modelRequests: string[] = [];
  page.on("request", (req) => {
    if (MODEL_HOSTS.test(req.url()) && /\.onnx/.test(req.url())) modelRequests.push(req.url());
  });

  await page.goto("/?live=wasm");
  await openVoice(page);
  await record(page, 2_000);
  await expect(voice(page).getByTestId("voice-clean")).not.toContainText("…", { timeout: 200_000 });
  const firstVisit = modelRequests.length;

  modelRequests.length = 0;
  await page.reload();
  await openVoice(page);
  await record(page, 2_000);
  await expect(voice(page).getByTestId("voice-clean")).not.toContainText("…", { timeout: 120_000 });
  expect(firstVisit).toBeGreaterThan(0); // the first visit really downloaded the model
  expect(modelRequests).toEqual([]); // the second came from the Cache API, no network
});

test("@slow without cross-origin isolation it still transcribes, single-threaded", async ({
  page,
}) => {
  test.setTimeout(240_000);
  // Browsers that ignore COEP "credentialless" (e.g. Safari) lose isolation: strip the headers.
  await page.route(/127\.0\.0\.1:\d+\/(\?.*)?$/, async (route) => {
    const response = await route.fetch();
    const headers = { ...response.headers() };
    delete headers["cross-origin-opener-policy"];
    delete headers["cross-origin-embedder-policy"];
    await route.fulfill({ response, headers });
  });
  await page.goto("/?live=wasm");
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(false);
  await openVoice(page);
  await record(page, 2_000);
  await expect(voice(page).getByTestId("voice-clean")).not.toContainText("…", { timeout: 200_000 });
});

test("@slow wide screens preload the model once the panel is in view, and recording reuses it", async ({
  page,
}) => {
  test.setTimeout(240_000);
  const modelRequests: string[] = [];
  page.on("request", (req) => {
    if (MODEL_HOSTS.test(req.url()) && /\.onnx/.test(req.url())) modelRequests.push(req.url());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/?live=wasm");
  await page.waitForTimeout(3_000);
  expect(modelRequests).toEqual([]); // nothing downloads before the panel is on screen
  await openVoice(page);
  await expect.poll(() => modelRequests.length, { timeout: 30_000 }).toBeGreaterThan(0);
  await page.waitForTimeout(15_000); // let the preload finish
  const preloaded = modelRequests.length;

  await record(page, 2_000);
  await expect(voice(page).getByTestId("voice-clean")).not.toContainText("…", { timeout: 120_000 });
  expect(modelRequests.length).toBe(preloaded); // nothing downloaded twice
});
