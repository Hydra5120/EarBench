import { defineConfig, devices } from "@playwright/test";
import { fileURLToPath } from "node:url";

// Tests run against the built site served by Cloudflare's local runtime (wrangler dev),
// so _headers, the 404 page and trailing slashes behave as they do in production.
const PORT = 8788;
// The your-voice panel records from a fake microphone that plays a Common Voice clip (CC0).
const FAKE_VOICE = fileURLToPath(new URL("tests/fixtures/voice.wav", import.meta.url));

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--autoplay-policy=no-user-gesture-required",
        "--mute-audio",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${FAKE_VOICE}`,
      ],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && npx wrangler dev --port ${PORT} --ip 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
