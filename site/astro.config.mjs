// Static output only: Cloudflare serves plain files from dist/ (no adapter, no SSR).
import { readFileSync } from "node:fs";
import { defineConfig } from "astro/config";
import svelte from "@astrojs/svelte";

// The live worker loads transformers.js from a CDN at exactly the installed version.
const transformersVersion = JSON.parse(
  readFileSync(new URL("./node_modules/@huggingface/transformers/package.json", import.meta.url), "utf-8"),
).version;

export default defineConfig({
  output: "static",
  integrations: [svelte()],
  build: { format: "directory", inlineStylesheets: "always" },
  trailingSlash: "ignore",
  // Data under public/data is fetched at runtime; never let Vite inline or rename it.
  vite: {
    build: { assetsInlineLimit: 0 },
    define: { __TRANSFORMERS_VERSION__: JSON.stringify(transformersVersion) },
    worker: { format: "es" },
  },
});
