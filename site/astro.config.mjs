// Static output only: Cloudflare serves plain files from dist/ (no adapter, no SSR).
import { defineConfig } from "astro/config";
import svelte from "@astrojs/svelte";

export default defineConfig({
  output: "static",
  integrations: [svelte()],
  build: { format: "directory", inlineStylesheets: "always" },
  trailingSlash: "ignore",
  // Data under public/data is fetched at runtime; never let Vite inline or rename it.
  vite: { build: { assetsInlineLimit: 0 } },
});
