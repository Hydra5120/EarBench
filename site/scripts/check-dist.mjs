// Cloudflare Workers static assets: every file under 25 MiB, at most 20,000 files.
// Also reports what a first visit downloads before Play (HTML + JS + CSS).
import { readFileSync, readdirSync, statSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { join, relative } from "node:path";

const DIST = new URL("../dist/", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const MAX_FILE = 25 * 1024 * 1024;
const MAX_FILES = 20_000;

const files = [];
(function walk(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const stat = statSync(path);
    if (stat.isDirectory()) walk(path);
    else files.push({ path: relative(DIST, path).replaceAll("\\", "/"), size: stat.size });
  }
})(DIST);

const errors = [];
if (files.length > MAX_FILES) errors.push(`${files.length} files (limit ${MAX_FILES})`);
for (const f of files) if (f.size > MAX_FILE) errors.push(`${f.path} is ${f.size} bytes (limit 25 MiB)`);
if (files.some((f) => /(^|\/)tv_[^/]*\.mp3$/.test(f.path))) errors.push("TV audio found in dist");

const kb = (n) => `${(n / 1024).toFixed(1)} kB`;
const gz = (pred) =>
  files
    .filter(pred)
    .reduce((a, f) => a + gzipSync(readFileSync(join(DIST, f.path)), { level: 9 }).length, 0);
// First load = the page plus every script it can run before Play (the live Whisper
// worker, added in Stage 4, loads only on "Run live" and lives under _astro/live-*).
const html = gz((f) => f.path === "index.html");
const js = gz((f) => f.path.endsWith(".js") && !f.path.includes("live-"));
const JS_BUDGET = 25 * 1024;
console.log(`dist: ${files.length} files, largest ${kb(Math.max(...files.map((f) => f.size)))}`);
console.log(`first load (gzip): index.html ${kb(html)}, js ${kb(js)} (budget ${kb(JS_BUDGET)})`);
if (js > JS_BUDGET) errors.push(`first-load JS ${kb(js)} gzip is over the ${kb(JS_BUDGET)} budget`);
if (errors.length) {
  console.error("check-dist failed:\n  " + errors.join("\n  "));
  process.exit(1);
}
