// Rasterizes region_map.svg to region_map.png, for PowerPoint. Playwright comes from seahag's evals, as the
// explorer's browser checks do (PLAYWRIGHT_FROM to point elsewhere).
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
const here = dirname(fileURLToPath(import.meta.url));
const from = process.env.PLAYWRIGHT_FROM ?? join(homedir(), "projects/seahag/evals");
const { chromium } = createRequire(join(from, "package.json"))("playwright");
const libs = join(homedir(), "projects/seahag/local/chromium-libs/root/usr/lib/x86_64-linux-gnu");
const browser = await chromium.launch({ env: { ...process.env, ...(existsSync(libs) ? { LD_LIBRARY_PATH: libs } : {}) } });
const page = await (await browser.newContext({ viewport: { width: 1600, height: 993 }, deviceScaleFactor: 2 })).newPage();
await page.setContent(`<body style="margin:0;background:transparent">${readFileSync(join(here, "region_map.svg"), "utf8")}</body>`);
await page.screenshot({ path: join(here, "region_map.png"), omitBackground: true });
await browser.close();
console.log("wrote region_map.png");
