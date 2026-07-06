#!/usr/bin/env node
/**
 * Capture FiNER-139 tab screenshots for PR artifacts.
 * Requires orchestrator (:8002) and frontend (:3000) running.
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const OUT_DIR = path.resolve("docs/research/findings/screenshots");
// Use localhost — 127.0.0.1 breaks Next.js dev HMR/hydration in headless runs.
const BASE_URL = process.env.FINER139_BASE_URL ?? "http://localhost:3000";

async function main() {
	await mkdir(OUT_DIR, { recursive: true });
	const browser = await chromium.launch({ headless: true });
	const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

	await page.goto(`${BASE_URL}/finer139`, { waitUntil: "networkidle" });
	await page.waitForTimeout(1500);
	await page.screenshot({
		path: path.join(OUT_DIR, "01-finer139-config.png"),
		fullPage: true,
	});

	// Deselect LLM methods for faster offline run; keep ontology + nlp
	for (const label of ["LLM-Based", "Hybrid", "Dynamic"]) {
		const btn = page.getByRole("button", { name: new RegExp(label) });
		if (await btn.count()) {
			const variant = await btn.first().getAttribute("data-variant");
			if (variant === "default") {
				await btn.first().click();
			}
		}
	}

	await page.getByRole("button", { name: /Run benchmark/i }).click();
	// CardTitle renders as div, not a heading role — wait for run completion.
	await page
		.getByRole("button", { name: /^Run benchmark$/i })
		.waitFor({ timeout: 120_000 });
	await page.getByText("F1 (strict)").waitFor({ timeout: 10_000 });
	await page.waitForTimeout(1000);
	await page.screenshot({
		path: path.join(OUT_DIR, "02-finer139-results.png"),
		fullPage: true,
	});

	// Examples section
	const examples = page.getByText("Examples").first();
	if (await examples.isVisible()) {
		await examples.scrollIntoViewIfNeeded();
		await page.waitForTimeout(500);
		await page.screenshot({
			path: path.join(OUT_DIR, "03-finer139-examples.png"),
			fullPage: true,
		});
	}

	// Sidebar nav showing FiNER-139 tab
	await page.setViewportSize({ width: 1440, height: 900 });
	await page.goto(`${BASE_URL}/finer139`, { waitUntil: "networkidle" });
	await page.waitForTimeout(500);
	await page.screenshot({
		path: path.join(OUT_DIR, "04-finer139-nav.png"),
		fullPage: false,
	});

	await browser.close();
	console.log("Screenshots saved to", OUT_DIR);
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
