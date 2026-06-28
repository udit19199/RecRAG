"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	exportBlueprint,
	getRun,
	type RunResponse,
	setRetention,
} from "@/lib/api/orchestrator";

const ACTIVE_STATUSES = new Set([
	"pending",
	"shortlisting",
	"indexing",
	"benchmarking",
]);

const POLL_MS = 3000;

type FetchRunResult =
	| { ok: true; run: RunResponse }
	| { ok: false; error: string };

type ExportResult = { ok: true } | { ok: false; error: string };

async function fetchRun(runId: string): Promise<FetchRunResult> {
	try {
		const run = await getRun(runId);
		return { ok: true, run };
	} catch (err) {
		return {
			ok: false,
			error: err instanceof Error ? err.message : "Failed to load run",
		};
	}
}

async function downloadBlueprint(runId: string): Promise<ExportResult> {
	try {
		const blueprint = await exportBlueprint(runId);
		const blob = new Blob([JSON.stringify(blueprint, null, 2)], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `pipeline-blueprint-${runId}.json`;
		a.click();
		URL.revokeObjectURL(url);
		return { ok: true };
	} catch (err) {
		return {
			ok: false,
			error: err instanceof Error ? err.message : "Export failed",
		};
	}
}

interface RecommendationPanelProps {
	runId: string;
}

export function RecommendationPanel({ runId }: RecommendationPanelProps) {
	const [run, setRun] = useState<RunResponse | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [exporting, setExporting] = useState(false);

	useEffect(() => {
		let cancelled = false;
		let timeoutId: ReturnType<typeof setTimeout> | undefined;

		const schedulePoll = (status: string) => {
			if (!ACTIVE_STATUSES.has(status)) {
				return;
			}
			timeoutId = setTimeout(() => {
				void poll();
			}, POLL_MS);
		};

		const poll = async () => {
			const result = await fetchRun(runId);
			if (cancelled) {
				return;
			}
			if (!result.ok) {
				setError(result.error);
				return;
			}
			setRun(result.run);
			setError(null);
			schedulePoll(result.run.status);
		};

		void poll();

		return () => {
			cancelled = true;
			if (timeoutId !== undefined) {
				clearTimeout(timeoutId);
			}
		};
	}, [runId]);

	const handleRefresh = async () => {
		const result = await fetchRun(runId);
		if (!result.ok) {
			setError(result.error);
			return;
		}
		setRun(result.run);
		setError(null);
	};

	const handleExport = async () => {
		setExporting(true);
		const result = await downloadBlueprint(runId);
		setExporting(false);
		if (!result.ok) {
			setError(result.error);
		}
	};

	const preliminary = run?.preliminary;
	const blueprint = run?.blueprint as Record<string, unknown> | undefined;
	const isRunning = run ? ACTIVE_STATUSES.has(run.status) : false;

	return (
		<Card>
			<CardHeader>
				<CardTitle>Recommendation</CardTitle>
			</CardHeader>
			<CardContent className="flex flex-col gap-4">
				<div className="flex gap-2">
					<Button
						type="button"
						variant="outline"
						onClick={() => void handleRefresh()}
						disabled={isRunning}
					>
						{isRunning ? "Updating…" : "Refresh status"}
					</Button>
					<Button
						type="button"
						onClick={() => void handleExport()}
						disabled={!blueprint || exporting}
					>
						Export blueprint
					</Button>
				</div>
				{error && <p className="text-sm text-destructive">{error}</p>}
				{run && (
					<p className="text-sm text-muted-foreground">Status: {run.status}</p>
				)}
				{isRunning && (
					<p className="text-sm text-muted-foreground">
						Shortlisting pipelines
						{run?.status === "indexing" || run?.status === "benchmarking"
							? " and benchmarking on your corpus"
							: ""}
						…
					</p>
				)}
				{run?.error_message && (
					<p className="text-sm text-destructive">{run.error_message}</p>
				)}
				{preliminary && !blueprint && (
					<div className="rounded-md border p-4 text-sm">
						<p className="font-medium">Preliminary (not exportable)</p>
						{preliminary.note ? (
							<p className="mt-2 text-muted-foreground">{preliminary.note}</p>
						) : null}
						<p className="mt-2">{preliminary.rationale}</p>
						<p className="mt-1 text-muted-foreground">
							Architecture: {preliminary.architecture}
						</p>
					</div>
				)}
				{blueprint && (
					<div className="rounded-md border p-4 text-sm">
						<p className="font-medium">Final recommendation</p>
						<p className="mt-2">{String(blueprint.rationale ?? "")}</p>
					</div>
				)}
				{blueprint && (
					<div className="flex flex-wrap gap-2">
						<Button
							type="button"
							variant="secondary"
							size="sm"
							onClick={() => void setRetention(runId, "yes")}
						>
							Keep for re-benchmark
						</Button>
						<Button
							type="button"
							variant="secondary"
							size="sm"
							onClick={() => void setRetention(runId, "later", "24h")}
						>
							Keep 24h
						</Button>
						<Button
							type="button"
							variant="ghost"
							size="sm"
							onClick={() => void setRetention(runId, "no")}
						>
							Tear down indexes
						</Button>
					</div>
				)}
			</CardContent>
		</Card>
	);
}
