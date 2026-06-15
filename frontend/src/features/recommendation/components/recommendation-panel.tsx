"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	exportBlueprint,
	getRun,
	setRetention,
	type Requirements,
	type RunResponse,
} from "@/lib/api/orchestrator";

const ACTIVE_STATUSES = new Set([
	"pending",
	"shortlisting",
	"indexing",
	"benchmarking",
]);

interface RecommendationPanelProps {
	runId: string;
}

export function RecommendationPanel({ runId }: RecommendationPanelProps) {
	const [run, setRun] = useState<RunResponse | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [exporting, setExporting] = useState(false);

	const refresh = useCallback(async () => {
		try {
			const next = await getRun(runId);
			setRun(next);
			setError(null);
			return next;
		} catch (err) {
			setError(err instanceof Error ? err.message : "Failed to load run");
			return null;
		}
	}, [runId]);

	useEffect(() => {
		void refresh();
	}, [refresh]);

	useEffect(() => {
		if (!run || !ACTIVE_STATUSES.has(run.status)) {
			return;
		}
		const timer = window.setInterval(() => {
			void refresh();
		}, 3000);
		return () => window.clearInterval(timer);
	}, [run?.status, refresh]);

	const handleExport = async () => {
		setExporting(true);
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
		} catch (err) {
			setError(err instanceof Error ? err.message : "Export failed");
		} finally {
			setExporting(false);
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
						onClick={() => void refresh()}
						disabled={isRunning}
					>
						{isRunning ? "Updating…" : "Refresh status"}
					</Button>
					<Button
						type="button"
						onClick={handleExport}
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
						Run in progress — indexing and benchmarking candidates when corpus
						files are available.
					</p>
				)}
				{run?.error_message && (
					<p className="text-sm text-destructive">{run.error_message}</p>
				)}
				{preliminary && !blueprint && (
					<div className="rounded-md border p-4 text-sm">
						<p className="font-medium">Preliminary (not exportable)</p>
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
							onClick={() => setRetention(runId, "yes")}
						>
							Keep for re-benchmark
						</Button>
						<Button
							type="button"
							variant="secondary"
							size="sm"
							onClick={() => setRetention(runId, "later", "24h")}
						>
							Keep 24h
						</Button>
						<Button
							type="button"
							variant="ghost"
							size="sm"
							onClick={() => setRetention(runId, "no")}
						>
							Tear down indexes
						</Button>
					</div>
				)}
			</CardContent>
		</Card>
	);
}

export async function startRecommendation(
	requirements: Requirements,
): Promise<string> {
	const { createRun } = await import("@/lib/api/orchestrator");
	const { run_id } = await createRun(requirements);
	return run_id;
}
