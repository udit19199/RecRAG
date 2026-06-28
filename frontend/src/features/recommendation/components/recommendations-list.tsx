"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { listRuns, type RunListItem } from "@/lib/api/orchestrator";

async function loadRuns(): Promise<RunListItem[] | { error: string }> {
	try {
		return await listRuns();
	} catch (err) {
		return {
			error:
				err instanceof Error ? err.message : "Failed to load recommendations",
		};
	}
}

function formatWhen(iso: string): string {
	const date = new Date(iso);
	if (Number.isNaN(date.getTime())) return iso;
	return date.toLocaleString();
}

export function RecommendationsList() {
	const [runs, setRuns] = useState<RunListItem[]>([]);
	const [error, setError] = useState<string | null>(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		let cancelled = false;
		const load = async () => {
			const result = await loadRuns();
			if (cancelled) return;
			setLoading(false);
			if ("error" in result) {
				setError(result.error);
				setRuns([]);
			} else {
				setError(null);
				setRuns(result);
			}
		};
		void load();
		return () => {
			cancelled = true;
		};
	}, []);

	if (loading) {
		return <p className="text-sm text-muted-foreground">Loading…</p>;
	}

	if (error) {
		return <p className="text-sm text-destructive">{error}</p>;
	}

	if (runs.length === 0) {
		return (
			<p className="text-sm text-muted-foreground">No recommendations yet.</p>
		);
	}

	return (
		<ul className="flex flex-col gap-2">
			{runs.map((run) => (
				<li key={run.run_id}>
					<Link
						href={`/recommendations/${run.run_id}`}
						className="block rounded-md border px-4 py-3 text-sm hover:bg-muted/50"
					>
						<p className="font-medium">{run.status}</p>
						<p className="mt-1 text-muted-foreground">
							{run.architecture ?? "—"} · {formatWhen(run.created_at)}
						</p>
					</Link>
				</li>
			))}
		</ul>
	);
}
