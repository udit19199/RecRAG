const ORCHESTRATOR_API = "/api/orchestrator";

export interface Requirements {
	use_case: string;
	audience?: string;
	document_modality?: string;
	budget_monthly_usd?: number | null;
	citations_required?: boolean;
	user_queries?: string[];
}

export interface RunResponse {
	run_id: string;
	status: string;
	requirements: Requirements;
	preliminary?: {
		architecture: string;
		rationale: string;
		estimated_monthly_usd?: number;
	};
	blueprint?: Record<string, unknown>;
	error_message?: string;
}

export async function createRun(
	requirements: Requirements,
): Promise<{ run_id: string; status: string }> {
	const res = await fetch(`${ORCHESTRATOR_API}/runs`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ requirements }),
	});
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

export async function getRun(runId: string): Promise<RunResponse> {
	const res = await fetch(`${ORCHESTRATOR_API}/runs/${runId}`);
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

export async function exportBlueprint(
	runId: string,
): Promise<Record<string, unknown>> {
	const res = await fetch(`${ORCHESTRATOR_API}/runs/${runId}/export`);
	if (!res.ok) throw new Error(await res.text());
	return res.json();
}

export async function setRetention(
	runId: string,
	choice: "yes" | "no" | "later",
	duration?: "1h" | "24h" | "7d",
): Promise<void> {
	const res = await fetch(`${ORCHESTRATOR_API}/runs/${runId}/retention`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ choice, duration: duration ?? null }),
	});
	if (!res.ok) throw new Error(await res.text());
}
