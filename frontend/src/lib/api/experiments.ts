import type {
	Finer139RunResponse,
	Finer139StartRequest,
} from "@/lib/api/types";

const ORCHESTRATOR_API = "/api/orchestrator";

export async function startFiner139Run(
	request: Finer139StartRequest,
): Promise<{ run_id: string; status: string }> {
	const res = await fetch(`${ORCHESTRATOR_API}/experiments/finer139/runs`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(request),
	});
	if (!res.ok) {
		const text = await res.text();
		throw new Error(text || `HTTP ${res.status}`);
	}
	return res.json();
}

export async function getFiner139Run(
	runId: string,
): Promise<Finer139RunResponse> {
	const res = await fetch(
		`${ORCHESTRATOR_API}/experiments/finer139/runs/${runId}`,
	);
	if (!res.ok) {
		const text = await res.text();
		throw new Error(text || `HTTP ${res.status}`);
	}
	return res.json();
}

export async function waitForFiner139Run(
	runId: string,
	intervalMs = 2000,
	timeoutMs = 600_000,
	onPoll?: (status: Finer139RunResponse) => void,
): Promise<Finer139RunResponse> {
	const start = Date.now();

	const poll = async (): Promise<Finer139RunResponse> => {
		const status = await getFiner139Run(runId);
		onPoll?.(status);

		if (status.status === "complete") {
			return status;
		}
		if (status.status === "error") {
			throw new Error(status.error || "Benchmark run failed");
		}
		if (Date.now() - start >= timeoutMs) {
			throw new Error(
				"Benchmark timeout — still running after maximum wait time",
			);
		}
		await new Promise((resolve) => setTimeout(resolve, intervalMs));
		return poll();
	};

	return poll();
}
