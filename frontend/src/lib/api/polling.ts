import { getIngestionStatus } from "@/lib/api/ingestion";
import { getEvalStatus } from "@/lib/api/retrieval";
import type { IngestionStatus } from "@/lib/api/types";

export async function waitForIngestionComplete(
	intervalMs: number = 2000,
	timeoutMs: number = 120000,
	onPoll?: (status: IngestionStatus) => void,
): Promise<IngestionStatus> {
	const startTime = Date.now();

	while (Date.now() - startTime < timeoutMs) {
		const status = await getIngestionStatus();
		onPoll?.(status);

		if (status.status === "complete") {
			return status;
		}

		if (status.status === "error") {
			throw new Error(status.error_message || "Ingestion failed");
		}

		await new Promise((resolve) => setTimeout(resolve, intervalMs));
	}

	throw new Error(
		"Ingestion timeout - still processing after maximum wait time",
	);
}

async function waitForEvalComplete(
	jobId: string,
	intervalMs: number = 1500,
	timeoutMs: number = 120000,
): Promise<void> {
	const startTime = Date.now();

	while (Date.now() - startTime < timeoutMs) {
		const status = await getEvalStatus(jobId);

		if (status.status === "complete" || status.status === "error") {
			return;
		}

		await new Promise((resolve) => setTimeout(resolve, intervalMs));
	}

	throw new Error(
		"Evaluation timeout - still processing after maximum wait time",
	);
}
