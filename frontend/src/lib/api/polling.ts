import { getIngestionStatus } from "@/lib/api/ingestion";
import type { IngestionStatus } from "@/lib/api/types";

export async function waitForIngestionComplete(
	intervalMs: number = 2000,
	timeoutMs: number = 120000,
	onPoll?: (status: IngestionStatus) => void,
): Promise<IngestionStatus> {
	const startTime = Date.now();

	const poll = async (): Promise<IngestionStatus> => {
		const status = await getIngestionStatus();
		onPoll?.(status);

		if (status.status === "complete") {
			return status;
		}

		if (status.status === "error") {
			throw new Error(status.error_message || "Ingestion failed");
		}

		if (Date.now() - startTime >= timeoutMs) {
			throw new Error(
				"Ingestion timeout - still processing after maximum wait time",
			);
		}

		await new Promise((resolve) => setTimeout(resolve, intervalMs));
		return poll();
	};

	return poll();
}
