import { getIngestionApiUrl, handleResponse } from "@/lib/api/client";
import type {
	AdapterConfig,
	HealthResponse,
	IngestionStatus,
	ReindexResponse,
	UploadResponse,
} from "@/lib/api/types";

export async function uploadPDFs(files: File[]): Promise<UploadResponse> {
	const formData = new FormData();

	for (const file of files) {
		formData.append("files", file);
	}

	const res = await fetch(getIngestionApiUrl("/upload"), {
		method: "POST",
		body: formData,
	});

	return handleResponse<UploadResponse>(res);
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
	const res = await fetch(getIngestionApiUrl("/status"));
	return handleResponse<IngestionStatus>(res);
}

export async function checkIngestionHealth(): Promise<HealthResponse> {
	const res = await fetch(getIngestionApiUrl("/health"));
	return handleResponse<HealthResponse>(res);
}

export async function setIngestionConfig(
	embedding: AdapterConfig,
): Promise<void> {
	const res = await fetch(getIngestionApiUrl("/config"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ embedding }),
	});

	await handleResponse<unknown>(res);
}

export async function triggerReindex(): Promise<ReindexResponse> {
	const res = await fetch(getIngestionApiUrl("/reindex"), {
		method: "POST",
	});

	return handleResponse<ReindexResponse>(res);
}
