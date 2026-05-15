import {
	authenticatedFetch,
	getIngestionApiUrl,
	handleResponse,
} from "@/lib/api/client";
import type {
	AdapterConfig,
	ExtractionOptions,
	FileListResponse,
	HealthResponse,
	IndexStatusRequest,
	IndexStatusResponse,
	IngestionStatus,
	ReindexResponse,
	TargetedIngestRequest,
	UploadResponse,
} from "@/lib/api/types";

export async function uploadPDFs(
	files: File[],
	options?: ExtractionOptions,
): Promise<UploadResponse> {
	const formData = new FormData();

	for (const file of files) {
		formData.append("files", file);
	}

	// Add extraction options to form data
	if (options) {
		formData.append("extraction_mode", options.extraction_mode);
		if (options.vision_provider) {
			formData.append("vision_provider", options.vision_provider);
		}
		if (options.vision_model) {
			formData.append("vision_model", options.vision_model);
		}
	}

	const res = await authenticatedFetch(getIngestionApiUrl("/upload"), {
		method: "POST",
		body: formData,
	});

	return handleResponse<UploadResponse>(res);
}

export async function getIngestionStatus(): Promise<IngestionStatus> {
	const res = await authenticatedFetch(getIngestionApiUrl("/status"));
	return handleResponse<IngestionStatus>(res);
}

export async function checkIngestionHealth(): Promise<HealthResponse> {
	const res = await fetch(getIngestionApiUrl("/health"));
	return handleResponse<HealthResponse>(res);
}

export async function getUploadedFiles(): Promise<FileListResponse> {
	const res = await authenticatedFetch(getIngestionApiUrl("/files"));
	return handleResponse<FileListResponse>(res);
}

export async function setIngestionConfig(
	embedding: AdapterConfig,
): Promise<void> {
	const res = await authenticatedFetch(getIngestionApiUrl("/config"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ embedding }),
	});

	await handleResponse<unknown>(res);
}

export async function triggerReindex(
	options?: ExtractionOptions,
): Promise<ReindexResponse> {
	const res = await authenticatedFetch(getIngestionApiUrl("/reindex"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: options ? JSON.stringify(options) : undefined,
	});

	return handleResponse<ReindexResponse>(res);
}

export async function checkIndexStatus(
	request: IndexStatusRequest,
): Promise<IndexStatusResponse> {
	const res = await authenticatedFetch(getIngestionApiUrl("/status/index"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(request),
	});

	return handleResponse<IndexStatusResponse>(res);
}

export async function triggerTargetedIngest(
	request: TargetedIngestRequest,
): Promise<ReindexResponse> {
	const res = await authenticatedFetch(getIngestionApiUrl("/ingest/target"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(request),
	});

	return handleResponse<ReindexResponse>(res);
}
