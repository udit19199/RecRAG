import {
	authenticatedFetch,
	getRetrievalApiUrl,
	handleResponse,
} from "@/lib/api/client";
import type {
	ConfigPatch,
	ConfigResponse,
	EvalJobStatus,
	HealthResponse,
	ProvidersResponse,
	QueryRequest,
	QueryWithEvalResponse,
	SetConfigResponse,
} from "@/lib/api/types";

export async function queryRAG(
	request: QueryRequest,
): Promise<QueryWithEvalResponse> {
	const res = await authenticatedFetch(getRetrievalApiUrl("/query"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(request),
	});

	return handleResponse<QueryWithEvalResponse>(res);
}

export async function getEvalStatus(jobId: string): Promise<EvalJobStatus> {
	const res = await fetch(getRetrievalApiUrl(`/evaluate/${jobId}`));
	return handleResponse<EvalJobStatus>(res);
}

export async function checkRetrievalHealth(): Promise<HealthResponse> {
	const res = await fetch(getRetrievalApiUrl("/health"));
	return handleResponse<HealthResponse>(res);
}

export async function getConfig(): Promise<ConfigResponse> {
	const res = await authenticatedFetch(getRetrievalApiUrl("/config"));
	return handleResponse<ConfigResponse>(res);
}

export async function getProviders(): Promise<ProvidersResponse> {
	const res = await authenticatedFetch(getRetrievalApiUrl("/providers"));
	return handleResponse<ProvidersResponse>(res);
}

export async function setConfig(
	patch: ConfigPatch,
): Promise<SetConfigResponse> {
	const res = await authenticatedFetch(getRetrievalApiUrl("/config"), {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(patch),
	});

	return handleResponse<SetConfigResponse>(res);
}
