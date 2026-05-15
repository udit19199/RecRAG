const RETRIEVAL_API_URL =
	process.env.NEXT_PUBLIC_RETRIEVAL_API_URL || "http://localhost:8000";
const INGESTION_API_URL =
	process.env.NEXT_PUBLIC_INGESTION_API_URL || "http://localhost:8001";

// API key for authenticated requests (set via environment variable)
const API_KEY = process.env.NEXT_PUBLIC_REC_RAG_API_KEY || "";

function getAuthHeaders(): Record<string, string> {
	if (API_KEY) {
		return { "RecRAG-API-Key": API_KEY };
	}
	return {};
}

export function getRetrievalApiUrl(path: string) {
	return `${RETRIEVAL_API_URL}${path}`;
}

export function getIngestionApiUrl(path: string) {
	return `${INGESTION_API_URL}${path}`;
}

export async function authenticatedFetch(
	url: string,
	options: RequestInit = {},
): Promise<Response> {
	const headers = {
		...getAuthHeaders(),
		...(options.headers || {}),
	};

	return fetch(url, {
		...options,
		headers,
	});
}

export async function handleResponse<T>(res: Response): Promise<T> {
	if (!res.ok) {
		const error = await res.json().catch(() => ({ detail: "Unknown error" }));
		throw new Error(error.detail || `HTTP ${res.status}`);
	}

	return res.json() as Promise<T>;
}
