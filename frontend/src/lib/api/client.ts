const RETRIEVAL_API_URL =
	process.env.NEXT_PUBLIC_RETRIEVAL_API_URL || "http://localhost:8000";
const INGESTION_API_URL =
	process.env.NEXT_PUBLIC_INGESTION_API_URL || "http://localhost:8001";

export function getRetrievalApiUrl(path: string) {
	return `${RETRIEVAL_API_URL}${path}`;
}

export function getIngestionApiUrl(path: string) {
	return `${INGESTION_API_URL}${path}`;
}

export async function handleResponse<T>(res: Response): Promise<T> {
	if (!res.ok) {
		const error = await res.json().catch(() => ({ detail: "Unknown error" }));
		throw new Error(error.detail || `HTTP ${res.status}`);
	}

	return res.json() as Promise<T>;
}
