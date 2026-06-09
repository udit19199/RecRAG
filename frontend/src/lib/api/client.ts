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

export function uploadFormData(
	url: string,
	formData: FormData,
	onProgress?: (percent: number) => void,
): Promise<Response> {
	return new Promise((resolve, reject) => {
		const xhr = new XMLHttpRequest();
		xhr.open("POST", url);

		for (const [key, value] of Object.entries(getAuthHeaders())) {
			xhr.setRequestHeader(key, value);
		}

		xhr.upload.onprogress = (event) => {
			if (event.lengthComputable && onProgress) {
				onProgress(Math.round((event.loaded / event.total) * 100));
			}
		};

		xhr.onload = () => {
			resolve(
				new Response(xhr.responseText, {
					status: xhr.status,
					statusText: xhr.statusText,
				}),
			);
		};

		xhr.onerror = () => reject(new Error("Network error during upload"));
		xhr.onabort = () => reject(new Error("Upload cancelled"));

		xhr.send(formData);
	});
}

export async function handleResponse<T>(res: Response): Promise<T> {
	if (!res.ok) {
		const error = await res.json().catch(() => ({ detail: "Unknown error" }));
		throw new Error(error.detail || `HTTP ${res.status}`);
	}

	return res.json() as Promise<T>;
}
