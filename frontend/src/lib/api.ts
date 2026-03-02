/**
 * RecRAG API Client
 *
 * Provides typed interfaces for communicating with the RecRAG backend services:
 * - Retrieval API: For querying documents with RAG
 * - Ingestion API: For uploading PDFs and checking status
 *
 * Environment variables:
 * - NEXT_PUBLIC_RETRIEVAL_API_URL: URL for retrieval API (default: http://localhost:8000)
 * - NEXT_PUBLIC_INGESTION_API_URL: URL for ingestion API (default: http://localhost:8001)
 */

const RETRIEVAL_API_URL =
  process.env.NEXT_PUBLIC_RETRIEVAL_API_URL || 'http://localhost:8000';
const INGESTION_API_URL =
  process.env.NEXT_PUBLIC_INGESTION_API_URL || 'http://localhost:8001';

// ── Core types ────────────────────────────────────────────────────────────────

/** Context item from RAG retrieval */
export interface ContextItem {
  text: string;
  source: string;
  distance: number;
}

/** Response from /query endpoint */
export interface QueryResponse {
  response: string;
  context: ContextItem[];
}

/** Request body for /query endpoint */
export interface QueryRequest {
  query: string;
}

/** Ingestion status response from /status */
export interface IngestionStatus {
  status: 'idle' | 'processing' | 'complete' | 'error';
  started_at?: string;
  completed_at?: string;
  files_processed?: number;
  error_message?: string;
}

/** Upload response from /upload */
export interface UploadResponse {
  success: boolean;
  filename: string;
  message: string;
}

/** Health check response */
export interface HealthResponse {
  status: string;
  service: string;
  pipeline_loaded?: boolean;
}

// ── Model / config types ──────────────────────────────────────────────────────

/** A single provider + model selection */
export interface AdapterConfig {
  provider: string;
  model: string;
}

/** Full current config from GET /config */
export interface ConfigResponse {
  embedding: AdapterConfig;
  llm: AdapterConfig;
}

/** Per-provider info from GET /providers */
export interface ProviderInfo {
  available: boolean;
  models: string[];
  reason?: string;
}

/** Full providers response from GET /providers */
export interface ProvidersResponse {
  embedders: Record<string, ProviderInfo>;
  llms: Record<string, ProviderInfo>;
}

/** Payload for POST /config (both fields optional) */
export interface ConfigPatch {
  embedding?: AdapterConfig;
  llm?: AdapterConfig;
}

/** Response from POST /config on the retrieval API */
export interface SetConfigResponse {
  applied: boolean;
  requires_reindex: boolean;
  embedding: AdapterConfig;
  llm: AdapterConfig;
}

/** Response from POST /reindex */
export interface ReindexResponse {
  started: boolean;
  message: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Retrieval API ─────────────────────────────────────────────────────────────

/**
 * Query the RAG pipeline.
 */
export async function queryRAG(query: string): Promise<QueryResponse> {
  const res = await fetch(`${RETRIEVAL_API_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  });
  return handleResponse<QueryResponse>(res);
}

/**
 * Check retrieval API health.
 */
export async function checkRetrievalHealth(): Promise<HealthResponse> {
  const res = await fetch(`${RETRIEVAL_API_URL}/health`);
  return handleResponse<HealthResponse>(res);
}

/**
 * Get currently active embedding and LLM configuration.
 */
export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${RETRIEVAL_API_URL}/config`);
  return handleResponse<ConfigResponse>(res);
}

/**
 * Get all available providers and their models.
 * Ollama and OpenAI/NIM models are fetched live when credentials are available.
 */
export async function getProviders(): Promise<ProvidersResponse> {
  const res = await fetch(`${RETRIEVAL_API_URL}/providers`);
  return handleResponse<ProvidersResponse>(res);
}

/**
 * Update the active LLM and/or embedding model at runtime.
 * Changing the embedding model may require a re-index (check `requires_reindex`).
 */
export async function setConfig(patch: ConfigPatch): Promise<SetConfigResponse> {
  const res = await fetch(`${RETRIEVAL_API_URL}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  });
  return handleResponse<SetConfigResponse>(res);
}

// ── Ingestion API ─────────────────────────────────────────────────────────────

/**
 * Upload a PDF file for ingestion.
 */
export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${INGESTION_API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  return handleResponse<UploadResponse>(res);
}

/**
 * Get current ingestion status.
 */
export async function getIngestionStatus(): Promise<IngestionStatus> {
  const res = await fetch(`${INGESTION_API_URL}/status`);
  return handleResponse<IngestionStatus>(res);
}

/**
 * Check ingestion API health.
 */
export async function checkIngestionHealth(): Promise<HealthResponse> {
  const res = await fetch(`${INGESTION_API_URL}/health`);
  return handleResponse<HealthResponse>(res);
}

/**
 * Update the embedding model on the ingestion pipeline at runtime.
 */
export async function setIngestionConfig(embedding: AdapterConfig): Promise<void> {
  const res = await fetch(`${INGESTION_API_URL}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding }),
  });
  await handleResponse<unknown>(res);
}

/**
 * Trigger a full re-index of all previously uploaded documents.
 * Uses the currently active embedding model.
 */
export async function triggerReindex(): Promise<ReindexResponse> {
  const res = await fetch(`${INGESTION_API_URL}/reindex`, {
    method: 'POST',
  });
  return handleResponse<ReindexResponse>(res);
}

/**
 * Poll ingestion status until complete or error.
 */
export async function waitForIngestionComplete(
  intervalMs: number = 2000,
  timeoutMs: number = 120000
): Promise<IngestionStatus> {
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    const status = await getIngestionStatus();

    if (status.status === 'complete') {
      return status;
    }

    if (status.status === 'error') {
      throw new Error(status.error_message || 'Ingestion failed');
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new Error('Ingestion timeout - still processing after maximum wait time');
}
