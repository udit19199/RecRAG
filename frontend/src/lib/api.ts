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

const RETRIEVAL_API_URL = process.env.NEXT_PUBLIC_RETRIEVAL_API_URL || 'http://localhost:8000';
const INGESTION_API_URL = process.env.NEXT_PUBLIC_INGESTION_API_URL || 'http://localhost:8001';

/**
 * Context item from RAG retrieval
 */
export interface ContextItem {
  text: string;
  source: string;
  distance: number;
}

/**
 * Response from query endpoint
 */
export interface QueryResponse {
  response: string;
  context: ContextItem[];
}

/**
 * Request body for query endpoint
 */
export interface QueryRequest {
  query: string;
}

/**
 * Ingestion status response
 */
export interface IngestionStatus {
  status: 'idle' | 'processing' | 'complete' | 'error';
  started_at?: string;
  completed_at?: string;
  files_processed?: number;
  error_message?: string;
}

/**
 * Upload response
 */
export interface UploadResponse {
  success: boolean;
  filename: string;
  message: string;
}

/**
 * Health check response
 */
export interface HealthResponse {
  status: string;
  service: string;
  pipeline_loaded?: boolean;
}

/**
 * Query the RAG pipeline
 * @param query - The question to ask
 * @returns QueryResponse with answer and context
 * @throws Error if the request fails
 */
export async function queryRAG(query: string): Promise<QueryResponse> {
  const response = await fetch(`${RETRIEVAL_API_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Check retrieval API health
 * @returns Health status
 */
export async function checkRetrievalHealth(): Promise<HealthResponse> {
  const response = await fetch(`${RETRIEVAL_API_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Upload a PDF file for ingestion
 * @param file - The PDF file to upload
 * @returns UploadResponse with status
 * @throws Error if upload fails
 */
export async function uploadPDF(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${INGESTION_API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get current ingestion status
 * @returns IngestionStatus
 * @throws Error if the request fails
 */
export async function getIngestionStatus(): Promise<IngestionStatus> {
  const response = await fetch(`${INGESTION_API_URL}/status`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Check ingestion API health
 * @returns Health status
 */
export async function checkIngestionHealth(): Promise<HealthResponse> {
  const response = await fetch(`${INGESTION_API_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Poll ingestion status until complete or error
 * @param intervalMs - Polling interval in milliseconds
 * @param timeoutMs - Maximum time to wait in milliseconds
 * @returns Final ingestion status
 * @throws Error if timeout or error
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

    // Still processing, wait and poll again
    await new Promise(resolve => setTimeout(resolve, intervalMs));
  }

  throw new Error('Ingestion timeout - still processing after maximum wait time');
}
