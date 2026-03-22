export interface ContextItem {
	text: string;
	source: string;
	distance: number;
}

export interface QueryResponse {
	response: string;
	context: ContextItem[];
}

export interface EvalScores {
	[metric: string]: number;
}

export interface EvalJobStatus {
	id: string;
	status: "pending" | "complete" | "error";
	query?: string;
	created_at?: string;
	updated_at?: string;
	scores?: EvalScores;
	error?: string;
}

export interface QueryWithEvalResponse extends QueryResponse {
	eval_job_id?: string;
}

export interface QueryRequest {
	query: string;
}

export interface IngestionStatus {
	status: "idle" | "processing" | "complete" | "error";
	started_at?: string;
	completed_at?: string;
	files_processed?: number;
	error_message?: string;
}

export interface UploadResponse {
	success: boolean;
	files_uploaded: number;
	message: string;
}

export interface HealthResponse {
	status: string;
	service: string;
	pipeline_loaded?: boolean;
}

export interface AdapterConfig {
	provider: string;
	model: string;
}

export interface ConfigResponse {
	embedding: AdapterConfig;
	llm: AdapterConfig;
}

export interface ProviderInfo {
	available: boolean;
	models: string[];
	reason?: string;
}

export interface ProvidersResponse {
	embedders: Record<string, ProviderInfo>;
	llms: Record<string, ProviderInfo>;
}

export interface ConfigPatch {
	embedding?: AdapterConfig;
	llm?: AdapterConfig;
}

export interface SetConfigResponse {
	applied: boolean;
	requires_reindex: boolean;
	embedding: AdapterConfig;
	llm: AdapterConfig;
}

export interface ReindexResponse {
	started: boolean;
	message: string;
}
