export interface ContextItem {
	text: string;
	source: string;
	distance: number;
	metadata?: Record<string, unknown> | null;
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
	llm?: AdapterConfig;
	embedding?: AdapterConfig;
	vision?: AdapterConfig;
}

export type ExtractionMode = "text_only" | "vision_assisted";

export type VisionProvider = "openai" | "ollama" | "nim" | "gemini";

export interface ExtractionOptions {
	extraction_mode: ExtractionMode;
	vision_provider?: VisionProvider;
	vision_model?: string;
}

export interface UploadOptions extends ExtractionOptions {
	/** When true, remove existing PDFs before saving the new batch. */
	replace?: boolean;
}

export interface IngestionStatus {
	status: "idle" | "processing" | "complete" | "error";
	started_at?: string;
	completed_at?: string;
	files_processed?: number;
	error_message?: string;
	extraction_mode?: ExtractionMode;
}

export interface UploadResponse {
	success: boolean;
	files_uploaded: number;
	message: string;
	extraction_mode: ExtractionMode;
}

export interface FileListResponse {
	files: string[];
}

export interface HealthResponse {
	status: string;
	service: string;
	pipeline_loaded?: boolean;
	has_documents?: boolean;
	error_message?: string | null;
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
	vision?: Record<string, ProviderInfo>;
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

export interface IndexStatusRequest {
	embedding?: AdapterConfig;
	vision?: AdapterConfig;
}

export interface IndexStatusResponse {
	has_documents: boolean;
}

export interface TargetedIngestRequest {
	extraction_mode: ExtractionMode;
	vision_provider?: string;
	vision_model?: string;
	embedding_provider?: string;
	embedding_model?: string;
}

export interface ReindexResponse {
	started: boolean;
	message: string;
	extraction_mode: ExtractionMode;
}

// --- FiNER-139 experiment (research-only) ---

export type Finer139Method = "llm" | "nlp" | "ontology" | "hybrid" | "dynamic";

export type Finer139RunStatus = "pending" | "running" | "complete" | "error";

export interface Finer139Progress {
	stage: string;
	current: number;
	total: number;
	message: string;
}

export interface Finer139Metrics {
	precision: number;
	recall: number;
	f1: number;
	tp: number;
	fp: number;
	fn: number;
}

export interface Finer139Diagnostics {
	strict: Finer139Metrics;
	relaxed: Finer139Metrics;
	partial: Finer139Metrics;
	macro_strict: Finer139Metrics;
	macro_relaxed: Finer139Metrics;
	macro_partial: Finer139Metrics;
	span_iou_mean: number;
	sentence_hit_rate: number;
	errors: {
		strict_tp?: number;
		boundary_fp?: number;
		spurious_fp?: number;
		missed_fn?: number;
	};
	bootstrap_strict_f1_ci: { low: number; high: number };
	concept_recall_top10: Array<{
		concept: string;
		gold_count: number;
		tp: number;
		recall: number;
	}>;
}

export interface Finer139MethodResult {
	name: Finer139Method;
	display_name: string;
	uses_llm: boolean;
	latency_s: number;
	llm_calls: number;
	error: string | null;
	strict?: Finer139Metrics;
	relaxed?: Finer139Metrics;
	partial?: Finer139Metrics;
	macro_strict?: Finer139Metrics;
	macro_partial?: Finer139Metrics;
	num_pred?: number;
	num_gold?: number;
	diagnostics?: Finer139Diagnostics;
}

export interface Finer139Example {
	index: number;
	tokens: string[];
	text: string;
	gold: [number, number][];
	predictions: Record<string, [number, number][]>;
}

export interface Finer139Results {
	params: {
		sample_size: number;
		seed: number;
		seeds?: number[];
		split?: string;
		stratified?: boolean;
		suite_path?: string | null;
		multi_seed?: boolean;
		methods: Finer139Method[];
		provider: string | null;
		model: string | null;
		spacy_model: string;
	};
	dataset: {
		id: string;
		split: string;
		num_sentences: number;
		num_gold_entities: number;
		stratified?: boolean;
		frozen?: boolean;
		suite_path?: string;
	};
	llm: {
		provider: string | null;
		model: string | null;
		error: string | null;
	};
	methods: Finer139MethodResult[];
	comparison?: {
		sentence_wins?: Record<string, number>;
		sentences_with_gold?: number;
		ties?: number;
	} | null;
	paired_comparisons?: Array<{
		method_a: string;
		method_b: string;
		delta_b_minus_a: {
			delta_mean: number;
			ci_low: number;
			ci_high: number;
			significant: boolean;
		};
		mcnemar: Record<string, number>;
		interpretation: string;
	}>;
	multi_seed_aggregate?: {
		seeds: number[];
		n_runs: number;
		by_method: Record<
			string,
			{
				strict_f1_mean: number;
				strict_f1_std: number;
				strict_f1_min: number;
				strict_f1_max: number;
				n_seeds: number;
				partial_f1_mean?: number;
			}
		>;
		ranking: string[];
	};
	per_seed_runs?: Finer139Results[];
	evaluation?: {
		protocol_version?: number;
		primary_metric?: string;
		secondary_metrics?: string[];
		partial_match_iou_threshold?: number;
		split?: string;
		stratified_sampling?: boolean;
		multi_seed?: boolean;
		n_seeds?: number;
	} | null;
	examples: Finer139Example[];
}

export interface Finer139RunResponse {
	run_id: string;
	status: Finer139RunStatus;
	progress?: Finer139Progress | null;
	params?: Finer139Results["params"] | null;
	results?: Finer139Results | null;
	error?: string | null;
	created_at?: string | null;
	updated_at?: string | null;
}

export interface Finer139StartRequest {
	sample_size?: number;
	seed?: number;
	seeds?: number[];
	split?: "train" | "validation" | "test";
	stratified?: boolean;
	suite_path?: string | null;
	methods?: Finer139Method[];
	provider?: string | null;
	model?: string | null;
}
