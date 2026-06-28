import {
	readStoredVisionConfig,
	type StoredVisionConfig,
} from "@/features/model-compare/lib";
import type { AdapterConfig, ProvidersResponse } from "@/lib/api";

export type VisionConfig = StoredVisionConfig;

export type PendingEmbedChange = { provider: string; model: string } | null;

export type ModelPickerState = {
	providers: ProvidersResponse | null;
	loadError: string | null;
	currentEmbedding: AdapterConfig | null;
	currentLLM: AdapterConfig | null;
	currentVision: VisionConfig | null;
	llmChanging: boolean;
	embedChanging: boolean;
	pendingEmbedChange: PendingEmbedChange;
};

export function createInitialModelPickerState(): ModelPickerState {
	return {
		providers: null,
		loadError: null,
		currentEmbedding: null,
		currentLLM: null,
		currentVision: readStoredVisionConfig(),
		llmChanging: false,
		embedChanging: false,
		pendingEmbedChange: null,
	};
}

export type ModelPickerAction =
	| {
			type: "bootstrap_success";
			providers: ProvidersResponse;
			embedding: AdapterConfig;
			llm: AdapterConfig;
	  }
	| { type: "bootstrap_error"; error: string }
	| { type: "set_llm_changing"; changing: boolean }
	| {
			type: "llm_changed";
			embedding: AdapterConfig;
			llm: AdapterConfig;
	  }
	| { type: "set_embed_pending"; change: PendingEmbedChange }
	| { type: "set_embed_changing"; changing: boolean }
	| {
			type: "embed_changed";
			embedding: AdapterConfig;
			llm: AdapterConfig;
	  }
	| { type: "set_vision"; vision: VisionConfig | null };

export function modelPickerReducer(
	state: ModelPickerState,
	action: ModelPickerAction,
): ModelPickerState {
	switch (action.type) {
		case "bootstrap_success":
			return {
				...state,
				providers: action.providers,
				currentEmbedding: action.embedding,
				currentLLM: action.llm,
				loadError: null,
			};
		case "bootstrap_error":
			return { ...state, loadError: action.error };
		case "set_llm_changing":
			return { ...state, llmChanging: action.changing };
		case "llm_changed":
			return {
				...state,
				currentEmbedding: action.embedding,
				currentLLM: action.llm,
			};
		case "set_embed_pending":
			return { ...state, pendingEmbedChange: action.change };
		case "set_embed_changing":
			return { ...state, embedChanging: action.changing };
		case "embed_changed":
			return {
				...state,
				currentEmbedding: action.embedding,
				currentLLM: action.llm,
			};
		case "set_vision":
			return { ...state, currentVision: action.vision };
		default:
			return state;
	}
}
