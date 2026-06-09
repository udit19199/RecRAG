"use client";

import { useEffect, useReducer } from "react";
import {
	createInitialModelPickerState,
	modelPickerReducer,
	type VisionConfig,
} from "@/features/model-compare/components/model-picker-types";
import { makeValue, parseValue } from "@/features/model-compare/lib";
import {
	getConfig,
	getProviders,
	setConfig,
	setIngestionConfig,
	triggerReindex,
} from "@/lib/api";
import type { ExtractionOptions } from "@/lib/api/types";

interface UseModelPickerOptions {
	onReindexStarted?: () => void;
	onConfigChanged?: (config: {
		embedding: { provider: string; model: string };
		llm: { provider: string; model: string };
	}) => void;
	onVisionConfigChanged?: (config: VisionConfig | null) => void;
}

function persistLastLLM(provider: string, model: string) {
	try {
		localStorage.setItem("lastLLM", makeValue(provider, model));
	} catch {
		// ignore storage errors
	}
}

function persistLastEmbedding(provider: string, model: string) {
	try {
		localStorage.setItem("lastEmbedding", makeValue(provider, model));
	} catch {
		// ignore storage errors
	}
}

function persistLastVision(value: string | null) {
	try {
		if (value) {
			localStorage.setItem("lastVision", value);
			return;
		}
		localStorage.removeItem("lastVision");
	} catch {
		// ignore storage errors
	}
}

export function useModelPicker({
	onReindexStarted,
	onConfigChanged,
	onVisionConfigChanged,
}: UseModelPickerOptions) {
	const [state, dispatch] = useReducer(
		modelPickerReducer,
		undefined,
		createInitialModelPickerState,
	);

	const isLoading = !state.providers && !state.loadError;

	useEffect(() => {
		let cancelled = false;

		async function bootstrap() {
			try {
				const [cfg, prov] = await Promise.all([getConfig(), getProviders()]);
				if (cancelled) return;

				dispatch({
					type: "bootstrap_success",
					providers: prov,
					embedding: cfg.embedding,
					llm: cfg.llm,
				});
			} catch (err) {
				if (cancelled) return;
				dispatch({
					type: "bootstrap_error",
					error:
						err instanceof Error
							? err.message
							: "Failed to load model options",
				});
			}
		}

		void bootstrap();
		return () => {
			cancelled = true;
		};
	}, []);

	const handleLLMChange = async (value: string) => {
		const [provider, model] = value.split("::");
		if (!provider || !model) return;
		if (
			provider === state.currentLLM?.provider &&
			model === state.currentLLM?.model
		) {
			return;
		}

		dispatch({ type: "set_llm_changing", changing: true });

		try {
			const result = await setConfig({ llm: { provider, model } });
			dispatch({
				type: "llm_changed",
				embedding: result.embedding,
				llm: result.llm,
			});
			persistLastLLM(result.llm.provider, result.llm.model);
			onConfigChanged?.({ embedding: result.embedding, llm: result.llm });
		} catch (err) {
			console.error("Failed to change LLM:", err);
		}

		dispatch({ type: "set_llm_changing", changing: false });
	};

	const handleEmbedChange = (value: string) => {
		const [provider, model] = value.split("::");
		if (!provider || !model) return;
		if (
			provider === state.currentEmbedding?.provider &&
			model === state.currentEmbedding?.model
		) {
			return;
		}
		dispatch({ type: "set_embed_pending", change: { provider, model } });
	};

	const handleVisionChange = (value: string | null) => {
		if (!value) {
			dispatch({ type: "set_vision", vision: null });
			onVisionConfigChanged?.(null);
			persistLastVision(null);
			return;
		}

		const parsed = parseValue(value);
		if (!parsed) return;

		const visionConfig: VisionConfig = {
			provider: parsed.provider as VisionConfig["provider"],
			model: parsed.model,
		};

		if (
			state.currentVision?.provider === visionConfig.provider &&
			state.currentVision?.model === visionConfig.model
		) {
			return;
		}

		dispatch({ type: "set_vision", vision: visionConfig });
		onVisionConfigChanged?.(visionConfig);
		persistLastVision(value);
	};

	const confirmEmbedChange = async () => {
		if (!state.pendingEmbedChange) return;

		const { provider, model } = state.pendingEmbedChange;
		const visionForReindex = state.currentVision;
		dispatch({ type: "set_embed_pending", change: null });
		dispatch({ type: "set_embed_changing", changing: true });

		try {
			const [result] = await Promise.all([
				setConfig({ embedding: { provider, model } }),
				setIngestionConfig({ provider, model }),
			]);

			dispatch({
				type: "embed_changed",
				embedding: result.embedding,
				llm: result.llm,
			});
			persistLastEmbedding(result.embedding.provider, result.embedding.model);
			onConfigChanged?.({ embedding: result.embedding, llm: result.llm });

			if (result.requires_reindex) {
				const extractionOptions: ExtractionOptions | undefined =
					visionForReindex
						? {
								extraction_mode: "vision_assisted",
								vision_provider: visionForReindex.provider,
								vision_model: visionForReindex.model,
							}
						: undefined;
				await triggerReindex(extractionOptions);
				onReindexStarted?.();
			}
		} catch (err) {
			console.error("Failed to change embedding model:", err);
		}

		dispatch({ type: "set_embed_changing", changing: false });
	};

	const dismissEmbedChange = () => {
		dispatch({ type: "set_embed_pending", change: null });
	};

	const currentLLMValue = state.currentLLM
		? makeValue(state.currentLLM.provider, state.currentLLM.model)
		: undefined;
	const currentEmbedValue = state.currentEmbedding
		? makeValue(state.currentEmbedding.provider, state.currentEmbedding.model)
		: undefined;
	const currentVisionValue = state.currentVision
		? makeValue(state.currentVision.provider, state.currentVision.model)
		: undefined;

	return {
		state,
		isLoading,
		currentLLMValue,
		currentEmbedValue,
		currentVisionValue,
		handleLLMChange,
		handleEmbedChange,
		handleVisionChange,
		confirmEmbedChange,
		dismissEmbedChange,
	};
}
