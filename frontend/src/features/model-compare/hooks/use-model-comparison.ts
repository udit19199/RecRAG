"use client";

import { useEffect, useReducer, useRef } from "react";
import type { ChatMessage } from "@/features/chat/types";
import {
	INITIAL_MODEL_COMPARISON_STATE,
	modelComparisonReducer,
	type SlotKey,
	type SlotState,
} from "@/features/model-compare/components/model-comparison-types";
import { makeValue, parseValue } from "@/features/model-compare/lib";
import { fetchSlotHasDocuments } from "@/features/model-compare/lib/slot-index";
import {
	getConfig,
	getIngestionStatus,
	getProviders,
	type QueryResponse,
	queryRAG,
	triggerTargetedIngest,
} from "@/lib/api";

export function useModelComparison() {
	const [state, dispatch] = useReducer(
		modelComparisonReducer,
		INITIAL_MODEL_COMPARISON_STATE,
	);
	const messageIdRef = useRef(0);
	const chatBottomRef = useRef<HTMLDivElement>(null);
	const stateRef = useRef(state);
	useEffect(() => {
		stateRef.current = state;
	});

	const isLoading = !state.providers && !state.error;
	const canChat =
		state.slotA.hasDocuments &&
		state.slotB.hasDocuments &&
		state.slotA.llm &&
		state.slotB.llm &&
		!state.isIngesting;

	const nextId = () => messageIdRef.current++;

	const refreshSlotIndexStatus = async (
		slotKey: SlotKey,
		embedding: string,
		vision: string | null,
	) => {
		await Promise.resolve();
		dispatch({
			type: "patch_slot",
			slot: slotKey,
			patch: { isChecking: true },
		});

		try {
			const hasDocuments = await fetchSlotHasDocuments(embedding, vision);
			dispatch({
				type: "patch_slot",
				slot: slotKey,
				patch: { hasDocuments, isChecking: false },
			});
		} catch (err) {
			console.error(`Failed to check index status for Slot ${slotKey}:`, err);
			dispatch({
				type: "patch_slot",
				slot: slotKey,
				patch: { isChecking: false },
			});
		}
	};

	useEffect(() => {
		let cancelled = false;

		async function bootstrap() {
			try {
				const [nextConfig, nextProviders] = await Promise.all([
					getConfig(),
					getProviders(),
				]);
				if (cancelled) return;

				dispatch({ type: "set_providers", providers: nextProviders });

				const defaultLlm = makeValue(
					nextConfig.llm.provider,
					nextConfig.llm.model,
				);
				const defaultEmbed = makeValue(
					nextConfig.embedding.provider,
					nextConfig.embedding.model,
				);

				const providerMap = nextProviders.llms;
				const keys = Object.entries(providerMap).flatMap(([provider, info]) =>
					info.available
						? info.models.map((model) => makeValue(provider, model))
						: [],
				);

				const defaultsSet = new Set<string>();
				if (keys.includes(defaultLlm)) {
					defaultsSet.add(defaultLlm);
				}

				for (const key of keys) {
					if (!defaultsSet.has(key)) defaultsSet.add(key);
					if (defaultsSet.size === 2) break;
				}

				const defaults = [...defaultsSet];

				dispatch({
					type: "patch_slot",
					slot: "A",
					patch: { embedding: defaultEmbed, llm: defaults[0] || null },
				});
				dispatch({
					type: "patch_slot",
					slot: "B",
					patch: { embedding: defaultEmbed, llm: defaults[1] || null },
				});

				await refreshSlotIndexStatus("A", defaultEmbed, null);
				if (!cancelled) {
					await refreshSlotIndexStatus("B", defaultEmbed, null);
				}
			} catch (err) {
				if (cancelled) return;
				dispatch({
					type: "set_error",
					error:
						err instanceof Error
							? err.message
							: "Unable to load model inventory",
				});
			}
		}

		void bootstrap();
		return () => {
			cancelled = true;
		};
		// biome-ignore lint/correctness/useExhaustiveDependencies: only bootstrap on mount
	}, []);

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const patchSlot = (slot: SlotKey, patch: Partial<SlotState>) => {
		dispatch({ type: "patch_slot", slot, patch });
	};

	const handleSlotVisionChange = (slot: SlotKey, vision: string | null) => {
		const slotState =
			slot === "A" ? stateRef.current.slotA : stateRef.current.slotB;
		patchSlot(slot, { vision });
		if (slotState.embedding) {
			void refreshSlotIndexStatus(slot, slotState.embedding, vision);
		}
	};

	const handleSlotEmbedChange = (slot: SlotKey, embedding: string) => {
		const slotState =
			slot === "A" ? stateRef.current.slotA : stateRef.current.slotB;
		patchSlot(slot, { embedding });
		void refreshSlotIndexStatus(slot, embedding, slotState.vision);
	};

	const handleSlotLlmChange = (slot: SlotKey, llm: string) => {
		patchSlot(slot, { llm });
	};

	const handleIngest = async (slotKey: SlotKey) => {
		const slotState =
			slotKey === "A" ? stateRef.current.slotA : stateRef.current.slotB;
		if (!slotState.embedding || stateRef.current.isIngesting) return;

		const embedConfig = parseValue(slotState.embedding);
		const visionConfig = slotState.vision ? parseValue(slotState.vision) : null;

		if (!embedConfig) return;

		dispatch({
			type: "set_ingestion",
			isIngesting: true,
			ingestingSlot: slotKey,
		});

		try {
			await triggerTargetedIngest({
				extraction_mode: visionConfig ? "vision_assisted" : "text_only",
				vision_provider: visionConfig?.provider,
				vision_model: visionConfig?.model,
				embedding_provider: embedConfig.provider,
				embedding_model: embedConfig.model,
			});

			const poll = async () => {
				const status = await getIngestionStatus();
				if (status.status === "complete") {
					const current = stateRef.current;
					if (current.slotA.embedding) {
						await refreshSlotIndexStatus(
							"A",
							current.slotA.embedding,
							current.slotA.vision,
						);
					}
					if (current.slotB.embedding) {
						await refreshSlotIndexStatus(
							"B",
							current.slotB.embedding,
							current.slotB.vision,
						);
					}
					dispatch({
						type: "set_ingestion",
						isIngesting: false,
						ingestingSlot: null,
					});
				} else if (status.status === "error") {
					dispatch({
						type: "set_error",
						error: `Ingestion failed: ${status.error_message}`,
					});
					dispatch({
						type: "set_ingestion",
						isIngesting: false,
						ingestingSlot: null,
					});
				} else {
					setTimeout(poll, 3000);
				}
			};

			setTimeout(poll, 3000);
		} catch (err) {
			dispatch({
				type: "set_error",
				error: err instanceof Error ? err.message : "Failed to start ingestion",
			});
			dispatch({
				type: "set_ingestion",
				isIngesting: false,
				ingestingSlot: null,
			});
		}
	};

	const appendAssistantMessage = (
		slot: SlotKey,
		result: PromiseSettledResult<QueryResponse>,
	) => {
		const message: ChatMessage =
			result.status === "fulfilled"
				? {
						id: nextId(),
						role: "assistant",
						content: result.value.response,
						context: result.value.context,
					}
				: {
						id: nextId(),
						role: "assistant",
						content: "",
						error:
							result.reason instanceof Error
								? result.reason.message
								: "Query failed",
					};
		dispatch({ type: "append_messages", slot, messages: [message] });
	};

	const handleQuery = async (query: string) => {
		const current = stateRef.current;
		if (
			!query.trim() ||
			current.isQuerying ||
			!current.slotA.llm ||
			!current.slotB.llm
		) {
			return;
		}

		const llmA = parseValue(current.slotA.llm);
		const llmB = parseValue(current.slotB.llm);
		const embedA = parseValue(current.slotA.embedding);
		const embedB = parseValue(current.slotB.embedding);
		const visionA = parseValue(current.slotA.vision);
		const visionB = parseValue(current.slotB.vision);

		if (!llmA || !llmB || !embedA || !embedB) return;

		const userMessageA: ChatMessage = {
			id: nextId(),
			role: "user",
			content: query,
		};
		const userMessageB: ChatMessage = {
			id: nextId(),
			role: "user",
			content: query,
		};

		dispatch({
			type: "append_messages",
			slot: "A",
			messages: [userMessageA],
		});
		dispatch({
			type: "append_messages",
			slot: "B",
			messages: [userMessageB],
		});
		dispatch({ type: "set_querying", isQuerying: true });

		const [resultA, resultB] = await Promise.allSettled([
			queryRAG({
				query,
				llm: llmA,
				embedding: embedA,
				vision: visionA
					? { provider: visionA.provider, model: visionA.model }
					: undefined,
			}),
			queryRAG({
				query,
				llm: llmB,
				embedding: embedB,
				vision: visionB
					? { provider: visionB.provider, model: visionB.model }
					: undefined,
			}),
		]);

		appendAssistantMessage("A", resultA);
		appendAssistantMessage("B", resultB);
		dispatch({ type: "set_querying", isQuerying: false });
	};

	return {
		state,
		isLoading,
		canChat,
		chatBottomRef,
		handleSlotVisionChange,
		handleSlotEmbedChange,
		handleSlotLlmChange,
		handleIngest,
		handleQuery,
	};
}
