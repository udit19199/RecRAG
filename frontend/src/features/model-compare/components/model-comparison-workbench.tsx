"use client";

import { Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Empty,
	EmptyDescription,
	EmptyHeader,
	EmptyMedia,
	EmptyTitle,
} from "@/components/ui/empty";
import { ChatInput } from "@/features/chat/components/chat-input";
import { ChatMessageView } from "@/features/chat/components/chat-message";
import { TypingIndicator } from "@/features/chat/components/typing-indicator";
import type { ChatMessage } from "@/features/chat/types";
import { makeValue, parseValue } from "@/features/model-compare/lib";
import {
	checkIndexStatus,
	getConfig,
	getIngestionStatus,
	getProviders,
	type ProvidersResponse,
	type QueryResponse,
	queryRAG,
	triggerTargetedIngest,
} from "@/lib/api";
import { SlotPicker } from "./slot-picker";

type SlotState = {
	vision: string | null;
	embedding: string | null;
	llm: string | null;
	hasDocuments: boolean;
	isChecking: boolean;
};

export function ModelComparisonWorkbench() {
	const [providers, setProviders] = useState<ProvidersResponse | null>(null);
	const [error, setError] = useState<string | null>(null);
	const isLoading = !providers && !error;

	const [slotA, setSlotA] = useState<SlotState>({
		vision: null,
		embedding: null,
		llm: null,
		hasDocuments: false,
		isChecking: false,
	});
	const [slotB, setSlotB] = useState<SlotState>({
		vision: null,
		embedding: null,
		llm: null,
		hasDocuments: false,
		isChecking: false,
	});

	const [ingestingSlot, setIngestingSlot] = useState<"A" | "B" | null>(null);
	const [isIngesting, setIsIngesting] = useState(false);

	// Chat state
	const [messagesA, setMessagesA] = useState<ChatMessage[]>([]);
	const [messagesB, setMessagesB] = useState<ChatMessage[]>([]);
	const [isQuerying, setIsQuerying] = useState(false);
	const messageIdRef = useRef(0);
	const chatBottomRef = useRef<HTMLDivElement>(null);

	const nextId = () => {
		return messageIdRef.current++;
	};

	const loadData = useCallback(async () => {
		setError(null);

		try {
			const [nextConfig, nextProviders] = await Promise.all([
				getConfig(),
				getProviders(),
			]);
			setProviders(nextProviders);

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

			setSlotA((prev) => ({
				...prev,
				embedding: defaultEmbed,
				llm: defaults[0] || null,
			}));
			setSlotB((prev) => ({
				...prev,
				embedding: defaultEmbed,
				llm: defaults[1] || null,
			}));
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Unable to load model inventory",
			);
		}
	}, []);

	useEffect(() => {
		loadData();
	}, [loadData]);

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const checkSlotStatus = useCallback(
		async (
			slotKey: "A" | "B",
			embedding: string | null,
			vision: string | null,
		) => {
			if (!embedding) return;

			const setSlot = slotKey === "A" ? setSlotA : setSlotB;
			setSlot((prev) => ({ ...prev, isChecking: true }));

			try {
				const embedConfig = parseValue(embedding);
				const visionConfig = vision ? parseValue(vision) : undefined;

				if (embedConfig) {
					const res = await checkIndexStatus({
						embedding: embedConfig,
						vision: visionConfig
							? { provider: visionConfig.provider, model: visionConfig.model }
							: undefined,
					});
					setSlot((prev) => ({ ...prev, hasDocuments: res.has_documents }));
				}
			} catch (err) {
				console.error(`Failed to check index status for Slot ${slotKey}:`, err);
			} finally {
				setSlot((prev) => ({ ...prev, isChecking: false }));
			}
		},
		[],
	);

	// Re-check status when vision or embedding changes
	useEffect(() => {
		if (slotA.embedding) checkSlotStatus("A", slotA.embedding, slotA.vision);
	}, [slotA.embedding, slotA.vision, checkSlotStatus]);

	useEffect(() => {
		if (slotB.embedding) checkSlotStatus("B", slotB.embedding, slotB.vision);
	}, [slotB.embedding, slotB.vision, checkSlotStatus]);

	const handleIngest = async (slotKey: "A" | "B", state: SlotState) => {
		if (!state.embedding || isIngesting) return;

		const embedConfig = parseValue(state.embedding);
		const visionConfig = state.vision ? parseValue(state.vision) : null;

		if (!embedConfig) return;

		setIngestingSlot(slotKey);
		setIsIngesting(true);

		try {
			await triggerTargetedIngest({
				extraction_mode: visionConfig ? "vision_assisted" : "text_only",
				vision_provider: visionConfig?.provider,
				vision_model: visionConfig?.model,
				embedding_provider: embedConfig.provider,
				embedding_model: embedConfig.model,
			});

			// Poll for completion
			const poll = async () => {
				const status = await getIngestionStatus();
				if (status.status === "complete") {
					await checkSlotStatus("A", slotA.embedding, slotA.vision);
					await checkSlotStatus("B", slotB.embedding, slotB.vision);
					setIsIngesting(false);
					setIngestingSlot(null);
				} else if (status.status === "error") {
					setError(`Ingestion failed: ${status.error_message}`);
					setIsIngesting(false);
					setIngestingSlot(null);
				} else {
					setTimeout(poll, 3000);
				}
			};

			setTimeout(poll, 3000);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Failed to start ingestion",
			);
			setIsIngesting(false);
			setIngestingSlot(null);
		}
	};

	const handleQuery = async (query: string) => {
		if (!query.trim() || isQuerying || !slotA.llm || !slotB.llm) return;

		const llmA = parseValue(slotA.llm);
		const llmB = parseValue(slotB.llm);
		const embedA = parseValue(slotA.embedding);
		const embedB = parseValue(slotB.embedding);
		const visionA = parseValue(slotA.vision);
		const visionB = parseValue(slotB.vision);

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

		setMessagesA((prev) => [...prev, userMessageA]);
		setMessagesB((prev) => [...prev, userMessageB]);
		setIsQuerying(true);

		const handleSlotResult = (
			result: PromiseSettledResult<QueryResponse>,
			setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
		) => {
			setMessages((prev) => [
				...prev,
				result.status === "fulfilled"
					? {
							id: nextId(),
							role: "assistant" as const,
							content: result.value.response,
							context: result.value.context,
						}
					: {
							id: nextId(),
							role: "assistant" as const,
							content: "",
							error:
								result.reason instanceof Error
									? result.reason.message
									: "Query failed",
						},
			]);
		};

		try {
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

			handleSlotResult(resultA, setMessagesA);
			handleSlotResult(resultB, setMessagesB);
		} finally {
			setIsQuerying(false);
		}
	};

	const renderSlotStatus = (slotKey: "A" | "B", state: SlotState) => {
		if (state.isChecking || !state.embedding || !state.llm) {
			return null;
		}

		if (isIngesting && ingestingSlot === slotKey) {
			return (
				<div className="flex items-center gap-2 text-sm text-muted-foreground mt-4 px-4 py-3 rounded-lg border bg-muted/50">
					<Loader2 className="size-4 animate-spin" />
					Processing documents…
				</div>
			);
		}

		if (!state.hasDocuments) {
			return (
				<div className="flex flex-col gap-2 mt-4 p-4 rounded-lg border border-destructive/30 bg-destructive/5 text-sm">
					<p className="text-destructive font-medium">Index not found</p>
					<p className="text-muted-foreground mb-2">
						This combination of Vision + Embedding models hasn't been used to
						index your files yet.
					</p>
					<Button
						onClick={() => handleIngest(slotKey, state)}
						disabled={isIngesting}
						size="sm"
					>
						Process Documents for Slot {slotKey}
					</Button>
				</div>
			);
		}

		return null;
	};

	const canChat =
		slotA.hasDocuments &&
		slotB.hasDocuments &&
		slotA.llm &&
		slotB.llm &&
		!isIngesting;

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30 p-4">
			<div className="mx-auto flex h-full w-full max-w-[1440px] flex-col gap-4">
				<header className="shrink-0 rounded-xl border bg-card p-5">
					<div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
						<div>
							<h1 className="text-xl font-semibold text-foreground">
								Model Comparison
							</h1>
							<p className="mt-1 text-sm text-muted-foreground">
								Select two configurations to compare their generation quality
								side-by-side.
							</p>
						</div>
						{error && <p className="text-sm text-destructive">{error}</p>}
					</div>

					<div className="mt-6 grid gap-6 md:grid-cols-2">
						<SlotPicker
							label="Slot A Configuration"
							providers={providers}
							isLoading={isLoading}
							disabled={isQuerying || isIngesting}
							visionValue={slotA.vision}
							embedValue={slotA.embedding}
							llmValue={slotA.llm}
							onVisionChange={(val) => setSlotA((p) => ({ ...p, vision: val }))}
							onEmbedChange={(val) =>
								setSlotA((p) => ({ ...p, embedding: val }))
							}
							onLlmChange={(val) => setSlotA((p) => ({ ...p, llm: val }))}
						/>
						<SlotPicker
							label="Slot B Configuration"
							providers={providers}
							isLoading={isLoading}
							disabled={isQuerying || isIngesting}
							visionValue={slotB.vision}
							embedValue={slotB.embedding}
							llmValue={slotB.llm}
							onVisionChange={(val) => setSlotB((p) => ({ ...p, vision: val }))}
							onEmbedChange={(val) =>
								setSlotB((p) => ({ ...p, embedding: val }))
							}
							onLlmChange={(val) => setSlotB((p) => ({ ...p, llm: val }))}
						/>
					</div>
				</header>

				{!slotA.llm || !slotB.llm || !slotA.embedding || !slotB.embedding ? (
					<div className="flex flex-1 items-center justify-center rounded-xl border bg-card p-8">
						<Empty>
							<EmptyHeader>
								<EmptyMedia variant="icon">{"//"}</EmptyMedia>
								<EmptyTitle>Select complete configurations</EmptyTitle>
								<EmptyDescription>
									Use the dropdowns above to choose the Vision, Embedding, and
									LLM for both slots.
								</EmptyDescription>
							</EmptyHeader>
						</Empty>
					</div>
				) : (
					<div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden">
						<div className="grid min-h-0 flex-1 grid-cols-1 gap-4 overflow-hidden md:grid-cols-2">
							{/* Chat A */}
							<div className="flex h-full flex-col overflow-y-auto rounded-xl border bg-card p-4">
								<h3 className="sticky top-0 z-10 bg-card pb-4 text-sm font-semibold tracking-wider text-foreground">
									Slot A
								</h3>
								{renderSlotStatus("A", slotA)}
								<div className="flex flex-col gap-5 mt-4">
									{messagesA.length === 0 ? (
										<p className="text-sm text-muted-foreground">
											No messages yet.
										</p>
									) : (
										messagesA.map((msg) => (
											<ChatMessageView key={msg.id} message={msg} />
										))
									)}
									{isQuerying && <TypingIndicator />}
									<div ref={chatBottomRef} />
								</div>
							</div>

							{/* Chat B */}
							<div className="flex h-full flex-col overflow-y-auto rounded-xl border bg-card p-4">
								<h3 className="sticky top-0 z-10 bg-card pb-4 text-sm font-semibold tracking-wider text-foreground">
									Slot B
								</h3>
								{renderSlotStatus("B", slotB)}
								<div className="flex flex-col gap-5 mt-4">
									{messagesB.length === 0 ? (
										<p className="text-sm text-muted-foreground">
											No messages yet.
										</p>
									) : (
										messagesB.map((msg) => (
											<ChatMessageView key={msg.id} message={msg} />
										))
									)}
									{isQuerying && <TypingIndicator />}
									<div ref={chatBottomRef} />
								</div>
							</div>
						</div>

						<div className="shrink-0 rounded-xl border bg-card p-4">
							<ChatInput
								onSubmit={handleQuery}
								onUpload={async () => {}} // Not supporting upload here
								isLoading={isQuerying}
								isUploading={false}
								disabled={!canChat || isQuerying}
								disabledReason={
									isIngesting
										? "Document processing in progress…"
										: !canChat
											? "Make sure both slots have documents processed."
											: undefined
								}
							/>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
