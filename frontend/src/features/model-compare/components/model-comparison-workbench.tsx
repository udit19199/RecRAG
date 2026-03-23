"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Combobox,
	ComboboxContent,
	ComboboxEmpty,
	ComboboxGroup,
	ComboboxInput,
	ComboboxItem,
	ComboboxLabel,
	ComboboxList,
} from "@/components/ui/combobox";
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
import {
	type AdapterConfig,
	getConfig,
	getProviders,
	type ProvidersResponse,
	queryRAG,
} from "@/lib/api";

type CompareRole = "llm" | "embedding";

type CompareModel = {
	id: string;
	provider: string;
	providerLabel: string;
	model: string;
	role: CompareRole;
	available: boolean;
	isActive: boolean;
};

const PROVIDER_LABELS: Record<string, string> = {
	ollama: "Ollama",
	openai: "OpenAI",
	nim: "NVIDIA NIM",
};

const _ROLE_LABELS: Record<CompareRole, string> = {
	llm: "LLM",
	embedding: "Embedding",
};

function makeValue(provider: string, model: string) {
	return `${provider}::${model}`;
}

function parseValue(value: string): AdapterConfig | null {
	const [provider, model] = value.split("::");
	if (!provider || !model) return null;
	return { provider, model };
}

function getModelFamily(model: string) {
	return model.split(/[:/]/)[0] || model;
}

function _getModelVariant(model: string) {
	const parts = model.split(":");
	return parts.length > 1 ? parts.slice(1).join(":") : "default tag";
}

export function ModelComparisonWorkbench() {
	const [providers, setProviders] = useState<ProvidersResponse | null>(null);
	const [config, setConfig] = useState<{
		embedding: AdapterConfig;
		llm: AdapterConfig;
	} | null>(null);

	const [slotA, setSlotA] = useState<string | null>(null);
	const [slotB, setSlotB] = useState<string | null>(null);

	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);

	// Chat state
	const [messagesA, setMessagesA] = useState<ChatMessage[]>([]);
	const [messagesB, setMessagesB] = useState<ChatMessage[]>([]);
	const [isQuerying, setIsQuerying] = useState(false);
	const messageIdRef = useRef(0);
	const chatBottomRef = useRef<HTMLDivElement>(null);

	const nextId = useCallback(() => {
		return messageIdRef.current++;
	}, []);

	const loadData = useCallback(async () => {
		setIsLoading(true);
		setError(null);

		try {
			const [nextConfig, nextProviders] = await Promise.all([
				getConfig(),
				getProviders(),
			]);
			setConfig(nextConfig);
			setProviders(nextProviders);

			// Pre-select defaults
			const providerMap = nextProviders.llms;
			const activeKey = makeValue(
				nextConfig.llm.provider,
				nextConfig.llm.model,
			);
			const keys = Object.entries(providerMap).flatMap(([provider, info]) =>
				info.available
					? info.models.map((model) => makeValue(provider, model))
					: [],
			);

			const defaults: string[] = [];
			if (keys.includes(activeKey)) {
				defaults.push(activeKey);
			}

			for (const key of keys) {
				if (!defaults.includes(key)) defaults.push(key);
				if (defaults.length === 2) break;
			}

			if (defaults[0]) setSlotA(defaults[0]);
			if (defaults[1]) setSlotB(defaults[1]);
		} catch (err) {
			setError(
				err instanceof Error ? err.message : "Unable to load model inventory",
			);
		} finally {
			setIsLoading(false);
		}
	}, []);

	useEffect(() => {
		loadData();
	}, [loadData]);

	useEffect(() => {
		chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
	});

	const providerMap = providers?.llms;

	const getCompareModel = useCallback(
		(value: string | null): CompareModel | null => {
			if (!value || !providerMap || !config) return null;

			const parsed = parseValue(value);
			if (!parsed) return null;

			const info = providerMap[parsed.provider];
			if (!info) return null;

			return {
				id: value,
				provider: parsed.provider,
				providerLabel: PROVIDER_LABELS[parsed.provider] ?? parsed.provider,
				model: parsed.model,
				role: "llm",
				available: info.available,
				isActive:
					config.llm.provider === parsed.provider &&
					config.llm.model === parsed.model,
			};
		},
		[providerMap, config],
	);

	const modelA = getCompareModel(slotA);
	const modelB = getCompareModel(slotB);

	const handleQuery = useCallback(
		async (query: string) => {
			if (!query.trim() || isQuerying || !slotA || !slotB) return;

			const configA = parseValue(slotA);
			const configB = parseValue(slotB);
			if (!configA || !configB) return;

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

			try {
				const [resultA, resultB] = await Promise.allSettled([
					queryRAG({ query, llm: configA }),
					queryRAG({ query, llm: configB }),
				]);

				if (resultA.status === "fulfilled") {
					setMessagesA((prev) => [
						...prev,
						{
							id: nextId(),
							role: "assistant",
							content: resultA.value.response,
							context: resultA.value.context,
						},
					]);
				} else {
					setMessagesA((prev) => [
						...prev,
						{
							id: nextId(),
							role: "assistant",
							content: "",
							error:
								resultA.reason instanceof Error
									? resultA.reason.message
									: "Query failed",
						},
					]);
				}

				if (resultB.status === "fulfilled") {
					setMessagesB((prev) => [
						...prev,
						{
							id: nextId(),
							role: "assistant",
							content: resultB.value.response,
							context: resultB.value.context,
						},
					]);
				} else {
					setMessagesB((prev) => [
						...prev,
						{
							id: nextId(),
							role: "assistant",
							content: "",
							error:
								resultB.reason instanceof Error
									? resultB.reason.message
									: "Query failed",
						},
					]);
				}
			} finally {
				setIsQuerying(false);
			}
		},
		[isQuerying, slotA, slotB, nextId],
	);

	const renderCombobox = (
		slot: "A" | "B",
		value: string | null,
		otherValue: string | null,
	) => {
		if (isLoading || !providerMap) {
			return (
				<Button variant="outline" disabled className="w-full justify-start">
					Loading...
				</Button>
			);
		}

		return (
			<Combobox
				items={Object.entries(providerMap).flatMap(([providerKey, info]) =>
					info.available
						? info.models
								.map((m) => makeValue(providerKey, m))
								.filter((val) => val !== otherValue) // Prevent duplicate selection
						: [],
				)}
				value={value}
				onValueChange={(nextValue) => {
					if (typeof nextValue === "string") {
						if (slot === "A") setSlotA(nextValue);
						else setSlotB(nextValue);
					}
				}}
				disabled={isQuerying}
			>
				<ComboboxInput placeholder="Select model" className="w-full" />
				<ComboboxContent className="w-[300px]">
					<ComboboxEmpty>No matching models.</ComboboxEmpty>
					<ComboboxList>
						{Object.entries(providerMap).map(([providerKey, info]) => {
							const providerLabel = PROVIDER_LABELS[providerKey] ?? providerKey;
							if (!info.available || info.models.length === 0) return null;

							const availableModels = info.models.filter(
								(m) => makeValue(providerKey, m) !== otherValue,
							);

							if (availableModels.length === 0) return null;

							return (
								<ComboboxGroup key={providerKey}>
									<ComboboxLabel>{providerLabel}</ComboboxLabel>
									{availableModels.map((model) => (
										<ComboboxItem
											key={model}
											value={makeValue(providerKey, model)}
										>
											{model}
										</ComboboxItem>
									))}
								</ComboboxGroup>
							);
						})}
					</ComboboxList>
				</ComboboxContent>
			</Combobox>
		);
	};

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
								Select two LLMs to compare their generation quality
								side-by-side.
							</p>
						</div>
						{error && <p className="text-sm text-destructive">{error}</p>}
					</div>

					<div className="mt-6 grid gap-6 md:grid-cols-2">
						<div className="flex flex-col gap-2">
							<span className="text-sm font-medium text-foreground">
								Slot A
							</span>
							{renderCombobox("A", slotA, slotB)}
							{modelA && (
								<div className="mt-2 text-xs text-muted-foreground">
									{modelA.providerLabel} &bull; {getModelFamily(modelA.model)}
								</div>
							)}
						</div>
						<div className="flex flex-col gap-2">
							<span className="text-sm font-medium text-foreground">
								Slot B
							</span>
							{renderCombobox("B", slotB, slotA)}
							{modelB && (
								<div className="mt-2 text-xs text-muted-foreground">
									{modelB.providerLabel} &bull; {getModelFamily(modelB.model)}
								</div>
							)}
						</div>
					</div>
				</header>

				{!modelA || !modelB ? (
					<div className="flex flex-1 items-center justify-center rounded-xl border bg-card p-8">
						<Empty>
							<EmptyHeader>
								<EmptyMedia variant="icon">{"//"}</EmptyMedia>
								<EmptyTitle>Select exactly two models</EmptyTitle>
								<EmptyDescription>
									Use the dropdowns above to choose two different LLM
									candidates.
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
									{modelA.model}
								</h3>
								<div className="flex flex-col gap-5">
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
									{modelB.model}
								</h3>
								<div className="flex flex-col gap-5">
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
								disabled={isQuerying}
							/>
						</div>
					</div>
				)}
			</div>
		</div>
	);
}
