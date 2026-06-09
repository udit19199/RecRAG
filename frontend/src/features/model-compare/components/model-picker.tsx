"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
	AlertDialog,
	AlertDialogAction,
	AlertDialogCancel,
	AlertDialogContent,
	AlertDialogDescription,
	AlertDialogFooter,
	AlertDialogHeader,
	AlertDialogTitle,
} from "@/components/ui/alert-dialog";
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
import { Skeleton } from "@/components/ui/skeleton";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import {
	makeValue,
	parseValue,
	uniqueModels,
} from "@/features/model-compare/lib";
import {
	type AdapterConfig,
	getConfig,
	getProviders,
	type ProvidersResponse,
	setConfig,
	setIngestionConfig,
	triggerReindex,
} from "@/lib/api";
import type { ExtractionOptions, VisionProvider } from "@/lib/api/types";

export interface VisionConfig {
	provider: VisionProvider;
	model: string;
}

interface ModelPickerProps {
	onReindexStarted?: () => void;
	onConfigChanged?: (config: {
		embedding: AdapterConfig;
		llm: AdapterConfig;
	}) => void;
	onVisionConfigChanged?: (config: VisionConfig | null) => void;
	disabled?: boolean;
}

type PendingEmbedChange = { provider: string; model: string } | null;

const PROVIDER_LABELS: Record<string, string> = {
	ollama: "Ollama",
	openai: "OpenAI",
	nim: "NVIDIA NIM",
	gemini: "Google Gemini",
};

export default function ModelPicker({
	onReindexStarted,
	onConfigChanged,
	onVisionConfigChanged,
	disabled = false,
}: ModelPickerProps) {
	const [providers, setProviders] = useState<ProvidersResponse | null>(null);
	const [currentEmbedding, setCurrentEmbedding] =
		useState<AdapterConfig | null>(null);
	const [currentLLM, setCurrentLLM] = useState<AdapterConfig | null>(null);
	const [currentVision, setCurrentVision] = useState<VisionConfig | null>(null);
	const isLoading = !providers;
	const [llmChanging, setLlmChanging] = useState(false);
	const [embedChanging, setEmbedChanging] = useState(false);
	const [pendingEmbedChange, setPendingEmbedChange] =
		useState<PendingEmbedChange>(null);
	const hydratedSelectionRef = useRef(false);

	const loadData = useCallback(async () => {
		try {
			const [cfg, prov] = await Promise.all([getConfig(), getProviders()]);
			setCurrentEmbedding(cfg.embedding);
			setCurrentLLM(cfg.llm);
			setProviders(prov);
		} catch {
			// ignore and render skeletons
		}
	}, []);

	useEffect(() => {
		loadData();
	}, [loadData]);

	useEffect(() => {
		if (hydratedSelectionRef.current) return;
		hydratedSelectionRef.current = true;

		try {
			const llm = localStorage.getItem("lastLLM");
			const emb = localStorage.getItem("lastEmbedding");
			const vision = localStorage.getItem("lastVision");

			if (llm) {
				const [provider, model] = llm.split("::");
				if (provider && model)
					setCurrentLLM((prev) => prev ?? { provider, model });
			}

			if (emb) {
				const [provider, model] = emb.split("::");
				if (provider && model)
					setCurrentEmbedding((prev) => prev ?? { provider, model });
			}

			if (vision) {
				const parsed = parseValue(vision);
				if (parsed) {
					const visionConfig = {
						provider: parsed.provider as VisionProvider,
						model: parsed.model,
					};
					setCurrentVision(visionConfig);
					onVisionConfigChanged?.(visionConfig);
				}
			}
		} catch {
			// ignore
		}
	}, [onVisionConfigChanged]);

	const handleLLMChange = async (value: string) => {
		const [provider, model] = value.split("::");
		if (!provider || !model) return;
		if (provider === currentLLM?.provider && model === currentLLM?.model)
			return;

		setLlmChanging(true);
		try {
			const result = await setConfig({ llm: { provider, model } });
			setCurrentLLM(result.llm);

			try {
				localStorage.setItem(
					"lastLLM",
					makeValue(result.llm.provider, result.llm.model),
				);
			} catch {}

			onConfigChanged?.({ embedding: result.embedding, llm: result.llm });
		} catch (err) {
			console.error("Failed to change LLM:", err);
		} finally {
			setLlmChanging(false);
		}
	};

	const handleEmbedChange = (value: string) => {
		const [provider, model] = value.split("::");
		if (!provider || !model) return;
		if (
			provider === currentEmbedding?.provider &&
			model === currentEmbedding?.model
		)
			return;
		setPendingEmbedChange({ provider, model });
	};

	const handleVisionChange = (value: string | null) => {
		if (!value) {
			// Clear vision model
			setCurrentVision(null);
			onVisionConfigChanged?.(null);
			try {
				localStorage.removeItem("lastVision");
			} catch {}
			return;
		}

		const parsed = parseValue(value);
		if (!parsed) return;

		const visionConfig: VisionConfig = {
			provider: parsed.provider as VisionProvider,
			model: parsed.model,
		};

		if (
			currentVision?.provider === visionConfig.provider &&
			currentVision?.model === visionConfig.model
		) {
			return;
		}

		setCurrentVision(visionConfig);
		onVisionConfigChanged?.(visionConfig);

		try {
			localStorage.setItem("lastVision", value);
		} catch {}
	};

	const confirmEmbedChange = async () => {
		if (!pendingEmbedChange) return;
		const { provider, model } = pendingEmbedChange;
		setPendingEmbedChange(null);
		setEmbedChanging(true);

		try {
			const [result] = await Promise.all([
				setConfig({ embedding: { provider, model } }),
				setIngestionConfig({ provider, model }),
			]);

			setCurrentEmbedding(result.embedding);

			try {
				localStorage.setItem(
					"lastEmbedding",
					makeValue(result.embedding.provider, result.embedding.model),
				);
			} catch {}

			onConfigChanged?.({ embedding: result.embedding, llm: result.llm });

			if (result.requires_reindex) {
				// Build extraction options based on current vision config
				const extractionOptions: ExtractionOptions | undefined = currentVision
					? {
							extraction_mode: "vision_assisted",
							vision_provider: currentVision.provider,
							vision_model: currentVision.model,
						}
					: undefined;
				await triggerReindex(extractionOptions);
				onReindexStarted?.();
			}
		} catch (err) {
			console.error("Failed to change embedding model:", err);
		} finally {
			setEmbedChanging(false);
		}
	};

	const currentLLMValue = currentLLM
		? makeValue(currentLLM.provider, currentLLM.model)
		: undefined;
	const currentEmbedValue = currentEmbedding
		? makeValue(currentEmbedding.provider, currentEmbedding.model)
		: undefined;
	const currentVisionValue = currentVision
		? makeValue(currentVision.provider, currentVision.model)
		: undefined;

	const renderSelect = (
		role: "llm" | "embedding" | "vision",
		label: string,
		value: string | undefined,
		onChange: (v: string) => void,
		isChanging: boolean,
		allowClear = false,
	) => {
		if (isLoading || !providers) {
			return (
				<div className="flex flex-col gap-1.5">
					<span className="text-sm font-medium text-muted-foreground">
						{label}
					</span>
					<Skeleton className="h-10 w-[13.5rem] rounded-md" />
				</div>
			);
		}

		const providerMap =
			role === "llm"
				? providers.llms
				: role === "embedding"
					? providers.embedders
					: (providers.vision ?? {});

		const hasAnyModels = Object.values(providerMap).some(
			(info) => info.available && info.models.length > 0,
		);

		// For vision, if no models available, show disabled state
		if (role === "vision" && !hasAnyModels) {
			return (
				<div className="flex flex-col gap-1.5">
					<span className="text-sm font-medium text-muted-foreground">
						{label}
					</span>
					<Tooltip>
						<TooltipTrigger asChild>
							<div className="flex h-10 w-[13.5rem] cursor-not-allowed items-center rounded-md border border-input bg-muted/50 px-3 text-sm text-muted-foreground">
								None (text only)
							</div>
						</TooltipTrigger>
						<TooltipContent>
							<p className="max-w-xs text-xs">
								No vision models available. Configure OpenAI, Ollama with LLaVA,
								or NVIDIA NIM to enable vision-assisted extraction.
							</p>
						</TooltipContent>
					</Tooltip>
				</div>
			);
		}

		return (
			<div className="flex flex-col gap-1.5">
				<span className="text-sm font-medium text-muted-foreground">
					{label}
				</span>
				<Combobox
					items={Object.entries(providerMap).flatMap(([providerKey, info]) =>
						info.available
							? uniqueModels(info.models).map((model) =>
									makeValue(providerKey, model),
								)
							: [],
					)}
					value={value ?? null}
					onValueChange={(nextValue) => {
						if (typeof nextValue === "string") {
							onChange(nextValue);
						} else if (nextValue === null && allowClear) {
							onChange("");
						}
					}}
					disabled={disabled || isChanging}
				>
					<ComboboxInput
						placeholder={
							isChanging
								? "Applying..."
								: allowClear
									? "None (text only)"
									: "Select model"
						}
						readOnly={isChanging}
						className={`w-[13.5rem] ${isChanging ? "opacity-60" : ""}`}
					/>
					<ComboboxContent>
						<ComboboxEmpty>No matching models.</ComboboxEmpty>
						<ComboboxList>
							{allowClear && (
								<ComboboxGroup>
									<ComboboxItem value="" className="text-muted-foreground">
										None (text only)
									</ComboboxItem>
								</ComboboxGroup>
							)}
							{Object.entries(providerMap).map(([providerKey, info]) => {
								const providerLabel =
									PROVIDER_LABELS[providerKey] ?? providerKey;
								const models = uniqueModels(info.models);

								if (!info.available || models.length === 0) {
									return (
										<ComboboxGroup key={providerKey}>
											<ComboboxLabel className="flex items-center justify-between">
												<span>{providerLabel}</span>
												<Tooltip>
													<TooltipTrigger asChild>
														<span className="ml-2 cursor-default text-xs text-muted-foreground">
															unavailable
														</span>
													</TooltipTrigger>
													<TooltipContent side="right">
														<p className="max-w-xs text-xs">
															{info.reason || "Provider not available"}
														</p>
													</TooltipContent>
												</Tooltip>
											</ComboboxLabel>
										</ComboboxGroup>
									);
								}

								return (
									<ComboboxGroup key={providerKey}>
										<ComboboxLabel>{providerLabel}</ComboboxLabel>
										{models.map((model) => (
											<ComboboxItem
												key={makeValue(providerKey, model)}
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
			</div>
		);
	};

	return (
		<>
			<div className="flex items-end gap-4">
				{renderSelect(
					"vision",
					"Vision Model",
					currentVisionValue,
					(v) => handleVisionChange(v || null),
					false,
					true,
				)}
				{renderSelect(
					"embedding",
					"Embedding",
					currentEmbedValue,
					handleEmbedChange,
					embedChanging,
				)}
				{renderSelect(
					"llm",
					"LLM",
					currentLLMValue,
					handleLLMChange,
					llmChanging,
				)}
			</div>

			<AlertDialog
				open={pendingEmbedChange !== null}
				onOpenChange={(open) => {
					if (!open) setPendingEmbedChange(null);
				}}
			>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>Change embedding model?</AlertDialogTitle>
						<AlertDialogDescription className="space-y-2">
							<span className="block">
								Switching to{" "}
								<span className="font-mono font-semibold">
									{pendingEmbedChange?.model}
								</span>{" "}
								will require re-indexing all your documents.
							</span>
							<span className="block text-muted-foreground">
								Existing search will be unavailable until re-indexing completes.
								This may take several minutes depending on the size of your
								document collection.
							</span>
						</AlertDialogDescription>
					</AlertDialogHeader>
					<AlertDialogFooter>
						<AlertDialogCancel>Cancel</AlertDialogCancel>
						<AlertDialogAction onClick={confirmEmbedChange}>
							Change &amp; Re-index
						</AlertDialogAction>
					</AlertDialogFooter>
				</AlertDialogContent>
			</AlertDialog>
		</>
	);
}
