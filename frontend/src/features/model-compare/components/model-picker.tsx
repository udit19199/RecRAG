"use client";

import { Alert, AlertDescription } from "@/components/ui/alert";
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
import type { VisionConfig } from "@/features/model-compare/components/model-picker-types";
import { ProviderModelSelect } from "@/features/model-compare/components/provider-model-select";
import { useModelPicker } from "@/features/model-compare/hooks/use-model-picker";
import type { AdapterConfig } from "@/lib/api";

export type { VisionConfig };

interface ModelPickerProps {
	onReindexStarted?: () => void;
	onConfigChanged?: (config: {
		embedding: AdapterConfig;
		llm: AdapterConfig;
	}) => void;
	onVisionConfigChanged?: (config: VisionConfig | null) => void;
	disabled?: boolean;
}

const PROVIDER_LABELS: Record<string, string> = {
	ollama: "Ollama",
	openai: "OpenAI",
	lmstudio: "LM Studio",
	nim: "NVIDIA NIM",
	gemini: "Google Gemini",
};

export default function ModelPicker({
	onReindexStarted,
	onConfigChanged,
	onVisionConfigChanged,
	disabled = false,
}: ModelPickerProps) {
	const {
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
	} = useModelPicker({
		onReindexStarted,
		onConfigChanged,
		onVisionConfigChanged,
	});

	return (
		<>
			{state.loadError ? (
				<Alert variant="destructive" className="mb-2">
					<AlertDescription>{state.loadError}</AlertDescription>
				</Alert>
			) : null}
			<div className="flex items-end gap-4">
				<ProviderModelSelect
					modelRole="vision"
					title="Vision Model"
					providers={state.providers}
					isLoading={isLoading}
					value={currentVisionValue}
					onValueChange={(value) => handleVisionChange(value)}
					providerLabels={PROVIDER_LABELS}
					allowClear
					showUnavailable
					disabled={disabled}
					inputClassName="w-[13.5rem]"
				/>
				<ProviderModelSelect
					modelRole="embedding"
					title="Embedding"
					providers={state.providers}
					isLoading={isLoading}
					value={currentEmbedValue}
					onValueChange={(value) => value && handleEmbedChange(value)}
					providerLabels={PROVIDER_LABELS}
					showUnavailable
					disabled={disabled}
					isChanging={state.embedChanging}
					inputClassName="w-[13.5rem]"
				/>
				<ProviderModelSelect
					modelRole="llm"
					title="LLM"
					providers={state.providers}
					isLoading={isLoading}
					value={currentLLMValue}
					onValueChange={(value) => value && handleLLMChange(value)}
					providerLabels={PROVIDER_LABELS}
					showUnavailable
					disabled={disabled}
					isChanging={state.llmChanging}
					inputClassName="w-[13.5rem]"
				/>
			</div>

			<AlertDialog
				open={state.pendingEmbedChange !== null}
				onOpenChange={(open) => {
					if (!open) dismissEmbedChange();
				}}
			>
				<AlertDialogContent>
					<AlertDialogHeader>
						<AlertDialogTitle>Change embedding model?</AlertDialogTitle>
						<AlertDialogDescription className="space-y-2">
							<span className="block">
								Switching to{" "}
								<span className="font-mono font-semibold">
									{state.pendingEmbedChange?.model}
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
