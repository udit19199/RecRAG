import { ProviderModelSelect } from "@/features/model-compare/components/provider-model-select";
import type { ProvidersResponse } from "@/lib/api";

interface SlotPickerProps {
	label: string;
	providers: ProvidersResponse | null;
	isLoading: boolean;
	disabled: boolean;
	llmValue: string | null;
	embedValue: string | null;
	visionValue: string | null;
	onLlmChange: (val: string) => void;
	onEmbedChange: (val: string) => void;
	onVisionChange: (val: string | null) => void;
}

const PROVIDER_LABELS: Record<string, string> = {
	ollama: "Ollama",
	openai: "OpenAI",
	lmstudio: "LM Studio",
	nim: "NVIDIA NIM",
	gemini: "Google Gemini",
};

export function SlotPicker({
	label,
	providers,
	isLoading,
	disabled,
	llmValue,
	embedValue,
	visionValue,
	onLlmChange,
	onEmbedChange,
	onVisionChange,
}: SlotPickerProps) {
	return (
		<div className="flex flex-col gap-4">
			<h3 className="text-sm font-semibold text-foreground border-b pb-2">
				{label}
			</h3>
			<div className="flex flex-col gap-4 xl:flex-row xl:items-end">
				<ProviderModelSelect
					modelRole="vision"
					title="Vision Model"
					providers={providers}
					isLoading={isLoading}
					value={visionValue}
					onValueChange={onVisionChange}
					providerLabels={PROVIDER_LABELS}
					allowClear
					disabled={disabled}
				/>
				<ProviderModelSelect
					modelRole="embedding"
					title="Embedding Model"
					providers={providers}
					isLoading={isLoading}
					value={embedValue}
					onValueChange={(value) => value && onEmbedChange(value)}
					providerLabels={PROVIDER_LABELS}
					disabled={disabled}
				/>
				<ProviderModelSelect
					modelRole="llm"
					title="LLM"
					providers={providers}
					isLoading={isLoading}
					value={llmValue}
					onValueChange={(value) => value && onLlmChange(value)}
					providerLabels={PROVIDER_LABELS}
					disabled={disabled}
				/>
			</div>
		</div>
	);
}
