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
import { makeValue, uniqueModels } from "@/features/model-compare/lib";
import type { ProvidersResponse } from "@/lib/api";

type Role = "llm" | "embedding" | "vision";

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
	nim: "NVIDIA NIM",
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
	const renderSelect = (
		role: Role,
		title: string,
		value: string | null,
		onChange: (v: string | null) => void,
		allowClear = false,
	) => {
		if (isLoading || !providers) {
			return (
				<div className="flex flex-col gap-1.5 flex-1">
					<span className="text-sm font-medium text-muted-foreground">
						{title}
					</span>
					<Skeleton className="h-10 w-full rounded-md" />
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

		if (role === "vision" && !hasAnyModels) {
			return (
				<div className="flex flex-col gap-1.5 flex-1">
					<span className="text-sm font-medium text-muted-foreground">
						{title}
					</span>
					<Tooltip>
						<TooltipTrigger asChild>
							<div className="flex h-10 w-full cursor-not-allowed items-center rounded-md border border-input bg-muted/50 px-3 text-sm text-muted-foreground">
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
			<div className="flex flex-col gap-1.5 flex-1">
				<span className="text-sm font-medium text-muted-foreground">
					{title}
				</span>
				<Combobox
					items={Object.entries(providerMap).flatMap(([providerKey, info]) =>
						info.available
							? uniqueModels(info.models).map((model) =>
									makeValue(providerKey, model),
								)
							: [],
					)}
					value={value}
					onValueChange={(nextValue) => {
						if (typeof nextValue === "string") {
							onChange(nextValue);
						} else if (nextValue === null && allowClear) {
							onChange(null);
						}
					}}
					disabled={disabled}
				>
					<ComboboxInput
						placeholder={allowClear ? "None (text only)" : "Select model"}
						className="w-full"
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

								if (!info.available || models.length === 0) return null;

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
		<div className="flex flex-col gap-4">
			<h3 className="text-sm font-semibold text-foreground border-b pb-2">
				{label}
			</h3>
			<div className="flex flex-col gap-4 xl:flex-row xl:items-end">
				{renderSelect(
					"vision",
					"Vision Model",
					visionValue,
					onVisionChange,
					true,
				)}
				{renderSelect(
					"embedding",
					"Embedding Model",
					embedValue,
					(v) => v && onEmbedChange(v),
				)}
				{renderSelect("llm", "LLM", llmValue, (v) => v && onLlmChange(v))}
			</div>
		</div>
	);
}
