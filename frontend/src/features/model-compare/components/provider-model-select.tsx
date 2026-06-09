import { Skeleton } from "@/components/ui/skeleton";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { ProviderModelCombobox } from "@/features/model-compare/components/provider-model-combobox";
import { buildProviderModelGroups } from "@/features/model-compare/lib";
import type { ProvidersResponse } from "@/lib/api";

type ModelRole = "llm" | "embedding" | "vision";

interface ProviderModelSelectProps {
	modelRole: ModelRole;
	title: string;
	providers: ProvidersResponse | null;
	isLoading: boolean;
	value: string | null | undefined;
	onValueChange: (value: string | null) => void;
	providerLabels: Record<string, string>;
	allowClear?: boolean;
	showUnavailable?: boolean;
	disabled?: boolean;
	isChanging?: boolean;
	inputClassName?: string;
}

export function ProviderModelSelect({
	modelRole,
	title,
	providers,
	isLoading,
	value,
	onValueChange,
	providerLabels,
	allowClear = false,
	showUnavailable = false,
	disabled = false,
	isChanging = false,
	inputClassName = "w-full",
}: ProviderModelSelectProps) {
	if (isLoading || !providers) {
		return (
			<div className="flex flex-col gap-1.5 flex-1">
				<span className="text-sm font-medium text-muted-foreground">
					{title}
				</span>
				<Skeleton className={`h-10 rounded-md ${inputClassName}`} />
			</div>
		);
	}

	const providerMap =
		modelRole === "llm"
			? providers.llms
			: modelRole === "embedding"
				? providers.embedders
				: (providers.vision ?? {});

	const hasAnyModels = Object.values(providerMap).some(
		(info) => info.available && info.models.length > 0,
	);

	if (modelRole === "vision" && !hasAnyModels) {
		return (
			<div className="flex flex-col gap-1.5 flex-1">
				<span className="text-sm font-medium text-muted-foreground">
					{title}
				</span>
				<Tooltip>
					<TooltipTrigger asChild>
						<div
							className={`flex h-10 cursor-not-allowed items-center rounded-md border border-input bg-muted/50 px-3 text-sm text-muted-foreground ${inputClassName}`}
						>
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

	const groups = buildProviderModelGroups(providerMap, providerLabels);

	return (
		<div className="flex flex-col gap-1.5 flex-1">
			<span className="text-sm font-medium text-muted-foreground">{title}</span>
			<ProviderModelCombobox
				groups={groups}
				value={value}
				onValueChange={onValueChange}
				allowClear={allowClear}
				showUnavailable={showUnavailable}
				disabled={disabled || isChanging}
				readOnly={isChanging}
				placeholder={
					isChanging
						? "Applying..."
						: allowClear
							? "None (text only)"
							: "Select model"
				}
				inputClassName={`${inputClassName} ${isChanging ? "opacity-60" : ""}`}
			/>
		</div>
	);
}
