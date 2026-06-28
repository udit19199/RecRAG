import type { ProviderInfo, VisionProvider } from "@/lib/api/types";

export interface StoredVisionConfig {
	provider: VisionProvider;
	model: string;
}

export function makeValue(provider: string, model: string) {
	return `${provider}::${model}`;
}

export function parseValue(value: string | null) {
	if (!value) return null;
	const [provider, model] = value.split("::");
	if (!provider || !model) return null;
	return { provider, model };
}

function uniqueModels(models: string[]) {
	return Array.from(new Set(models));
}

export function readStoredVisionConfig(): StoredVisionConfig | null {
	if (typeof window === "undefined") return null;

	try {
		const vision = localStorage.getItem("lastVision");
		if (!vision) return null;

		const parsed = parseValue(vision);
		if (!parsed) return null;

		return {
			provider: parsed.provider as VisionProvider,
			model: parsed.model,
		};
	} catch {
		return null;
	}
}

export function modelValueLabel(value: string) {
	if (!value) return "None (text only)";
	return parseValue(value)?.model ?? value;
}

export type ProviderModelGroup = {
	key: string;
	label: string;
	items: string[];
	available: boolean;
	reason: string | null;
};

export function buildProviderModelGroups(
	providerMap: Record<string, ProviderInfo>,
	labels: Record<string, string>,
): ProviderModelGroup[] {
	return Object.entries(providerMap).map(([providerKey, info]) => ({
		key: providerKey,
		label: labels[providerKey] ?? providerKey,
		items: info.available
			? uniqueModels(info.models).map((model) => makeValue(providerKey, model))
			: [],
		available: info.available,
		reason: info.reason ?? null,
	}));
}
