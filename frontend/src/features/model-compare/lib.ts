export function makeValue(provider: string, model: string) {
	return `${provider}::${model}`;
}

export function parseValue(value: string | null) {
	if (!value) return null;
	const [provider, model] = value.split("::");
	if (!provider || !model) return null;
	return { provider, model };
}

export function uniqueModels(models: string[]) {
	return Array.from(new Set(models));
}
