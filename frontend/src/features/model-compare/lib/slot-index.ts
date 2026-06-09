import { parseValue } from "@/features/model-compare/lib";
import { checkIndexStatus } from "@/lib/api";

export async function fetchSlotHasDocuments(
	embedding: string,
	vision: string | null,
): Promise<boolean> {
	const embedConfig = parseValue(embedding);
	if (!embedConfig) return false;

	const visionConfig = vision ? parseValue(vision) : undefined;
	const res = await checkIndexStatus({
		embedding: embedConfig,
		vision: visionConfig
			? { provider: visionConfig.provider, model: visionConfig.model }
			: undefined,
	});
	return res.has_documents;
}
