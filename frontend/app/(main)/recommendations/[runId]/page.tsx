import type { Metadata } from "next";
import { RecommendationPanel } from "@/features/recommendation/components/recommendation-panel";

export const metadata: Metadata = {
	title: "Recommendation | RecRAG",
};

export default async function RecommendationDetailPage({
	params,
}: {
	params: Promise<{ runId: string }>;
}) {
	const { runId } = await params;

	return (
		<div className="mx-auto w-full max-w-3xl p-8">
			<RecommendationPanel runId={runId} />
		</div>
	);
}
