import type { Metadata } from "next";
import { RecommendationsList } from "@/features/recommendation/components/recommendations-list";

export const metadata: Metadata = {
	title: "Recommendations | RecRAG",
};

export default function RecommendationsPage() {
	return (
		<div className="mx-auto w-full max-w-2xl p-8">
			<RecommendationsList />
		</div>
	);
}
