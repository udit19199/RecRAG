import type { Metadata } from "next";
import { GenerateRecommendationForm } from "@/features/recommendation/components/generate-recommendation-form";

export const metadata: Metadata = {
	title: "Generate recommendation | RecRAG",
};

export default function GeneratePage() {
	return (
		<div className="mx-auto w-full max-w-2xl p-8">
			<GenerateRecommendationForm />
		</div>
	);
}
