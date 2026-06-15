"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { RecommendationPanel } from "@/features/recommendation/components/recommendation-panel";

function RecommendContent() {
	const params = useSearchParams();
	const runId = params.get("run_id");

	if (!runId) {
		return (
			<div className="p-8 text-sm text-muted-foreground">
				No recommendation run selected.
			</div>
		);
	}

	return (
		<div className="mx-auto max-w-3xl p-8">
			<RecommendationPanel runId={runId} />
		</div>
	);
}

export default function RecommendPage() {
	return (
		<Suspense
			fallback={
				<div className="p-8 text-sm text-muted-foreground">Loading...</div>
			}
		>
			<RecommendContent />
		</Suspense>
	);
}
