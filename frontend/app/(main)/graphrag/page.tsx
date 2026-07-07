import type { Metadata } from "next";
import { Suspense } from "react";
import { GraphRagResearchHub } from "@/features/graphrag/components/graphrag-research-hub";

export const metadata: Metadata = {
	title: "GraphRAG — RecRAG",
	description:
		"GraphRAG retrieval patterns and FiNER-139 graph-construction benchmark.",
};

export default function GraphRagPage() {
	return (
		<Suspense fallback={null}>
			<GraphRagResearchHub />
		</Suspense>
	);
}
