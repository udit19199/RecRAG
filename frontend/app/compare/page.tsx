import type { Metadata } from "next";
import { ModelComparisonWorkbench } from "@/features/model-compare/components/model-comparison-workbench";

export const metadata: Metadata = {
	title: "Model Comparison — RecRAG",
	description: "Compare AI model configurations side-by-side for generation quality.",
};

export default function ComparePage() {
	return <ModelComparisonWorkbench />;
}
