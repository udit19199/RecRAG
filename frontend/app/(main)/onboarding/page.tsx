import type { Metadata } from "next";
import { OnboardingFlow } from "@/components/onboarding/onboarding-flow";

export const metadata: Metadata = {
	title: "New recommendation | RecRAG",
	description: "Describe your RAG use case and constraints.",
};

export default function OnboardingPage() {
	return (
		<div className="flex flex-1 items-center justify-center p-6">
			<OnboardingFlow />
		</div>
	);
}
