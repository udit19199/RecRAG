import type { Metadata } from "next";
import { OnboardingFlow } from "./onboarding-flow";

export const metadata: Metadata = {
	title: "Onboarding | RecRAG",
	description: "Configure your RecRAG workspace and connect data sources.",
};

export default function OnboardingPage() {
	return <OnboardingFlow />;
}
