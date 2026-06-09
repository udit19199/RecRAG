"use client";

import { useState } from "react";
import { RoleSelection } from "@/components/onboarding/RoleSelection";
import { ToolsIntegration } from "@/components/onboarding/ToolsIntegration";
import { WorkspaceSetup } from "@/components/onboarding/WorkspaceSetup";
import { Card, CardContent } from "@/components/ui/card";

function completeOnboarding() {
	window.location.href = "/chat";
}

export function OnboardingFlow() {
	const [step, setStep] = useState(0);

	const nextStep = () => setStep((s) => s + 1);
	const prevStep = () => setStep((s) => s - 1);

	return (
		<Card className="relative w-full max-w-2xl animate-in p-8 duration-500 ease-out fade-in slide-in-from-bottom-4 md:p-12">
			<CardContent className="relative z-10 px-0">
				{step === 0 && <RoleSelection onNext={nextStep} />}
				{step === 1 && <ToolsIntegration onNext={nextStep} onPrev={prevStep} />}
				{step === 2 && (
					<WorkspaceSetup onComplete={completeOnboarding} onPrev={prevStep} />
				)}
			</CardContent>
		</Card>
	);
}
