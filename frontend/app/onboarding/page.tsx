"use client";

import { useState } from "react";
import { RoleSelection } from "@/components/onboarding/RoleSelection";
import { ToolsIntegration } from "@/components/onboarding/ToolsIntegration";
import { WorkspaceSetup } from "@/components/onboarding/WorkspaceSetup";

export default function OnboardingPage() {
	const [step, setStep] = useState(0);

	const nextStep = () => setStep((s) => s + 1);
	const prevStep = () => setStep((s) => s - 1);
	const handleComplete = () => {
		window.location.href = "/chat";
	};

	return (
		<div className="w-full max-w-2xl bg-card text-card-foreground border border-border rounded-md p-8 md:p-12 animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out relative">
			<div className="relative z-10">
				{step === 0 && <RoleSelection onNext={nextStep} />}
				{step === 1 && <ToolsIntegration onNext={nextStep} onPrev={prevStep} />}
				{step === 2 && (
					<WorkspaceSetup onComplete={handleComplete} onPrev={prevStep} />
				)}
			</div>
		</div>
	);
}
