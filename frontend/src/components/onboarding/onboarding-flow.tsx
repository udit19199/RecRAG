"use client";

import { RequirementsWizard } from "@/components/onboarding/RequirementsWizard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function OnboardingFlow() {
	return (
		<Card className="relative w-full max-w-2xl p-8 md:p-12">
			<CardHeader className="px-0">
				<CardTitle>Pipeline recommendation</CardTitle>
				<p className="text-sm text-muted-foreground">
					Define the constraints that matter for your use case. Skip anything
					you do not want to specify.
				</p>
			</CardHeader>
			<CardContent className="px-0">
				<RequirementsWizard />
			</CardContent>
		</Card>
	);
}
