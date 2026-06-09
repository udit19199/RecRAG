"use client";

import {
	IconBriefcase,
	IconCrown,
	IconUserCircle,
	IconUsers,
} from "@tabler/icons-react";
import { Button } from "@/components/ui/button";

interface RoleSelectionProps {
	onNext: () => void;
}

const ROLES = [
	{ id: "solo", label: "Working solo", icon: IconUserCircle },
	{ id: "member", label: "Team member", icon: IconUsers },
	{ id: "manager", label: "Team manager", icon: IconBriefcase },
	{ id: "executive", label: "Executive (C-level / VP)", icon: IconCrown },
];

export function RoleSelection({ onNext }: RoleSelectionProps) {
	return (
		<div className="flex animate-in flex-col duration-300 fade-in">
			<div className="mb-8 text-left">
				<h1 className="mb-1 text-2xl font-semibold tracking-tight text-foreground">
					Select Operating Mode
				</h1>
				<p className="text-sm text-muted-foreground">
					We will tailor the setup process based on your operational scale.
				</p>
			</div>

			<div className="flex flex-col gap-3">
				{ROLES.map((role) => {
					const Icon = role.icon;
					return (
						<Button
							key={role.id}
							type="button"
							variant="outline"
							onClick={onNext}
							className="group h-auto w-full justify-start gap-4 p-3"
						>
							<div className="rounded-sm border border-transparent bg-muted p-2 text-muted-foreground transition-colors group-hover:border-border group-hover:bg-background group-hover:text-foreground">
								<Icon data-icon="inline-start" />
							</div>
							<span className="text-sm font-medium text-foreground">
								{role.label}
							</span>
						</Button>
					);
				})}
			</div>

			<div className="mt-8 flex justify-end">
				<Button type="button" variant="ghost" onClick={onNext}>
					Skip
				</Button>
			</div>
		</div>
	);
}
