"use client";

import { Briefcase, Crown, UserCircle, Users } from "lucide-react";

interface RoleSelectionProps {
	onNext: () => void;
}

const ROLES = [
	{ id: "solo", label: "Working solo", icon: UserCircle },
	{ id: "member", label: "Team member", icon: Users },
	{ id: "manager", label: "Team manager", icon: Briefcase },
	{ id: "executive", label: "Executive (C-level / VP)", icon: Crown },
];

export function RoleSelection({ onNext }: RoleSelectionProps) {
	return (
		<div className="flex flex-col animate-in fade-in duration-300">
			<div className="text-left mb-8">
				<h1 className="text-2xl font-semibold text-foreground mb-1 tracking-tight">
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
						<button
							key={role.id}
							type="button"
							onClick={onNext}
							className="flex items-center gap-4 w-full p-3 text-left border border-border rounded-md hover:border-foreground hover:bg-muted/50 transition-all duration-150 group bg-card"
						>
							<div className="p-2 border border-transparent rounded-sm bg-muted text-muted-foreground group-hover:border-border group-hover:bg-background group-hover:text-foreground transition-colors">
								<Icon className="w-5 h-5" aria-hidden="true" />
							</div>
							<span className="font-medium text-foreground text-sm">
								{role.label}
							</span>
						</button>
					);
				})}
			</div>

			<div className="mt-8 flex justify-end">
				<button
					type="button"
					onClick={onNext}
					className="text-muted-foreground hover:text-foreground text-sm font-medium transition-colors"
				>
					Skip
				</button>
			</div>
		</div>
	);
}
