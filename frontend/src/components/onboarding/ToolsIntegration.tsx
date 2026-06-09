"use client";

import {
	CheckCircle2,
	Circle,
	Cloud,
	Database,
	FileText,
	HardDrive,
	LayoutList,
	MessageSquare,
} from "lucide-react";
import { useState } from "react";

interface ToolsIntegrationProps {
	onNext: () => void;
	onPrev: () => void;
}

const TOOLS = [
	{ id: "s3", label: "Amazon S3", icon: Cloud, color: "text-chart-1" },
	{
		id: "drive",
		label: "Google Drive",
		icon: HardDrive,
		color: "text-chart-2",
	},
	{
		id: "confluence",
		label: "Confluence",
		icon: LayoutList,
		color: "text-chart-3",
	},
	{ id: "notion", label: "Notion", icon: FileText, color: "text-foreground" },
	{ id: "slack", label: "Slack", icon: MessageSquare, color: "text-chart-4" },
	{
		id: "snowflake",
		label: "Snowflake",
		icon: Database,
		color: "text-chart-5",
	},
];

export function ToolsIntegration({ onNext, onPrev }: ToolsIntegrationProps) {
	const [selected, setSelected] = useState<Set<string>>(new Set());

	const toggleTool = (id: string) => {
		const newSelected = new Set(selected);
		if (newSelected.has(id)) {
			newSelected.delete(id);
		} else {
			newSelected.add(id);
		}
		setSelected(newSelected);
	};

	return (
		<div className="flex flex-col animate-in fade-in duration-300">
			<div className="text-left mb-8">
				<h1 className="text-2xl font-semibold text-foreground mb-1 tracking-tight">
					Connect Data Sources
				</h1>
				<p className="text-sm text-muted-foreground">
					Select the initial pipelines you wish to configure.
				</p>
			</div>

			<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
				{TOOLS.map((tool) => {
					const isSelected = selected.has(tool.id);
					const Icon = tool.icon;
					return (
						<button
							key={tool.id}
							type="button"
							onClick={() => toggleTool(tool.id)}
							className={`flex items-center justify-between p-3 text-left border rounded-md transition-all duration-150 ${
								isSelected
									? "border-foreground bg-muted/30"
									: "border-border bg-card hover:border-foreground/50 hover:bg-muted/10"
							}`}
						>
							<div className="flex items-center gap-3">
								<Icon
									className={`w-4 h-4 text-foreground`}
									aria-hidden="true"
								/>
								<span className="font-medium text-sm text-foreground">
									{tool.label}
								</span>
							</div>
							{isSelected ? (
								<CheckCircle2
									className="w-4 h-4 text-foreground"
									aria-hidden="true"
								/>
							) : (
								<Circle
									className="w-4 h-4 text-muted-foreground/30"
									aria-hidden="true"
								/>
							)}
						</button>
					);
				})}
			</div>

			<div className="mt-10 flex items-center justify-between">
				<button
					type="button"
					onClick={onPrev}
					className="text-muted-foreground hover:text-foreground text-sm font-medium transition-colors"
				>
					Back
				</button>
				<button
					type="button"
					onClick={onNext}
					className="bg-foreground hover:bg-foreground/90 text-background px-6 py-2 rounded-md font-medium transition-colors"
				>
					Next
				</button>
			</div>
		</div>
	);
}
