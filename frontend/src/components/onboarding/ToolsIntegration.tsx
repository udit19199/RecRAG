"use client";

import {
	IconBrandGoogleDrive,
	IconCircle,
	IconCircleCheck,
	IconCloud,
	IconDatabase,
	IconFileText,
	IconLayoutList,
	IconMessage,
} from "@tabler/icons-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ToolsIntegrationProps {
	onNext: () => void;
	onPrev: () => void;
}

const TOOLS = [
	{ id: "s3", label: "Amazon S3", icon: IconCloud },
	{ id: "drive", label: "Google Drive", icon: IconBrandGoogleDrive },
	{ id: "confluence", label: "Confluence", icon: IconLayoutList },
	{ id: "notion", label: "Notion", icon: IconFileText },
	{ id: "slack", label: "Slack", icon: IconMessage },
	{ id: "snowflake", label: "Snowflake", icon: IconDatabase },
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
		<div className="flex animate-in flex-col duration-300 fade-in">
			<div className="mb-8 text-left">
				<h1 className="mb-1 text-2xl font-semibold tracking-tight text-foreground">
					Connect Data Sources
				</h1>
				<p className="text-sm text-muted-foreground">
					Select the initial pipelines you wish to configure.
				</p>
			</div>

			<div className="grid grid-cols-1 gap-4 md:grid-cols-2">
				{TOOLS.map((tool) => {
					const isSelected = selected.has(tool.id);
					const Icon = tool.icon;
					return (
						<Button
							key={tool.id}
							type="button"
							variant="outline"
							onClick={() => toggleTool(tool.id)}
							className={cn(
								"h-auto justify-between p-3",
								isSelected && "border-foreground bg-muted/30",
							)}
						>
							<div className="flex items-center gap-3">
								<Icon data-icon="inline-start" />
								<span className="text-sm font-medium text-foreground">
									{tool.label}
								</span>
							</div>
							{isSelected ? (
								<IconCircleCheck data-icon="inline-end" />
							) : (
								<IconCircle
									data-icon="inline-end"
									className="text-muted-foreground/30"
								/>
							)}
						</Button>
					);
				})}
			</div>

			<div className="mt-10 flex items-center justify-between">
				<Button type="button" variant="ghost" onClick={onPrev}>
					Back
				</Button>
				<Button type="button" onClick={onNext}>
					Next
				</Button>
			</div>
		</div>
	);
}
