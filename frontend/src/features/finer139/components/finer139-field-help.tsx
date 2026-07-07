"use client";

import { IconHelpCircle } from "@tabler/icons-react";
import { FieldDescription, FieldLabel } from "@/components/ui/field";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";

interface Finer139FieldHelpProps {
	htmlFor: string;
	label: string;
	help: string;
	children: React.ReactNode;
}

export function Finer139FieldHelp({
	htmlFor,
	label,
	help,
	children,
}: Finer139FieldHelpProps) {
	return (
		<div className="space-y-1.5">
			<div className="flex items-center gap-1.5">
				<FieldLabel htmlFor={htmlFor}>{label}</FieldLabel>
				<Tooltip>
					<TooltipTrigger asChild>
						<button
							type="button"
							className="text-muted-foreground hover:text-foreground"
							aria-label={`Help: ${label}`}
						>
							<IconHelpCircle className="size-3.5" />
						</button>
					</TooltipTrigger>
					<TooltipContent className="max-w-xs text-left">{help}</TooltipContent>
				</Tooltip>
			</div>
			{children}
		</div>
	);
}

export const FIELD_HELP = {
	sampleSize:
		"How many FiNER-139 sentences to score in this run (1–500). Smaller samples finish faster; use 100+ for stable comparisons.",
	seed: "Random seed for reproducibility. The same seed + sample size always picks the same sentences.",
	split:
		"Which HuggingFace split to draw from. Validation is the default for research; test is held-out for final checks.",
	stratified:
		"When enabled, samples are balanced across the top XBRL concept types so rare concepts are not under-represented.",
} as const;

export function Finer139IntroLink({
	onOpenRetrieval,
}: {
	onOpenRetrieval?: () => void;
} = {}) {
	return (
		<FieldDescription>
			Benchmarks which <strong>graph construction</strong> method best
			recognizes numeric financial entities — a prerequisite for any{" "}
			{onOpenRetrieval ? (
				<button
					type="button"
					className="underline underline-offset-2"
					onClick={onOpenRetrieval}
				>
					GraphRAG retrieval pattern
				</button>
			) : (
				"GraphRAG retrieval pattern"
			)}
			.
		</FieldDescription>
	);
}
