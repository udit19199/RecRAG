"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import type { Finer139Example } from "@/lib/api/types";

function spanClass(kind: "gold" | "pred") {
	return kind === "gold"
		? "rounded bg-amber-200/80 px-0.5 text-amber-950 dark:bg-amber-900/50 dark:text-amber-100"
		: "rounded bg-sky-200/80 px-0.5 text-sky-950 dark:bg-sky-900/50 dark:text-sky-100";
}

interface HighlightedTokensProps {
	tokens: string[];
	highlight: [number, number][];
	className: string;
	tokenOffsets?: [number, number][];
}

function HighlightedTokens({
	tokens,
	highlight,
	className,
	tokenOffsets,
}: HighlightedTokensProps) {
	const marked = new Set<number>();
	for (const [start, end] of highlight) {
		for (let i = start; i < end; i++) marked.add(i);
	}
	return (
		<>
			{tokens.map((tok, i) => {
				const key = tokenOffsets
					? `${tokenOffsets[i]?.[0] ?? i}-${tok}`
					: `${tok}-${i}`;
				return (
					<span key={key}>
						{i > 0 ? " " : ""}
						<span className={marked.has(i) ? className : undefined}>{tok}</span>
					</span>
				);
			})}
		</>
	);
}

interface Finer139ExamplesPanelProps {
	examples: Finer139Example[];
}

export function Finer139ExamplesPanel({
	examples,
}: Finer139ExamplesPanelProps) {
	const [index, setIndex] = useState(0);
	if (examples.length === 0) return null;

	const ex = examples[index] ?? examples[0];
	const methodNames = Object.keys(ex.predictions);

	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-lg">Examples</CardTitle>
				<CardDescription>
					Gold spans (amber) vs predicted numeric spans per method (sky). Sample
					{index + 1} of {examples.length}.
				</CardDescription>
			</CardHeader>
			<CardContent className="space-y-4">
				<div className="flex flex-wrap gap-2">
					{examples.map((_, i) => (
						<button
							key={examples[i].index}
							type="button"
							className={`rounded-md border px-2 py-1 text-xs ${
								i === index
									? "border-primary bg-primary/10"
									: "border-border hover:bg-muted"
							}`}
							onClick={() => setIndex(i)}
						>
							#{i + 1}
						</button>
					))}
				</div>

				<div className="rounded-md border bg-muted/30 p-3 text-sm leading-relaxed">
					<p className="mb-2 text-xs font-medium text-muted-foreground">Gold</p>
					<HighlightedTokens
						tokens={ex.tokens}
						highlight={ex.gold}
						className={spanClass("gold")}
					/>
				</div>

				{methodNames.map((name) => (
					<div
						key={name}
						className="rounded-md border bg-muted/20 p-3 text-sm leading-relaxed"
					>
						<div className="mb-2 flex items-center gap-2">
							<Badge variant="outline" className="text-xs">
								{name}
							</Badge>
						</div>
						<HighlightedTokens
							tokens={ex.tokens}
							highlight={ex.predictions[name] ?? []}
							className={spanClass("pred")}
						/>
					</div>
				))}
			</CardContent>
		</Card>
	);
}
