"use client";

import { IconChevronDown, IconSparkles } from "@tabler/icons-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { CONSTRUCTION_METHODS } from "@/features/finer139/lib/method-descriptions";
import type { Finer139Method } from "@/lib/api/types";
import { cn } from "@/lib/utils";

interface Finer139MethodCardsProps {
	selected: Finer139Method[];
	onToggle: (id: Finer139Method) => void;
	disabled?: boolean;
	onOpenRetrieval?: () => void;
}

export function Finer139MethodCards({
	selected,
	onToggle,
	disabled,
	onOpenRetrieval,
}: Finer139MethodCardsProps) {
	const [expanded, setExpanded] = useState<Set<Finer139Method>>(new Set());

	const toggleExpanded = (id: Finer139Method) => {
		setExpanded((prev) => {
			const next = new Set(prev);
			if (next.has(id)) next.delete(id);
			else next.add(id);
			return next;
		});
	};

	return (
		<div className="space-y-2">
			<div className="flex flex-wrap items-center justify-between gap-2">
				<p className="text-sm font-medium text-muted-foreground">
					Graph construction methods
				</p>
				<p className="text-xs text-muted-foreground">
					Not retrieval hybrid
					{onOpenRetrieval ? (
						<>
							{" "}
							—{" "}
							<button
								type="button"
								className="underline underline-offset-2"
								onClick={onOpenRetrieval}
							>
								see retrieval tab
							</button>
						</>
					) : null}
				</p>
			</div>
			<div className="grid gap-2 sm:grid-cols-2">
				{CONSTRUCTION_METHODS.map((method) => {
					const isSelected = selected.includes(method.id);
					const isOpen = expanded.has(method.id);
					return (
						<Collapsible
							key={method.id}
							open={isOpen}
							onOpenChange={() => toggleExpanded(method.id)}
						>
							<div
								className={cn(
									"rounded-lg border transition-colors",
									isSelected && "border-primary/40 bg-primary/5",
								)}
							>
								<div className="flex items-start gap-2 p-3">
									<input
										type="checkbox"
										checked={isSelected}
										disabled={disabled}
										onChange={() => onToggle(method.id)}
										className="mt-1"
										aria-label={`Select ${method.label}`}
									/>
									<div className="min-w-0 flex-1">
										<div className="flex flex-wrap items-center gap-1.5">
											<p className="text-sm font-medium">{method.label}</p>
											{method.needsLlm && (
												<Badge
													variant="secondary"
													className="h-5 gap-0.5 px-1.5 text-[10px]"
												>
													<IconSparkles className="size-3" />
													LLM
												</Badge>
											)}
										</div>
										<p className="mt-0.5 text-xs text-muted-foreground">
											{method.tagline}
										</p>
									</div>
									<CollapsibleTrigger
										className="shrink-0 rounded p-1 text-muted-foreground hover:bg-muted"
										aria-label={`Details for ${method.label}`}
									>
										<IconChevronDown
											className={cn(
												"size-4 transition-transform",
												isOpen && "rotate-180",
											)}
										/>
									</CollapsibleTrigger>
								</div>
								<CollapsibleContent>
									<p className="border-t px-3 py-2 text-sm text-muted-foreground">
										{method.description}
									</p>
								</CollapsibleContent>
							</div>
						</Collapsible>
					);
				})}
			</div>
		</div>
	);
}
