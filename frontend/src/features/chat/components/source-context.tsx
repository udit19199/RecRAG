import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import type { QueryResponse } from "@/lib/api";

export function SourceContext({
	context,
}: {
	context: QueryResponse["context"];
}) {
	const [openIndexes, setOpenIndexes] = useState<Set<number>>(new Set());

	const toggle = (index: number) => {
		setOpenIndexes((prev) => {
			const next = new Set(prev);
			if (next.has(index)) next.delete(index);
			else next.add(index);
			return next;
		});
	};

	return (
		<div className="overflow-hidden rounded-xl border bg-background">
			<div className="flex items-center gap-2 border-b px-4 py-2.5">
				<span className="text-sm font-semibold text-foreground">Sources</span>
				<Badge variant="secondary" className="h-4 min-w-4 px-1 text-[10px]">
					{context.length}
				</Badge>
			</div>
			<div className="divide-y divide-border">
				{context.map((item, index) => (
					<Collapsible
						key={`${item.source}-${item.distance}-${item.text}`}
						open={openIndexes.has(index)}
						onOpenChange={() => toggle(index)}
					>
						<CollapsibleTrigger className="flex w-full items-center justify-between px-4 py-2.5 text-left transition-colors hover:bg-muted/50">
							<div className="flex min-w-0 items-center gap-2.5">
								<Badge
									variant="outline"
									className="size-5 rounded-full p-0 text-[10px]"
								>
									{index + 1}
								</Badge>
								<span className="truncate text-xs font-medium text-foreground">
									{item.source}
								</span>
							</div>
							<div className="ml-3 flex shrink-0 items-center gap-2">
								<Tooltip>
									<TooltipTrigger asChild>
										<span className="cursor-help text-[10px] text-muted-foreground">
											{item.distance.toFixed(4)}
										</span>
									</TooltipTrigger>
									<TooltipContent>
										<p>Vector similarity score</p>
									</TooltipContent>
								</Tooltip>
								<svg
									aria-hidden="true"
									className={`size-3.5 text-muted-foreground transition-transform ${openIndexes.has(index) ? "rotate-180" : ""}`}
									viewBox="0 0 20 20"
									fill="currentColor"
								>
									<path
										fillRule="evenodd"
										d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
										clipRule="evenodd"
									/>
								</svg>
							</div>
						</CollapsibleTrigger>
						<CollapsibleContent>
							<div className="border-t bg-muted/40 px-4 py-3">
								<p className="whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
									{item.text}
								</p>
							</div>
						</CollapsibleContent>
					</Collapsible>
				))}
			</div>
		</div>
	);
}
