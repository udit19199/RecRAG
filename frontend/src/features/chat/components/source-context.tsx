import { IconChevronDown } from "@tabler/icons-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import {
	Tooltip,
	TooltipContent,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import type { QueryResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

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
		<Card className="overflow-hidden py-0">
			<CardHeader className="flex-row items-center gap-2 border-b py-2.5">
				<CardTitle className="text-sm font-semibold">Sources</CardTitle>
				<Badge variant="secondary" className="h-4 min-w-4 px-1 text-[10px]">
					{context.length}
				</Badge>
			</CardHeader>
			<CardContent className="flex flex-col gap-0 p-0">
				{context.map((item, index) => (
					<div key={`${item.source}-${item.distance}-${item.text}`}>
						{index > 0 ? <Separator /> : null}
						<Collapsible
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
									<IconChevronDown
										className={cn(
											"text-muted-foreground transition-transform",
											openIndexes.has(index) && "rotate-180",
										)}
									/>
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
					</div>
				))}
			</CardContent>
		</Card>
	);
}
