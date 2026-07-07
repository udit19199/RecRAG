"use client";

import { IconChartBar, IconTopologyStar3 } from "@tabler/icons-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Finer139Workbench } from "@/features/finer139/components/finer139-workbench";
import { GraphRagExplorer } from "@/features/graphrag/components/graphrag-explorer";
import { cn } from "@/lib/utils";

export type GraphRagTab = "retrieval" | "benchmark";

const TABS: {
	id: GraphRagTab;
	label: string;
	short: string;
	icon: typeof IconTopologyStar3;
}[] = [
	{
		id: "retrieval",
		label: "Retrieval patterns",
		short: "How to query a graph",
		icon: IconTopologyStar3,
	},
	{
		id: "benchmark",
		label: "FiNER-139 benchmark",
		short: "How to build the graph",
		icon: IconChartBar,
	},
];

function parseTab(value: string | null): GraphRagTab {
	return value === "benchmark" ? "benchmark" : "retrieval";
}

export function GraphRagResearchHub() {
	const router = useRouter();
	const searchParams = useSearchParams();
	const tab = parseTab(searchParams.get("tab"));

	const setTab = (next: GraphRagTab) => {
		const href = next === "benchmark" ? "/graphrag?tab=benchmark" : "/graphrag";
		router.replace(href, { scroll: false });
	};

	return (
		<div className="flex h-full min-h-0 flex-col overflow-hidden bg-muted/30">
			<div className="shrink-0 border-b bg-background/95 px-4 py-3 backdrop-blur">
				<div className="mx-auto flex max-w-[1200px] flex-col gap-3">
					<div>
						<h1 className="text-lg font-semibold">GraphRAG research</h1>
						<p className="text-sm text-muted-foreground">
							Retrieval patterns and FiNER-139 graph-construction benchmark —
							two sides of the same GraphRAG analysis.
						</p>
					</div>
					<div
						className="flex flex-wrap gap-2"
						role="tablist"
						aria-label="GraphRAG research sections"
					>
						{TABS.map(({ id, label, short, icon: Icon }) => {
							const active = tab === id;
							return (
								<button
									key={id}
									type="button"
									role="tab"
									aria-selected={active}
									onClick={() => setTab(id)}
									className={cn(
										"flex min-w-[200px] flex-1 items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors sm:max-w-xs",
										active
											? "border-primary/40 bg-primary/5 text-foreground"
											: "border-border bg-card text-muted-foreground hover:border-border hover:bg-muted/50",
									)}
								>
									<Icon
										className={cn(
											"mt-0.5 size-5 shrink-0",
											active ? "text-primary" : "text-muted-foreground",
										)}
									/>
									<span>
										<span className="block text-sm font-medium">{label}</span>
										<span className="block text-xs opacity-80">{short}</span>
									</span>
								</button>
							);
						})}
					</div>
				</div>
			</div>

			<div className="min-h-0 flex-1 overflow-hidden">
				{tab === "retrieval" ? (
					<GraphRagExplorer onOpenBenchmark={() => setTab("benchmark")} />
				) : (
					<Finer139Workbench
						embedded
						onOpenRetrieval={() => setTab("retrieval")}
					/>
				)}
			</div>
		</div>
	);
}
