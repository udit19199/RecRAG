"use client";

import {
	IconArrowRight,
	IconChevronDown,
	IconRoute,
	IconTopologyStar3,
} from "@tabler/icons-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import {
	Collapsible,
	CollapsibleContent,
	CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";

const APPROACHES = [
	{
		name: "Vector RAG",
		retrieves: "Top-k similar chunks",
		bestFor: "Single-fact lookup, definitions",
	},
	{
		name: "Neighborhood",
		retrieves: "Ego-network around seed entities (k-hop)",
		bestFor: "Local multi-entity context",
	},
	{
		name: "Community graph",
		retrieves: "Pre-summarized clusters (local or global)",
		bestFor: "Thematic / corpus-wide questions",
	},
	{
		name: "PathRAG",
		retrieves: "Pruned relational paths between entities",
		bestFor: "Multi-hop reasoning with less noise",
	},
	{
		name: "Hybrid retrieval",
		retrieves: "Vector + graph combined",
		bestFor: "Mixed query types in one product",
	},
] as const;

const PATH_VS_NEIGHBORHOOD = [
	{
		dimension: "Unit retrieved",
		neighborhood: "Entire k-hop subgraph (nodes + all edges)",
		pathrag: "Key relational paths only (pruned sequences)",
	},
	{
		dimension: "Redundancy",
		neighborhood: "High — duplicate facts via multiple edges",
		pathrag: "Low — flow-based pruning drops weak paths",
	},
	{
		dimension: "Prompt structure",
		neighborhood: "Flat triple list or node descriptions",
		pathrag: "Ordered paths (A → rel → B → rel → C)",
	},
	{
		dimension: "Token efficiency",
		neighborhood: "Poor at larger k",
		pathrag: "~40% fewer tokens vs full subgraph",
	},
	{
		dimension: "Best queries",
		neighborhood: '"What is near X?"',
		pathrag: '"How does A connect to B?" / causal chains',
	},
] as const;

const HYBRID_STRATEGIES = [
	{
		name: "Selection",
		mechanism:
			"Router classifies query → run either vector RAG or graph retrieval",
		example:
			'Simple fact → Milvus; "compare themes across filings" → community global',
	},
	{
		name: "Integration",
		mechanism: "Run both, merge/dedupe contexts, single LLM call",
		example:
			"Graph local search + embedding-similar chunks for peer comparison",
	},
] as const;

function SectionHeading({
	id,
	title,
	description,
}: {
	id: string;
	title: string;
	description: string;
}) {
	return (
		<div id={id} className="scroll-mt-4 space-y-1">
			<h2 className="font-heading text-lg font-medium">{title}</h2>
			<p className="text-sm text-muted-foreground">{description}</p>
		</div>
	);
}

function CodeBlock({ children }: { children: string }) {
	return (
		<pre className="overflow-x-auto rounded-md border bg-muted/50 p-3 font-mono text-xs leading-relaxed text-foreground">
			{children}
		</pre>
	);
}

export function GraphRagExplorer({
	onOpenBenchmark,
}: {
	onOpenBenchmark?: () => void;
} = {}) {
	const [communityOpen, setCommunityOpen] = useState(true);
	const [pathOpen, setPathOpen] = useState(true);

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-y-auto p-4">
			<div className="mx-auto flex h-full w-full max-w-[1100px] flex-col gap-4 overflow-y-auto">
				<Card className="shrink-0">
					<CardHeader>
						<div className="flex items-start gap-3">
							<div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
								<IconTopologyStar3 className="size-5" />
							</div>
							<div className="space-y-1">
								<CardTitle className="text-xl">
									GraphRAG Retrieval Patterns
								</CardTitle>
								<CardDescription>
									How graph-based retrieval differs from flat vector RAG —
									community search, neighborhood expansion, PathRAG pruning, and
									hybrid strategies. Aligned with the GraphRAG analysis
									taxonomy.
								</CardDescription>
							</div>
						</div>
					</CardHeader>
					<CardContent className="flex flex-wrap gap-2">
						<Badge variant="secondary">Research reference</Badge>
						<Badge variant="outline">D15 GRAPH (planned)</Badge>
						{onOpenBenchmark && (
							<Button
								variant="link"
								className="h-auto p-0 text-sm"
								type="button"
								onClick={onOpenBenchmark}
							>
								Why entity recognition matters
								<IconArrowRight className="ml-1 size-3.5" />
							</Button>
						)}
					</CardContent>
				</Card>

				<Card>
					<CardHeader className="pb-3">
						<CardTitle className="text-base">
							Retrieval approaches at a glance
						</CardTitle>
						<CardDescription>
							Flat RAG retrieves isolated chunks. GraphRAG builds a knowledge
							graph first, then retrieves structured context.
						</CardDescription>
					</CardHeader>
					<CardContent className="overflow-x-auto">
						<table className="w-full min-w-[640px] border-collapse text-sm">
							<thead>
								<tr className="border-b text-left text-muted-foreground">
									<th className="py-2 pr-4 font-medium">Approach</th>
									<th className="px-2 font-medium">What gets retrieved</th>
									<th className="px-2 font-medium">Best for</th>
								</tr>
							</thead>
							<tbody>
								{APPROACHES.map((row) => (
									<tr key={row.name} className="border-b last:border-0">
										<td className="py-2.5 pr-4 font-medium">{row.name}</td>
										<td className="px-2 text-muted-foreground">
											{row.retrieves}
										</td>
										<td className="px-2 text-muted-foreground">
											{row.bestFor}
										</td>
									</tr>
								))}
							</tbody>
						</table>
					</CardContent>
				</Card>

				<Collapsible open={communityOpen} onOpenChange={setCommunityOpen}>
					<Card>
						<CollapsibleTrigger className="flex w-full items-center justify-between px-6 py-4 text-left">
							<SectionHeading
								id="community"
								title="Community-graph retrieval"
								description="Microsoft GraphRAG style — local vs global search over pre-summarized clusters."
							/>
							<IconChevronDown
								className={cn(
									"ml-4 shrink-0 text-muted-foreground transition-transform",
									communityOpen && "rotate-180",
								)}
							/>
						</CollapsibleTrigger>
						<CollapsibleContent>
							<CardContent className="space-y-4 pt-0">
								<div className="grid gap-3 sm:grid-cols-2">
									<div className="rounded-lg border p-3">
										<p className="text-sm font-medium">Local search</p>
										<p className="mt-1 text-sm text-muted-foreground">
											Matched entities, incident edges, and lower-level
											community reports. Fits entity-centric questions in one
											filing.
										</p>
									</div>
									<div className="rounded-lg border p-3">
										<p className="text-sm font-medium">Global search</p>
										<p className="mt-1 text-sm text-muted-foreground">
											High-level community summaries only. Fits thematic
											questions across the whole corpus without reading every
											chunk.
										</p>
									</div>
								</div>

								<Separator />

								<div>
									<p className="mb-2 text-sm font-medium">
										Example — 50 SEC 10-K filings (technology sector)
									</p>
									<CodeBlock>{`Community C1 (Cloud revenue)     Community C2 (Supply chain)
  ├─ AWS, Azure, GCP summaries      ├─ semiconductor shortage
  └─ segment revenue tables           └─ vendor concentration

Community C3 (Regulatory) — "Cross-filing compliance themes"`}</CodeBlock>
								</div>

								<div className="grid gap-3 md:grid-cols-2">
									<div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
										<p className="text-xs font-medium uppercase tracking-wide text-primary">
											Query A — local
										</p>
										<p className="mt-1 text-sm italic">
											"What revenue did Company X report for cloud services in
											FY2023?"
										</p>
										<p className="mt-2 text-sm text-muted-foreground">
											Seed <code className="text-xs">Company X</code>, pull
											REPORTS_REVENUE edges and the C1 community report snippet.
										</p>
									</div>
									<div className="rounded-lg border p-3">
										<p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
											Query B — global
										</p>
										<p className="mt-1 text-sm italic">
											"What are the dominant risk themes across all filings this
											year?"
										</p>
										<p className="mt-2 text-sm text-muted-foreground">
											Retrieve top-level summaries from C1, C2, C3 — not every
											chunk — then synthesize themes.
										</p>
									</div>
								</div>
							</CardContent>
						</CollapsibleContent>
					</Card>
				</Collapsible>

				<Collapsible open={pathOpen} onOpenChange={setPathOpen}>
					<Card>
						<CollapsibleTrigger className="flex w-full items-center justify-between px-6 py-4 text-left">
							<div className="flex items-start gap-2">
								<IconRoute className="mt-0.5 size-5 shrink-0 text-muted-foreground" />
								<SectionHeading
									id="path-vs-neighborhood"
									title="PathRAG vs neighborhood retrieval"
									description="Both start from seed entities. The difference is what you keep after expansion."
								/>
							</div>
							<IconChevronDown
								className={cn(
									"ml-4 shrink-0 text-muted-foreground transition-transform",
									pathOpen && "rotate-180",
								)}
							/>
						</CollapsibleTrigger>
						<CollapsibleContent>
							<CardContent className="space-y-4 pt-0">
								<div className="grid gap-3 sm:grid-cols-2">
									<div className="rounded-lg border p-3">
										<p className="text-sm font-medium">Neighborhood</p>
										<p className="mt-1 text-sm text-muted-foreground">
											"Give me everything within distance k of these seeds."
										</p>
									</div>
									<div className="rounded-lg border p-3">
										<p className="text-sm font-medium">PathRAG</p>
										<p className="mt-1 text-sm text-muted-foreground">
											"Give me the few most reliable paths that explain how
											seeds connect to answers."
										</p>
									</div>
								</div>

								<div className="overflow-x-auto">
									<table className="w-full min-w-[640px] border-collapse text-sm">
										<thead>
											<tr className="border-b text-left text-muted-foreground">
												<th className="py-2 pr-4 font-medium">Dimension</th>
												<th className="px-2 font-medium">Neighborhood</th>
												<th className="px-2 font-medium">PathRAG</th>
											</tr>
										</thead>
										<tbody>
											{PATH_VS_NEIGHBORHOOD.map((row) => (
												<tr
													key={row.dimension}
													className="border-b last:border-0"
												>
													<td className="py-2.5 pr-4 font-medium">
														{row.dimension}
													</td>
													<td className="px-2 text-muted-foreground">
														{row.neighborhood}
													</td>
													<td className="px-2 text-muted-foreground">
														{row.pathrag}
													</td>
												</tr>
											))}
										</tbody>
									</table>
								</div>

								<Separator />

								<div>
									<p className="mb-2 text-sm font-medium">
										Side-by-side example
									</p>
									<p className="mb-3 text-sm italic text-muted-foreground">
										"What is the link between stock-based compensation expense
										and unrecognized compensation cost for Company X?"
									</p>
									<div className="grid gap-3 lg:grid-cols-2">
										<div className="space-y-2">
											<p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
												Neighborhood (k=2) — ~45 triples
											</p>
											<CodeBlock>{`(Company X) --RECORDED--> stock-based compensation ($45M)
(Company X) --HAS--> unrecognized compensation cost ($12M)
(Company X) --OPERATES_IN--> North America
(Company X) --FILED--> 10-K 2023
(North America) --HAS_REGULATION--> SEC Rule 402
... 40 more incidental edges ...`}</CodeBlock>
											<p className="text-sm text-muted-foreground">
												LLM must find the two compensation facts among noise.
											</p>
										</div>
										<div className="space-y-2">
											<p className="text-xs font-medium uppercase tracking-wide text-primary">
												PathRAG — top paths only
											</p>
											<CodeBlock>{`Path 1 (score 0.91):
  Company X --RECORDED--> stock-based compensation ($45M)
  Company X --HAS--> unrecognized compensation cost ($12M)

Path 2 (score 0.12, dropped):
  Company X --OPERATES_IN--> North America
    --HAS_REGULATION--> SEC Rule 402`}</CodeBlock>
											<p className="text-sm text-muted-foreground">
												Prompt contains 2 on-topic hops, not 45 triples.
											</p>
										</div>
									</div>
								</div>
							</CardContent>
						</CollapsibleContent>
					</Card>
				</Collapsible>

				<Card>
					<CardHeader>
						<SectionHeading
							id="hybrid"
							title="Hybrid retrieval"
							description="Combines vector (semantic) and graph (structural) retrieval — not the same as graph construction."
						/>
					</CardHeader>
					<CardContent className="space-y-4">
						<div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">
							<p className="font-medium text-amber-900 dark:text-amber-200">
								Not FiNER-139 &quot;Hybrid Construction&quot;
							</p>
							<p className="mt-1 text-muted-foreground">
								<strong>Hybrid retrieval</strong> merges vector search with
								graph traversal at query time.{" "}
								<strong>Hybrid (Schema-Guided LLM)</strong> in the{" "}
								{onOpenBenchmark ? (
									<button
										type="button"
										className="underline underline-offset-2"
										onClick={onOpenBenchmark}
									>
										FiNER-139 benchmark
									</button>
								) : (
									"FiNER-139 benchmark"
								)}{" "}
								tab is a <em>graph construction</em> method — how entities are
								extracted when building the graph.
							</p>
						</div>

						<div className="grid gap-3 sm:grid-cols-2">
							{HYBRID_STRATEGIES.map((s) => (
								<div key={s.name} className="rounded-lg border p-3">
									<p className="text-sm font-medium">{s.name}</p>
									<p className="mt-1 text-sm text-muted-foreground">
										{s.mechanism}
									</p>
									<p className="mt-2 text-xs text-muted-foreground">
										<span className="font-medium text-foreground">
											Example:{" "}
										</span>
										{s.example}
									</p>
								</div>
							))}
						</div>

						<CodeBlock>{`Query router (production stack)

                    ┌─ vector RAG (naive)
Query ── router ────┼─ community local (neighborhood + reports)
                    ├─ community global (thematic)
                    └─ PathRAG (pruned paths for multi-hop)`}</CodeBlock>
					</CardContent>
				</Card>

				<Card className="shrink-0 border-dashed">
					<CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between">
						<div>
							<p className="text-sm font-medium">
								Prerequisite: accurate entity recognition
							</p>
							<p className="text-sm text-muted-foreground">
								Every retrieval pattern above depends on how well the graph was
								built. Compare construction methods on FiNER-139.
							</p>
						</div>
						{onOpenBenchmark && (
							<Button variant="outline" type="button" onClick={onOpenBenchmark}>
								Open FiNER-139 benchmark
								<IconArrowRight className="ml-2 size-4" />
							</Button>
						)}
					</CardContent>
				</Card>
			</div>
		</div>
	);
}
