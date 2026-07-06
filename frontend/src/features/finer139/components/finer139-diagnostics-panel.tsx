"use client";

import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
} from "@/components/ui/card";
import type { Finer139MethodResult } from "@/lib/api/types";

function pct(n: number | undefined) {
	if (n === undefined) return "—";
	return `${(n * 100).toFixed(1)}%`;
}

interface Finer139DiagnosticsPanelProps {
	methods: Finer139MethodResult[];
	comparison?: {
		sentence_wins?: Record<string, number>;
		sentences_with_gold?: number;
		ties?: number;
	} | null;
}

export function Finer139DiagnosticsPanel({
	methods,
	comparison,
}: Finer139DiagnosticsPanelProps) {
	const scored = methods.filter((m) => m.diagnostics && !m.error);
	if (scored.length === 0) return null;

	return (
		<div className="space-y-4">
			<Card>
				<CardHeader>
					<h2 className="font-heading text-lg leading-snug font-medium">
						Extended evaluation
					</h2>
					<CardDescription>
						Macro F1 (per-sentence average), partial match (IoU ≥ 0.5),
						bootstrap 95% CI, error taxonomy, and concept-stratified recall.
					</CardDescription>
				</CardHeader>
				<CardContent className="overflow-x-auto">
					<table className="w-full min-w-[960px] border-collapse text-sm">
						<thead>
							<tr className="border-b text-left text-muted-foreground">
								<th className="py-2 pr-4 font-medium">Method</th>
								<th className="px-2 font-medium">Partial F1</th>
								<th className="px-2 font-medium">Macro strict F1</th>
								<th className="px-2 font-medium">95% CI (strict)</th>
								<th className="px-2 font-medium">Sentence hit rate</th>
								<th className="px-2 font-medium">Mean span IoU</th>
								<th className="px-2 font-medium">Boundary FP</th>
								<th className="px-2 font-medium">Spurious FP</th>
							</tr>
						</thead>
						<tbody>
							{scored.map((m) => {
								const d = m.diagnostics;
								if (!d) return null;
								const ci = d.bootstrap_strict_f1_ci;
								return (
									<tr key={m.name} className="border-b border-border/60">
										<td className="py-2.5 pr-4 font-medium">
											{m.display_name}
										</td>
										<td className="px-2 tabular-nums">{pct(m.partial?.f1)}</td>
										<td className="px-2 tabular-nums">
											{pct(m.macro_strict?.f1)}
										</td>
										<td className="px-2 tabular-nums text-xs">
											{pct(ci?.low)} – {pct(ci?.high)}
										</td>
										<td className="px-2 tabular-nums">
											{pct(d.sentence_hit_rate)}
										</td>
										<td className="px-2 tabular-nums">
											{pct(d.span_iou_mean)}
										</td>
										<td className="px-2 tabular-nums">
											{d.errors?.boundary_fp ?? "—"}
										</td>
										<td className="px-2 tabular-nums">
											{d.errors?.spurious_fp ?? "—"}
										</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</CardContent>
			</Card>

			{comparison?.sentence_wins &&
				Object.keys(comparison.sentence_wins).length > 0 && (
					<Card>
						<CardHeader>
							<h3 className="text-base font-medium">Head-to-head</h3>
							<CardDescription>
								Sentences where each method achieved the highest strict F1 (
								{comparison.sentences_with_gold ?? 0} sentences with gold;
								{comparison.ties ?? 0} ties).
							</CardDescription>
						</CardHeader>
						<CardContent className="flex flex-wrap gap-3">
							{Object.entries(comparison.sentence_wins).map(([name, wins]) => (
								<div
									key={name}
									className="rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm"
								>
									<span className="font-medium">{name}</span>
									<span className="ml-2 tabular-nums text-muted-foreground">
										{wins} wins
									</span>
								</div>
							))}
						</CardContent>
					</Card>
				)}
		</div>
	);
}
