import { Badge } from "@/components/ui/badge";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import type { Finer139MethodResult } from "@/lib/api/types";

function pct(n: number | undefined) {
	if (n === undefined) return "—";
	return `${(n * 100).toFixed(1)}%`;
}

interface Finer139ResultsTableProps {
	methods: Finer139MethodResult[];
}

export function Finer139ResultsTable({ methods }: Finer139ResultsTableProps) {
	if (methods.length === 0) {
		return null;
	}

	return (
		<Card>
			<CardHeader>
				<CardTitle className="text-lg">Results</CardTitle>
				<CardDescription>
					Numeric-only evaluation universe; strict token-span match (primary)
					and relaxed overlap. Higher F1 is better.
				</CardDescription>
			</CardHeader>
			<CardContent className="overflow-x-auto">
				<table className="w-full min-w-[720px] border-collapse text-sm">
					<thead>
						<tr className="border-b text-left text-muted-foreground">
							<th className="py-2 pr-4 font-medium">Method</th>
							<th className="py-2 px-2 font-medium">P (strict)</th>
							<th className="py-2 px-2 font-medium">R (strict)</th>
							<th className="py-2 px-2 font-medium">F1 (strict)</th>
							<th className="py-2 px-2 font-medium">F1 (relaxed)</th>
							<th className="py-2 px-2 font-medium">Latency</th>
							<th className="py-2 pl-2 font-medium">LLM calls</th>
						</tr>
					</thead>
					<tbody>
						{methods.map((m) => (
							<tr key={m.name} className="border-b border-border/60">
								<td className="py-2.5 pr-4">
									<div className="flex flex-wrap items-center gap-2">
										<span className="font-medium">{m.display_name}</span>
										{m.uses_llm && (
											<Badge variant="secondary" className="text-xs">
												LLM
											</Badge>
										)}
									</div>
									{m.error && (
										<p className="mt-1 text-xs text-destructive">{m.error}</p>
									)}
								</td>
								<td className="px-2 tabular-nums">
									{pct(m.strict?.precision)}
								</td>
								<td className="px-2 tabular-nums">{pct(m.strict?.recall)}</td>
								<td className="px-2 tabular-nums font-medium">
									{pct(m.strict?.f1)}
								</td>
								<td className="px-2 tabular-nums">{pct(m.relaxed?.f1)}</td>
								<td className="px-2 tabular-nums">{m.latency_s.toFixed(1)}s</td>
								<td className="pl-2 tabular-nums">{m.llm_calls || "—"}</td>
							</tr>
						))}
					</tbody>
				</table>
			</CardContent>
		</Card>
	);
}
