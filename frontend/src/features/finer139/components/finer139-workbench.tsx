"use client";

import { IconPlayerPlay } from "@tabler/icons-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Finer139ExamplesPanel } from "@/features/finer139/components/finer139-examples-panel";
import { Finer139ResultsTable } from "@/features/finer139/components/finer139-results-table";
import {
	METHOD_OPTIONS,
	useFiner139Run,
} from "@/features/finer139/hooks/use-finer139-run";
import { ProviderModelSelect } from "@/features/model-compare/components/provider-model-select";

const PROVIDER_LABELS: Record<string, string> = {
	openai: "OpenAI",
	ollama: "Ollama",
	nim: "NVIDIA NIM",
	gemini: "Google Gemini",
	lmstudio: "LM Studio",
};

export function Finer139Workbench() {
	const {
		providers,
		providersLoading,
		sampleSize,
		setSampleSize,
		seed,
		setSeed,
		methods,
		toggleMethod,
		llmValue,
		setLlmValue,
		run,
		isRunning,
		error,
		progressPct,
		startRun,
	} = useFiner139Run();

	const results = run?.results;
	const busy = isRunning;

	return (
		<div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30 p-4">
			<div className="mx-auto flex h-full w-full max-w-[1200px] flex-col gap-4 overflow-y-auto">
				<Card className="shrink-0">
					<CardHeader>
						<CardTitle className="text-xl">
							FiNER-139 Graph Construction
						</CardTitle>
						<CardDescription>
							Compare five graph-construction methods on entity recognition in
							the FiNER-139 financial dataset. Scored as type-agnostic numeric
							span detection (precision / recall / F1).
						</CardDescription>
					</CardHeader>
					<CardContent className="space-y-6">
						<FieldGroup className="grid gap-4 md:grid-cols-2">
							<Field>
								<FieldLabel htmlFor="sample-size">Sample size</FieldLabel>
								<Input
									id="sample-size"
									type="number"
									min={1}
									max={500}
									value={sampleSize}
									disabled={busy}
									onChange={(e) =>
										setSampleSize(
											Math.min(500, Math.max(1, Number(e.target.value) || 1)),
										)
									}
								/>
							</Field>
							<Field>
								<FieldLabel htmlFor="seed">Random seed</FieldLabel>
								<Input
									id="seed"
									type="number"
									min={0}
									value={seed}
									disabled={busy}
									onChange={(e) => setSeed(Number(e.target.value) || 0)}
								/>
							</Field>
						</FieldGroup>

						<div>
							<p className="mb-2 text-sm font-medium text-muted-foreground">
								Methods
							</p>
							<div className="flex flex-wrap gap-2">
								{METHOD_OPTIONS.map((opt) => {
									const on = methods.includes(opt.id);
									return (
										<Button
											key={opt.id}
											type="button"
											size="sm"
											variant={on ? "default" : "outline"}
											disabled={busy}
											onClick={() => toggleMethod(opt.id)}
										>
											{opt.label}
										</Button>
									);
								})}
							</div>
						</div>

						<div className="max-w-md">
							<ProviderModelSelect
								modelRole="llm"
								title="LLM (for LLM / Hybrid / Dynamic)"
								providers={providers}
								isLoading={providersLoading}
								value={llmValue}
								onValueChange={(v) => setLlmValue(v ?? undefined)}
								providerLabels={PROVIDER_LABELS}
								disabled={busy}
							/>
						</div>

						{busy && (
							<div className="space-y-2">
								<div className="flex items-center gap-2 text-sm text-muted-foreground">
									<Spinner className="size-4" />
									<span>
										{run?.progress?.message ?? "Running benchmark..."}
									</span>
								</div>
								{progressPct !== undefined && (
									<Progress value={progressPct} className="h-2" />
								)}
							</div>
						)}

						{error && (
							<Alert variant="destructive">
								<AlertDescription>{error}</AlertDescription>
							</Alert>
						)}
					</CardContent>
					<CardFooter>
						<Button
							type="button"
							disabled={busy || methods.length === 0}
							onClick={() => void startRun()}
						>
							<IconPlayerPlay className="mr-2 size-4" />
							{busy ? "Running..." : "Run benchmark"}
						</Button>
					</CardFooter>
				</Card>

				{results && (
					<>
						<Card>
							<CardHeader className="pb-2">
								<CardTitle className="text-base">Dataset</CardTitle>
								<CardDescription>
									{results.dataset.num_sentences} sentences,{" "}
									{results.dataset.num_gold_entities} gold numeric entities (
									{results.dataset.split} split)
								</CardDescription>
							</CardHeader>
						</Card>
						<Finer139ResultsTable methods={results.methods} />
						<Finer139ExamplesPanel examples={results.examples} />
					</>
				)}
			</div>
		</div>
	);
}
