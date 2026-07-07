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
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import { Finer139DiagnosticsPanel } from "@/features/finer139/components/finer139-diagnostics-panel";
import { Finer139ExamplesPanel } from "@/features/finer139/components/finer139-examples-panel";
import {
	FIELD_HELP,
	Finer139FieldHelp,
	Finer139IntroLink,
} from "@/features/finer139/components/finer139-field-help";
import { Finer139MethodCards } from "@/features/finer139/components/finer139-method-cards";
import { Finer139ResultsTable } from "@/features/finer139/components/finer139-results-table";
import { useFiner139Run } from "@/features/finer139/hooks/use-finer139-run";
import { ProviderModelSelect } from "@/features/model-compare/components/provider-model-select";

const PROVIDER_LABELS: Record<string, string> = {
	openai: "OpenAI",
	ollama: "Ollama",
	nim: "NVIDIA NIM",
	gemini: "Google Gemini",
	lmstudio: "LM Studio",
};

export function Finer139Workbench({
	embedded = false,
	onOpenRetrieval,
}: {
	embedded?: boolean;
	onOpenRetrieval?: () => void;
} = {}) {
	const {
		providers,
		providersLoading,
		sampleSize,
		setSampleSize,
		seed,
		setSeed,
		split,
		setSplit,
		stratified,
		setStratified,
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
	const needsLlm = methods.some(
		(m) => m === "llm" || m === "hybrid" || m === "dynamic",
	);

	return (
		<div
			className={
				embedded
					? "flex h-full min-h-0 w-full flex-col overflow-y-auto p-4"
					: "flex h-full min-h-0 w-full flex-col overflow-hidden bg-muted/30 p-4"
			}
		>
			<div className="mx-auto flex h-full w-full max-w-[1200px] flex-col gap-4 overflow-y-auto">
				<Card className="shrink-0">
					<CardHeader>
						<CardTitle className={embedded ? "text-lg" : "text-xl"}>
							FiNER-139 graph construction
						</CardTitle>
						<CardDescription>
							Compare five graph-construction methods on entity recognition in
							the FiNER-139 financial dataset. Scored as type-agnostic numeric
							span detection (precision / recall / F1).
						</CardDescription>
						<Finer139IntroLink onOpenRetrieval={onOpenRetrieval} />
					</CardHeader>
					<CardContent className="space-y-6">
						<FieldGroup className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
							<Finer139FieldHelp
								htmlFor="sample-size"
								label="Sample size"
								help={FIELD_HELP.sampleSize}
							>
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
							</Finer139FieldHelp>
							<Finer139FieldHelp
								htmlFor="seed"
								label="Random seed"
								help={FIELD_HELP.seed}
							>
								<Input
									id="seed"
									type="number"
									min={0}
									value={seed}
									disabled={busy}
									onChange={(e) => setSeed(Number(e.target.value) || 0)}
								/>
							</Finer139FieldHelp>
							<Finer139FieldHelp
								htmlFor="split"
								label="Data split"
								help={FIELD_HELP.split}
							>
								<select
									id="split"
									className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs"
									value={split}
									disabled={busy}
									onChange={(e) =>
										setSplit(e.target.value as "train" | "validation" | "test")
									}
								>
									<option value="validation">Validation</option>
									<option value="test">Test (held-out)</option>
									<option value="train">Train</option>
								</select>
							</Finer139FieldHelp>
							<Finer139FieldHelp
								htmlFor="stratified"
								label="Stratified sampling"
								help={FIELD_HELP.stratified}
							>
								<label className="flex h-9 items-center gap-2 text-sm">
									<input
										id="stratified"
										type="checkbox"
										checked={stratified}
										disabled={busy}
										onChange={(e) => setStratified(e.target.checked)}
									/>
									By primary XBRL concept
								</label>
							</Finer139FieldHelp>
						</FieldGroup>

						<Finer139MethodCards
							selected={methods}
							onToggle={toggleMethod}
							disabled={busy}
							onOpenRetrieval={onOpenRetrieval}
						/>

						{needsLlm && (
							<div className="max-w-md space-y-2">
								<ProviderModelSelect
									modelRole="llm"
									title="LLM (for LLM / Hybrid Construction / Dynamic)"
									providers={providers}
									isLoading={providersLoading}
									value={llmValue}
									onValueChange={(v) => setLlmValue(v ?? undefined)}
									providerLabels={PROVIDER_LABELS}
									disabled={busy}
								/>
								<p className="text-xs text-muted-foreground">
									Requires a configured provider API key in{" "}
									<code className="text-[11px]">.env</code> (e.g. OpenAI).
								</p>
							</div>
						)}

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
					<CardFooter className="justify-end">
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
									{results.dataset.split} split
									{results.evaluation?.protocol_version
										? ` · protocol v${results.evaluation.protocol_version}`
										: ""}
									)
								</CardDescription>
							</CardHeader>
						</Card>
						<Finer139ResultsTable methods={results.methods} />
						<Finer139DiagnosticsPanel
							methods={results.methods}
							comparison={results.comparison}
						/>
						<Finer139ExamplesPanel examples={results.examples} />
					</>
				)}
			</div>
		</div>
	);
}
