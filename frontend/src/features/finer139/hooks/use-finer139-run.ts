"use client";

import { useCallback, useEffect, useState } from "react";
import { parseValue } from "@/features/model-compare/lib";
import { getProviders, startFiner139Run, waitForFiner139Run } from "@/lib/api";
import type {
	Finer139Method,
	Finer139RunResponse,
	ProvidersResponse,
} from "@/lib/api/types";

export const METHOD_OPTIONS: { id: Finer139Method; label: string }[] = [
	{ id: "llm", label: "LLM-Based (open-ended)" },
	{ id: "nlp", label: "NLP / OpenIE (spaCy)" },
	{ id: "ontology", label: "Ontology / Schema-Driven" },
	{ id: "hybrid", label: "Hybrid (Schema-Guided LLM)" },
	{ id: "dynamic", label: "Dynamic / Incremental" },
];

const DEFAULT_METHODS: Finer139Method[] = METHOD_OPTIONS.map((m) => m.id);
const DEFAULT_LLM_PROVIDER = "openai";
const DEFAULT_LLM_MODEL = "gpt-4o-mini";

function pickDefaultLlmValue(prov: ProvidersResponse): string | undefined {
	const openai = prov.llms.openai;
	if (openai?.available && openai.models.length > 0) {
		const preferred = openai.models.includes(DEFAULT_LLM_MODEL)
			? DEFAULT_LLM_MODEL
			: openai.models[0];
		return `${DEFAULT_LLM_PROVIDER}::${preferred}`;
	}
	for (const [providerKey, info] of Object.entries(prov.llms)) {
		if (info.available && info.models.length > 0) {
			return `${providerKey}::${info.models[0]}`;
		}
	}
	return undefined;
}

export function useFiner139Run() {
	const [providers, setProviders] = useState<ProvidersResponse | null>(null);
	const [providersLoading, setProvidersLoading] = useState(true);
	const [sampleSize, setSampleSize] = useState(100);
	const [seed, setSeed] = useState(42);
	const [methods, setMethods] = useState<Finer139Method[]>(DEFAULT_METHODS);
	const [llmValue, setLlmValue] = useState<string | undefined>(undefined);
	const [run, setRun] = useState<Finer139RunResponse | null>(null);
	const [isRunning, setIsRunning] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		void (async () => {
			try {
				const prov = await getProviders();
				if (!cancelled) {
					setProviders(prov);
					setLlmValue(pickDefaultLlmValue(prov));
				}
			} catch (err) {
				if (!cancelled) {
					setError(
						err instanceof Error ? err.message : "Failed to load providers",
					);
				}
			} finally {
				if (!cancelled) setProvidersLoading(false);
			}
		})();
		return () => {
			cancelled = true;
		};
	}, []);

	const toggleMethod = useCallback((id: Finer139Method) => {
		setMethods((prev) =>
			prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id],
		);
	}, []);

	const startRun = useCallback(async () => {
		if (methods.length === 0) {
			setError("Select at least one method");
			return;
		}
		setError(null);
		setIsRunning(true);
		setRun(null);

		const parsed = llmValue ? parseValue(llmValue) : null;

		try {
			const { run_id } = await startFiner139Run({
				sample_size: sampleSize,
				seed,
				methods,
				provider: parsed?.provider ?? null,
				model: parsed?.model ?? null,
			});
			const final = await waitForFiner139Run(run_id, 1500, 600_000, setRun);
			setRun(final);
		} catch (err) {
			setError(err instanceof Error ? err.message : "Run failed");
		} finally {
			setIsRunning(false);
		}
	}, [llmValue, methods, sampleSize, seed]);

	const progressPct =
		run?.progress && run.progress.total > 0
			? Math.round((run.progress.current / run.progress.total) * 100)
			: run?.status === "complete"
				? 100
				: undefined;

	return {
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
	};
}
