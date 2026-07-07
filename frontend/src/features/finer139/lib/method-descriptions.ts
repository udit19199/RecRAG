import type { Finer139Method } from "@/lib/api/types";

export type MethodDescription = {
	id: Finer139Method;
	label: string;
	shortLabel: string;
	tagline: string;
	description: string;
	needsLlm: boolean;
};

/** Graph *construction* methods — not retrieval hybrid (see /graphrag). */
export const CONSTRUCTION_METHODS: MethodDescription[] = [
	{
		id: "llm",
		label: "LLM-Based (open-ended)",
		shortLabel: "LLM",
		tagline: "Open-ended LLM extraction",
		description:
			"Prompts an LLM to extract entities and relations from each sentence with no schema constraint. Flexible and high recall for implicit relationships, but can hallucinate entities or over-extract non-numeric spans.",
		needsLlm: true,
	},
	{
		id: "nlp",
		label: "NLP / OpenIE (spaCy)",
		shortLabel: "NLP",
		tagline: "Traditional NER + OpenIE",
		description:
			"Uses spaCy to tag MONEY, PERCENT, CARDINAL, and related labels. Deterministic, fast, and offline — but misses domain-specific XBRL concepts and produces many overlapping spans.",
		needsLlm: false,
	},
	{
		id: "ontology",
		label: "Ontology / Schema-Driven",
		shortLabel: "Ontology",
		tagline: "XBRL keyword gazetteer",
		description:
			"Matches CamelCase XBRL concept keywords in context, then flags nearby numeric tokens. Highly consistent for regulated financial text; strong recall without any LLM cost.",
		needsLlm: false,
	},
	{
		id: "hybrid",
		label: "Hybrid Construction (Schema-Guided LLM)",
		shortLabel: "Hybrid construction",
		tagline: "Schema-guided LLM — not retrieval hybrid",
		description:
			"Combines the full 139-concept XBRL schema with LLM reasoning, then filters to numeric spans. This is a graph construction technique — not the same as hybrid retrieval (vector + graph at query time). See the GraphRAG page for retrieval hybrid.",
		needsLlm: true,
	},
	{
		id: "dynamic",
		label: "Dynamic / Incremental",
		shortLabel: "Dynamic",
		tagline: "Memory-augmented construction",
		description:
			"Wraps Hybrid construction with a memory layer that learns context words from prior hits and flags nearby numerics incrementally. Designed for evolving graphs, not one-shot batch extraction.",
		needsLlm: true,
	},
];

export const CONSTRUCTION_BY_ID = Object.fromEntries(
	CONSTRUCTION_METHODS.map((m) => [m.id, m]),
) as Record<Finer139Method, MethodDescription>;
