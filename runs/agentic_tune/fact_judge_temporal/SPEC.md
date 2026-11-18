# Temporal gold facts, Claude Haiku judge: finish run

> complete the judging and update the slides

dataset: MultiHop-RAG held-out 400, temporal_query, 100 questions
methods: Normal RAG r1-r3, Agentic GraphRAG 2 Nov r1-r3, Agentic GraphRAG 10 Nov r1-r3 (temporal100, _rep2, _rep3)
judge model: haiku (Claude Code Agent subagent, general-purpose), one subagent per file
prompt version: 2026-11-10.haiku.2 (unchanged; guide inside each items file)
files to judge: judge/items-132.md .. items-189.md (58); items-000..131 already judged 11 Nov 12:07-12:33
done when: 190 verdict files, `score` passes its id/fact checks, slide 05 gold-fact cells and matrices filled
