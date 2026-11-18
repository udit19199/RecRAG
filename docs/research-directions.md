# Research directions

The paper asks what drives Agentic Hybrid RAG's gains over plain RAG: wider
search, the agent choosing its evidence, or the graph. This note describes the
two runs the question rests on, then the plan to answer it. The plan started as
a team sketch ("Selection, Search, or Structure?"); every number here is checked
against the run data, and where the sketch disagrees with the data, the data
wins (see the last section).

## The two runs

Both runs answer with GPT-6 Luna (medium reasoning), use the same
256-token chunks, the same indexes and graphs (built once), and the same
answer prompt. Each method passes at most 10 items to the answer step.

| | Previous run | Latest run |
| --- | --- | --- |
| Folder | `runs/heldout400/` | `runs/final500/` (`run_registry.json` lists its run folders) |
| Questions | 400 MultiHop-RAG, 100 per type | 500 MultiHop-RAG (the 400 + 100 new, 125 per type), 500 HotpotQA (250 bridge, 250 comparison), 500 Natural Questions |
| Repeats | 3 | 3 |
| Normal RAG, RAPTOR, HippoRAG | Answered all 400 | Reused their 400 previous answers and retrieved text, answered the 100 new MultiHop-RAG questions and both new datasets |
| Agentic Hybrid RAG search | Vector search with the question, one graph query on the names found, then search again or stop (up to 5 model calls) | Same |
| Which 10 items reach the answer step | A fixed rule: top passage of the first search, top graph row, then the rest of the first search. Later searches almost never got in. | The agent lists the tagged results the answer needs; those go first, the best-scoring passages fill the rest. Agent picked on 4,500 of 4,500 answers. |
| Graph query rule | Find passages through `FROM_CHUNK`; rows had no size limit (largest about 66,000 tokens) | Graph facts only; a query reading article text is refused; lists cut to 20 items |
| What the answer step reads | Each passage as JSON with its entity names and graph facts | Plain passage text, like Normal RAG |
| Agent trace saved | No (only the final items), so its graph-query failures cannot be counted | Yes: 337 of 4,569 graph queries failed (7.4%), 1,121 returned nothing (24.5%) |
| Answered right | Exact match, then Claude Haiku | Exact match, then a fresh Claude Haiku judge |
| Gold facts found | GPT-6 Luna splits and checks facts | Claude Haiku splits and checks facts; Claude Opus agreed on 196 of 200 |

What changed for Agentic Hybrid RAG, MultiHop-RAG answered right: comparison
50.3% → 70.4%, temporal 56.0% → 76.5%. Wrong "Insufficient Information"
answers fell from 24.0% to 9.1% (comparison) and 37.0% to 11.5% (temporal),
while Normal RAG stayed flat (32.0% → 31.7%, 45.7% → 44.3%). On the shared
questions the new judges moved the other methods by under 3 points. Sources:
`runs/final500/two_run_analysis/SUMMARY.md`, `runs/final500/cypher_failures/SUMMARY.md`.

Two caveats the paper has to carry:
- **Selection and graph format changed together.** Between the runs the
  agent started picking its evidence, and graph rows were cut to facts and
  passages sent as plain text. Trimming alone barely moved answers on 38 hard
  comparison questions (26, 28, 27 right before; 27, 26, 25 after), which
  points at selection, but the 2×2 below is what separates them.
- **Tuning overlap.** The agent-picks change was developed on 38 previous-run
  comparison questions, after the 400 previous-run results had been analysed.
  Natural Questions rows 0–99 were the tuning pilot, and 27 HotpotQA questions
  appeared in older runs. Questions never used in tuning: the 100 new
  MultiHop-RAG, 473 HotpotQA and 400 Natural Questions
  (`runs/final500/tuning_questions.json`, DECISIONS 20).

## Paper plan: selection, search or structure

### Research questions

- **RQ1, attribution:** when Agentic Hybrid RAG beats Normal RAG, how much comes
  from wider search, from the agent choosing its evidence, and from the graph?
- **RQ2, where it helps:** comparison and temporal versus inference, null and
  single-hop questions.
- **RQ3, cost:** cost per correct answer including builds, and latency.
- **RQ4, diagnosis:** why the graph helps or not: how often it is used, how
  often its queries fail, and whether its results reach the answer.

### Core experiment: graph × selection

Every variant runs the same agent and sends at most 10 items to the same
answer step; only what sits inside the agent and how the 10 are chosen change.

| Inside the agent | Fixed top-10 rule | Agent picks by tag |
| --- | --- | --- |
| Vector only | A0 | A1 |
| Vector + graph | A2 (previous run's selection) | A3 (latest run) |
| Vector + cleaned graph | A2c | A3c |

- Selection effect = A1 − A0; graph effect = A3 − A1; graph quality effect =
  A3c − A3; interaction: does the graph help only when the agent picks?
- A2 is the previous run's rule on the latest run's graph format, so the
  previous run itself is not a clean A2 (see the caveat above). A2 and A1 can
  be replays: the agent's saved searches re-selected under the other rule,
  without new searches.

### Variant ladder, from plain RAG to the full method

| Variant | Wider search | Selection | Graph |
| --- | --- | --- | --- |
| B0 Normal RAG, top 10 | no | no | no |
| B1 + reranker, 50 → 10 | no | reranker | no |
| B2 + LLM selection, 50 → 10 | no | GPT-6 Luna | no |
| B3 several searches + selection | yes | GPT-6 Luna | no |
| A0 agent, vector, top 10 | yes | no | no |
| A1 agent, vector, picks | yes | agent | no |
| A2 agent, hybrid, top 10 | yes | no | yes |
| A3 agent, hybrid, picks | yes | agent | yes |
| A3 with the general ontology | yes | agent | yes |

### Measures

The deck's rules apply: answered right (exact match, then Claude Haiku) and
gold facts found at ≥50%, ≥75% and 100%, judged by Claude Haiku; tokens sent
to the answer step and seen by the agent; cost per correct answer including
builds; latency. Main comparisons are fixed before running; paired bootstrap
over questions, 3 repeats, Holm correction. The deck shows every result per
question type; whether the paper also reports pooled numbers is an open
question below.

### Evidence funnel (headline figure)

Where the agent loses evidence: gold facts in everything the agent saw (about
50 candidates) → gold facts in the 10 items kept (selection loss) → answered
right (answer loss). The latest run saves the full trace, so this can be
computed from saved runs without new model calls; the previous run cannot,
because it saved only the final 10 items.

### Robustness

- **Graph quality:** the current graph against a cleaned one (one string per
  name, entity resolution, de-duplication). The current graph has list-valued
  names (88 entities in the October check) and duplicate outlets (315
  Publisher nodes for 49 outlets); in the latest run, `toLower` on a list
  value caused 104 of 337 graph-query failures. Report graph-query success for
  both graphs, so a weak graph cannot explain a small graph effect.
- **Selector size:** GPT-6 Luna against a smaller model and the cross-encoder,
  to test whether selection works only because a strong model half-solves the
  question.

### Data and models

MultiHop-RAG (all four types), HotpotQA and Natural Questions as a single-hop
control; possibly MuSiQue or 2WikiMultiHopQA, since inference and null sit near
95% for every method and cannot separate them. A development set for tuning and
a fresh test set nobody looks at. GPT-6 Luna plus one open model; HippoRAG and
RAPTOR as outside reference points.

### Next steps

1. Trace check and evidence funnel from the latest run's saved traces.
2. Replays: the fixed top-10 rule and picks without the graph, on saved searches.
3. Matched baselines (B1–B3), the cleaned graph, a smaller selector.
4. A fresh test set, and agree the main comparisons before running it.
5. Run once on the test set.
6. Write up; release question ids and run files.

### Open questions

- The exact wording of the research question.
- Whether the paper reports pooled numbers as well as per type (the deck
  never does).
- Which open model can run the agent.
- Workshop or findings track, and the deadline.
- Answered from the data: the latest run does overlap tuning (38 MultiHop-RAG
  comparison questions, Natural Questions rows 0–99, 27 HotpotQA questions;
  see the caveat above).

### Positioning

[RAG vs. GraphRAG](https://arxiv.org/abs/2502.11371) finds reranking helps
every method; [SetR](https://aclanthology.org/2025.acl-long.861/) studies
choosing a set of passages instead of ranking them; broad comparisons are
covered by [GraphRAG-Bench](https://arxiv.org/abs/2506.05690) and [Dissecting
GraphRAG](https://aclanthology.org/2026.tacl-1.29/). Our angle: the agent's own
selection, separated from the graph in a 2×2, with full cost on held-out
questions.

## Where the sketch and the data disagree

- **"~40% Cypher failures".** That figure is from a 40-question study before
  the previous run, on an older graph. The latest run: 7.4% failed, 24.5%
  returned nothing. The previous run has no trace to count.
- **"~3% graph use under free choice".** Also from that early study (first 100
  temporal questions). Both runs force one graph query after the first vector
  search, so neither measures free choice.
- **"Graph results never reached the answer before selection".** True for the
  test version in between (graph rows only filled empty slots: 0 of 300
  comparison runs). The previous run's rule put the top graph row in slot 2
  (796 of its retrievals), so graph rows did reach the answer there.
- **"Routing by question type matched the agent at about 1/10 the cost".**
  From the previous run (75.3% vs 74.8%, assuming the type is known). Not yet
  redone on the latest run, where Agentic Hybrid RAG leads on comparison and
  temporal.

## Other directions, not in the current plan

- **Typed evidence decisions (RLCD).** Test whether a calibrated decision model
  such as TypeSafe's Jev judges graph evidence (relevant, sufficient,
  contradictory) better and more cheaply than an LLM judge, with GPT-6 Luna
  still writing the answer. Calibration must be measured, not assumed. Jev is
  an early-access hosted API.
- **Ontology × retrieval interaction.** Ontology-guided construction may help
  retrieval that uses types and relations and do little for vector retrieval.
  Run every construction choice with every retrieval choice at a fixed
  context budget and test the interaction rather than pick one winner. The
  "A3 with the general ontology" variant above is a first slice of this.
