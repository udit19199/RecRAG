"""Per-question-type results for the held-out study (reuses heldout_analysis scoring)."""
import json, random, statistics as st, sys, io, contextlib
from collections import defaultdict
with contextlib.redirect_stdout(io.StringIO()):
    import runpy
    A = runpy.run_path("runs/heldout400/scripts/heldout_analysis.py")
runs, reps, NAMES, BUILD = A["runs"], A["reps"], A["NAMES"], A["BUILD"]
TYPES = ["comparison_query", "temporal_query", "inference_query", "null_query"]
qtype = {q: row["type"] for (t, m), rows in runs.items() for q, row in rows.items()}
out = {"types": []}
rnd = random.Random(1)
for qt in TYPES:
    block = {"type": qt, "methods": [], "paired": []}
    per_q = {}
    for m in NAMES:
        rows = [r for t in reps for q, r in runs[(t, m)].items() if qtype[q] == qt]
        lat = [r for q, r in runs[("latency", m)].items() if qtype[q] == qt]
        quality_lat = [r["sec"] for r in rows]
        rec = [r["recall"] for r in rows if r["recall"] is not None]
        acc = st.mean(r["correct"] for r in rows)
        usd = st.mean(r["usd"] for r in rows)
        block["methods"].append(dict(
            method=NAMES[m], answers=len(rows), accuracy=acc,
            accuracy_by_repeat=[st.mean(r["correct"] for q, r in runs[(t, m)].items() if qtype[q] == qt) for t in reps],
            retrieval=st.mean(rec) if rec else None, context_tokens=st.mean(r["ctx"] for r in rows),
            query_usd=usd, usd_per_1000_correct=1000 * (usd + BUILD[m] / 2556) / acc if acc else None,
            latency_median_1worker=st.median(r["sec"] for r in lat) if lat else None, latency_n=len(lat),
            latency_median_4workers=st.median(quality_lat), abstain=st.mean(r["abstain"] for r in rows),
            model_calls=None))
        pq = defaultdict(list)
        for t in reps:
            for q, r in runs[(t, m)].items():
                if qtype[q] == qt:
                    pq[q].append(r)
        per_q[m] = pq
    for key in ("correct", "recall"):
        for b in ("rag", "hipporag", "raptor"):
            qs = [q for q in per_q["graphrag"] if q in per_q[b] and per_q["graphrag"][q][0][key] is not None]
            if not qs:
                continue
            d = [st.mean(r[key] for r in per_q["graphrag"][q]) - st.mean(r[key] for r in per_q[b][q]) for q in qs]
            boots = sorted(st.mean(rnd.choice(d) for _ in d) for _ in range(4000))
            block["paired"].append(dict(metric=key, vs=NAMES[b], n=len(qs), diff=st.mean(d), lo=boots[100], hi=boots[3899],
                                        wins=sum(x > 0 for x in d), losses=sum(x < 0 for x in d)))
    out["types"].append(block)
json.dump(out, open("runs/heldout400/analysis_by_type.json", "w"), indent=1)
for b in out["types"]:
    print(f"\n=== {b['type']}")
    for r in b["methods"]:
        rt = f"{r['retrieval']:.1%}" if r["retrieval"] is not None else "  n/a"
        print(f"{r['method']:17} acc={r['accuracy']:.1%} reps={[round(x*100,1) for x in r['accuracy_by_repeat']]} retr={rt} ctx={r['context_tokens']:.0f} "
              f"q$={r['query_usd']*1000:.2f}/1k $/1kcorrect={r['usd_per_1000_correct']:.2f} lat1w={r['latency_median_1worker']:.1f}s(n={r['latency_n']}) lat4w={r['latency_median_4workers']:.1f}s abstain={r['abstain']:.0%}")
    for p in b["paired"]:
        print(f"  {p['metric']:7} vs {p['vs']:9} n={p['n']} {p['diff']*100:+.1f}pts 95%CI[{p['lo']*100:+.1f},{p['hi']*100:+.1f}] wins={p['wins']} losses={p['losses']}")
