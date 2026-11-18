"""Pick 100 held-out MultiHop-RAG questions per type, frozen before any run.
Excludes every question id that appears in any saved run (tuning, baselines, smoke tests)."""
import glob, json, random, re, collections
from dataset import load_multihop_rag_record, _multihop_queries
used = set()
for f in glob.glob("runs/**/retrieval.jsonl", recursive=True) + glob.glob("runs/**/results.jsonl", recursive=True):
    for line in open(f):
        m = re.search(r'"record_id":\s*"multihop-rag-(\d+)"', line)
        if m: used.add(int(m.group(1)))
n = len(_multihop_queries())
by = collections.defaultdict(list)
for i in range(n):
    if i not in used:
        by[load_multihop_rag_record(i).question_type].append(i)
rnd = random.Random(20261102)
picked = {t: sorted(rnd.sample(v, 100)) for t, v in sorted(by.items())}
# Interleave types so a partial run still covers every type evenly.
order = [i for group in zip(*picked.values()) for i in group]
json.dump({"seed": 20261102, "excluded_used": len(used), "available": {t: len(v) for t, v in by.items()},
           "by_type": picked, "order": order}, open("runs/heldout400/questions.json", "w"), indent=1)
print(len(used), "used;", {t: len(v) for t, v in by.items()}, "->", len(order))
