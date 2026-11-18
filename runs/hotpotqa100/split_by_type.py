import json
qs=json.load(open('datasets/hotpotqa/hotpotqa_distractor_validation_hard_bridge_1000_seed_42.json'))[:100]
T={**{i:'temporal' for i in (2,6,17,31,51,75,78,85)},**{i:'comparison' for i in (18,25,54)}}
s=json.load(open('runs/hotpotqa100/scores_per_question.json'))
M=['rag','raptor','hipporag','graphrag']
res={}
for t in ['comparison','temporal','inference','null']:
    ids=[q['id'] for i,q in enumerate(qs) if T.get(i,'inference')==t]
    res[t]={'n':len(ids)}
    for m in M:
        if not ids: continue
        v=[s[f'{m}|{r}'] for r in ids]
        res[t][m]={'support':round(100*sum(x[0][-1] for x in v)/(3*len(v))),'full_support':round(100*sum(x[0][-1]==3 for x in v)/len(v)),'gold':round(100*sum(x[1] for x in v)/(3*len(v)))}
print(json.dumps(res,indent=0))
json.dump(res,open('runs/hotpotqa100/scores_by_type.json','w'))
